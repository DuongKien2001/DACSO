import argparse
import os
import sys
import random
import timeit
import datetime
import copy 
import numpy as np
import pickle
import scipy.misc
from core.configs import cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from core.utils.prototype_dist_estimator import prototype_dist_estimator

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from core.utils.loss import PrototypeContrastiveLoss

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate

import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    return parser.parse_args()


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('dacs/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('dacs/', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model
"""
def prototype_dist_init(cfg, trainloader, model):
    feature_num = 2048
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    torch.cuda.empty_cache()

    iteration = 0
    model.eval()
    end = time.time()
    start_time = time.time()
    max_iters = len(trainloader)
    
    with torch.no_grad():
        for i, (src_input, src_label, _, _) in enumerate(trainloader):
            data_time = time.time() - end

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            src_out, src_feat = model(src_input)
            B, N, Hs, Ws = src_feat.size()
            _, C, _, _ = src_out.size()

            # source mask: downsample the ground-truth label
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )

            # feature level
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_estimator.update(features=src_feat.detach().clone(), labels=src_mask)

            # output level
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, C)
            out_estimator.update(features=src_out.detach().clone(), labels=src_mask)

            batch_time = time.time() - end
            end = time.time()
            #meters.update(time=batch_time, data=data_time)

            iteration = iteration + 1
            #eta_seconds = meters.time.global_avg * (max_iters - iteration)
            #eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            if iteration == max_iters:
                feat_estimator.save(name='prototype_feat_dist.pth')
                out_estimator.save(name='prototype_out_dist.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_time / max_iters))
    """
def main():
    print(config)
    list_name = []
    best_mIoU = 0
    feature_num = 2048

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)
    
    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    
    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    
    
    cudnn.benchmark = True
    """
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    pcl_criterion_src = PrototypeContrastiveLoss(cfg)
    pcl_criterion_tgt = PrototypeContrastiveLoss(cfg)
    """
    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        np.random.seed(random_seed)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    #New loader for Domain transfer
    a = ['10812.png', '15152.png', '21507.png', '24780.png', '08400.png', '20708.png', '13741.png', '16527.png', '22181.png', '10446.png', '17969.png', '08446.png', '06923.png', '16403.png', '19947.png', '24117.png', '06042.png', '14311.png', '19472.png', '24965.png', '18084.png', '07288.png', '14292.png', '09999.png', '20720.png', '14025.png', '19915.png', '17904.png', '16131.png', '08130.png', '14263.png', '19542.png', '07578.png', '02067.png', '08846.png', '21529.png', '13165.png', '19708.png', '17316.png', '00542.png', '10438.png', '24534.png', '04642.png', '23550.png', '16893.png', '13658.png', '22140.png', '15532.png', '04385.png', '03915.png', '10443.png', '05884.png', '04201.png', '19515.png', '08473.png', '10569.png', '01671.png', '00333.png', '24630.png', '02946.png', '20017.png', '22883.png', '16532.png', '13011.png', '16115.png', '14799.png', '20570.png', '21010.png', '14122.png', '21444.png', '21509.png', '17862.png', '19921.png', '16024.png', '07574.png', '24707.png', '23902.png', '21729.png', '18652.png', '02217.png', '11962.png', '13896.png', '23042.png', '15881.png', '16372.png', '12661.png', '02278.png', '05274.png', '03756.png', '11477.png', '19668.png', '11976.png', '04892.png', '21130.png', '04869.png', '06815.png', '03854.png', '20602.png', '01542.png', '17225.png', '06542.png', '17362.png', '15135.png', '05480.png', '18494.png', '02706.png', '03000.png', '21988.png', '22526.png', '09230.png', '08206.png', '18640.png', '04859.png', '02740.png', '23178.png', '08363.png', '04789.png', '05293.png', '21916.png', '01023.png', '16582.png', '24174.png', '05693.png', '15504.png', '23126.png', '14901.png', '01637.png', '05359.png', '00280.png', '15997.png', '13549.png', '20055.png', '09783.png', '08325.png', '23183.png', '22488.png', '02275.png', '18673.png', '20787.png', '07246.png', '18920.png', '14023.png', '07678.png', '04458.png', '08239.png', '19154.png', '22784.png', '15461.png', '15444.png', '16130.png', '08850.png', '15255.png', '03198.png', '23450.png', '05505.png', '00793.png', '16906.png', '07733.png', '23199.png', '13533.png', '04324.png', '15009.png', '12192.png', '06624.png', '19406.png', '01308.png', '03375.png', '24138.png', '10251.png', '22626.png', '10195.png', '02151.png', '07264.png', '01479.png', '23737.png', '18768.png', '16711.png', '23272.png', '11758.png', '02660.png', '04478.png', '09788.png', '05715.png', '08515.png', '00184.png', '00587.png', '09867.png', '01730.png', '15380.png', '22443.png', '13352.png', '19413.png', '05157.png', '14283.png', '15950.png', '05063.png', '23088.png', '10753.png', '21871.png', '10164.png', '00857.png', '12134.png', '13682.png', '12935.png', '06319.png', '12061.png', '04045.png', '15106.png', '18844.png', '18899.png', '23240.png', '08851.png', '22682.png', '02489.png', '23261.png', '15811.png', '00256.png', '14093.png', '16123.png', '06419.png', '01740.png', '12698.png', '15480.png', '09435.png', '19692.png', '14369.png', '19934.png', '13286.png', '07798.png', '20879.png', '11877.png', '02156.png', '19303.png', '14252.png', '01470.png', '21043.png', '03638.png', '22826.png', '15381.png', '24793.png', '24661.png', '13541.png', '14404.png', '24448.png', '24193.png', '19015.png', '18155.png', '09384.png', '16733.png', '24574.png', '22009.png', '17321.png', '14318.png', '22663.png', '01936.png', '02522.png', '16820.png', '19901.png', '03674.png', '09813.png', '23103.png', '18832.png', '20454.png', '07383.png', '12681.png', '05642.png', '23805.png', '23770.png', '09983.png', '05352.png', '04147.png', '07977.png', '08801.png', '12194.png', '19907.png', '04236.png', '16589.png', '14889.png', '21721.png', '03906.png', '03177.png', '07430.png', '23383.png', '04479.png', '22226.png', '01160.png', '08042.png', '19017.png', '05690.png', '14673.png', '06946.png', '08056.png', '17824.png', '04320.png', '06132.png', '05767.png', '04533.png', '23911.png', '01884.png', '05411.png', '24504.png', '15955.png', '14766.png', '16350.png', '05905.png', '23883.png', '07159.png', '09210.png', '05304.png', '17723.png', '01534.png', '22182.png', '00886.png', '03544.png', '12364.png', '14255.png', '19906.png', '18634.png', '14074.png', '12276.png', '15160.png', '16212.png', '24766.png', '14132.png', '24823.png', '20067.png', '03758.png', '15693.png', '01280.png', '06621.png', '08638.png', '24702.png', '02550.png', '02722.png', '16688.png', '04016.png', '00797.png', '17078.png', '19346.png', '09362.png', '06112.png', '23735.png', '17328.png', '21720.png', '08087.png', '02301.png', '08748.png', '12941.png', '15521.png', '03502.png', '21847.png', '00547.png', '06148.png', '03543.png', '10093.png', '16235.png', '08820.png', '02666.png', '10548.png', '00183.png', '11206.png', '20554.png', '02287.png', '02312.png', '10064.png', '07334.png', '07634.png', '10323.png', '12045.png', '21394.png', '07755.png', '24493.png', '07416.png', '23946.png', '19705.png', '04298.png', '15808.png', '14075.png', '20503.png', '18580.png', '03312.png', '11823.png', '08632.png', '14759.png', '20848.png', '24461.png', '02677.png', '05141.png', '04777.png', '15643.png', '09924.png', '01223.png', '03213.png', '09281.png', '12937.png', '23055.png', '06795.png', '14487.png', '23755.png', '22196.png', '04470.png', '15607.png', '14470.png', '24253.png', '02031.png', '19595.png', '04886.png', '15756.png', '17532.png', '01856.png', '09541.png', '03296.png', '06771.png', '13367.png', '15554.png', '09214.png', '01263.png', '13843.png', '08628.png', '23320.png', '07440.png', '10792.png', '09583.png', '10880.png', '14683.png', '17980.png', '11394.png', '09916.png', '03890.png', '01519.png', '17390.png', '20952.png', '11361.png', '07319.png', '02247.png', '14655.png', '19375.png', '22543.png', '14994.png', '12558.png', '00339.png', '24535.png', '01103.png', '20531.png', '06085.png', '04601.png', '21485.png', '10952.png', '24717.png', '18953.png', '16794.png', '05846.png', '04932.png', '21236.png', '00494.png', '00512.png', '05872.png', '16379.png', '10182.png', '20954.png', '16247.png', '19464.png', '19194.png', '01153.png', '20667.png', '16091.png', '19893.png', '01780.png', '22299.png', '08261.png', '07541.png', '11590.png', '09076.png', '20821.png', '02109.png', '19909.png', '21100.png', '02903.png', '17365.png', '08202.png', '02504.png', '02631.png', '21082.png', '02223.png', '06406.png', '13082.png', '17493.png', '10541.png', '07998.png', '00439.png', '18136.png', '13544.png', '08385.png', '21557.png', '20126.png', '21559.png', '03327.png', '17836.png', '24712.png', '15477.png', '13614.png', '10853.png', '05020.png', '18448.png', '06790.png', '02647.png', '11045.png', '20774.png', '00874.png', '01558.png', '20254.png', '02292.png', '20395.png', '01486.png', '17630.png', '16697.png', '16214.png', '07689.png', '00152.png', '18108.png', '23256.png', '06142.png', '07213.png', '04691.png', '15487.png', '07449.png', '04140.png', '19473.png', '06768.png', '11985.png', '10069.png', '15203.png', '20520.png', '11147.png', '11754.png', '00518.png', '18949.png', '22901.png', '21576.png', '08211.png', '03597.png', '04141.png', '04094.png', '10106.png', '17534.png', '24200.png', '05932.png', '04338.png', '23965.png', '04102.png', '06816.png', '16606.png', '09421.png', '12574.png', '12459.png', '12603.png', '00570.png', '08302.png', '19500.png', '21650.png', '09673.png', '20180.png', '13017.png', '17469.png', '05808.png', '24716.png', '05566.png', '15836.png', '18914.png', '16361.png', '09498.png', '10394.png', '14343.png', '20486.png', '16490.png', '18287.png', '18212.png', '23323.png', '15820.png', '01437.png', '09171.png', '14534.png', '18739.png', '18493.png', '11203.png', '02644.png', '14078.png', '10808.png', '19814.png', '03011.png', '14147.png', '19637.png', '08462.png', '15345.png', '13164.png', '19918.png', '16011.png', '08190.png', '05343.png', '05416.png', '11887.png', '15667.png', '18371.png', '08888.png', '12629.png', '06472.png', '03475.png', '13949.png', '21080.png', '03271.png', '21696.png', '11557.png', '18744.png', '06261.png', '19596.png', '12315.png', '11580.png', '11335.png', '01678.png', '12447.png', '01729.png', '22315.png', '01077.png', '14129.png', '19424.png', '22712.png', '22313.png', '00888.png', '15181.png', '18707.png', '15661.png', '16418.png', '03105.png', '03383.png', '12779.png', '22620.png', '10990.png', '24329.png', '22400.png', '06655.png', '00992.png', '07126.png', '16765.png', '09150.png', '09116.png', '14377.png', '04973.png', '09890.png', '11404.png', '12838.png', '05477.png', '03615.png', '22841.png', '17419.png', '22450.png', '24013.png', '06481.png', '13084.png', '00325.png', '16610.png', '15988.png', '09182.png', '08104.png', '08397.png', '14076.png', '14557.png', '23459.png', '09233.png', '20211.png', '21443.png', '22060.png', '19718.png', '12463.png', '21901.png', '07351.png', '18803.png', '20065.png', '07935.png', '18229.png', '07155.png', '21758.png', '04780.png', '23596.png', '08877.png', '24376.png', '01237.png', '09485.png', '08345.png', '01107.png', '23619.png', '07563.png', '13360.png', '21003.png', '21925.png', '23893.png', '15690.png', '06214.png', '04731.png', '13271.png', '21825.png', '08595.png', '03647.png', '11100.png', '07693.png', '03077.png', '23137.png', '23877.png', '00582.png', '18394.png', '07024.png', '17924.png', '11397.png', '17366.png', '12207.png', '23460.png', '09802.png', '02695.png', '14867.png', '14405.png', '13551.png', '09457.png', '05972.png', '03689.png', '16672.png', '24396.png', '23723.png', '24916.png', '10741.png', '15388.png', '00242.png', '18978.png', '17597.png', '10216.png', '01980.png', '00624.png', '18137.png', '11455.png', '20087.png', '04394.png', '00225.png', '10373.png', '22106.png', '01017.png', '02500.png', '20590.png', '15039.png', '10965.png', '22062.png', '14105.png', '12228.png', '13734.png', '05365.png', '08696.png', '05973.png', '10521.png', '07555.png', '21291.png', '05497.png', '02300.png', '05417.png', '14108.png', '03884.png', '08887.png', '05962.png', '14607.png', '05160.png', '05403.png', '23198.png', '09763.png', '19838.png', '17983.png', '15159.png', '08848.png', '07692.png', '10565.png', '10375.png', '19777.png', '23466.png', '21735.png', '05006.png', '19072.png', '13744.png', '01330.png', '03550.png', '19054.png', '16140.png', '17747.png', '21322.png', '12445.png', '16160.png', '05008.png', '21151.png', '13370.png', '07112.png', '11433.png', '24012.png', '17909.png', '16915.png', '20576.png', '00309.png', '22336.png', '22858.png', '14131.png', '07871.png', '02725.png', '08422.png', '01252.png', '17500.png', '10629.png', '18715.png', '09973.png', '06474.png', '00810.png', '17674.png', '19635.png', '15138.png', '06200.png', '14553.png', '22595.png', '22774.png', '04850.png', '06543.png', '16547.png', '09985.png', '14825.png', '13285.png', '15783.png', '03287.png', '17039.png', '19112.png', '13966.png', '04706.png', '11312.png', '12551.png', '11220.png', '12722.png', '00818.png', '07631.png', '11895.png', '10234.png', '04075.png', '11998.png', '07245.png', '19687.png', '11267.png', '18585.png', '04475.png', '02573.png', '06159.png', '13277.png', '17991.png', '06284.png', '23102.png', '12633.png', '01496.png', '11629.png', '22224.png', '15147.png', '17595.png', '22977.png', '18196.png', '19976.png', '14462.png', '02940.png', '11690.png', '03876.png', '18987.png', '16567.png', '13702.png', '13803.png', '08870.png', '09278.png', '20861.png', '01835.png', '05458.png', '19678.png', '13514.png', '23151.png', '04257.png', '11131.png', '14691.png', '18894.png', '08596.png', '20566.png', '10570.png', '12488.png', '24525.png', '18205.png', '04374.png', '24876.png', '21255.png', '15957.png', '01949.png', '21588.png', '21744.png', '06980.png', '22335.png', '10646.png', '03713.png', '13856.png', '08547.png', '02044.png', '22430.png', '23163.png', '20212.png', '17049.png', '04853.png', '13484.png', '19119.png', '05881.png', '11410.png', '13236.png', '14157.png', '03737.png', '04607.png', '13369.png', '18596.png', '18945.png', '02782.png', '14461.png', '07710.png', '06982.png', '15842.png', '03777.png', '24159.png', '23937.png', '18052.png', '18150.png', '06722.png', '22320.png', '21379.png', '18128.png', '24210.png', '07766.png', '09951.png', '07149.png', '05258.png', '21023.png', '17131.png', '09744.png', '17352.png', '22585.png', '00437.png', '13120.png', '23885.png', '16328.png', '20760.png', '10397.png', '16784.png', '00785.png', '01791.png', '07388.png', '12220.png', '05545.png', '00009.png', '04186.png', '20819.png', '04134.png', '04308.png', '10520.png', '16363.png', '10848.png', '13747.png', '08231.png', '13356.png', '14253.png', '09378.png', '03058.png', '11966.png', '15389.png', '19209.png', '12354.png', '12962.png', '04663.png', '16755.png', '17491.png', '06692.png', '21252.png', '20094.png', '24227.png', '07651.png', '10780.png', '00247.png', '17205.png', '24005.png', '20483.png', '10068.png', '14416.png', '09007.png', '10757.png', '06905.png', '03355.png', '17738.png', '10951.png', '10402.png', '18681.png', '23920.png', '09704.png', '17580.png', '21042.png', '23414.png', '17885.png', '00654.png', '20771.png', '05059.png', '08871.png', '14586.png', '13302.png', '07826.png', '00870.png', '24169.png', '04229.png', '04245.png', '05814.png', '01593.png', '00910.png', '15803.png', '13961.png', '17097.png', '08713.png', '17264.png', '08587.png', '19568.png', '19565.png', '02236.png', '10236.png', '20056.png', '19338.png', '01734.png', '01501.png', '09345.png', '18642.png', '14846.png', '13965.png', '05124.png', '09402.png', '09078.png', '19335.png', '20399.png', '14480.png', '17879.png', '09240.png', '09868.png', '20500.png', '00546.png', '13437.png', '22757.png', '02997.png', '15437.png', '17910.png', '16346.png', '07565.png', '16448.png', '23931.png', '20927.png', '08002.png', '07453.png', '22829.png', '02577.png', '21482.png', '02128.png', '10923.png', '24436.png', '12477.png', '24154.png', '00739.png', '23667.png', '05307.png', '03343.png', '03022.png', '18562.png', '18314.png', '11405.png', '09354.png', '04208.png', '03401.png', '20088.png', '20674.png', '21346.png', '20891.png', '11198.png', '23880.png', '16256.png', '12186.png', '11907.png', '11173.png', '16981.png', '18234.png', '08241.png', '05631.png', '00773.png', '18053.png', '13227.png', '18837.png', '17514.png', '22478.png', '24119.png', '07892.png', '16462.png', '13967.png', '05325.png', '20462.png', '00627.png', '05220.png', '15048.png', '02136.png', '19527.png', '04834.png', '06039.png', '12882.png', '00131.png', '12036.png', '13600.png', '16767.png', '15299.png', '15481.png', '09809.png', '20856.png', '17330.png', '20888.png', '21736.png', '02069.png', '05234.png', '13478.png', '13848.png', '14266.png', '06627.png', '13274.png', '16000.png', '15290.png', '02470.png', '24834.png', '02189.png', '14503.png', '05781.png', '12120.png', '07080.png', '01807.png', '20810.png', '14278.png', '15608.png', '15723.png', '04617.png', '02327.png', '12374.png', '17057.png', '16756.png', '16754.png', '20996.png', '16598.png', '08963.png', '07840.png', '17498.png', '15677.png', '12819.png', '11310.png', '06959.png', '21399.png', '07741.png', '01011.png', '11107.png', '12430.png', '22141.png', '06301.png', '12877.png', '19374.png', '24598.png', '17273.png', '06001.png', '11872.png', '02955.png', '02968.png', '15309.png', '10266.png', '11066.png', '21335.png', '12217.png', '03524.png', '04501.png', '16401.png', '02971.png', '17000.png', '02823.png', '22042.png', '14869.png', '01597.png', '15030.png', '18796.png', '11789.png', '22095.png', '20668.png', '15570.png', '09415.png', '06133.png', '23360.png', '15429.png', '22583.png', '17650.png', '03849.png', '14394.png', '19632.png', '02547.png', '00996.png', '17241.png', '17128.png', '09332.png', '00539.png', '15619.png', '13572.png', '06453.png', '13977.png', '06154.png', '07992.png', '13110.png', '00707.png', '03962.png', '17556.png', '03618.png', '05740.png', '24475.png', '09054.png', '21962.png', '01772.png', '23281.png', '02326.png', '10418.png', '22203.png', '16074.png', '09369.png', '17860.png', '04100.png', '01456.png', '20678.png', '10703.png', '06268.png', '16357.png', '17533.png', '12756.png', '02468.png', '17272.png', '24090.png', '09654.png', '07803.png', '14248.png', '11670.png', '01417.png', '03124.png', '15155.png', '24487.png', '18570.png', '22350.png', '07999.png', '16406.png', '00632.png', '18637.png', '22185.png', '22980.png', '02591.png', '15859.png', '19597.png', '05374.png', '12555.png', '10840.png', '23825.png', '24385.png', '06719.png', '17808.png', '24298.png', '20416.png', '05358.png', '19849.png', '19612.png', '07209.png', '11237.png', '17994.png', '06606.png', '05735.png', '19037.png', '23050.png', '17218.png', '19492.png', '17610.png', '23934.png', '24559.png', '23370.png', '12729.png', '05061.png', '10360.png', '22172.png', '07393.png', '15201.png', '05326.png', '17382.png', '07743.png', '22025.png', '19549.png', '19324.png', '08590.png', '01018.png', '08138.png', '16753.png', '05588.png', '08089.png', '03891.png', '21314.png', '12403.png', '08949.png', '00316.png', '02039.png', '24311.png', '24010.png', '09070.png', '00921.png', '20791.png', '17219.png', '12940.png', '10975.png', '20853.png', '14575.png', '00222.png', '06059.png', '21924.png', '03265.png', '15557.png', '13556.png', '22065.png', '24251.png', '17784.png', '08838.png', '05678.png', '24925.png', '14505.png', '01719.png', '19392.png', '08531.png', '09472.png', '24678.png', '21201.png', '13390.png', '15188.png', '06456.png', '24016.png', '04261.png', '07807.png', '15602.png', '00983.png', '04059.png', '15124.png', '10887.png', '18881.png', '05451.png', '09295.png', '20669.png', '01794.png', '05438.png', '10379.png', '19031.png', '06067.png', '03656.png', '17615.png', '23861.png', '12897.png', '15592.png', '20144.png', '21326.png', '12252.png', '06537.png', '01509.png', '12130.png', '11516.png', '09760.png', '07536.png', '02580.png', '24145.png', '05840.png', '12539.png', '19558.png', '12412.png', '01342.png', '18208.png', '23161.png', '22905.png', '18291.png', '13689.png', '19060.png', '10517.png', '00979.png', '02481.png', '24204.png', '23062.png', '24854.png', '20343.png', '08391.png', '11802.png', '02225.png', '03099.png', '16444.png', '15078.png', '02540.png', '10294.png', '01195.png', '05518.png', '01283.png', '00072.png', '14852.png', '16367.png', '13717.png', '07137.png', '09605.png', '14500.png', '03418.png', '10956.png', '13495.png', '01230.png', '16788.png', '18711.png', '08240.png', '13619.png', '19503.png', '17758.png', '01803.png', '09893.png', '12802.png', '01732.png', '02218.png', '13168.png', '18466.png', '02022.png', '21711.png', '10787.png', '22809.png', '19535.png', '20752.png', '00612.png', '18436.png', '20031.png', '09048.png', '13350.png', '19830.png', '09448.png', '20206.png', '11127.png', '05620.png', '18255.png', '21671.png', '03205.png', '21546.png', '11957.png', '23687.png', '14740.png', '22418.png', '01123.png', '18540.png', '23474.png', '18781.png', '11296.png', '22214.png', '11356.png', '04497.png', '13640.png', '08382.png', '04052.png', '05052.png', '07768.png', '00356.png', '23478.png', '01689.png', '07078.png', '18345.png', '08290.png', '16904.png', '22153.png', '24444.png', '22993.png', '24305.png', '01741.png', '05252.png', '02294.png', '13365.png', '04955.png', '16603.png', '14668.png', '23158.png', '20141.png', '01785.png', '23523.png', '06189.png', '23051.png', '23378.png', '13867.png', '04540.png', '14903.png', '21265.png', '07304.png', '11524.png', '19050.png', '12777.png', '11596.png', '15744.png', '19255.png', '15194.png', '22263.png', '01827.png', '08047.png', '23750.png', '20150.png', '03553.png', '03577.png', '16777.png', '14018.png', '02002.png', '24411.png', '20521.png', '00201.png', '05570.png', '16309.png', '16064.png', '05785.png', '07053.png', '20200.png', '22057.png', '24934.png', '07094.png', '09877.png', '11573.png', '18068.png', '03368.png', '20386.png', '19100.png', '07681.png', '12671.png', '03977.png', '05585.png', '17571.png', '05988.png', '23970.png', '09151.png', '11844.png', '24288.png', '18206.png', '21782.png', '21122.png', '09066.png', '11012.png', '02125.png', '23940.png', '03722.png', '11239.png', '22958.png', '06162.png', '16186.png', '09625.png', '08222.png', '23306.png', '16209.png', '05859.png', '22866.png', '04047.png', '07747.png', '16634.png', '23008.png', '04061.png', '04693.png', '03248.png', '10400.png', '06256.png', '20967.png', '06804.png', '21378.png', '00781.png', '07026.png', '16303.png', '07996.png', '05546.png', '18491.png', '15206.png', '24929.png', '07127.png', '22936.png', '14316.png', '20469.png', '20370.png', '17876.png', '17620.png', '24224.png', '02238.png', '00817.png', '04381.png', '02373.png', '01746.png', '08200.png', '09112.png', '04674.png', '22014.png', '12078.png', '07591.png', '12435.png', '16415.png', '14049.png', '07259.png', '08965.png', '14272.png', '15430.png', '15423.png', '21270.png', '00754.png', '19877.png', '01665.png', '19442.png', '17794.png', '18276.png', '21775.png', '18492.png', '00964.png', '15779.png', '11951.png', '20271.png', '18867.png', '22970.png', '24231.png', '03664.png', '10237.png', '00367.png', '12613.png', '08454.png', '23098.png', '14842.png', '22898.png', '07187.png', '02738.png', '14902.png', '11392.png', '23710.png', '11213.png', '03273.png', '17676.png', '17091.png', '04743.png', '08266.png', '05870.png', '24144.png', '18418.png', '02908.png', '14362.png', '18863.png', '19423.png', '14764.png', '07419.png', '15605.png', '01474.png', '24919.png', '24312.png', '19748.png', '15797.png', '17857.png', '23811.png', '01173.png', '10258.png', '10021.png', '00959.png', '16564.png', '00204.png', '06770.png', '04287.png', '01616.png', '06183.png', '11008.png', '22012.png', '09120.png', '02286.png', '18627.png', '21277.png', '09647.png', '23777.png', '23324.png', '16938.png', '03395.png', '14540.png', '14164.png', '19267.png', '23116.png', '02641.png', '07107.png', '09174.png', '11298.png', '16394.png', '21478.png', '10304.png', '16137.png', '06696.png', '05302.png', '20948.png', '13587.png', '17379.png', '18431.png', '07680.png', '04655.png', '22247.png', '14706.png', '19616.png', '05101.png', '19358.png', '19961.png', '14648.png', '00735.png', '10348.png', '03981.png', '02366.png', '22972.png', '17058.png', '09789.png', '03545.png', '24864.png', '13439.png', '20487.png', '02076.png', '00791.png', '14682.png', '20247.png', '12154.png', '18109.png', '23906.png', '18259.png', '12788.png', '06846.png', '12280.png', '05719.png', '09573.png', '01289.png', '13131.png', '03621.png', '10970.png', '22486.png', '12598.png', '24541.png', '20194.png', '21814.png', '04950.png', '14941.png', '10655.png', '08721.png', '15977.png', '22265.png', '11478.png', '15548.png', '07588.png', '17327.png', '06126.png', '10026.png', '22661.png', '21954.png', '04354.png', '17019.png', '08216.png', '08466.png', '03488.png', '03344.png', '12369.png', '01602.png', '20844.png', '08361.png', '22967.png', '08423.png', '05587.png', '01231.png', '18474.png', '13001.png', '15567.png', '02471.png', '20973.png', '00476.png', '04815.png', '24151.png', '18531.png', '07775.png', '20936.png', '09910.png', '15267.png', '21718.png', '21203.png', '00436.png', '18180.png', '16559.png', '19528.png', '10405.png', '03893.png', '09896.png', '24059.png', '01407.png', '07260.png', '14060.png', '12974.png', '08725.png', '03083.png', '14287.png', '06956.png', '21170.png', '04840.png', '19526.png', '19458.png', '00328.png', '01818.png', '12919.png', '01462.png', '14601.png', '04770.png', '11839.png', '16770.png', '17292.png', '06566.png', '19009.png', '22345.png', '03189.png', '16257.png', '02708.png', '14644.png', '17726.png', '15038.png', '23170.png', '04362.png', '19805.png', '03028.png', '06779.png', '04897.png', '08135.png', '09818.png', '22458.png', '23487.png', '17822.png', '21275.png', '06826.png', '23215.png', '16630.png', '12258.png', '24391.png', '06964.png', '04464.png', '21853.png', '14549.png', '21618.png', '09718.png', '19496.png', '08983.png', '00678.png', '08279.png', '06375.png', '01630.png', '17152.png', '02429.png', '16145.png', '00438.png', '07257.png', '21286.png', '04910.png', '22579.png', '10814.png', '22672.png', '24647.png', '24657.png', '11800.png', '19490.png', '23144.png', '14327.png', '24619.png', '19453.png', '03566.png', '15684.png', '18704.png', '10631.png', '22766.png', '06478.png', '16034.png', '10354.png', '02744.png', '15680.png', '05237.png', '18890.png', '18698.png', '12550.png', '03858.png', '02457.png', '07986.png', '16297.png', '23219.png', '07698.png', '16828.png', '16057.png', '17692.png', '18286.png', '05192.png', '17664.png', '23539.png', '08664.png', '05873.png', '09227.png', '10365.png', '05489.png', '19771.png', '02951.png', '19610.png', '01528.png', '14545.png', '06358.png', '07586.png', '02028.png', '04779.png', '00059.png', '22285.png', '01898.png', '19664.png', '22917.png', '24033.png', '01522.png', '01869.png', '04697.png', '22760.png', '06564.png', '06665.png', '15075.png', '19716.png', '17655.png', '13258.png', '14125.png', '21062.png', '13315.png', '22354.png', '16664.png', '15236.png', '01070.png', '09739.png', '09110.png', '01500.png', '19722.png', '23517.png', '05264.png', '11713.png', '23015.png', '02969.png', '16723.png', '11175.png', '03352.png', '23772.png', '11439.png', '13699.png', '04670.png', '13142.png', '04020.png', '22058.png', '19772.png', '14887.png', '12332.png', '11154.png', '09943.png', '17881.png', '18338.png', '06873.png', '15027.png', '05111.png', '01376.png', '04082.png', '09259.png', '05407.png', '20849.png', '23535.png', '13913.png', '23888.png', '13669.png', '21473.png', '23477.png', '11363.png', '06245.png', '22135.png', '16692.png', '06441.png', '05764.png', '08974.png', '24522.png', '15307.png', '09323.png', '18397.png', '04801.png', '18967.png', '19903.png', '06764.png', '07702.png', '15216.png', '22414.png', '09038.png', '08348.png', '02453.png', '09337.png', '24775.png', '18470.png', '24337.png', '23915.png', '19494.png', '19061.png', '21611.png', '10769.png', '00084.png', '17695.png', '08005.png', '15676.png', '09267.png', '13954.png', '06464.png', '02627.png', '00061.png', '10849.png', '16340.png', '22406.png', '05529.png', '06475.png', '15984.png', '24802.png', '10431.png', '04175.png', '14829.png', '00314.png', '03944.png', '15915.png', '20411.png', '00920.png', '00417.png', '20911.png', '20375.png', '02320.png', '02851.png', '11911.png', '04334.png', '09190.png', '13187.png', '15850.png', '10638.png', '03990.png', '06625.png', '04366.png', '17529.png', '00376.png', '16588.png', '00002.png', '09825.png', '11084.png', '07456.png', '07991.png', '12794.png', '15633.png', '06366.png', '12182.png', '05738.png', '05926.png', '09201.png', '08866.png', '10342.png', '22688.png', '05335.png', '01013.png', '18199.png', '08761.png', '15021.png', '11709.png', '02082.png', '13112.png', '19674.png', '01977.png', '24907.png', '06775.png', '08326.png', '00976.png', '24491.png', '12969.png', '07250.png', '17149.png', '08223.png', '10349.png', '04291.png', '16341.png', '22603.png', '24765.png', '00825.png', '24577.png', '19006.png', '03627.png', '22754.png', '08199.png', '16941.png', '04631.png', '07715.png', '16531.png', '17778.png', '10760.png', '02781.png', '15925.png', '17274.png', '19213.png', '16543.png', '04123.png', '05364.png', '22059.png', '05382.png', '04314.png', '01482.png', '07045.png', '08708.png', '18754.png', '19944.png', '07594.png', '18407.png', '15018.png', '01968.png', '21631.png', '17811.png', '17997.png', '19619.png', '10513.png', '10696.png', '23642.png', '23363.png', '01346.png', '18805.png', '19578.png', '17531.png', '00019.png', '04040.png', '14714.png', '08606.png', '10217.png', '07352.png', '20197.png', '13714.png', '07888.png', '02646.png', '15655.png', '13748.png', '09843.png', '13243.png', '05591.png', '07040.png', '00651.png', '05139.png', '21727.png', '11585.png', '15788.png', '09155.png', '14459.png', '22137.png', '19641.png', '01363.png', '10002.png', '21506.png', '14045.png', '10708.png', '06559.png', '00056.png', '23797.png', '11721.png', '24748.png', '13240.png', '15453.png', '21570.png', '05822.png', '23753.png', '05521.png', '01756.png', '16730.png', '04328.png', '24498.png', '22824.png', '08649.png', '07549.png', '14578.png', '15789.png', '16602.png', '20143.png', '18387.png', '09544.png', '20205.png', '09044.png', '13626.png', '22668.png', '21161.png', '01190.png', '10478.png', '07912.png', '03417.png', '02083.png', '20061.png', '06287.png', '24825.png', '09273.png', '21073.png', '05535.png', '04991.png', '23803.png', '01028.png', '13330.png', '11258.png', '23950.png', '12081.png', '23738.png', '01322.png', '19187.png', '01487.png', '10107.png', '09804.png', '18112.png', '23009.png', '03581.png', '23090.png', '24001.png', '08837.png', '12259.png', '10847.png', '24616.png', '03426.png', '18908.png', '02559.png', '12752.png', '20998.png', '00564.png', '17018.png', '01228.png', '08132.png', '23849.png', '19623.png', '13314.png', '24007.png', '24328.png', '22310.png', '03616.png', '04506.png', '24723.png', '05243.png', '19662.png', '17130.png', '13359.png', '11946.png', '14688.png', '17788.png', '11423.png', '04416.png', '20905.png', '05266.png', '00561.png', '23203.png', '17795.png', '01372.png', '11118.png', '11803.png', '19117.png', '06060.png', '23143.png', '06837.png', '10911.png', '03024.png', '08832.png', '10623.png', '18887.png', '13141.png', '01303.png', '08009.png', '03637.png', '05602.png', '13795.png', '00946.png', '04312.png', '07483.png', '14619.png', '10662.png', '24176.png', '05290.png', '09585.png', '09724.png', '23406.png', '05199.png', '18504.png', '11481.png', '01119.png', '16997.png', '10826.png', '24417.png', '02279.png', '18834.png', '00943.png', '16264.png', '14039.png', '17244.png', '12366.png', '03332.png', '17214.png', '19125.png', '07785.png', '00087.png', '13764.png', '01923.png', '09938.png', '21592.png', '20429.png', '18066.png', '17411.png', '12260.png', '04163.png', '03815.png', '20558.png', '07878.png', '07664.png', '00950.png', '23252.png', '22997.png', '00315.png', '15952.png', '06096.png', '21932.png', '04435.png', '17609.png', '01591.png', '01029.png', '16874.png', '14061.png', '00232.png', '13175.png', '13591.png', '24633.png', '24708.png', '09745.png', '11643.png', '21260.png', '03651.png', '16907.png', '06244.png', '05102.png', '00218.png', '21137.png', '10366.png', '17188.png', '07772.png', '05083.png', '02186.png', '23821.png', '03489.png', '13155.png', '00828.png', '09017.png', '17667.png', '11257.png', '10994.png', '14517.png', '11969.png', '23966.png', '01357.png', '11854.png', '06950.png', '08693.png', '16921.png', '07432.png', '20328.png', '00039.png', '23214.png', '15919.png', '08686.png', '17657.png', '16922.png', '21543.png', '04284.png', '09839.png', '20664.png', '19724.png', '12431.png', '24830.png', '08282.png', '24769.png', '24721.png', '14148.png', '16377.png', '01724.png', '04431.png', '02391.png', '17608.png', '12189.png', '11080.png', '09480.png', '10331.png', '19340.png', '11563.png', '02064.png', '23923.png', '17393.png', '04523.png', '06551.png', '20748.png', '12305.png', '05560.png', '16952.png', '20902.png', '19987.png', '00755.png', '06103.png', '06560.png', '11728.png', '09291.png', '20589.png', '00285.png', '20577.png', '02273.png', '07884.png', '04602.png', '13313.png', '24057.png', '03954.png', '23850.png', '12587.png', '05001.png', '09903.png', '20188.png', '17743.png', '17198.png', '22861.png', '12845.png', '21731.png', '05068.png', '10001.png', '01366.png', '02876.png', '06749.png', '11091.png', '05791.png', '00815.png', '04678.png', '17467.png', '05708.png', '15638.png', '12669.png', '15927.png', '23181.png', '08629.png', '14559.png', '01733.png', '02065.png', '18226.png', '21732.png', '03303.png', '17752.png', '02682.png', '06654.png', '00375.png', '02153.png', '12352.png', '18771.png', '19410.png', '09389.png', '08237.png', '13295.png', '17320.png', '22474.png', '18001.png', '00123.png', '13869.png', '04672.png', '06839.png', '10447.png', '23639.png', '21129.png', '06286.png', '20258.png', '04647.png', '03848.png', '07143.png', '13606.png', '18447.png', '16191.png', '03862.png', '12918.png', '22612.png', '14746.png', '04521.png', '13604.png', '02704.png', '00915.png', '04614.png', '03243.png', '17144.png', '12975.png', '20079.png', '06728.png', '21964.png', '01084.png', '12127.png', '16101.png', '06880.png', '20782.png', '09321.png', '11671.png', '23292.png', '17601.png', '03971.png', '20034.png', '23421.png', '21312.png', '09851.png', '06334.png', '01493.png', '13565.png', '02524.png', '03464.png', '00831.png', '18443.png', '06743.png', '09125.png', '21586.png', '17623.png', '17092.png', '23052.png', '08468.png', '15726.png', '24350.png', '23464.png', '19063.png', '22955.png', '24113.png', '18618.png', '02874.png', '16864.png', '00378.png', '11874.png', '24949.png', '21525.png', '16945.png', '02762.png', '15450.png', '17720.png', '05271.png', '07100.png', '05893.png', '21657.png', '02346.png', '14622.png', '15710.png', '24506.png', '16867.png', '22372.png', '12494.png', '16010.png', '20202.png', '14742.png', '16792.png', '09085.png', '08824.png', '19933.png', '02579.png', '07438.png', '11031.png', '01049.png', '05214.png', '21501.png', '12129.png', '04325.png', '17182.png', '15620.png', '03451.png', '09284.png', '15065.png', '07300.png', '20567.png', '01598.png', '13876.png', '09423.png', '07948.png', '06425.png', '06827.png', '09101.png', '23286.png', '12022.png', '16333.png', '14215.png', '13665.png', '02699.png', '07956.png', '17148.png', '21301.png', '09603.png', '10305.png', '02548.png', '00068.png', '16166.png', '13813.png', '01010.png', '14314.png', '23224.png', '21111.png', '02078.png', '04390.png', '14597.png', '18940.png', '21117.png', '00745.png', '14180.png', '14169.png', '10935.png', '09239.png', '00890.png', '01008.png', '00550.png', '04474.png', '18611.png', '06291.png', '13629.png', '12830.png', '09994.png', '20968.png', '24360.png', '18249.png', '17791.png', '16545.png', '10637.png', '15720.png', '13781.png', '03070.png', '11736.png', '17968.png', '02909.png', '15425.png', '01182.png', '23016.png', '23630.png', '17557.png', '11818.png', '06213.png', '23129.png', '12758.png', '11191.png', '06193.png', '02814.png', '13953.png', '21965.png', '12437.png', '14533.png', '08410.png', '17562.png', '14421.png', '05971.png', '03320.png', '08982.png', '13933.png', '04644.png', '13844.png', '03614.png', '02578.png', '02008.png', '11860.png', '19269.png', '23037.png', '09725.png', '16615.png', '12675.png', '17281.png', '21369.png', '13081.png', '15327.png', '17775.png', '12183.png', '13357.png', '00523.png', '04919.png', '16818.png', '04349.png', '05963.png', '00318.png', '08953.png', '22981.png', '06600.png', '01672.png', '02727.png', '14708.png', '14786.png', '08905.png', '06227.png', '00180.png', '06383.png', '13808.png', '08619.png', '04742.png', '10830.png', '18078.png', '11856.png', '08094.png', '22701.png', '01135.png', '12429.png', '13375.png', '13123.png', '22005.png', '07232.png', '10707.png', '17830.png', '23356.png', '13185.png', '21982.png', '17783.png', '07619.png', '01225.png', '24278.png', '10682.png', '03871.png', '08677.png', '09523.png', '15623.png', '22130.png', '00267.png', '21861.png', '05158.png', '01454.png', '15752.png', '07742.png', '19804.png', '21380.png', '10497.png', '21678.png', '03974.png', '08486.png', '10645.png', '16139.png', '21210.png', '16028.png', '10428.png', '05648.png', '18780.png', '10824.png', '12284.png', '11981.png', '08756.png', '22711.png', '18083.png', '10359.png', '02772.png', '18005.png', '08267.png', '05419.png', '12340.png', '20265.png', '08445.png', '22522.png', '06164.png', '23461.png', '09779.png', '11708.png', '19999.png', '05978.png', '00571.png', '02796.png', '10616.png', '02799.png', '05402.png', '09696.png', '16625.png', '00710.png', '00116.png', '15291.png', '22938.png', '13272.png', '07241.png', '10139.png', '05053.png', '10695.png', '17973.png', '08945.png', '00730.png', '16516.png', '14410.png', '18703.png', '21219.png', '13104.png', '00297.png', '13615.png', '24243.png', '23961.png', '17719.png', '22755.png', '15727.png', '09682.png', '02589.png', '19501.png', '15914.png', '17385.png', '15585.png', '21489.png', '02452.png', '21381.png', '18175.png', '06705.png', '12617.png', '06152.png', '13245.png', '06421.png', '19111.png', '04443.png', '19650.png', '16197.png', '21469.png', '08032.png', '03988.png', '21753.png', '07725.png', '08308.png', '15473.png', '08044.png', '00290.png', '06597.png', '09864.png', '22305.png', '13580.png', '20604.png', '13190.png', '20714.png', '23581.png', '08878.png', '01444.png', '16804.png', '01693.png', '09468.png', '07482.png', '02061.png', '03188.png', '10937.png', '16597.png', '14483.png', '05526.png', '03924.png', '04099.png', '16218.png', '13353.png', '01126.png', '12834.png', '07018.png', '18770.png', '15587.png', '24163.png', '00420.png', '13427.png', '19974.png', '19376.png', '23234.png', '10284.png', '24679.png', '00179.png', '03191.png', '15431.png', '03774.png', '19589.png', '10489.png', '05552.png', '13121.png', '07463.png', '14139.png', '21939.png', '09886.png', '06028.png', '21407.png', '13202.png', '02364.png', '12246.png', '24241.png', '10858.png', '00162.png', '12227.png', '02491.png', '11673.png', '11717.png', '01145.png', '02731.png', '04609.png', '04462.png', '21412.png', '17813.png', '23515.png', '22477.png', '13872.png', '06389.png', '14914.png', '24382.png', '16159.png', '18678.png', '21060.png', '16517.png', '09266.png', '15778.png', '11714.png', '11152.png', '03879.png', '08417.png', '00094.png', '12862.png', '02585.png', '19042.png', '04716.png', '05780.png', '08607.png', '14003.png', '10602.png', '18146.png', '20466.png', '20830.png', '18355.png', '12982.png', '16889.png', '17435.png', '03595.png', '23057.png', '14465.png', '13773.png', '05804.png', '01370.png', '01325.png', '04571.png', '00503.png', '08149.png', '00655.png', '10585.png', '06575.png', '23591.png', '20050.png', '12145.png', '17394.png', '05607.png', '16840.png', '13186.png', '10908.png', '18030.png', '21452.png', '23315.png', '14582.png', '11429.png', '08736.png', '15824.png', '23193.png', '21109.png', '10730.png', '09401.png', '23863.png', '14595.png', '22468.png', '23348.png', '17207.png', '13026.png', '05164.png', '17801.png', '02673.png', '12534.png', '17211.png', '20953.png', '05624.png', '18972.png', '18594.png', '00657.png', '17154.png', '22545.png', '10894.png', '19440.png', '08176.png', '18639.png', '10825.png', '24537.png', '10433.png', '16071.png', '01218.png', '11654.png', '03065.png', '18556.png', '06435.png', '14732.png', '18982.png', '23419.png', '15688.png', '12091.png', '07520.png', '17022.png', '21019.png', '05772.png', '05991.png', '19661.png', '04701.png', '21905.png', '08722.png', '22259.png', '04137.png', '15709.png', '22575.png', '11288.png', '24316.png', '21865.png', '17117.png', '15466.png', '00053.png', '17168.png', '10797.png', '00978.png', '15621.png', '12171.png', '07926.png', '18050.png', '24235.png', '09944.png', '22262.png', '13100.png', '18622.png', '24490.png', '00971.png', '12031.png', '07851.png', '05043.png', '04066.png', '00841.png', '09703.png', '24271.png', '08124.png', '17318.png', '23010.png', '04668.png', '08869.png', '06336.png', '20747.png', '04947.png', '05351.png', '16740.png', '02957.png', '10110.png', '12590.png', '05903.png', '01942.png', '16875.png', '06229.png', '03897.png', '05049.png', '18279.png', '00122.png', '11929.png', '16201.png', '02789.png', '20690.png', '18437.png', '05085.png', '21027.png', '17954.png', '23624.png', '22996.png', '17363.png', '12971.png', '07444.png', '15244.png', '06860.png', '11289.png', '23319.png', '14809.png', '05140.png', '22515.png', '10099.png', '18476.png', '06870.png', '06363.png', '16407.png', '09927.png', '22024.png', '18261.png', '08507.png', '06938.png', '14014.png', '14243.png', '17563.png', '17277.png', '12012.png', '07896.png', '16868.png', '13655.png', '05400.png', '19241.png', '19715.png', '20817.png', '13894.png', '11442.png', '00045.png', '09343.png', '24301.png', '20160.png', '19023.png', '14952.png', '17966.png', '14845.png', '16133.png', '23402.png', '20850.png', '05815.png', '13905.png', '10715.png', '01787.png', '05646.png', '03364.png', '04772.png', '06869.png', '14811.png', '00759.png', '07058.png', '14873.png', '13907.png', '01588.png', '08886.png', '17150.png', '02119.png', '17409.png', '11934.png', '02148.png', '22391.png', '10516.png', '13476.png', '03631.png', '21625.png', '22451.png', '07214.png', '05542.png', '10562.png', '09959.png', '07953.png', '04987.png', '07614.png', '00645.png', '11419.png', '15440.png', '18523.png', '23798.png', '17067.png', '14244.png', '11752.png', '08112.png', '17009.png', '24963.png', '05731.png', '19208.png', '10820.png', '24756.png', '07307.png', '23271.png', '24581.png', '23403.png', '01666.png', '02490.png', '14654.png', '13158.png', '23192.png', '24100.png', '24102.png', '12006.png', '07476.png', '18210.png', '01189.png', '08129.png', '18376.png', '17011.png', '18414.png', '04159.png', '17613.png', '15614.png', '22653.png', '14265.png', '10238.png', '02784.png', '22370.png', '15882.png', '17497.png', '01240.png', '11646.png', '14834.png', '23117.png', '21047.png', '20980.png', '16284.png', '08357.png', '17596.png', '23658.png', '20988.png', '22332.png', '08765.png', '19210.png', '01307.png', '10410.png', '04522.png', '07789.png', '12953.png', '17482.png', '13840.png', '06220.png', '03898.png', '20754.png', '18769.png', '12658.png', '18070.png', '08635.png', '24340.png', '01928.png', '01238.png', '10903.png', '19773.png', '05421.png', '16493.png', '19842.png', '03047.png', '21403.png', '08145.png', '22353.png', '23739.png', '06373.png', '18487.png', '05913.png', '07356.png', '23577.png', '18265.png', '14520.png', '23212.png', '05734.png', '15366.png', '19470.png', '19847.png', '15053.png', '24310.png', '16360.png', '20546.png', '00081.png', '24027.png', '05349.png', '04447.png', '02285.png', '18225.png', '09591.png', '20049.png', '20657.png', '08051.png', '12626.png', '06226.png', '10312.png', '02906.png', '17119.png', '22383.png', '15356.png', '19548.png', '08694.png', '12231.png', '06247.png', '20106.png', '12326.png', '21607.png', '11648.png', '04817.png', '11862.png', '18404.png', '06168.png', '18382.png', '20960.png', '02458.png', '16961.png', '13786.png', '05115.png', '14170.png', '20847.png', '20710.png', '12244.png', '12222.png', '22799.png', '18648.png', '23699.png', '04669.png', '15578.png', '13846.png', '04612.png', '17577.png', '04794.png', '17696.png', '16714.png', '11735.png', '11215.png', '07890.png', '19551.png', '04606.png', '08667.png', '15182.png', '09550.png', '09270.png', '00270.png', '04967.png', '20494.png', '07008.png', '19926.png', '03567.png', '22847.png', '16699.png', '11511.png', '15109.png', '16557.png', '18930.png', '11848.png', '05087.png', '07475.png', '12714.png', '11628.png', '19726.png', '10455.png', '09649.png', '10713.png', '21496.png', '02643.png', '19439.png', '20675.png', '04145.png', '16897.png', '07544.png', '16916.png', '05746.png', '08351.png', '03677.png', '23091.png', '12762.png', '00237.png', '03733.png', '13297.png', '09358.png', '10429.png', '20609.png', '06136.png', '20620.png', '02659.png', '17487.png', '23677.png', '24019.png', '18227.png', '16595.png', '03416.png', '17179.png', '22537.png', '12367.png', '21641.png', '17586.png', '18304.png', '04025.png', '03878.png', '02203.png', '20736.png', '05285.png', '01498.png', '18386.png', '16888.png', '04397.png', '10932.png', '00953.png', '02090.png', '24955.png', '03379.png', '05623.png', '04292.png', '02216.png', '08681.png', '19647.png', '03289.png', '04603.png', '23485.png', '17507.png', '00802.png', '10324.png', '10273.png', '14140.png', '12201.png', '02703.png', '11457.png', '20468.png', '24605.png', '22257.png', '19360.png', '01744.png', '19337.png', '24030.png', '14946.png', '09730.png', '13820.png', '16991.png', '17017.png', '19388.png', '02121.png', '18664.png', '01463.png', '18384.png', '17319.png', '07295.png', '13506.png', '10401.png', '18877.png', '02049.png', '13785.png', '17260.png', '06188.png', '23171.png', '07195.png', '20711.png', '05360.png', '05942.png', '14506.png', '14288.png', '18475.png', '23734.png', '19053.png', '11567.png', '16566.png', '00924.png', '02570.png', '18956.png', '12208.png', '03419.png', '13677.png', '11473.png', '06960.png', '12810.png', '22569.png', '10404.png', '16477.png', '01022.png', '05498.png', '16142.png', '06339.png', '14027.png', '16260.png', '23298.png', '20820.png', '14189.png', '20290.png', '07951.png', '23069.png', '07887.png', '02926.png', '03349.png', '15670.png', '21132.png', '03161.png', '00010.png', '06100.png', '21553.png', '03549.png', '10468.png', '01105.png', '09161.png', '04758.png', '14780.png', '22780.png', '01279.png', '23499.png', '05682.png', '06081.png', '05515.png', '04277.png', '04204.png', '15897.png', '08957.png', '12683.png', '12074.png', '20129.png', '13881.png', '23527.png', '20086.png', '04114.png', '12663.png', '04937.png', '14770.png', '23189.png', '14876.png', '23926.png', '22823.png', '13547.png', '00914.png', '10336.png', '05660.png', '17024.png', '23916.png', '22145.png', '14333.png', '21505.png', '17424.png', '02665.png', '13645.png', '12460.png', '22924.png', '14473.png', '16738.png', '06072.png', '15571.png', '07866.png', '04301.png', '08083.png', '18814.png', '14371.png', '11901.png', '10072.png', '11251.png', '11594.png', '18306.png', '15956.png', '08055.png', '24484.png', '04555.png', '05257.png', '13052.png', '00399.png', '18785.png', '06212.png', '10346.png', '04905.png', '17513.png', '22715.png', '04295.png', '02029.png', '18852.png', '06963.png', '12468.png', '16020.png', '06733.png', '22922.png', '12774.png', '20449.png', '10350.png', '04496.png', '23691.png', '09500.png', '22404.png', '07518.png', '11734.png', '05148.png', '00723.png', '02783.png', '16192.png', '12218.png', '14720.png', '17456.png', '19659.png', '20154.png', '04627.png', '10104.png', '07821.png', '10126.png', '01887.png', '16018.png', '17575.png', '05637.png', '17202.png', '14336.png', '02353.png', '20537.png', '07203.png', '20560.png', '19825.png', '19114.png', '20346.png', '24127.png', '07944.png', '06536.png', '01468.png', '24613.png', '03972.png', '03928.png', '20919.png', '14310.png', '13952.png', '10881.png', '24892.png', '17430.png', '12652.png', '03593.png', '12711.png', '12266.png', '12351.png', '11247.png', '15524.png', '23113.png', '12342.png', '20756.png', '13060.png', '10580.png', '13964.png', '01726.png', '01224.png', '04778.png', '19514.png', '00530.png', '19123.png', '22942.png', '10007.png', '02889.png', '13162.png', '18025.png', '14906.png', '00109.png', '21737.png', '02617.png', '03215.png', '14466.png', '12254.png', '08909.png', '00617.png', '06393.png', '11677.png', '03776.png', '07397.png', '09543.png', '03633.png', '15861.png', '15941.png', '16741.png', '21987.png', '13194.png', '00115.png', '11995.png', '17996.png', '15800.png', '17799.png', '16207.png', '02520.png', '22807.png', '07906.png', '13563.png', '10494.png', '12586.png', '10205.png', '14613.png', '10083.png', '06671.png', '07863.png', '04403.png', '03690.png', '22128.png', '14428.png', '05696.png', '23757.png', '16165.png', '09784.png', '12050.png', '07841.png', '15718.png', '22016.png', '19789.png', '09866.png', '23268.png', '11505.png', '04845.png', '04907.png', '05680.png', '04934.png', '02093.png', '23518.png', '00985.png', '09736.png', '00734.png', '13157.png', '16498.png', '00658.png', '06071.png', '00696.png', '16335.png', '11042.png', '09622.png', '01975.png', '08117.png', '16039.png', '07152.png', '03982.png', '13296.png', '18519.png', '20338.png', '06221.png', '18074.png', '05670.png', '23554.png', '19355.png', '03846.png', '08402.png', '05844.png', '06750.png', '14273.png', '12349.png', '03518.png', '03552.png', '12666.png', '11488.png', '02146.png', '09857.png', '22651.png', '16660.png', '10810.png', '14957.png', '18094.png', '23829.png', '12285.png', '00142.png', '19487.png', '13733.png', '20772.png', '03993.png', '22951.png', '18928.png', '23043.png', '12407.png', '08753.png', '17889.png', '18850.png', '21120.png', '08935.png', '02607.png', '00533.png', '14760.png', '15606.png', '10949.png', '20835.png', '05640.png', '24869.png', '08227.png', '23637.png', '22819.png', '20199.png', '02776.png', '24286.png', '17183.png', '19228.png', '11973.png', '16234.png', '08663.png', '18624.png', '07661.png', '01574.png', '21031.png', '20024.png', '12616.png', '19057.png', '17550.png', '10999.png', '17790.png', '03095.png', '22249.png', '15734.png', '02228.png', '22684.png', '13628.png', '04442.png', '15297.png', '04921.png', '18811.png', '02566.png', '18040.png', '18358.png', '04865.png', '13707.png', '20390.png', '22539.png', '16237.png', '10784.png', '10882.png', '12741.png', '01278.png', '04423.png', '07433.png', '06273.png', '16222.png', '15365.png', '06090.png', '04969.png', '24290.png', '10496.png', '16088.png', '03814.png', '17031.png', '11801.png', '24546.png', '23317.png', '21619.png', '02652.png', '14785.png', '03589.png', '02274.png', '16512.png', '03701.png', '17228.png', '18011.png', '09663.png', '20989.png', '04077.png', '04710.png', '03397.png', '20414.png', '13171.png', '19465.png', '08298.png', '23678.png', '22564.png', '05970.png', '21662.png', '20235.png', '04483.png', '10995.png', '21668.png', '21374.png', '09292.png', '16859.png', '09899.png', '14392.png', '08347.png', '11069.png', '21009.png', '22725.png', '18174.png', '06335.png', '15376.png', '00941.png', '14836.png', '14699.png', '17197.png', '00510.png', '00101.png', '18902.png', '16194.png', '11137.png', '21014.png', '21531.png', '19745.png', '19043.png', '09426.png', '13283.png', '08966.png', '19939.png', '20991.png', '03219.png', '21536.png', '10437.png', '15195.png', '02438.png', '17222.png', '00192.png', '22035.png', '23680.png', '12383.png', '13399.png', '24894.png', '10460.png', '20527.png', '24022.png', '19102.png', '15692.png', '20910.png', '00365.png', '21614.png', '16420.png', '14993.png', '13550.png', '07849.png', '04280.png', '08811.png', '06763.png', '23122.png', '00034.png', '19232.png', '11040.png', '05964.png', '13496.png', '02014.png', '15596.png', '10380.png', '20614.png', '11913.png', '15839.png', '22670.png', '03525.png', '15781.png', '00136.png', '18181.png', '16525.png', '15492.png', '22749.png', '24691.png', '03686.png', '21693.png', '20276.png', '24279.png', '20617.png', '16152.png', '14281.png', '22528.png', '01750.png', '02885.png', '01504.png', '00238.png', '06647.png', '20460.png', '10910.png', '21200.png', '10080.png', '03393.png', '12884.png', '07526.png', '08098.png', '12178.png', '06953.png', '18677.png', '08984.png', '10153.png', '01457.png', '13382.png', '19253.png', '20340.png', '10590.png', '03716.png', '21992.png', '05098.png', '01192.png', '09963.png', '14988.png', '11122.png', '24737.png', '01829.png', '04078.png', '09051.png', '09502.png', '20337.png', '02995.png', '14463.png', '12515.png', '11022.png', '02362.png', '18332.png', '16773.png', '15029.png', '18426.png', '11878.png', '12032.png', '00011.png', '17929.png', '21605.png', '13573.png', '10900.png', '17504.png', '17184.png', '18512.png', '20251.png', '24539.png', '04998.png', '04249.png', '15197.png', '16353.png', '13893.png', '16075.png', '14161.png', '05697.png', '15409.png', '05817.png', '12335.png', '00246.png', '05661.png', '07925.png', '11015.png', '10943.png', '00040.png', '14801.png', '18537.png', '02616.png', '05051.png', '04398.png', '06619.png', '07354.png', '22514.png', '21608.png', '22444.png', '10648.png', '07491.png', '12300.png', '15935.png', '08080.png', '04952.png', '24762.png', '22364.png', '06002.png', '11349.png', '18559.png', '02045.png', '19680.png', '13833.png', '00392.png', '03385.png', '24062.png', '02561.png', '18170.png', '03128.png', '18113.png', '24746.png', '03087.png', '14017.png', '15348.png', '00891.png', '03263.png', '06418.png', '00465.png', '21006.png', '16903.png', '17872.png', '21804.png', '21642.png', '18115.png', '02670.png', '17511.png', '14264.png', '24639.png', '04111.png', '24954.png', '03985.png', '18827.png', '04129.png', '09347.png', '19161.png', '20916.png', '15324.png', '12588.png', '11287.png', '14305.png', '14087.png', '04255.png', '08143.png', '12744.png', '17235.png', '01652.png', '18222.png', '08496.png', '02164.png', '21961.png', '05272.png', '00812.png', '11440.png', '09114.png', '12797.png', '19555.png', '13657.png', '00347.png', '07039.png', '08804.png', '07406.png', '19212.png', '04005.png', '12842.png', '17056.png', '01695.png', '01492.png', '05651.png', '11176.png', '02123.png', '15871.png', '04329.png', '11722.png', '12612.png', '11339.png', '17287.png', '09684.png', '05666.png', '13674.png', '14956.png', '09575.png', '24696.png', '02771.png', '08849.png', '03264.png', '20275.png', '09733.png', '02335.png', '08153.png', '08455.png', '16263.png', '14354.png', '09555.png', '10511.png', '03799.png', '15994.png', '15026.png', '08517.png', '24358.png', '07814.png', '21224.png', '13051.png', '15766.png', '17704.png', '10627.png', '19446.png', '06556.png', '17547.png', '19922.png', '24777.png', '20839.png', '22960.png', '10492.png', '03117.png', '19752.png', '10831.png', '11085.png', '16449.png', '05441.png', '01203.png', '03787.png', '08269.png', '06293.png', '16414.png', '21698.png', '06592.png', '19557.png', '17084.png', '00974.png', '23068.png', '17073.png', '21820.png', '16084.png', '08528.png', '20582.png', '10967.png', '07752.png', '03053.png', '12966.png', '01604.png', '22655.png', '15003.png', '06405.png', '18822.png', '14181.png', '04125.png', '24645.png', '19172.png', '23002.png', '19137.png', '16613.png', '12689.png', '00643.png', '24223.png', '04981.png', '05167.png', '02239.png', '20257.png', '21818.png', '17251.png', '04943.png', '00997.png', '06836.png', '01021.png', '00574.png', '18380.png', '17376.png', '15646.png', '21889.png', '03908.png', '10006.png', '03324.png', '20716.png', '02839.png', '03337.png', '18913.png', '22169.png', '01216.png', '06359.png', '01518.png', '16900.png', '17800.png', '05609.png', '08228.png', '19629.png', '09016.png', '00468.png', '12957.png', '15354.png', '07495.png', '07639.png', '02435.png', '09835.png', '22618.png', '11638.png', '03029.png', '04624.png', '16058.png', '03810.png', '15876.png', '19544.png', '10936.png', '04881.png', '23653.png', '12646.png', '16854.png', '14935.png', '12622.png', '04513.png', '23809.png', '17555.png', '21598.png', '24041.png', '10075.png', '22516.png', '01382.png', '22669.png', '15397.png', '18036.png', '10011.png', '24905.png', '18331.png', '09322.png', '01849.png', '03439.png', '12271.png', '16199.png', '23017.png', '17412.png', '06371.png', '22179.png', '07511.png', '08579.png', '20619.png', '07650.png', '19594.png', '07777.png', '19076.png', '20427.png', '06851.png', '12837.png', '12956.png', '13659.png', '14639.png', '05968.png', '01344.png', '05450.png', '09242.png', '24052.png', '07750.png', '00991.png', '12977.png', '07198.png', '10196.png', '14046.png', '18840.png', '06822.png', '06628.png', '08249.png', '11459.png', '01246.png', '22352.png', '23054.png', '15131.png', '18503.png', '10771.png', '06243.png', '07602.png', '14718.png', '18767.png', '19626.png', '17247.png', '02924.png', '12148.png', '20718.png', '15020.png', '24281.png', '01234.png', '23447.png', '17187.png', '05946.png', '00906.png', '02107.png', '05797.png', '01932.png', '15269.png', '24527.png', '02608.png', '02034.png', '06582.png', '11815.png', '06167.png', '16549.png', '11921.png', '00071.png', '00380.png', '17708.png', '24713.png', '09050.png', '00482.png', '16203.png', '17847.png', '09117.png', '03540.png', '16939.png', '02568.png', '19792.png', '18818.png', '20485.png', '19185.png', '11284.png', '06589.png', '18712.png', '09133.png', '04057.png', '20316.png', '07313.png', '04382.png', '15565.png', '14372.png', '13341.png', '08526.png', '17342.png', '06364.png', '19139.png', '23685.png', '07041.png', '24377.png', '06891.png', '10270.png', '22691.png', '11644.png', '06402.png', '11967.png', '08931.png', '24034.png', '09952.png', '06208.png', '04822.png', '12847.png', '14631.png', '23228.png', '24086.png', '13410.png', '24808.png', '08006.png', '05802.png', '22843.png', '14975.png', '16200.png', '06855.png', '03138.png', '17369.png', '24198.png', '07059.png', '05174.png', '03297.png', '09371.png', '24906.png', '24456.png', '20463.png', '03771.png', '19890.png', '21411.png', '23303.png', '22568.png', '16362.png', '15792.png', '02923.png', '22606.png', '02948.png', '10530.png', '17109.png', '15503.png', '24186.png', '05720.png', '04906.png', '08353.png', '13327.png', '09022.png', '12417.png', '02896.png', '12206.png', '03997.png', '07846.png', '21332.png', '23570.png', '01165.png', '16829.png', '02841.png', '09883.png', '05476.png', '15218.png', '18579.png', '17485.png', '18578.png', '19245.png', '05888.png', '03571.png', '07090.png', '05889.png', '03307.png', '00879.png', '22118.png', '21211.png', '02352.png', '17669.png', '11770.png', '05223.png', '13094.png', '06195.png', '23511.png', '20204.png', '00838.png', '09966.png', '00340.png', '00093.png', '02669.png', '07065.png', '01269.png', '03903.png', '17065.png', '18411.png', '05750.png', '16285.png', '02721.png', '03603.png', '06374.png', '03272.png', '23720.png', '11438.png', '16928.png', '04988.png', '02253.png', '03659.png', '09327.png', '21157.png', '23040.png', '04218.png', '10796.png', '04378.png', '13663.png', '13500.png', '05226.png', '06442.png', '17312.png', '12564.png', '06015.png', '24700.png', '12627.png', '10879.png', '13433.png', '13687.png', '04621.png', '05732.png', '12136.png', '20157.png', '12553.png', '09503.png', '08374.png', '12249.png', '06794.png', '19469.png', '08724.png', '06565.png', '23580.png', '06789.png', '02157.png', '13483.png', '00669.png', '00589.png', '08930.png', '10174.png', '01815.png', '19819.png', '03821.png', '02077.png', '23971.png', '18457.png', '21108.png', '05314.png', '00942.png', '17527.png', '02139.png', '03102.png', '15489.png', '21383.png', '17622.png', '20887.png', '00794.png', '06643.png', '24137.png', '05491.png', '21705.png', '06471.png', '10372.png', '14424.png', '19018.png', '22091.png', '09463.png', '11053.png', '03709.png', '17417.png', '08894.png', '09770.png', '04830.png', '17753.png', '03933.png', '05313.png', '04660.png', '22507.png', '04702.png', '23924.png', '00673.png', '08273.png', '00753.png', '09377.png', '14611.png', '09018.png', '10277.png', '11242.png', '14621.png', '19940.png', '03350.png', '24082.png', '04453.png', '00016.png', '24367.png', '09317.png', '10063.png', '21897.png', '09915.png', '03019.png', '08678.png', '21149.png', '07612.png', '18853.png', '02650.png', '12157.png', '03396.png', '24489.png', '15275.png', '21764.png', '16691.png', '03855.png', '07510.png', '16373.png', '01427.png', '01201.png', '12115.png', '09382.png', '05392.png', '18045.png', '18684.png', '18869.png', '13458.png', '21075.png', '20080.png', '10117.png', '18207.png', '04547.png', '09061.png', '01057.png', '13012.png', '18085.png', '08653.png', '12874.png', '02444.png', '23823.png', '01143.png', '08992.png', '16102.png', '23791.png', '18533.png', '00317.png', '20559.png', '04451.png', '00962.png', '15013.png', '15279.png', '16665.png', '17501.png', '05176.png', '12373.png', '09894.png', '03875.png', '21313.png', '04551.png', '10285.png', '16596.png', '18362.png', '08623.png', '08793.png', '00384.png', '23605.png', '23405.png', '00668.png', '22029.png', '13959.png', '21234.png', '04361.png', '06194.png', '03139.png', '22339.png', '08250.png', '17629.png', '03274.png', '22685.png', '05175.png', '16334.png', '02991.png', '23233.png', '18463.png', '06079.png', '22582.png', '22371.png', '10465.png', '17326.png', '24206.png', '05996.png', '10729.png', '19488.png', '13053.png', '19386.png', '19858.png', '07763.png', '16719.png', '15129.png', '10828.png', '13298.png', '24548.png', '07200.png', '08275.png', '21032.png', '03782.png', '22647.png', '17174.png', '05894.png', '08260.png', '12938.png', '02319.png', '12805.png', '08428.png', '06781.png', '00490.png', '14133.png', '00158.png', '14107.png', '15493.png', '17906.png', '00050.png', '04856.png', '07883.png', '10287.png', '08001.png', '23266.png', '07359.png', '00770.png', '07825.png', '02075.png', '07797.png', '07490.png', '02719.png', '15191.png', '11399.png', '22399.png', '09861.png', '22689.png', '22004.png', '14100.png', '11004.png', '23128.png', '05090.png', '01425.png', '02691.png', '02323.png', '06636.png', '15367.png', '20731.png', '19176.png', '18799.png', '10212.png', '03989.png', '01590.png', '00409.png', '06337.png', '24283.png', '07346.png', '06288.png', '13857.png', '15775.png', '03598.png', '06999.png', '02073.png', '11642.png', '01886.png', '17978.png', '04289.png', '21946.png', '20121.png', '19130.png', '00967.png', '01319.png', '15459.png', '11128.png', '01115.png', '24597.png', '07581.png', '16409.png', '11763.png', '13398.png', '00165.png', '22147.png', '24789.png', '05951.png', '22080.png', '00929.png', '12581.png', '22554.png', '24216.png', '04214.png', '19363.png', '14238.png', '22763.png', '06727.png', '08764.png', '12112.png', '23783.png', '06427.png', '22919.png', '20138.png', '09246.png', '06740.png', '22501.png', '10023.png', '08370.png', '16629.png', '02437.png', '07399.png', '23543.png', '06053.png', '23028.png', '16635.png', '03456.png', '15262.png', '11622.png', '16354.png', '20765.png', '20213.png', '00202.png', '11799.png', '17880.png', '13610.png', '02244.png', '05569.png', '01512.png', '04418.png', '01861.png', '16586.png', '16869.png', '13727.png', '06109.png', '21034.png', '00928.png', '18383.png', '10185.png', '12924.png', '03394.png', '23864.png', '12382.png', '07809.png', '20534.png', '24441.png', '01677.png', '14875.png', '17305.png', '03995.png', '00880.png', '19742.png', '14026.png', '19673.png', '21051.png', '02280.png', '12301.png', '03973.png', '13828.png', '18606.png', '12093.png', '12747.png', '12950.png', '06439.png', '05341.png', '10692.png', '05517.png', '04029.png', '14543.png', '20630.png', '01000.png', '18927.png', '24036.png', '11406.png', '21017.png', '12724.png', '04629.png', '02769.png', '06634.png', '10526.png', '20448.png', '13521.png', '24722.png', '18419.png', '23483.png', '23763.png', '04544.png', '02967.png', '21934.png', '10145.png', '21832.png', '18080.png', '08728.png', '01610.png', '06289.png', '12455.png', '15249.png', '11304.png', '11114.png', '09431.png', '17746.png', '13449.png', '11942.png', '03301.png', '02026.png', '14136.png', '22558.png', '06428.png', '21168.png', '14433.png', '13328.png', '04471.png', '05299.png', '24095.png', '14804.png', '16571.png', '00298.png', '16696.png', '11680.png', '01703.png', '01613.png', '22052.png', '24424.png', '22405.png', '18985.png', '01300.png', '20909.png', '03591.png', '24682.png', '01981.png', '06943.png', '07682.png', '02560.png', '22467.png', '12985.png', '23162.png', '04721.png', '14210.png', '11579.png', '14022.png', '16865.png', '03904.png', '07448.png', '04774.png', '19222.png', '17538.png', '18473.png', '17448.png', '04285.png', '03883.png', '09069.png', '11163.png', '04092.png', '07949.png', '22963.png', '16577.png', '16515.png', '05462.png', '15603.png', '05291.png', '23617.png', '06480.png', '20636.png', '05751.png', '19152.png', '22549.png', '06228.png', '10246.png', '17953.png', '17106.png', '02241.png', '11909.png', '03754.png', '16877.png', '11462.png', '13701.png', '00720.png', '14195.png', '21910.png', '18666.png', '23743.png', '05296.png', '08815.png', '23647.png', '20695.png', '20651.png', '05525.png', '20701.png', '04482.png', '04835.png', '05901.png', '05728.png', '00220.png', '11559.png', '12609.png', '15156.png', '09098.png', '05908.png', '23812.png', '01847.png', '08201.png', '16505.png', '12282.png', '06893.png', '20938.png', '21591.png', '14618.png', '22327.png', '03084.png', '19134.png', '13520.png', '04553.png', '23416.png', '18465.png', '09416.png', '02406.png', '03267.png', '15157.png', '20876.png', '18791.png', '18708.png', '07611.png', '20007.png', '19281.png', '21089.png', '14573.png', '03495.png', '01526.png', '07236.png', '04802.png', '24284.png', '09141.png', '24635.png', '07113.png', '21008.png', '17995.png', '17984.png', '10114.png', '06080.png', '10837.png', '16806.png', '09302.png', '07402.png', '02877.png', '02803.png', '03607.png', '17098.png', '03217.png', '19382.png', '17945.png', '20501.png', '24609.png', '15031.png', '14982.png', '03517.png', '19124.png', '03630.png', '05843.png', '16466.png', '17055.png', '21059.png', '11297.png', '15801.png', '02871.png', '18944.png', '10586.png', '24272.png', '05284.png', '15844.png', '01406.png', '03801.png', '23194.png', '23330.png', '19051.png', '21548.png', '02509.png', '24947.png', '00444.png', '15375.png', '13760.png', '11153.png', '17727.png', '04488.png', '22171.png', '10767.png', '17585.png', '11547.png', '06037.png', '06423.png', '08552.png', '16580.png', '24564.png', '15143.png', '24219.png', '22053.png', '19802.png', '14315.png', '10659.png', '08993.png', '21415.png', '20433.png', '23881.png', '02654.png', '00475.png', '16843.png', '15326.png', '03126.png', '17605.png', '08716.png', '05136.png', '15782.png', '06181.png', '19350.png', '15219.png', '18012.png', '23718.png', '13253.png', '15900.png', '21319.png', '19096.png', '20762.png', '15705.png', '14922.png', '02396.png', '23751.png', '15336.png', '03994.png', '19248.png', '00747.png', '23391.png', '16286.png', '14163.png', '00642.png', '12030.png', '10317.png', '01350.png', '15669.png', '19383.png', '07284.png', '00819.png', '04688.png', '16648.png', '10140.png', '24402.png', '04115.png', '15874.png', '21176.png', '15118.png', '10679.png', '21147.png', '13465.png', '05114.png', '20286.png', '03080.png', '05601.png', '20528.png', '17810.png', '10832.png', '20648.png', '03168.png', '00098.png', '16148.png', '13237.png', '21938.png', '11182.png', '08142.png', '02870.png', '12065.png', '24733.png', '13726.png', '20053.png', '17804.png', '02544.png', '13384.png', '19839.png', '15985.png', '04510.png', '15040.png', '12829.png', '05062.png', '06885.png', '01443.png', '11757.png', '17059.png', '15478.png', '07447.png', '02637.png', '07027.png', '15061.png', '09806.png', '17345.png', '17930.png', '07685.png', '10788.png', '12709.png', '05754.png', '05747.png', '05057.png', '05917.png', '09074.png', '21596.png', '05721.png', '04686.png', '12000.png', '21779.png', '05595.png', '16618.png', '02897.png', '01059.png', '01623.png', '21768.png', '23634.png', '00262.png', '17196.png', '24348.png', '00541.png', '18179.png', '23439.png', '10439.png', '14510.png', '21874.png', '10345.png', '17088.png', '02112.png', '14523.png', '10374.png', '06910.png', '10527.png', '21296.png', '02461.png', '19240.png', '23576.png', '09507.png', '22614.png', '08641.png', '07788.png', '11621.png', '22910.png', '05253.png', '09570.png', '17736.png', '06462.png', '15113.png', '19975.png', '05426.png', '23263.png', '04819.png', '17114.png', '24890.png', '16179.png', '20176.png', '20066.png', '22151.png', '23379.png', '06903.png', '11891.png', '22601.png', '08306.png', '10996.png', '16796.png', '16300.png', '04148.png', '08847.png', '16413.png', '23856.png', '09698.png', '19395.png', '03206.png', '17551.png', '11922.png', '13941.png', '21285.png', '24218.png', '06239.png', '01072.png', '02729.png', '23516.png', '15736.png', '03870.png', '05013.png', '02071.png', '17454.png', '02460.png', '24600.png', '06450.png', '03048.png', '00352.png', '19467.png', '11116.png', '23875.png', '03389.png', '04376.png', '05440.png', '24664.png', '03466.png', '07605.png', '13021.png', '21740.png', '13488.png', '15433.png', '24421.png', '15568.png', '03945.png', '03062.png', '02988.png', '05004.png', '06342.png', '03266.png', '13103.png', '16026.png', '17431.png', '08321.png', '12939.png', '16511.png', '12422.png', '18655.png', '22822.png', '09611.png', '07344.png', '23614.png', '09748.png', '06191.png', '24262.png', '14445.png', '00323.png', '21918.png', '03640.png', '18324.png', '24359.png', '17181.png', '08604.png', '19147.png', '07560.png', '08095.png', '22830.png', '02638.png', '11052.png', '13205.png', '13354.png', '11055.png', '05375.png', '23607.png', '18613.png', '00895.png', '16338.png', '07646.png', '24323.png', '07545.png', '17325.png', '10024.png', '18193.png', '11115.png', '05452.png', '08019.png', '16518.png', '22407.png', '18372.png', '07015.png', '07292.png', '08884.png', '16475.png', '04515.png', '20381.png', '13432.png', '09429.png', '08554.png', '07711.png', '06082.png', '01755.png', '18500.png', '16509.png', '07894.png', '15695.png', '10148.png', '06517.png', '12421.png', '23792.png', '00100.png', '18449.png', '21208.png', '10265.png', '18101.png', '02742.png', '19608.png', '19864.png', '24668.png', '06163.png', '24266.png', '07486.png', '11020.png', '05733.png', '10194.png', '13838.png', '24735.png', '17310.png', '18224.png', '14294.png', '02798.png', '21446.png', '13409.png', '05028.png', '13037.png', '04262.png', '05262.png', '09252.png', '21366.png', '11430.png', '03744.png', '16659.png', '15222.png', '24361.png', '11076.png', '19563.png', '03014.png', '06723.png', '10009.png', '02657.png', '20514.png', '22466.png', '19916.png', '16007.png', '06502.png', '19307.png', '11150.png', '06540.png', '16076.png', '14791.png', '16957.png', '04870.png', '07806.png', '23067.png', '23080.png', '07239.png', '22641.png', '24419.png', '09029.png', '02449.png', '16524.png', '11073.png', '22337.png', '00583.png', '05357.png', '20398.png', '13849.png', '18723.png', '00155.png', '01720.png', '08035.png', '13537.png', '07271.png', '07237.png', '05657.png', '19370.png', '07993.png', '02516.png', '11733.png', '02557.png', '01074.png', '11366.png', '22473.png', '08372.png', '23793.png', '15237.png', '19741.png', '22071.png', '20957.png', '16204.png', '20591.png', '19835.png', '12770.png', '02392.png', '08845.png', '15407.png', '04713.png', '15471.png', '16426.png', '23891.png', '12262.png', '18091.png', '10170.png', '07364.png', '02606.png', '16650.png', '03916.png', '00334.png', '18608.png', '04677.png', '16587.png', '22272.png', '21803.png', '19495.png', '14275.png', '21409.png', '11923.png', '21310.png', '00858.png', '02200.png', '03671.png', '11954.png', '08068.png', '14984.png', '17762.png', '05653.png', '18727.png', '05371.png', '03244.png', '00292.png', '21096.png', '23616.png', '10298.png', '00758.png', '05718.png', '13791.png', '10086.png', '19572.png', '00848.png', '19266.png', '03430.png', '06043.png', '15981.png', '00988.png', '19254.png', '01751.png', '17075.png', '14849.png', '09152.png', '16980.png', '13899.png', '21269.png', '09508.png', '07512.png', '23833.png', '21667.png', '11665.png', '14063.png', '18742.png', '04480.png', '05615.png', '09036.png', '16369.png', '23651.png', '13680.png', '18248.png', '24942.png', '14697.png', '11379.png', '03792.png', '10945.png', '18699.png', '10426.png', '00418.png', '09109.png', '12785.png', '18929.png', '14806.png', '02196.png', '19870.png', '18132.png', '22530.png', '12684.png', '12552.png', '13528.png', '08893.png', '11571.png', '13524.png', '02271.png', '07429.png', '14319.png', '06818.png', '13999.png', '09997.png', '08066.png', '21192.png', '24128.png', '13249.png', '15447.png', '19099.png', '18312.png', '20593.png', '03802.png', '01178.png', '04900.png', '01256.png', '09086.png', '07762.png', '06635.png', '15865.png', '15563.png', '13230.png', '21826.png', '22393.png', '21348.png', '09662.png', '10071.png', '22664.png', '17666.png', '12390.png', '14938.png', '09580.png', '10643.png', '15852.png', '04503.png', '19144.png', '13222.png', '15387.png', '22064.png', '06604.png', '00024.png', '18793.png', '11417.png', '03723.png', '06757.png', '03321.png', '14468.png', '07461.png', '22657.png', '18471.png', '19881.png', '13073.png', '06204.png', '14853.png', '00513.png', '10229.png', '15241.png', '06834.png', '06416.png', '06380.png', '24320.png', '13381.png', '19304.png', '18296.png', '06532.png', '15543.png', '08738.png', '09624.png', '10484.png', '03187.png', '19547.png', '12917.png', '03154.png', '06641.png', '22067.png', '07708.png', '15177.png', '13426.png', '05499.png', '19586.png', '00048.png', '18564.png', '02588.png', '17142.png', '04138.png', '01614.png', '07521.png', '04228.png', '18629.png', '07240.png', '05229.png', '05575.png', '21018.png', '15932.png', '22356.png', '06766.png', '20705.png', '12749.png', '17471.png', '15853.png', '20306.png', '04548.png', '02398.png', '20026.png', '20729.png', '04156.png', '00940.png', '05749.png', '21589.png', '11121.png', '16318.png', '18313.png', '00989.png', '12247.png', '22159.png', '04344.png', '10739.png', '22447.png', '22041.png', '11650.png', '24118.png', '23095.png', '20227.png', '04189.png', '23697.png', '20900.png', '08540.png', '13910.png', '14204.png', '07353.png', '01760.png', '20072.png', '16001.png', '04193.png', '17384.png', '08766.png', '20993.png', '08309.png', '15595.png', '22639.png', '24457.png', '17233.png', '16396.png', '02403.png', '09471.png', '09333.png', '12341.png', '12662.png', '04054.png', '12839.png', '09678.png', '22827.png', '23085.png', '08692.png', '16766.png', '15300.png', '11620.png', '21261.png', '18498.png', '07425.png', '04946.png', '16617.png', '18567.png', '03790.png', '13982.png', '01199.png', '24446.png', '16125.png', '07350.png', '02257.png', '08205.png', '12857.png', '03976.png', '06706.png', '07527.png', '13963.png', '23318.png', '20013.png', '20825.png', '21898.png', '14324.png', '02981.png', '23626.png', '21188.png', '09359.png', '06022.png', '16669.png', '13891.png', '09005.png', '19859.png', '15809.png', '08189.png', '05548.png', '09716.png', '07644.png', '08427.png', '13729.png', '23519.png', '17343.png', '22258.png', '05153.png', '02404.png', '12546.png', '18766.png', '21228.png', '03785.png', '23046.png', '14456.png', '06224.png', '03049.png', '05713.png', '13244.png', '21072.png', '06548.png', '16178.png', '20252.png', '11999.png', '06451.png', '22751.png', '19291.png', '19312.png', '11809.png', '22954.png', '24248.png', '12773.png', '14511.png', '06196.png', '09524.png', '00863.png', '14481.png', '20697.png', '19259.png', '12087.png', '18907.png', '04948.png', '18280.png', '21707.png', '07069.png', '05035.png', '19422.png', '23273.png', '00663.png', '22877.png', '13782.png', '09854.png', '00429.png', '11773.png', '07265.png', '23705.png', '15333.png', '03119.png', '07290.png', '15488.png', '02890.png', '12123.png', '09674.png', '24035.png', '13624.png', '06704.png', '19977.png', '00208.png', '24867.png', '00368.png', '05641.png', '19002.png', '06933.png', '02621.png', '18104.png', '11655.png', '01093.png', '09617.png', '11771.png', '14224.png', '23168.png', '02110.png', '05672.png', '03468.png', '09968.png', '08061.png', '14340.png', '22166.png', '00320.png', '07751.png', '14537.png', '23623.png', '16326.png', '05395.png', '05278.png', '02702.png', '10917.png', '18631.png', '08532.png', '21831.png', '18095.png', '00460.png', '13710.png', '05999.png', '17457.png', '03427.png', '16772.png', '11840.png', '00209.png', '18980.png', '24758.png', '09422.png', '22511.png', '10329.png', '20828.png', '05628.png', '18446.png', '21340.png', '06304.png', '22022.png', '14698.png', '11230.png', '02676.png', '09288.png', '00091.png', '10550.png', '05536.png', '22677.png', '13898.png', '20529.png', '07219.png', '11948.png', '05613.png', '06467.png', '15717.png', '23107.png', '24956.png', '21999.png', '17302.png', '22210.png', '01705.png', '04200.png', '21749.png', '05430.png', '10218.png', '14464.png', '14554.png', '08123.png', '06697.png', '10742.png', '01392.png', '08049.png', '02230.png', '16800.png', '15739.png', '13832.png', '19380.png', '06202.png', '12137.png', '12682.png', '21229.png', '20293.png', '24258.png', '24226.png', '08224.png', '12336.png', '05583.png', '24319.png', '06922.png', '22949.png', '04796.png', '11841.png', '09452.png', '07624.png', '17464.png', '06350.png', '02367.png', '12621.png', '08996.png', '20940.png', '11274.png', '19570.png', '08243.png', '11453.png', '20997.png', '22020.png', '22535.png', '02446.png', '05722.png', '14358.png', '03764.png', '05249.png', '06003.png', '08181.png', '19888.png', '14652.png', '03793.png', '05092.png', '09392.png', '12650.png', '00578.png', '06265.png', '15153.png', '02302.png', '06201.png', '20914.png', '12668.png', '09832.png', '17602.png', '23745.png', '17525.png', '11689.png', '02133.png', '10456.png', '03529.png', '05857.png', '03648.png', '13630.png', '07175.png', '15141.png', '04318.png', '07266.png', '20264.png', '06856.png', '03118.png', '08196.png', '12160.png', '23019.png', '03650.png', '18574.png', '03227.png', '04346.png', '02910.png', '13192.png', '04028.png', '17673.png', '17886.png', '19229.png', '08940.png', '14207.png', '18008.png', '18496.png', '16382.png', '04708.png', '16447.png', '08438.png', '08287.png', '11652.png', '13642.png', '20365.png', '16232.png', '23025.png', '09853.png', '22284.png', '19810.png', '19198.png', '17495.png', '03506.png', '11384.png', '23901.png', '03885.png', '07156.png', '07816.png', '24871.png', '03662.png', '18892.png', '19260.png', '05564.png', '00076.png', '07606.png', '18021.png', '24637.png', '13220.png', '22433.png', '21843.png', '05706.png', '23475.png', '00358.png', '18439.png', '08430.png', '18970.png', '06379.png', '04430.png', '09304.png', '23773.png', '07166.png', '00544.png', '22082.png', '08827.png', '11489.png', '24184.png', '06031.png', '17021.png', '00980.png', '06362.png', '05231.png', '22204.png', '03978.png', '16651.png', '22101.png', '08487.png', '16287.png', '01446.png', '01897.png', '04273.png', '04679.png', '00172.png', '20151.png', '00300.png', '10070.png', '05592.png', '08553.png', '17731.png', '17453.png', '07342.png', '02338.png', '16445.png', '05097.png', '14058.png', '20763.png', '17685.png', '19080.png', '02716.png', '08620.png', '02205.png', '02685.png', '06633.png', '17619.png', '18127.png', '08537.png', '06372.png', '15055.png', '19230.png', '17229.png', '19497.png', '14190.png', '08071.png', '13218.png', '03783.png', '00221.png', '07019.png', '18342.png', '15610.png', '04909.png', '15825.png', '10102.png', '09107.png', '14482.png', '06030.png', '06500.png', '03948.png', '01775.png', '11632.png', '04021.png', '00389.png', '24676.png', '21621.png', '00738.png', '14945.png', '24502.png', '12005.png', '00425.png', '14330.png', '04925.png', '04072.png', '17294.png', '13009.png', '03715.png', '11824.png', '10353.png', '20444.png', '10886.png', '07569.png', '10578.png', '22417.png', '01679.png', '13334.png', '21828.png', '07387.png', '11097.png', '07670.png', '05094.png', '04191.png', '10765.png', '13884.png', '22833.png', '15998.png', '22212.png', '00017.png', '19645.png', '22731.png', '15172.png', '23919.png', '19003.png', '13618.png', '23148.png', '16467.png', '09058.png', '01736.png', '19755.png', '04317.png', '24883.png', '20730.png', '11705.png', '03285.png', '06840.png', '24625.png', '00150.png', '21796.png', '22040.png', '00545.png', '01447.png', '23846.png', '09454.png', '17863.png', '05283.png', '08048.png', '01254.png', '21944.png', '12454.png', '03861.png', '11248.png', '13992.png', '23664.png', '10852.png', '14099.png', '09342.png', '02191.png', '13176.png', '01467.png', '10528.png', '13182.png', '00062.png', '23657.png', '02289.png', '03109.png', '19631.png', '17230.png', '23218.png', '08478.png', '24393.png', '01399.png', '07004.png', '06917.png', '00675.png', '02833.png', '20240.png', '12894.png', '18736.png', '02966.png', '21951.png', '11534.png', '15962.png', '18915.png', '06225.png', '10939.png', '01187.png', '22584.png', '14229.png', '05058.png', '13761.png', '21502.png', '03093.png', '20145.png', '13758.png', '20932.png', '04726.png', '13383.png', '00324.png', '21733.png', '06443.png', '23936.png', '21785.png', '02998.png', '09615.png', '07129.png', '08673.png', '21851.png', '11454.png', '17190.png', '10050.png', '12815.png', '05007.png', '13115.png', '02515.png', '05597.png', '24038.png', '06413.png', '13336.png', '15170.png', '09949.png', '07498.png', '08389.png', '19417.png', '24917.png', '07778.png', '14970.png', '13479.png', '03151.png', '09475.png', '21995.png', '16551.png', '13284.png', '22368.png', '05645.png', '14467.png', '17672.png', '00766.png', '12883.png', '04341.png', '18573.png', '10138.png', '23316.png', '23852.png', '05887.png', '11723.png', '13711.png', '09740.png', '08564.png', '22199.png', '03460.png', '14963.png', '13061.png', '14006.png', '06737.png', '05511.png', '20156.png', '06092.png', '20990.png', '19790.png', '06240.png', '10160.png', '00269.png', '03378.png', '24785.png', '09006.png', '24122.png', '01164.png', '19566.png', '10088.png', '07539.png', '02753.png', '09020.png', '22903.png', '04459.png', '01758.png', '05502.png', '22260.png', '22761.png', '13787.png', '15972.png', '05170.png', '12511.png', '23835.png', '06698.png', '14096.png', '24693.png', '03180.png', '05282.png', '01345.png', '05686.png', '23423.png', '05869.png', '03184.png', '20715.png', '08215.png', '15584.png', '10149.png', '00350.png', '18788.png', '15432.png', '01151.png', '19347.png', '00709.png', '22747.png', '12371.png', '09026.png', '03811.png', '01624.png', '17459.png', '19143.png', '20522.png', '05189.png', '01565.png', '22796.png', '09526.png', '23656.png', '13673.png', '12766.png', '11827.png', '11645.png', '18125.png', '24155.png', '18829.png', '14083.png', '18966.png', '23059.png', '01473.png', '04866.png', '23761.png', '04781.png', '04313.png', '11832.png', '18032.png', '10202.png', '20383.png', '24249.png', '14246.png', '18959.png', '15566.png', '11062.png', '23533.png', '22559.png', '23854.png', '07861.png', '17134.png', '07548.png', '14347.png', '01109.png', '21563.png', '07853.png', '06074.png', '18061.png', '24366.png', '12156.png', '18801.png', '24670.png', '22728.png', '12278.png', '15041.png', '06099.png', '00507.png', '20671.png', '14007.png', '21321.png', '06507.png', '23887.png', '11371.png', '02409.png', '14440.png', '11533.png', '16703.png', '18073.png', '10473.png', '19853.png', '22276.png', '07630.png', '18667.png', '05210.png', '08178.png', '22434.png', '04321.png', '08593.png', '20218.png', '22536.png', '06004.png', '03850.png', '20633.png', '02710.png', '04428.png', '00572.png', '20743.png', '19912.png', '10597.png', '18301.png', '01337.png', '20551.png', '14397.png', '20588.png', '02058.png', '14793.png', '19959.png', '04687.png', '00282.png', '05075.png', '15523.png', '15120.png', '07829.png', '15704.png', '22904.png', '07969.png', '22739.png', '23587.png', '19990.png', '22283.png', '14048.png', '12784.png', '08601.png', '07372.png', '15420.png', '03170.png', '02583.png', '01315.png', '05434.png', '18263.png', '13934.png', '20054.png', '09935.png', '16793.png', '02019.png', '12561.png', '08661.png', '04872.png', '00603.png', '20926.png', '10940.png', '21054.png', '00648.png', '13275.png', '09626.png', '17427.png', '08262.png', '08171.png', '22112.png', '11982.png', '16775.png', '06806.png', '21165.png', '02354.png', '01383.png', '06520.png', '04190.png', '08494.png', '17036.png', '02590.png', '21949.png', '04585.png', '22906.png', '21245.png', '04747.png', '23595.png', '10868.png', '07442.png', '03458.png', '07739.png', '07573.png', '13261.png', '17285.png', '04232.png', '08377.png', '04761.png', '00349.png', '19404.png', '05024.png', '04132.png', '06095.png', '17517.png', '17974.png', '10836.png', '01548.png', '21754.png', '14689.png', '21997.png', '22237.png', '13344.png', '02315.png', '09655.png', '01420.png', '23701.png', '13361.png', '07827.png', '24662.png', '10411.png', '07633.png', '04903.png', '12193.png', '15893.png', '02013.png', '01296.png', '14800.png', '12531.png', '18133.png', '11858.png', '01127.png', '11836.png', '03486.png', '09939.png', '02003.png', '13386.png', '03866.png', '18551.png', '12462.png', '03340.png', '19738.png', '06401.png', '16927.png', '24467.png', '01157.png', '13373.png', '23428.png', '15505.png', '22476.png', '09045.png', '03318.png', '15085.png', '05993.png', '17265.png', '09604.png', '06955.png', '16895.png', '01783.png', '03203.png', '10263.png', '19862.png', '19184.png', '17064.png', '23503.png', '08492.png', '23248.png', '07748.png', '05439.png', '11552.png', '18096.png', '05490.png', '15178.png', '13028.png', '04182.png', '22449.png', '21652.png', '14236.png', '09011.png', '05016.png', '15785.png', '13372.png', '16671.png', '22362.png', '14758.png', '14966.png', '21595.png', '14858.png', '14391.png', '10701.png', '00515.png', '12700.png', '17215.png', '00069.png', '16255.png', '02265.png', '18932.png', '02905.png', '01529.png', '08608.png', '12563.png', '23362.png', '08773.png', '04740.png', '08749.png', '08676.png', '16971.png', '16305.png', '22777.png', '03697.png', '03820.png', '09262.png', '19739.png', '23138.png', '07355.png', '14909.png', '04829.png', '22632.png', '12869.png', '24234.png', '12209.png', '01516.png', '06276.png', '19714.png', '13351.png', '22115.png', '04785.png', '12816.png', '07455.png', '03683.png', '04562.png', '10884.png', '11588.png', '10027.png', '10215.png', '19892.png', '14943.png', '11825.png', '19951.png', '19289.png', '01721.png', '13946.png', '16835.png', '01227.png', '01481.png', '06584.png', '11211.png', '12111.png', '01403.png', '05737.png', '16495.png', '19942.png', '24232.png', '01789.png', '15753.png', '02925.png', '02807.png', '17895.png', '15240.png', '06522.png', '00652.png', '20780.png', '09364.png', '16736.png', '17959.png', '11021.png', '16657.png', '02959.png', '06432.png', '11466.png', '01453.png', '14950.png', '10625.png', '08165.png', '21866.png', '08809.png', '19905.png', '15746.png', '14954.png', '07947.png', '13050.png', '22803.png', '10272.png', '07513.png', '04494.png', '20353.png', '07691.png', '08907.png', '07398.png', '00712.png', '16620.png', '06069.png', '23357.png', '05332.png', '19411.png', '19167.png', '07140.png', '20592.png', '07847.png', '01041.png', '18721.png', '23610.png', '01124.png', '14753.png', '07869.png', '01131.png', '03736.png', '21675.png', '19166.png', '10930.png', '02316.png', '14550.png', '04158.png', '24495.png', '12313.png', '08076.png', '18575.png', '05084.png', '07792.png', '15230.png', '11263.png', '05744.png', '11498.png', '17254.png', '10675.png', '14321.png', '21994.png', '23709.png', '01660.png', '08255.png', '04281.png', '08772.png', '12891.png', '13970.png', '06032.png', '05916.png', '18016.png', '00319.png', '10529.png', '08994.png', '04093.png', '00650.png', '07971.png', '09547.png', '11330.png', '06558.png', '01998.png', '12809.png', '08403.png', '17399.png', '03353.png', '22369.png', '11943.png', '18946.png', '14296.png', '14810.png', '06675.png', '22469.png', '21695.png', '04783.png', '03968.png', '21574.png', '13938.png', '00798.png', '17561.png', '10537.png', '03938.png', '09080.png', '09404.png', '24831.png', '21465.png', '19874.png', '23649.png', '05897.png', '19753.png', '13873.png', '17253.png', '04044.png', '14816.png', '01327.png', '09258.png', '18129.png', '12331.png', '00132.png', '15867.png', '18482.png', '04630.png', '16055.png', '18071.png', '02872.png', '04168.png', '21713.png', '15084.png', '05704.png', '17897.png', '16182.png', '18947.png', '21515.png', '05086.png', '12576.png', '20322.png', '13216.png', '03222.png', '17386.png', '00253.png', '01200.png', '11900.png', '00374.png', '24930.png', '20660.png', '04837.png', '08435.png', '12976.png', '07347.png', '23141.png', '17942.png', '22644.png', '07904.png', '14095.png', '17716.png', '18757.png', '17632.png', '01451.png', '13609.png', '17115.png', '10184.png', '10252.png', '19666.png', '12330.png', '10721.png', '03420.png', '17964.png', '14203.png', '18185.png', '18821.png', '10800.png', '22250.png', '04620.png', '02309.png', '11063.png', '14968.png', '02098.png', '02804.png', '17083.png', '01728.png', '07608.png', '01316.png', '03663.png', '17282.png', '00182.png', '06149.png', '11143.png', '06934.png', '20010.png', '23222.png', '11624.png', '13459.png', '15673.png', '10854.png', '07221.png', '14066.png', '03433.png', '18440.png', '21646.png', '05066.png', '20137.png', '02450.png', '24023.png', '18421.png', '00926.png', '15823.png', '05119.png', '13633.png', '04651.png', '14643.png', '23867.png', '08107.png', '03235.png', '11806.png', '15860.png', '15364.png', '15322.png', '14894.png', '24784.png', '21980.png', '11184.png', '22338.png', '14201.png', '22028.png', '03476.png', '09610.png', '17834.png', '07583.png', '21048.png', '00423.png', '03592.png', '21416.png', '02574.png', '13926.png', '02893.png', '07168.png', '00477.png', '16171.png', '19510.png', '08264.png', '14438.png', '20934.png', '19770.png', '20379.png', '07852.png', '00401.png', '16296.png', '09974.png', '08122.png', '24135.png', '05586.png', '06320.png', '24518.png', '12723.png', '17509.png', '06077.png', '22043.png', '12453.png', '13136.png', '18353.png', '15996.png', '04353.png', '14109.png', '04225.png', '24658.png', '20518.png', '01759.png', '00113.png', '01786.png', '17044.png', '14267.png', '04415.png', '07030.png', '14054.png', '01338.png', '04683.png', '22217.png', '15099.png', '12800.png', '24074.png', '20645.png', '06570.png', '08173.png', '05418.png', '13124.png', '23598.png', '03721.png', '10201.png', '05342.png', '17592.png', '08079.png', '15817.png', '16051.png', '11811.png', '03513.png', '07939.png', '19466.png', '06156.png', '08768.png', '09097.png', '07029.png', '16319.png', '05002.png', '13834.png', '11280.png', '05218.png', '14675.png', '11540.png', '05380.png', '08294.png', '12754.png', '24004.png', '17042.png', '01636.png', '20496.png', '07109.png', '11975.png', '10167.png', '07091.png', '06650.png', '24429.png', '09640.png', '18064.png', '20339.png', '02159.png', '01499.png', '10953.png', '20450.png', '16402.png', '06862.png', '15485.png', '24732.png', '01423.png', '24142.png', '04880.png', '09576.png', '16371.png', '05455.png', '09517.png', '17951.png', '21712.png', '09146.png', '13994.png', '14919.png', '06803.png', '16393.png', '20917.png', '12833.png', '02539.png', '17519.png', '10163.png', '02010.png', '02414.png', '09679.png', '19683.png', '16154.png', '13183.png', '24464.png', '08970.png', '03309.png', '11674.png', '15979.png', '22790.png', '12902.png', '24240.png', '16345.png', '05338.png', '08816.png', '01961.png', '03892.png', '03044.png', '02140.png', '05227.png', '23297.png', '14435.png', '07420.png', '00543.png', '15413.png', '15615.png', '23011.png', '23041.png', '11684.png', '08303.png', '11994.png', '04755.png', '00239.png', '02059.png', '08583.png', '04791.png', '22068.png', '20865.png', '16443.png', '17578.png', '18422.png', '07597.png', '00686.png', '01816.png', '12308.png', '14359.png', '21959.png', '04035.png', '10490.png', '14144.png', '11262.png', '19541.png', '08855.png', '11181.png', '18106.png', '20985.png', '00820.png', '11611.png', '06068.png', '05047.png', '10254.png', '21697.png', '08000.png', '18710.png', '13700.png', '23482.png', '01913.png', '13797.png', '15037.png', '03328.png', '15909.png', '15698.png', '02970.png', '12400.png', '23119.png', '14783.png', '11328.png', '10334.png', '14998.png', '06147.png', '02361.png', '17101.png', '00716.png', '17728.png', '11614.png', '12216.png', '09519.png', '11303.png', '02199.png', '10240.png', '06751.png', '05081.png', '18713.png', '05023.png', '24631.png', '21035.png', '01324.png', '21679.png', '02118.png', '10127.png', '04793.png', '22744.png', '16883.png', '01711.png', '18679.png', '02675.png', '18232.png', '16926.png', '23635.png', '09563.png', '17947.png', '19238.png', '02101.png', '17151.png', '04268.png', '22706.png', '19250.png', '18725.png', '05135.png', '19983.png', '13062.png', '20299.png', '24888.png', '17192.png', '14969.png', '07396.png', '07036.png', '19316.png', '12496.png', '16968.png', '08315.png', '17378.png', '13105.png', '16119.png', '02365.png', '15087.png', '24698.png', '17714.png', '09849.png', '20605.png', '09987.png', '14728.png', '08695.png', '06875.png', '15738.png', '16073.png', '16169.png', '09840.png', '07746.png', '14627.png', '16164.png', '20302.png', '07652.png', '02418.png', '04922.png', '08085.png', '05966.png', '01881.png', '24960.png', '19186.png', '20317.png', '20231.png', '13242.png', '14479.png', '00156.png', '23810.png', '11346.png', '22658.png', '02826.png', '24950.png', '22102.png', '20608.png', '01349.png', '11146.png', '08509.png', '19506.png', '15923.png', '23788.png', '04018.png', '18656.png', '02983.png', '21174.png', '11139.png', '10316.png', '16499.png', '01583.png', '17236.png', '18223.png', '03576.png', '15257.png', '23781.png', '01568.png', '06884.png', '04212.png', '11805.png', '08808.png', '23246.png', '05373.png', '23659.png', '20684.png', '24742.png', '18041.png', '02651.png', '14002.png', '21539.png', '04043.png', '14042.png', '00718.png', '19069.png', '24333.png', '06686.png', '13502.png', '01896.png', '17639.png', '14345.png', '19029.png', '14274.png', '04106.png', '00412.png', '01341.png', '23740.png', '04573.png', '19751.png', '08233.png', '10066.png', '18236.png', '20965.png', '04951.png', '03445.png', '21933.png', '23977.png', '12457.png', '04520.png', '18871.png', '05040.png', '17076.png', '03358.png', '07365.png', '10584.png', '05643.png', '06452.png', '14872.png', '10786.png', '04392.png', '07620.png', '10726.png', '23164.png', '16306.png', '24936.png', '24519.png', '02408.png', '11314.png', '22729.png', '20203.png', '14326.png', '21453.png', '18941.png', '11894.png', '01553.png', '20435.png', '07720.png', '02261.png', '21247.png', '10817.png', '20397.png', '05120.png', '06659.png', '03415.png', '18560.png', '07942.png', '02793.png', '19485.png', '11484.png', '24015.png', '16094.png', '00219.png', '20971.png', '11949.png', '16042.png', '21360.png', '23027.png', '17850.png', '23352.png', '01937.png', '02185.png', '01465.png', '14620.png', '03156.png', '01513.png', '07035.png', '12165.png', '09352.png', '20892.png', '10843.png', '24451.png', '07061.png', '24228.png', '08096.png', '23146.png', '19760.png', '16964.png', '24835.png', '01248.png', '02857.png', '04293.png', '03295.png', '14947.png', '21238.png', '24399.png', '11082.png', '09856.png', '15261.png', '22520.png', '11023.png', '09962.png', '16458.png', '17985.png', '10067.png', '20074.png', '14212.png', '06996.png', '20196.png', '14496.png', '06184.png', '22075.png', '12166.png', '03511.png', '15091.png', '05251.png', '02770.png', '02079.png', '18162.png', '20298.png', '05050.png', '04757.png', '22767.png', '07067.png', '14183.png', '05874.png', '08508.png', '06400.png', '08337.png', '13800.png', '08477.png', '18244.png', '22085.png', '18792.png', '11787.png', '23455.png', '08812.png', '15199.png', '22577.png', '03391.png', '18375.png', '11899.png', '19955.png', '09558.png', '08818.png', '07515.png', '16849.png', '08633.png', '07311.png', '06948.png', '14230.png', '12248.png', '16082.png', '13321.png', '22533.png', '07730.png', '17833.png', '09154.png', '06408.png', '02511.png', '20418.png', '09408.png', '04580.png', '02596.png', '14636.png', '06473.png', '24270.png', '19560.png', '00228.png', '07060.png', '02466.png', '14344.png', '03822.png', '09928.png', '00744.png', '08779.png', '01371.png', '23564.png', '18676.png', '00126.png', '08959.png', '23989.png', '15306.png', '00717.png', '10442.png', '09002.png', '13996.png', '21184.png', '11424.png', '00619.png', '03075.png', '07012.png', '23339.png', '19573.png', '18251.png', '17414.png', '07781.png', '06887.png', '08323.png', '08714.png', '12768.png', '01448.png', '15111.png', '18777.png', '02211.png', '20184.png', '02182.png', '24410.png', '08289.png', '24583.png', '22734.png', '02174.png', '22713.png', '07609.png', '17875.png', '05711.png', '06119.png', '12211.png', '08551.png', '16351.png', '17516.png', '05454.png', '04248.png', '17372.png', '10435.png', '18204.png', '08324.png', '19183.png', '15119.png', '12047.png', '09945.png', '24191.png', '00130.png', '19151.png', '22081.png', '18152.png', '21941.png', '18665.png', '18926.png', '02960.png', '22168.png', '16534.png', '00693.png', '16819.png', '09108.png', '21700.png', '08156.png', '16422.png', '14117.png', '14676.png', '00538.png', '17950.png', '21305.png', '23346.png', '02775.png', '08839.png', '06378.png', '02687.png', '17870.png', '17882.png', '03481.png', '23851.png', '23719.png', '16408.png', '03079.png', '03706.png', '15870.png', '10736.png', '23799.png', '13519.png', '24486.png', '00035.png', '21013.png', '07933.png', '14693.png', '09901.png', '11290.png', '02655.png', '04999.png', '11865.png', '19064.png', '17162.png', '15848.png', '14332.png', '17809.png', '11597.png', '05486.png', '13818.png', '02317.png', '09008.png', '18273.png', '03574.png', '03594.png', '20380.png', '13995.png', '17295.png', '23746.png', '05667.png', '10670.png', '00160.png', '06797.png', '10299.png', '10745.png', '05984.png', '15112.png', '11525.png', '20368.png', '07717.png', '18709.png', '13232.png', '04766.png', '11299.png', '00884.png', '23827.png', '12899.png', '00498.png', '20345.png', '04359.png', '05195.png', '09211.png', '23496.png', '02697.png', '24650.png', '17436.png', '15814.png', '18589.png', '06772.png', '20712.png', '02838.png', '23082.png', '16419.png', '07412.png', '05453.png', '01202.png', '13526.png', '08788.png', '06250.png', '12149.png', '18651.png', '06088.png', '23553.png', '01217.png', '14942.png', '11058.png', '05333.png', '04463.png', '00630.png', '16978.png', '17072.png', '23732.png', '00893.png', '03240.png', '11202.png', '10883.png', '05200.png', '19785.png', '16103.png', '21320.png', '03509.png', '14401.png', '04534.png', '24764.png', '22578.png', '03860.png', '11407.png', '14031.png', '00346.png', '10122.png', '11915.png', '16054.png', '06639.png', '18993.png', '07979.png', '22096.png', '05461.png', '10156.png', '04838.png', '08709.png', '08853.png', '21751.png', '24604.png', '06313.png', '04626.png', '06329.png', '21240.png', '18731.png', '17270.png', '23537.png', '24468.png', '02143.png', '22206.png', '13406.png', '14560.png', '17691.png', '11468.png', '11355.png', '03824.png', '19351.png', '23955.png', '23578.png', '07900.png', '06782.png', '18364.png', '21173.png', '17410.png', '01024.png', '08010.png', '23465.png', '01647.png', '23712.png', '13412.png', '03806.png', '05681.png', '24680.png', '15721.png', '10343.png', '21794.png', '19004.png', '15749.png', '01076.png', '20073.png', '16560.png', '23260.png', '08334.png', '14538.png', '02720.png', '12067.png', '04608.png', '06483.png', '05240.png', '18321.png', '08600.png', '09863.png', '04210.png', '08194.png', '12058.png', '03178.png', '16321.png', '21215.png', '14455.png', '18630.png', '15974.png', '00414.png', '02845.png', '11937.png', '15617.png', '18858.png', '10821.png', '00511.png', '20382.png', '10171.png', '17846.png', '13322.png', '12949.png', '06755.png', '16937.png', '12019.png', '05573.png', '13234.png', '06056.png', '08510.png', '03241.png', '23613.png', '10058.png', '21756.png', '22498.png', '11081.png', '01768.png', '19136.png', '00842.png', '03237.png', '01461.png', '13033.png', '24499.png', '21442.png', '01586.png', '24659.png', '05685.png', '17038.png', '04369.png', '08720.png', '21046.png', '18425.png', '12265.png', '06977.png', '05487.png', '14686.png', '04294.png', '18860.png', '08822.png', '18309.png', '06714.png', '09747.png', '10772.png', '04775.png', '10268.png', '00364.png', '19654.png', '09533.png', '03069.png', '12318.png', '02144.png', '03936.png', '06991.png', '14763.png', '01241.png', '20921.png', '22342.png', '22945.png', '17798.png', '01710.png', '16969.png', '00445.png', '23114.png', '02761.png', '07642.png', '18812.png', '00585.png', '05547.png', '12567.png', '16442.png', '00705.png', '24237.png', '06333.png', '10038.png', '04852.png', '20580.png', '16526.png', '01293.png', '01889.png', '18738.png', '13553.png', '13472.png', '21632.png', '04878.png', '12105.png', '15468.png', '09356.png', '02464.png', '15894.png', '12203.png', '04395.png', '05537.png', '19055.png', '14589.png', '20297.png', '11286.png', '03242.png', '02328.png', '18644.png', '14571.png', '21770.png', '06311.png', '18952.png', '22937.png', '01321.png', '09511.png', '17957.png', '24173.png', '08179.png', '01332.png', '01273.png', '19633.png', '04131.png', '18794.png', '18412.png', '00284.png', '00736.png', '12339.png', '17061.png', '05950.png', '18722.png', '19348.png', '08482.png', '04449.png', '21694.png', '15460.png', '24704.png', '11418.png', '06936.png', '20038.png', '22431.png', '17765.png', '07017.png', '09028.png', '16311.png', '00371.png', '04633.png', '01155.png', '22121.png', '08388.png', '13221.png', '02937.png', '08817.png', '01056.png', '16423.png', '01239.png', '14749.png', '23217.png', '03508.png', '04557.png', '02342.png', '22380.png', '03579.png', '09732.png', '09419.png', '11536.png', '20436.png', '19697.png', '11527.png', '03943.png', '04877.png', '01931.png', '08821.png', '03444.png', '19545.png', '06296.png', '20703.png', '15205.png', '08825.png', '22021.png', '14823.png', '10531.png', '17010.png', '02412.png', '16647.png', '17358.png', '10618.png', '24341.png', '15225.png', '20027.png', '07564.png', '08740.png', '23506.png', '14610.png', '19693.png', '11838.png', '20021.png', '02103.png', '14659.png', '01572.png', '12319.png', '15511.png', '18297.png', '18516.png', '03478.png', '05669.png', '19882.png', '06735.png', '01097.png', '15414.png', '07146.png', '11694.png', '12139.png', '07409.png', '11095.png', '24644.png', '22402.png', '09177.png', '09329.png', '21645.png', '13468.png', '19299.png', '24953.png', '09395.png', '00966.png', '07669.png', '20680.png', '12413.png', '00665.png', '08521.png', '23801.png', '04700.png', '20431.png', '17658.png', '19419.png', '03951.png', '03449.png', '16990.png', '23190.png', '11340.png', '10082.png', '05133.png', '21462.png', '22026.png', '19084.png', '13794.png', '13847.png', '16994.png', '23301.png', '10957.png', '23628.png', '17146.png', '14587.png', '12213.png', '06913.png', '13248.png', '07323.png', '05327.png', '23972.png', '00839.png', '07600.png', '06577.png', '10297.png', '22862.png', '15674.png', '03902.png', '05741.png', '01221.png', '21966.png', '02562.png', '02883.png', '11756.png', '23759.png', '18728.png', '21414.png', '00348.png', '10694.png', '04575.png', '23488.png', '24029.png', '16680.png', '12469.png', '20465.png', '14874.png', '12674.png', '15515.png', '13763.png', '14552.png', '22229.png', '21504.png', '18974.png', '07542.png', '18873.png', '08899.png', '19387.png', '24129.png', '02827.png', '24477.png', '19298.png', '01158.png', '19588.png', '14270.png', '14493.png', '01397.png', '11712.png', '12440.png', '12915.png', '14960.png', '04098.png', '21840.png', '01015.png', '16985.png', '16242.png', '00726.png', '05320.png', '09719.png']
    epochs_since_start = 0
    if True:
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None
        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = a)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('gta size:',len(trainloader))
    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()
    model.cuda()
    model.train()
    #prototype_dist_init(cfg, trainloader, model)
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0
    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)
    
    """
    if True:
        model.eval()
        if dataset == 'cityscapes':
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024))

        model.train()
        print("mIoU: ",mIoU, eval_loss)
    """
    
    accumulated_loss_l = []
    accumulated_loss_u = []
    #accumulated_loss_feat = []
    #accumulated_loss_out = []   
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    
    print(epochs_since_start)
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        #loss_feat_value = 0
        #loss_out_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            if epochs_since_start >= 2:
                list_name = []
            if epochs_since_start == 1:
                data_loader = get_loader('gta')
                data_path = get_data_path('gta')
                if random_crop:
                    data_aug = Compose([RandomCrop_gta(input_size)])
                else:
                    data_aug = None        
                train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = None)
                trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                print('gta size:',len(trainloader))
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        
        weak_parameters={"flip": 0}


        images, labels, _, names = batch
        images = images.cuda()
        labels = labels.cuda().long()
        if epochs_since_start >= 2:
            for name in names:
                list_name.append(name)

        #images, labels = weakTransform(weak_parameters, data = images, target = labels)

        src_pred, src_feat= model(images)
        pred = interp(src_pred)
        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
        #L_l = torch.Tensor([0.0]).cuda()

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            #inputs_u_w = inputs_u_w.clone()
            logits_u_w = interp(ema_model(inputs_u_w)[0])
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            if mix_mask == "class":
                for image_i in range(batch_size):
                    classes = torch.unique(labels[image_i])
                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            inputs_u_s1, _ = strongTransform(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
            inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
            logits_u_s_tgt, tgt_feat = model(inputs_u_s)
            logits_u_s = interp(logits_u_s_tgt)

            strong_parameters["Mix"] = MixMask0
            _, targets_u0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, targets_u1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
            targets_u = torch.cat((targets_u0,targets_u1)).long()
            
            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().cuda()
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).cuda()

            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight)

            loss = L_l + L_u

        else:
            loss = L_l
        """
        # source mask: downsample the ground-truth label
        src_out_ema, src_feat_ema = ema_model(images)
        tgt_out_ema, tgt_feat_ema = ema_model(inputs_u_s)
        B, A, Hs, Ws = src_feat.size()
        src_mask = F.interpolate(labels.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        assert not src_mask.requires_grad
        pseudo_weight = F.interpolate(pixelWiseWeight.unsqueeze(1),
                                         size=(65,65), mode='bilinear',
                                         align_corners=True).squeeze(1)
        
        _, _, Ht, Wt = tgt_feat.size()
        tgt_out_maxvalue, tgt_mask_st = torch.max(tgt_feat_ema, dim=1)
        tgt_mask = F.interpolate(targets_u.unsqueeze(1).float(), size=(65,65), mode='nearest').squeeze(1).long()
        tgt_mask_upt = copy.deepcopy(tgt_mask)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_mask_upt[(((tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask_st == i)).int() + (pseudo_weight != 1.0).int()) == 2] = 255

        tgt_mask = tgt_mask.contiguous().view(B * Hs * Ws, )
        pseudo_weight = pseudo_weight.contiguous().view(B * Hs * Ws, )
        tgt_mask_upt = tgt_mask_upt.contiguous().view(B * Hs * Ws, )
        src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
        src_feat_ema = src_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat_ema = tgt_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        # update feature-level statistics
        feat_estimator.update(features=tgt_feat_ema.detach(), labels=tgt_mask_upt)
        feat_estimator.update(features=src_feat_ema.detach(), labels=src_mask)

        # contrastive loss on both domains
        
        loss_feat = pcl_criterion_src(Proto=feat_estimator.Proto.detach(),
                                  feat=src_feat,
                                  labels=src_mask) \
                    + pcl_criterion_tgt(Proto=feat_estimator.Proto.detach(),
                                  feat=tgt_feat,
                                  labels=tgt_mask, pixelWiseWeight=pseudo_weight)
        #meters.update(loss_feat=loss_feat.item())

        if cfg.SOLVER.MULTI_LEVEL:
            src_out = src_pred.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out = logits_u_s_tgt.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            src_out_ema = src_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out_ema = tgt_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            # update output-level statistics
            out_estimator.update(features=tgt_out_ema.detach(), labels=tgt_mask_upt)
            out_estimator.update(features=src_out_ema.detach(), labels=src_mask)

            # the proposed contrastive loss on prediction map
            loss_out = pcl_criterion_src(Proto=out_estimator.Proto.detach(),
                                     feat=src_out,
                                     labels=src_mask) \
                       + pcl_criterion_tgt(Proto=out_estimator.Proto.detach(),
                                       feat=tgt_out,
                                       labels=tgt_mask, pixelWiseWeight=pseudo_weight)
            #meters.update(loss_out=loss_out.item())

            loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat + cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat
        """
        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_u_value += L_u.item()
            #loss_feat_value += loss_feat.item()
            #loss_out_value += loss_out.item()
        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        #print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))
        
        if i_iter % save_checkpoint_every == 1486 and i_iter!=0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            #feat_estimator.save(name='prototype_feat_dist.pth')
            #out_estimator.save(name='prototype_out_dist.pth')
            print('save_prototype')
        
        if i_iter == 179926:
            print(list_name)
        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

        accumulated_loss_l.append(loss_l_value)
        #accumulated_loss_feat.append(loss_feat_value)
        #accumulated_loss_out.append(loss_out_value)
        if train_unlabeled:
            accumulated_loss_u.append(loss_u_value)
            
        if i_iter % log_per_iter == 0 and i_iter != 0:
                #tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
            #print('Training/contrastive_feat_loss', np.mean(accumulated_loss_feat), 'Training/contrastive_out_loss', np.mean(accumulated_loss_out), i_iter)
            #accumulated_loss_feat = []
            #accumulated_loss_out = []
            if train_unlabeled:
                #tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                print('Training/Supervised loss', np.mean(accumulated_loss_l), 'Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                accumulated_loss_u = []
                accumulated_loss_l = []
        """
        if save_unlabeled_images and train_unlabeled and (i_iter == 5650600):
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input_s1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input_s2',palette.CityScpates_palette)
            save_image(inputs_u_w[0].cpu(),i_iter,'input_w1',palette.CityScpates_palette)
            save_image(inputs_u_w[1].cpu(),i_iter,'input_w2',palette.CityScpates_palette)
            save_image(images[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(images[1].cpu(),i_iter,'input2',palette.CityScpates_palette)

            _, pred_u_s = torch.max(logits_u_w, dim=1)
            #save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            #save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

        """
    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        mIoU, val_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024), save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)


    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()
    

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    #if args.save_images:
    print('Saving unlabeled images')
    save_unlabeled_images = True
    #else:
    #save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]
    #print(args.config_file)    
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()


    main()
