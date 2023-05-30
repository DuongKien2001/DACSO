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
    a = ['22092.png', '22848.png', '02218.png', '03294.png', '03474.png', '12572.png', '02421.png', '00302.png', '16445.png', '12795.png', '05742.png', '13343.png', '02025.png', '18142.png', '20995.png', '08076.png', '08346.png', '23625.png', '11500.png', '08959.png', '09247.png', '24447.png', '22337.png', '08041.png', '21500.png', '23488.png', '01929.png', '21455.png', '21076.png', '06442.png', '03309.png', '13915.png', '07679.png', '01248.png', '05303.png', '20749.png', '00710.png', '17542.png', '06893.png', '00131.png', '12522.png', '14311.png', '02663.png', '00267.png', '01323.png', '23953.png', '04446.png', '05798.png', '19105.png', '01412.png', '04893.png', '21634.png', '09590.png', '14543.png', '22842.png', '11431.png', '16133.png', '20181.png', '04224.png', '20672.png', '15565.png', '11408.png', '13497.png', '21874.png', '14083.png', '01610.png', '23949.png', '00288.png', '22334.png', '19272.png', '23662.png', '00219.png', '14774.png', '07090.png', '12029.png', '02105.png', '04982.png', '08190.png', '07283.png', '01106.png', '13219.png', '02291.png', '05223.png', '12772.png', '05106.png', '11537.png', '16552.png', '14881.png', '18038.png', '23701.png', '10580.png', '12659.png', '10941.png', '01028.png', '24946.png', '11141.png', '00934.png', '14245.png', '00864.png', '01203.png', '12083.png', '24921.png', '09699.png', '18191.png', '21112.png', '12566.png', '09688.png', '06769.png', '09454.png', '05016.png', '18024.png', '18638.png', '11578.png', '12126.png', '22856.png', '08700.png', '00153.png', '11680.png', '02081.png', '13294.png', '06822.png', '20568.png', '10257.png', '04360.png', '11149.png', '00946.png', '06960.png', '02032.png', '18176.png', '21128.png', '19320.png', '09457.png', '12831.png', '22918.png', '09903.png', '13054.png', '12383.png', '21070.png', '02378.png', '14281.png', '24382.png', '07914.png', '19136.png', '04941.png', '05377.png', '08342.png', '13311.png', '24878.png', '10491.png', '15200.png', '21060.png', '07730.png', '08057.png', '16574.png', '02724.png', '04946.png', '03609.png', '03536.png', '12595.png', '18936.png', '19951.png', '15936.png', '24060.png', '20741.png', '12926.png', '05859.png', '24415.png', '18534.png', '20760.png', '11571.png', '04066.png', '02173.png', '07162.png', '17944.png', '19463.png', '16301.png', '02888.png', '03220.png', '24107.png', '03065.png', '15581.png', '18231.png', '05108.png', '23178.png', '11198.png', '09710.png', '24144.png', '00997.png', '15631.png', '07784.png', '19976.png', '23137.png', '06074.png', '06339.png', '03057.png', '05374.png', '18379.png', '24046.png', '00557.png', '00858.png', '01526.png', '20082.png', '24555.png', '18298.png', '08590.png', '19399.png', '03537.png', '00161.png', '03683.png', '12802.png', '21580.png', '01021.png', '08085.png', '06085.png', '22609.png', '24646.png', '02521.png', '02179.png', '22358.png', '15481.png', '12060.png', '08927.png', '09788.png', '13577.png', '13748.png', '01271.png', '09941.png', '05669.png', '15786.png', '03258.png', '17482.png', '13226.png', '10961.png', '23500.png', '21590.png', '21353.png', '13717.png', '23193.png', '05498.png', '03466.png', '12204.png', '13150.png', '21560.png', '21667.png', '15777.png', '13394.png', '05517.png', '19258.png', '12074.png', '05825.png', '05170.png', '14239.png', '14756.png', '13063.png', '00775.png', '00698.png', '23409.png', '20858.png', '12413.png', '00319.png', '12388.png', '02131.png', '16033.png', '03428.png', '03348.png', '16776.png', '10774.png', '13331.png', '07110.png', '11334.png', '24387.png', '08751.png', '04803.png', '19800.png', '02816.png', '09237.png', '01126.png', '18632.png', '04029.png', '14445.png', '21031.png', '02181.png', '13014.png', '14853.png', '24403.png', '03688.png', '21050.png', '11903.png', '05905.png', '18433.png', '05994.png', '07174.png', '06080.png', '18795.png', '17671.png', '15585.png', '05697.png', '24287.png', '24616.png', '02087.png', '18230.png', '14134.png', '11688.png', '24670.png', '02007.png', '19180.png', '13190.png', '22235.png', '05266.png', '11850.png', '17080.png', '20352.png', '20676.png', '07802.png', '24540.png', '20355.png', '04620.png', '09645.png', '04375.png', '17230.png', '08970.png', '00376.png', '24960.png', '05185.png', '01739.png', '13756.png', '13908.png', '06880.png', '11462.png', '16916.png', '13233.png', '09240.png', '15053.png', '06371.png', '21528.png', '13943.png', '15877.png', '12664.png', '03944.png', '04862.png', '17491.png', '15369.png', '24301.png', '17332.png', '19101.png', '05426.png', '03828.png', '07214.png', '19945.png', '00280.png', '11583.png', '16965.png', '11306.png', '00321.png', '18349.png', '07850.png', '22615.png', '00084.png', '18277.png', '19020.png', '09577.png', '15951.png', '16499.png', '06311.png', '20225.png', '12645.png', '15686.png', '10633.png', '18684.png', '19307.png', '04628.png', '01094.png', '02423.png', '16583.png', '16091.png', '14060.png', '04515.png', '09529.png', '17031.png', '17548.png', '01884.png', '06366.png', '21354.png', '07727.png', '12546.png', '18320.png', '04240.png', '03206.png', '08047.png', '21212.png', '19552.png', '09392.png', '22428.png', '03204.png', '12540.png', '09129.png', '09886.png', '02478.png', '04189.png', '02839.png', '01494.png', '20640.png', '17572.png', '16745.png', '21991.png', '14408.png', '10485.png', '21296.png', '09542.png', '12086.png', '05042.png', '05624.png', '17360.png', '00862.png', '03028.png', '15947.png', '08760.png', '01053.png', '14746.png', '22267.png', '16642.png', '08783.png', '13122.png', '01706.png', '02835.png', '10129.png', '09007.png', '08045.png', '12808.png', '07216.png', '15794.png', '08721.png', '12041.png', '19076.png', '05479.png', '14642.png', '15462.png', '12731.png', '22925.png', '20560.png', '09323.png', '21885.png', '01296.png', '01312.png', '24929.png', '01753.png', '09852.png', '24392.png', '17187.png', '00963.png', '08134.png', '05796.png', '22504.png', '20610.png', '14426.png', '20177.png', '22430.png', '17764.png', '23963.png', '18688.png', '24866.png', '09288.png', '09018.png', '23185.png', '05548.png', '21438.png', '23776.png', '14966.png', '19594.png', '01432.png', '08118.png', '01152.png', '05702.png', '12563.png', '13456.png', '04852.png', '09508.png', '05085.png', '14830.png', '22142.png', '01833.png', '00884.png', '13901.png', '01957.png', '18836.png', '11834.png', '04589.png', '23675.png', '02206.png', '09836.png', '19560.png', '14815.png', '17425.png', '23726.png', '21284.png', '01083.png', '16165.png', '08822.png', '19197.png', '08161.png', '00178.png', '03862.png', '13112.png', '03630.png', '07121.png', '12709.png', '22464.png', '16512.png', '23945.png', '23322.png', '20119.png', '12426.png', '16692.png', '17211.png', '02801.png', '15478.png', '15434.png', '18355.png', '13752.png', '11201.png', '00344.png', '22971.png', '11336.png', '13652.png', '06187.png', '23656.png', '20192.png', '21931.png', '02901.png', '20544.png', '08541.png', '06090.png', '22554.png', '05439.png', '23111.png', '08627.png', '02817.png', '21052.png', '18151.png', '14070.png', '24738.png', '11762.png', '12994.png', '19207.png', '20784.png', '10609.png', '23005.png', '07924.png', '05705.png', '12829.png', '14662.png', '19879.png', '01431.png', '17881.png', '08349.png', '01145.png', '05538.png', '22613.png', '06383.png', '24826.png', '17850.png', '06021.png', '03996.png', '20722.png', '07712.png', '23176.png', '15996.png', '01843.png', '01620.png', '12652.png', '19744.png', '06139.png', '21494.png', '01589.png', '09540.png', '04747.png', '04583.png', '22572.png', '05772.png', '21659.png', '17839.png', '19323.png', '17867.png', '16443.png', '07231.png', '09135.png', '07034.png', '07495.png', '17365.png', '08244.png', '21803.png', '20027.png', '18007.png', '15069.png', '22295.png', '16951.png', '02848.png', '21905.png', '18122.png', '09136.png', '23258.png', '19499.png', '19504.png', '01800.png', '03724.png', '19039.png', '05116.png', '06034.png', '21988.png', '09490.png', '17647.png', '19396.png', '06749.png', '22558.png', '18692.png', '18953.png', '05397.png', '11517.png', '13937.png', '12940.png', '13644.png', '15025.png', '09842.png', '22347.png', '13669.png', '09068.png', '13441.png', '06785.png', '13552.png', '14934.png', '11961.png', '13098.png', '24917.png', '08673.png', '18118.png', '07631.png', '08639.png', '14748.png', '01439.png', '14420.png', '16205.png', '14405.png', '01694.png', '17240.png', '09562.png', '10266.png', '04004.png', '10560.png', '16254.png', '24239.png', '11369.png', '09946.png', '09062.png', '11146.png', '19143.png', '05059.png', '23733.png', '19586.png', '10087.png', '18492.png', '22898.png', '13601.png', '20221.png', '23066.png', '19753.png', '08609.png', '18290.png', '04807.png', '02988.png', '14945.png', '22491.png', '06459.png', '20189.png', '16960.png', '01425.png', '08499.png', '12417.png', '05101.png', '24407.png', '03674.png', '23639.png', '12521.png', '16673.png', '17135.png', '04353.png', '02018.png', '19549.png', '20249.png', '24411.png', '07943.png', '07264.png', '11101.png', '04590.png', '07576.png', '09594.png', '02425.png', '13593.png', '08250.png', '15646.png', '16964.png', '04357.png', '04162.png', '09473.png', '06565.png', '22799.png', '06432.png', '12481.png', '15855.png', '16492.png', '06552.png', '23045.png', '05077.png', '15425.png', '12077.png', '20704.png', '01775.png', '07281.png', '24585.png', '23974.png', '01676.png', '11099.png', '01598.png', '06790.png', '21054.png', '07780.png', '15896.png', '15361.png', '17062.png', '23082.png', '03184.png', '08171.png', '01124.png', '06726.png', '00824.png', '14163.png', '20907.png', '12199.png', '20588.png', '02705.png', '02711.png', '18040.png', '15791.png', '19428.png', '01101.png', '09259.png', '02042.png', '14425.png', '11542.png', '22654.png', '15943.png', '12978.png', '18474.png', '09801.png', '08369.png', '19902.png', '18648.png', '13318.png', '24648.png', '15149.png', '05716.png', '23035.png', '01995.png', '09449.png', '17825.png', '22191.png', '14835.png', '23537.png', '13829.png', '20000.png', '17700.png', '11613.png', '04408.png', '19978.png', '06191.png', '03949.png', '11446.png', '18551.png', '21062.png', '16013.png', '22126.png', '21908.png', '19113.png', '19275.png', '05667.png', '04309.png', '08388.png', '17635.png', '02332.png', '11397.png', '17151.png', '12713.png', '04332.png', '22501.png', '12683.png', '05508.png', '01914.png', '14465.png', '22309.png', '18808.png', '10240.png', '15031.png', '16723.png', '01292.png', '24543.png', '05901.png', '22458.png', '09204.png', '19406.png', '06043.png', '06248.png', '13540.png', '13103.png', '04088.png', '14517.png', '16811.png', '19059.png', '04425.png', '09641.png', '23740.png', '04873.png', '03897.png', '16922.png', '12756.png', '20660.png', '05703.png', '03929.png', '00482.png', '18214.png', '08755.png', '22114.png', '22717.png', '12130.png', '10169.png', '20869.png', '20866.png', '12485.png', '10934.png', '08048.png', '10404.png', '24754.png', '19835.png', '00797.png', '01035.png', '21668.png', '05855.png', '21079.png', '23985.png', '12072.png', '23287.png', '14253.png', '02891.png', '07522.png', '07074.png', '15368.png', '09679.png', '04970.png', '24748.png', '23436.png', '19588.png', '02264.png', '10531.png', '00959.png', '09656.png', '23459.png', '14098.png', '24160.png', '04129.png', '01789.png', '09644.png', '09076.png', '06219.png', '14992.png', '05204.png', '22625.png', '01727.png', '04409.png', '15909.png', '03635.png', '16137.png', '19412.png', '15897.png', '11084.png', '24707.png', '11345.png', '22535.png', '19990.png', '13779.png', '08021.png', '13636.png', '18933.png', '10871.png', '22559.png', '09375.png', '13979.png', '21422.png', '15349.png', '10556.png', '04700.png', '07186.png', '23107.png', '09122.png', '07650.png', '01941.png', '09162.png', '04902.png', '23312.png', '21147.png', '02284.png', '20509.png', '17068.png', '24052.png', '14313.png', '20096.png', '21359.png', '16451.png', '06280.png', '23486.png', '05687.png', '00600.png', '12517.png', '11227.png', '01497.png', '18284.png', '07357.png', '04416.png', '10410.png', '10686.png', '09554.png', '05353.png', '01878.png', '08154.png', '00312.png', '10622.png', '15337.png', '21357.png', '04773.png', '21530.png', '16706.png', '20819.png', '20055.png', '11651.png', '02825.png', '22376.png', '08061.png', '17566.png', '00314.png', '20549.png', '03369.png', '04260.png', '02809.png', '17355.png', '08318.png', '04943.png', '09199.png', '02772.png', '17782.png', '14946.png', '00652.png', '09959.png', '13916.png', '20752.png', '09144.png', '18282.png', '21686.png', '11095.png', '06735.png', '13401.png', '20120.png', '04002.png', '15760.png', '16993.png', '11073.png', '14086.png', '24813.png', '21047.png', '11722.png', '22317.png', '10585.png', '17315.png', '17172.png', '09279.png', '03886.png', '07472.png', '18210.png', '05370.png', '14069.png', '17609.png', '13589.png', '11771.png', '22108.png', '08414.png', '00291.png', '06489.png', '17016.png', '15410.png', '09153.png', '18928.png', '23891.png', '17116.png', '10465.png', '24132.png', '00041.png', '21685.png', '23250.png', '16805.png', '13407.png', '23454.png', '05523.png', '09652.png', '11530.png', '04955.png', '19000.png', '22416.png', '07208.png', '24014.png', '23259.png', '19005.png', '12325.png', '14667.png', '13183.png', '05121.png', '23881.png', '17295.png', '18578.png', '19171.png', '02248.png', '24751.png', '15281.png', '11846.png', '01834.png', '09778.png', '20893.png', '18572.png', '09604.png', '20272.png', '00192.png', '16220.png', '03721.png', '19260.png', '06652.png', '15952.png', '21291.png', '02237.png', '20619.png', '18046.png', '06666.png', '14090.png', '11825.png', '12175.png', '23295.png', '14504.png', '13064.png', '04570.png', '04520.png', '16004.png', '17800.png', '10407.png', '11240.png', '19512.png', '16863.png', '16874.png', '24692.png', '04959.png', '16659.png', '17676.png', '24562.png', '18940.png', '00888.png', '21899.png', '12180.png', '20797.png', '10430.png', '06471.png', '07950.png', '15455.png', '20548.png', '05734.png', '15372.png', '17980.png', '09738.png', '01928.png', '01118.png', '13686.png', '06005.png', '09549.png', '07896.png', '19484.png', '07511.png', '22026.png', '21600.png', '21456.png', '14153.png', '13363.png', '19386.png', '06376.png', '03427.png', '17117.png', '08534.png', '00691.png', '08025.png', '09469.png', '08099.png', '10099.png', '02968.png', '17393.png', '02528.png', '11657.png', '24424.png', '17483.png', '24252.png', '01317.png', '13562.png', '19776.png', '22894.png', '10133.png', '15366.png', '03472.png', '21706.png', '08095.png', '05586.png', '19047.png', '09335.png', '10945.png', '04600.png', '14582.png', '14727.png', '09648.png', '09513.png', '10147.png', '00392.png', '00804.png', '23852.png', '08539.png', '06690.png', '24928.png', '24941.png', '08433.png', '10992.png', '10661.png', '10592.png', '09147.png', '24093.png', '16040.png', '09397.png', '01960.png', '01364.png', '19417.png', '21983.png', '24069.png', '07807.png', '12507.png', '11943.png', '12433.png', '22057.png', '04870.png', '02062.png', '04116.png', '03587.png', '18063.png', '04096.png', '14867.png', '15559.png', '10468.png', '02387.png', '11995.png', '02665.png', '06404.png', '09799.png', '08759.png', '17835.png', '00315.png', '01273.png', '05653.png', '21216.png', '09939.png', '24097.png', '11871.png', '10752.png', '24254.png', '02483.png', '19085.png', '02896.png', '16950.png', '23505.png', '13959.png', '00953.png', '22177.png', '16501.png', '24426.png', '08517.png', '08629.png', '05366.png', '13935.png', '16011.png', '22629.png', '20376.png', '18752.png', '18623.png', '00094.png', '12598.png', '08450.png', '04551.png', '12625.png', '15277.png', '13971.png', '22822.png', '11045.png', '20659.png', '03913.png', '23969.png', '00346.png', '17585.png', '22666.png', '17841.png', '02180.png', '01681.png', '08971.png', '10066.png', '08059.png', '07012.png', '06082.png', '15889.png', '22802.png', '01661.png', '18180.png', '06185.png', '06623.png', '10530.png', '22482.png', '05731.png', '12987.png', '03938.png', '07902.png', '13921.png', '01999.png', '03954.png', '02773.png', '03368.png', '17246.png', '16250.png', '15007.png', '13925.png', '22774.png', '13703.png', '19444.png', '13420.png', '17516.png', '12254.png', '10371.png', '12596.png', '04944.png', '16389.png', '03357.png', '14826.png', '07439.png', '07955.png', '19424.png', '16983.png', '21751.png', '11164.png', '03651.png', '09804.png', '01297.png', '01185.png', '06174.png', '22220.png', '06949.png', '01440.png', '09505.png', '16679.png', '18228.png', '24247.png', '10694.png', '19073.png', '19497.png', '24450.png', '14939.png', '16725.png', '23226.png', '20025.png', '07454.png', '17000.png', '11848.png', '09128.png', '16355.png', '21656.png', '22215.png', '16570.png', '08885.png', '07375.png', '10461.png', '20851.png', '22391.png', '17106.png', '22469.png', '21234.png', '00759.png', '20847.png', '06285.png', '12326.png', '06124.png', '23266.png', '09029.png', '05874.png', '01806.png', '13525.png', '22935.png', '21607.png', '10254.png', '24964.png', '20013.png', '18408.png', '12266.png', '02292.png', '08392.png', '02564.png', '17045.png', '10764.png', '22996.png', '05036.png', '11996.png', '09124.png', '08067.png', '23591.png', '04358.png', '23062.png', '00721.png', '10844.png', '24179.png', '01724.png', '04641.png', '11148.png', '19930.png', '12110.png', '09666.png', '20863.png', '06466.png', '03902.png', '11589.png', '21958.png', '18234.png', '02586.png', '18871.png', '12622.png', '11661.png', '08542.png', '12195.png', '14506.png', '20131.png', '23230.png', '17346.png', '19959.png', '02826.png', '02151.png', '18626.png', '10793.png', '14200.png', '15474.png', '04137.png', '01899.png', '15234.png', '09966.png', '19044.png', '19127.png', '07412.png', '03326.png', '11874.png', '13002.png', '02338.png', '09059.png', '13392.png', '02242.png', '17364.png', '01514.png', '16658.png', '13936.png', '21228.png', '11694.png', '01450.png', '10060.png', '22248.png', '13575.png', '19728.png', '02993.png', '09376.png', '01340.png', '03112.png', '09832.png', '07965.png', '14493.png', '06556.png', '20053.png', '10164.png', '21133.png', '19357.png', '08119.png', '24664.png', '07905.png', '19009.png', '11287.png', '02351.png', '09632.png', '12591.png', '04524.png', '05595.png', '09192.png', '07015.png', '15446.png', '23234.png', '18136.png', '16664.png', '08306.png', '19334.png', '14119.png', '03143.png', '07365.png', '13744.png', '24559.png', '19509.png', '10415.png', '22388.png', '23508.png', '11655.png', '00567.png', '24820.png', '23243.png', '19954.png', '14641.png', '04386.png', '22121.png', '13820.png', '10790.png', '22744.png', '09600.png', '04577.png', '09373.png', '03005.png', '08399.png', '20813.png', '14683.png', '08191.png', '05960.png', '19367.png', '24527.png', '12896.png', '08563.png', '19352.png', '16258.png', '19108.png', '08160.png', '09271.png', '12212.png', '05120.png', '14910.png', '01309.png', '16097.png', '19968.png', '01885.png', '04186.png', '06157.png', '10517.png', '01236.png', '00746.png', '03969.png', '05555.png', '21341.png', '01822.png', '20574.png', '12285.png', '12894.png', '03507.png', '20031.png', '20372.png', '10848.png', '17435.png', '10500.png', '13794.png', '24473.png', '10510.png', '14699.png', '16087.png', '16514.png', '06967.png', '23965.png', '08679.png', '10432.png', '17862.png', '09006.png', '11274.png', '24485.png', '07370.png', '22672.png', '18754.png', '19868.png', '06796.png', '19156.png', '12937.png', '15106.png', '12275.png', '23585.png', '23866.png', '10775.png', '15900.png', '12423.png', '05307.png', '16101.png', '11109.png', '12619.png', '00458.png', '12363.png', '22534.png', '24113.png', '18929.png', '07000.png', '04367.png', '00681.png', '12368.png', '13046.png', '17569.png', '10587.png', '01392.png', '10166.png', '06170.png', '20821.png', '08879.png', '11396.png', '22465.png', '24774.png', '03550.png', '24253.png', '11837.png', '04667.png', '13143.png', '13186.png', '03479.png', '18627.png', '01024.png', '10658.png', '00497.png', '08937.png', '01920.png', '09681.png', '23843.png', '05979.png', '18980.png', '17175.png', '19691.png', '22397.png', '04321.png', '12480.png', '04199.png', '12044.png', '12635.png', '15261.png', '15058.png', '06873.png', '01108.png', '16394.png', '15323.png', '17858.png', '24839.png', '01667.png', '17380.png', '10477.png', '01906.png', '05551.png', '12301.png', '00976.png', '11982.png', '10504.png', '03035.png', '15511.png', '07664.png', '22389.png', '17702.png', '13128.png', '06745.png', '24281.png', '17104.png', '23521.png', '10255.png', '01169.png', '13099.png', '06675.png', '08778.png', '09694.png', '15348.png', '18462.png', '10955.png', '06148.png', '10886.png', '19775.png', '19342.png', '23276.png', '22421.png', '01265.png', '06766.png', '13022.png', '08451.png', '08533.png', '15656.png', '21140.png', '14432.png', '18749.png', '01347.png', '04616.png', '20469.png', '08744.png', '20946.png', '03552.png', '11645.png', '11520.png', '24279.png', '03619.png', '00708.png', '01821.png', '06401.png', '19791.png', '19746.png', '23910.png', '16588.png', '13523.png', '18075.png', '07235.png', '09118.png', '13587.png', '09769.png', '23381.png', '17459.png', '05564.png', '16731.png', '03905.png', '24720.png', '02879.png', '10789.png', '07900.png', '13004.png', '11010.png', '17883.png', '23606.png', '19986.png', '23700.png', '00179.png', '20981.png', '08641.png', '13701.png', '17772.png', '19432.png', '04805.png', '09667.png', '18733.png', '17523.png', '02231.png', '22035.png', '07220.png', '24153.png', '16869.png', '21960.png', '04237.png', '00705.png', '02734.png', '20381.png', '00114.png', '05112.png', '05770.png', '11492.png', '21134.png', '09216.png', '04291.png', '07933.png', '08933.png', '24801.png', '08536.png', '03737.png', '21197.png', '13697.png', '19046.png', '12964.png', '19288.png', '24200.png', '20258.png', '16575.png', '17345.png', '20092.png', '03300.png', '15702.png', '17886.png', '19149.png', '13782.png', '17768.png', '04077.png', '08891.png', '24461.png', '13879.png', '04687.png', '18120.png', '09808.png', '23850.png', '16603.png', '17017.png', '11527.png', '09100.png', '04427.png', '17498.png', '04568.png', '10818.png', '10019.png', '00250.png', '09790.png', '22450.png', '12564.png', '11915.png', '16396.png', '02571.png', '21511.png', '01155.png', '23127.png', '10289.png', '21335.png', '21039.png', '24270.png', '13830.png', '10808.png', '13858.png', '23610.png', '03764.png', '17073.png', '09926.png', '14908.png', '07462.png', '12651.png', '20769.png', '19465.png', '05409.png', '23807.png', '12164.png', '06409.png', '03726.png', '18566.png', '06065.png', '20007.png', '00855.png', '06453.png', '17025.png', '03808.png', '21266.png', '20964.png', '03556.png', '22999.png', '24665.png', '06917.png', '14247.png', '00583.png', '11748.png', '12444.png', '09927.png', '07022.png', '02085.png', '24797.png', '07363.png', '20786.png', '22791.png', '17521.png', '01325.png', '15276.png', '13743.png', '21172.png', '10207.png', '04974.png', '07003.png', '02184.png', '24898.png', '11889.png', '08423.png', '18305.png', '01937.png', '04118.png', '17929.png', '05092.png', '15333.png', '20692.png', '16894.png', '20511.png', '20241.png', '22957.png', '09751.png', '17529.png', '14131.png', '23766.png', '22462.png', '21741.png', '16644.png', '24274.png', '13818.png', '01290.png', '00502.png', '08592.png', '12001.png', '12765.png', '17243.png', '10359.png', '13160.png', '16238.png', '13810.png', '17469.png', '23868.png', '24761.png', '07531.png', '12497.png', '07172.png', '09732.png', '08842.png', '02943.png', '14898.png', '13157.png', '03876.png', '03602.png', '10442.png', '13723.png', '04304.png', '19453.png', '06101.png', '17856.png', '16363.png', '02667.png', '01170.png', '19561.png', '00851.png', '09708.png', '19632.png', '18246.png', '07897.png', '10481.png', '11887.png', '02313.png', '08069.png', '07101.png', '14212.png', '02777.png', '23374.png', '24213.png', '04695.png', '05213.png', '04381.png', '13364.png', '19801.png', '12381.png', '03451.png', '13650.png', '02588.png', '01070.png', '04938.png', '12123.png', '16318.png', '00807.png', '10327.png', '14932.png', '08754.png', '01331.png', '23619.png', '18784.png', '17851.png', '04796.png', '21768.png', '21379.png', '09715.png', '17200.png', '14160.png', '15542.png', '20024.png', '13013.png', '07317.png', '18437.png', '22954.png', '14384.png', '21021.png', '08666.png', '09784.png', '07346.png', '06458.png', '17272.png', '04051.png', '08942.png', '16468.png', '14578.png', '12609.png', '23200.png', '15890.png', '06093.png', '08488.png', '24802.png', '19577.png', '23349.png', '12146.png', '20912.png', '13718.png', '21194.png', '24769.png', '09997.png', '05172.png', '19940.png', '04932.png', '12476.png', '15340.png', '12367.png', '15850.png', '20756.png', '16891.png', '00916.png', '06257.png', '21954.png', '21855.png', '04105.png', '19138.png', '02281.png', '11620.png', '22292.png', '08793.png', '14768.png', '07259.png', '13105.png', '05334.png', '20570.png', '06025.png', '14410.png', '02751.png', '08616.png', '02080.png', '24265.png', '07452.png', '11301.png', '03384.png', '16162.png', '18064.png', '09882.png', '05549.png', '17066.png', '05315.png', '15468.png', '07032.png', '17183.png', '12434.png', '18103.png', '08272.png', '12733.png', '05145.png', '06644.png', '18562.png', '20902.png', '10599.png', '00839.png', '20378.png', '12970.png', '17961.png', '06297.png', '23895.png', '16525.png', '13168.png', '03525.png', '14614.png', '14780.png', '17860.png', '17949.png', '08121.png', '24208.png', '21654.png', '17463.png', '05561.png', '01346.png', '24731.png', '09522.png', '05768.png', '11593.png', '02876.png', '04964.png', '08236.png', '11728.png', '22712.png', '03570.png', '13066.png', '12545.png', '10605.png', '03989.png', '07611.png', '20532.png', '09853.png', '09759.png', '11634.png', '08404.png', '00735.png', '05888.png', '20106.png', '12073.png', '22308.png', '13604.png', '06071.png', '01139.png', '08939.png', '07285.png', '02894.png', '14354.png', '09106.png', '21962.png', '01250.png', '19336.png', '01680.png', '00137.png', '24857.png', '03972.png', '20877.png', '14449.png', '04382.png', '17803.png', '15759.png', '14177.png', '14594.png', '11175.png', '24689.png', '15442.png', '07722.png', '15160.png', '01329.png', '09559.png', '05545.png', '11217.png', '21316.png', '13250.png', '23305.png', '14817.png', '17932.png', '17347.png', '18553.png', '12198.png', '05491.png', '03414.png', '10777.png', '14332.png', '05601.png', '10762.png', '01381.png', '22455.png', '12118.png', '10109.png', '12412.png', '04400.png', '02726.png', '12114.png', '17375.png', '10154.png', '10868.png', '17872.png', '18855.png', '13855.png', '14630.png', '17570.png', '21653.png', '20665.png', '21838.png', '24296.png', '08143.png', '06595.png', '14404.png', '16893.png', '24204.png', '11059.png', '02979.png', '15887.png', '05258.png', '07093.png', '08210.png', '22110.png', '19947.png', '23473.png', '16114.png', '02989.png', '10899.png', '18829.png', '15254.png', '15459.png', '20239.png', '14515.png', '16797.png', '22190.png', '01738.png', '01324.png', '19794.png', '04954.png', '23552.png', '16082.png', '10719.png', '11604.png', '00561.png', '01588.png', '14698.png', '20179.png', '00871.png', '12220.png', '02316.png', '01974.png', '14007.png', '23560.png', '06830.png', '07696.png', '05140.png', '01970.png', '22988.png', '01868.png', '07942.png', '11185.png', '01850.png', '18473.png', '04462.png', '02539.png', '14509.png', '17567.png', '12184.png', '02470.png', '24836.png', '07668.png', '08632.png', '10810.png', '16417.png', '12065.png', '02493.png', '07304.png', '18451.png', '14793.png', '09528.png', '09589.png', '13193.png', '22903.png', '05459.png', '19057.png', '24094.png', '11407.png', '04017.png', '16987.png', '13309.png', '10639.png', '11555.png', '15447.png', '11356.png', '22058.png', '18053.png', '06754.png', '03104.png', '14718.png', '08018.png', '11739.png', '13997.png', '06289.png', '01088.png', '10140.png', '20058.png', '02361.png', '06117.png', '16821.png', '21952.png', '00390.png', '23851.png', '10746.png', '08690.png', '16111.png', '02962.png', '14103.png', '12921.png', '02174.png', '03743.png', '19804.png', '22148.png', '24622.png', '21927.png', '16601.png', '14267.png', '11586.png', '00193.png', '02837.png', '24446.png', '13492.png', '21012.png', '10823.png', '01173.png', '07632.png', '16767.png', '20885.png', '19885.png', '24908.png', '01215.png', '12009.png', '19102.png', '24875.png', '19211.png', '09451.png', '20174.png', '23267.png', '02132.png', '11125.png', '19024.png', '06370.png', '07248.png', '04947.png', '22041.png', '01202.png', '22632.png', '11490.png', '11569.png', '16886.png', '01580.png', '23497.png', '14081.png', '05423.png', '12661.png', '09885.png', '24953.png', '09764.png', '06573.png', '23732.png', '18042.png', '21150.png', '17899.png', '00499.png', '02686.png', '22585.png', '05918.png', '02360.png', '21097.png', '13378.png', '24151.png', '21099.png', '24771.png', '20419.png', '07181.png', '00265.png', '03392.png', '22673.png', '23323.png', '12177.png', '06655.png', '09565.png', '04594.png', '15942.png', '05385.png', '18101.png', '11954.png', '14469.png', '01886.png', '09677.png', '17381.png', '05208.png', '21945.png', '16604.png', '22266.png', '11616.png', '17593.png', '23076.png', '00573.png', '07279.png', '23109.png', '21985.png', '24847.png', '09909.png', '01470.png', '18367.png', '06177.png', '19919.png', '03444.png', '04986.png', '01921.png', '00679.png', '05568.png', '16100.png', '08671.png', '01734.png', '08701.png', '22093.png', '04769.png', '10943.png', '14909.png', '22804.png', '04368.png', '23722.png', '15750.png', '15618.png', '12602.png', '11378.png', '22720.png', '00519.png', '05767.png', '08976.png', '08589.png', '21919.png', '17201.png', '12649.png', '22152.png', '20772.png', '22962.png', '06403.png', '01430.png', '10153.png', '23064.png', '20430.png', '10494.png', '02605.png', '11147.png', '22306.png', '11383.png', '00292.png', '20811.png', '18375.png', '00149.png', '08795.png', '11990.png', '16039.png', '06664.png', '19416.png', '17659.png', '03887.png', '02732.png', '12446.png', '22016.png', '02659.png', '15771.png', '14333.png', '02895.png', '12046.png', '06524.png', '10397.png', '06780.png', '21201.png', '16479.png', '22345.png', '21916.png', '04324.png', '04629.png', '24130.png', '07835.png', '20176.png', '14556.png', '10409.png', '21745.png', '18480.png', '04254.png', '06405.png', '23778.png', '14776.png', '06901.png', '02575.png', '21248.png', '04929.png', '12873.png', '08332.png', '07358.png', '19106.png', '05157.png', '15001.png', '02357.png', '24211.png', '09283.png', '17721.png', '24389.png', '19892.png', '24809.png', '19262.png', '11416.png', '02662.png', '13228.png', '01020.png', '22003.png', '06009.png', '06715.png', '24674.png', '02066.png', '18482.png', '23609.png', '14353.png', '10737.png', '17876.png', '07872.png', '04557.png', '24507.png', '04804.png', '06490.png', '09561.png', '24808.png', '05889.png', '07637.png', '02011.png', '04095.png', '14825.png', '04267.png', '07306.png', '04791.png', '13578.png', '14944.png', '01064.png', '04061.png', '17114.png', '18174.png', '24535.png', '23710.png', '08957.png', '13024.png', '02619.png', '20311.png', '11610.png', '00364.png', '20609.png', '03029.png', '13436.png', '13504.png', '12631.png', '23341.png', '07990.png', '23654.png', '03448.png', '03628.png', '15587.png', '12391.png', '16995.png', '07938.png', '07757.png', '13814.png', '07504.png', '14904.png', '13372.png', '03254.png', '05429.png', '18010.png', '13418.png', '01154.png', '06557.png', '05975.png', '23724.png', '18014.png', '05051.png', '16956.png', '23430.png', '23418.png', '03882.png', '18185.png', '00258.png', '01005.png', '13006.png', '21924.png', '11178.png', '11658.png', '13214.png', '04820.png', '17204.png', '16132.png', '11672.png', '22232.png', '17669.png', '13234.png', '13699.png', '08917.png', '14567.png', '05493.png', '10831.png', '05032.png', '19483.png', '23338.png', '14838.png', '20725.png', '24397.png', '11403.png', '04983.png', '14248.png', '19922.png', '12601.png', '06426.png', '06342.png', '21283.png', '05269.png', '02325.png', '15038.png', '10619.png', '22023.png', '04571.png', '00304.png', '22032.png', '14190.png', '02503.png', '07571.png', '06902.png', '03797.png', '04733.png', '02931.png', '05948.png', '16062.png', '00587.png', '09506.png', '02965.png', '16764.png', '20939.png', '06991.png', '15250.png', '06669.png', '08734.png', '08338.png', '02557.png', '18383.png', '18844.png', '23106.png', '14462.png', '13569.png', '08797.png', '21465.png', '08587.png', '14581.png', '14142.png', '22185.png', '23016.png', '02114.png', '09025.png', '15270.png', '12796.png', '02121.png', '15508.png', '22401.png', '06055.png', '01911.png', '03970.png', '03147.png', '06589.png', '12158.png', '15306.png', '22622.png', '03910.png', '10367.png', '22408.png', '11086.png', '20744.png', '14320.png', '05741.png', '14286.png', '07460.png', '01978.png', '24841.png', '17442.png', '03449.png', '00406.png', '01685.png', '23635.png', '11365.png', '06714.png', '17238.png', '24350.png', '16247.png', '24636.png', '09736.png', '07210.png', '18702.png', '06151.png', '03481.png', '04478.png', '17923.png', '18164.png', '08853.png', '19895.png', '08323.png', '03180.png', '10383.png', '03595.png', '22591.png', '21175.png', '01946.png', '20129.png', '05152.png', '15501.png', '11910.png', '19011.png', '00490.png', '07312.png', '13754.png', '20499.png', '23254.png', '05602.png', '21977.png', '05368.png', '08253.png', '18483.png', '24460.png', '05559.png', '00335.png', '16887.png', '15256.png', '01938.png', '04930.png', '01268.png', '00494.png', '13245.png', '19290.png', '04144.png', '17967.png', '08073.png', '11395.png', '01429.png', '24603.png', '05931.png', '01464.png', '15644.png', '21586.png', '24167.png', '08291.png', '12724.png', '19647.png', '16507.png', '20649.png', '22028.png', '00799.png', '02455.png', '00240.png', '23072.png', '14075.png', '09661.png', '01874.png', '06153.png', '07083.png', '06598.png', '07935.png', '03216.png', '20357.png', '21605.png', '05069.png', '14570.png', '24380.png', '03083.png', '15047.png', '13057.png', '06643.png', '00429.png', '02086.png', '08310.png', '14174.png', '11858.png', '15752.png', '06810.png', '08267.png', '08420.png', '04599.png', '09611.png', '14806.png', '23379.png', '01814.png', '02875.png', '00917.png', '12839.png', '00398.png', '23260.png', '01898.png', '20529.png', '21223.png', '00491.png', '23003.png', '04754.png', '09861.png', '13704.png', '01897.png', '10483.png', '05737.png', '20853.png', '14241.png', '00287.png', '03712.png', '12648.png', '11167.png', '02244.png', '13176.png', '12559.png', '17423.png', '05218.png', '17232.png', '20296.png', '22072.png', '13755.png', '07228.png', '07224.png', '13551.png', '22179.png', '17122.png', '21858.png', '22062.png', '07019.png', '02451.png', '14155.png', '07099.png', '20795.png', '21724.png', '15859.png', '06007.png', '21036.png', '11068.png', '09550.png', '14339.png', '22691.png', '13733.png', '07789.png', '02768.png', '05006.png', '24047.png', '24365.png', '05180.png', '04559.png', '20489.png', '00715.png', '22496.png', '18768.png', '13955.png', '01533.png', '16789.png', '11799.png', '15763.png', '09395.png', '04126.png', '17214.png', '15792.png', '07233.png', '23648.png', '08408.png', '06444.png', '12574.png', '15698.png', '01507.png', '09892.png', '01377.png', '14246.png', '18188.png', '19605.png', '09995.png', '01279.png', '09614.png', '05319.png', '23856.png', '05475.png', '02709.png', '18961.png', '19673.png', '24955.png', '03685.png', '14646.png', '03872.png', '18202.png', '04619.png', '20591.png', '01713.png', '22669.png', '15444.png', '04779.png', '23159.png', '16966.png', '18687.png', '18115.png', '21174.png', '17300.png', '21890.png', '04855.png', '04003.png', '22628.png', '02945.png', '18247.png', '17507.png', '21299.png', '19182.png', '24845.png', '09622.png', '16714.png', '02513.png', '03199.png', '15235.png', '20138.png', '04201.png', '06274.png', '21305.png', '21239.png', '10741.png', '20205.png', '09711.png', '04302.png', '07824.png', '20484.png', '15795.png', '10568.png', '02324.png', '17329.png', '06116.png', '19295.png', '20029.png', '23166.png', '09757.png', '18811.png', '21336.png', '14839.png', '12061.png', '00130.png', '17495.png', '17350.png', '07372.png', '23319.png', '04552.png', '15766.png', '00975.png', '14097.png', '20891.png', '19153.png', '13410.png', '02122.png', '21356.png', '21660.png', '14052.png', '18239.png', '19439.png', '14959.png', '04813.png', '24437.png', '12436.png', '10588.png', '06301.png', '00867.png', '09419.png', '11756.png', '17450.png', '13114.png', '22758.png', '15117.png', '21892.png', '20298.png', '23638.png', '18340.png', '17466.png', '14152.png', '07249.png', '11543.png', '16757.png', '03285.png', '07456.png', '16882.png', '19618.png', '06487.png', '01633.png', '20673.png', '24193.png', '22620.png', '22080.png', '23231.png', '00831.png', '21956.png', '20701.png', '05156.png', '02984.png', '20076.png', '03281.png', '09121.png', '01972.png', '10950.png', '09389.png', '13380.png', '21011.png', '12447.png', '10375.png', '10083.png', '01352.png', '14118.png', '01256.png', '10888.png', '20172.png', '18222.png', '19402.png', '07255.png', '11252.png', '11548.png', '11528.png', '19607.png', '23755.png', '07764.png', '08152.png', '19703.png', '24610.png', '14303.png', '18173.png', '06599.png', '24315.png', '00035.png', '07615.png', '00037.png', '23642.png', '18991.png', '06461.png', '06865.png', '17595.png', '18041.png', '09369.png', '22998.png', '16327.png', '13414.png', '08262.png', '10191.png', '17128.png', '19950.png', '08706.png', '10557.png', '21018.png', '05173.png', '16743.png', '06975.png', '00175.png', '15445.png', '05642.png', '08649.png', '11975.png', '14691.png', '21554.png', '15155.png', '16361.png', '07194.png', '17723.png', '18029.png', '15753.png', '11043.png', '17660.png', '24374.png', '11177.png', '00895.png', '01305.png', '07102.png', '18236.png', '13680.png', '00431.png', '05265.png', '02372.png', '15593.png', '15586.png', '14357.png', '14929.png', '16707.png', '12858.png', '01434.png', '03398.png', '05433.png', '23643.png', '09938.png', '19761.png', '00928.png', '23434.png', '14447.png', '03963.png', '21015.png', '16756.png', '16108.png', '09887.png', '04789.png', '18410.png', '05021.png', '15717.png', '10516.png', '12395.png', '02186.png', '02998.png', '07409.png', '18013.png', '19012.png', '11734.png', '20475.png', '14928.png', '05205.png', '07024.png', '09556.png', '07908.png', '22882.png', '03279.png', '19575.png', '17624.png', '12311.png', '09567.png', '05752.png', '15041.png', '19364.png', '22710.png', '21141.png', '08403.png', '06165.png', '16708.png', '12379.png', '19646.png', '15224.png', '07774.png', '19226.png', '18873.png', '09229.png', '14224.png', '15694.png', '12579.png', '09700.png', '06577.png', '20739.png', '01501.png', '17684.png', '20294.png', '07743.png', '18212.png', '05492.png', '02052.png', '09923.png', '20644.png', '03679.png', '05360.png', '08129.png', '01278.png', '14849.png', '19341.png', '05937.png', '17744.png', '19898.png', '24714.png', '19393.png', '10767.png', '02607.png', '03085.png', '14366.png', '23396.png', '11912.png', '09489.png', '02937.png', '13466.png', '23236.png', '15081.png', '01863.png', '18259.png', '09762.png', '17368.png', '11350.png', '13455.png', '03715.png', '20585.png', '17078.png', '22631.png', '00160.png', '01517.png', '12486.png', '07693.png', '04372.png', '22081.png', '09872.png', '19853.png', '08801.png', '12035.png', '07444.png', '14478.png', '01002.png', '23506.png', '10039.png', '06615.png', '20399.png', '18429.png', '07801.png', '16145.png', '06081.png', '04259.png', '20551.png', '06898.png', '14605.png', '24867.png', '04670.png', '00126.png', '12257.png', '03379.png', '19221.png', '15873.png', '05665.png', '04510.png', '03053.png', '14722.png', '04207.png', '14663.png', '06511.png', '00511.png', '03089.png', '14255.png', '00487.png', '14047.png', '11791.png', '23209.png', '12537.png', '19521.png', '03514.png', '06150.png', '17554.png', '16528.png', '04766.png', '00025.png', '02560.png', '08611.png', '03088.png', '12904.png', '24548.png', '16222.png', '12615.png', '15063.png', '00743.png', '23636.png', '01790.png', '03773.png', '09744.png', '14850.png', '22924.png', '01318.png', '06520.png', '24187.png', '19629.png', '00348.png', '15781.png', '20556.png', '12319.png', '21262.png', '24196.png', '18015.png', '21496.png', '18119.png', '17970.png', '06828.png', '21789.png', '24490.png', '03367.png', '15772.png', '17574.png', '05476.png', '10528.png', '16044.png', '08003.png', '23309.png', '06305.png', '16554.png', '10662.png', '24728.png', '03325.png', '19600.png', '19553.png', '21662.png', '04147.png', '13076.png', '01888.png', '16974.png', '11037.png', '18958.png', '23325.png', '19546.png', '11842.png', '13163.png', '19212.png', '22379.png', '15811.png', '23314.png', '02921.png', '04226.png', '12394.png', '06582.png', '16002.png', '12768.png', '18528.png', '24335.png', '17082.png', '04411.png', '20197.png', '15004.png', '08950.png', '18676.png', '23719.png', '11872.png', '11230.png', '10838.png', '24546.png', '10966.png', '15076.png', '03380.png', '02016.png', '17160.png', '13878.png', '20788.png', '04562.png', '03139.png', '24643.png', '02269.png', '20857.png', '07257.png', '10433.png', '23624.png', '04245.png', '21413.png', '01246.png', '06279.png', '19811.png', '21731.png', '15084.png', '02410.png', '17203.png', '09536.png', '01787.png', '13020.png', '05082.png', '14214.png', '19582.png', '21445.png', '18094.png', '03787.png', '21385.png', '19655.png', '24373.png', '24442.png', '04232.png', '01718.png', '00875.png', '06445.png', '12690.png', '20519.png', '08518.png', '13791.png', '18645.png', '17001.png', '17305.png', '22657.png', '12770.png', '18555.png', '15976.png', '19053.png', '17626.png', '10680.png', '00644.png', '00484.png', '07904.png', '24176.png', '00868.png', '07783.png', '07332.png', '17555.png', '12357.png', '09588.png', '03951.png', '02356.png', '20167.png', '24523.png', '08216.png', '01264.png', '03561.png', '07051.png', '23756.png', '02929.png', '11460.png', '01625.png', '17196.png', '18587.png', '21523.png', '09098.png', '05117.png', '02226.png', '02230.png', '00324.png', '24965.png', '04148.png', '05746.png', '06373.png', '18071.png', '13158.png', '15142.png', '15805.png', '04112.png', '18459.png', '12818.png', '08283.png', '18505.png', '00515.png', '04141.png', '16237.png', '14913.png', '01723.png', '02182.png', '04214.png', '10729.png', '18819.png', '22970.png', '22352.png', '06500.png', '07845.png', '16656.png', '03358.png', '14195.png', '22049.png', '07954.png', '16001.png', '11299.png', '10234.png', '07982.png', '05791.png', '18968.png', '00813.png', '13334.png', '03400.png', '21882.png', '08651.png', '20963.png', '09849.png', '20304.png', '21544.png', '03642.png', '04781.png', '12583.png', '17108.png', '20969.png', '07113.png', '21139.png', '18803.png', '08133.png', '12458.png', '00948.png', '07694.png', '23671.png', '20292.png', '04584.png', '03243.png', '14315.png', '21375.png', '23982.png', '00097.png', '12427.png', '20061.png', '13944.png', '22468.png', '12402.png', '20986.png', '06803.png', '08406.png', '23156.png', '15207.png', '07303.png', '09371.png', '00826.png', '12798.png', '15109.png', '11972.png', '22648.png', '04109.png', '24361.png', '20545.png', '16164.png', '24042.png', '03203.png', '01857.png', '18256.png', '09615.png', '05562.png', '15435.png', '16646.png', '08800.png', '17885.png', '02091.png', '07202.png', '02118.png', '01869.png', '15997.png', '23421.png', '19285.png', '15722.png', '08015.png', '20015.png', '07939.png', '21317.png', '00870.png', '22681.png', '03232.png', '17303.png', '07482.png', '15180.png', '15721.png', '16666.png', '15037.png', '03349.png', '04120.png', '11113.png', '21427.png', '20596.png', '13225.png', '04885.png', '15689.png', '06848.png', '10126.png', '20831.png', '10880.png', '20603.png', '22206.png', '19537.png', '23962.png', '02166.png', '14223.png', '00236.png', '00954.png', '02343.png', '06360.png', '22378.png', '08841.png', '17668.png', '02973.png', '22582.png', '00060.png', '12193.png', '00912.png', '19391.png', '18533.png', '08233.png', '16610.png', '22192.png', '04451.png', '22942.png', '11947.png', '04633.png', '24879.png', '11337.png', '20783.png', '21404.png', '21178.png', '19223.png', '12549.png', '22763.png', '03226.png', '04580.png', '19653.png', '10721.png', '24903.png', '01029.png', '20695.png', '18612.png', '05550.png', '04210.png', '15619.png', '18380.png', '17650.png', '18880.png', '01077.png', '23165.png', '14694.png', '09967.png', '18369.png', '09976.png', '21680.png', '01690.png', '19139.png', '00647.png', '03455.png', '06358.png', '12973.png', '10036.png', '09706.png', '24576.png', '15745.png', '07263.png', '14979.png', '24730.png', '17033.png', '14210.png', '23742.png', '07301.png', '01791.png', '20432.png', '10678.png', '18511.png', '13564.png', '18987.png', '24506.png', '10978.png', '19090.png', '03150.png', '00690.png', '13558.png', '06896.png', '08194.png', '13448.png', '06609.png', '14829.png', '09875.png', '13999.png', '03122.png', '10787.png', '24458.png', '22765.png', '00505.png', '06613.png', '09859.png', '01847.png', '11539.png', '19598.png', '23705.png', '24629.png', '11291.png', '06859.png', '10591.png', '17894.png', '12194.png', '05574.png', '08988.png', '07137.png', '14762.png', '09231.png', '08621.png', '02611.png', '18051.png', '16297.png', '16878.png', '13725.png', '08481.png', '23650.png', '22271.png', '15714.png', '14424.png', '15397.png', '24749.png', '22901.png', '01336.png', '19995.png', '05887.png', '15077.png', '07154.png', '01701.png', '22088.png', '07477.png', '09010.png', '24314.png', '03316.png', '06860.png', '21798.png', '02124.png', '16273.png', '22643.png', '14000.png', '03318.png', '00056.png', '08851.png', '08610.png', '08953.png', '11331.png', '05933.png', '01654.png', '11611.png', '00251.png', '23873.png', '02390.png', '11752.png', '24379.png', '12095.png', '14870.png', '10744.png', '05854.png', '20931.png', '08773.png', '01426.png', '03101.png', '19210.png', '24623.png', '15036.png', '24829.png', '01074.png', '00714.png', '14068.png', '10469.png', '18598.png', '13964.png', '00604.png', '20424.png', '02862.png', '16890.png', '22348.png', '13566.png', '07699.png', '23731.png', '22307.png', '05186.png', '22551.png', '20097.png', '15116.png', '04448.png', '24062.png', '24688.png', '24404.png', '16521.png', '20546.png', '20155.png', '02890.png', '21386.png', '02509.png', '16493.png', '11254.png', '16189.png', '02814.png', '04108.png', '08846.png', '14880.png', '13326.png', '16898.png', '03947.png', '24901.png', '17791.png', '12002.png', '01275.png', '18979.png', '05320.png', '22474.png', '11777.png', '20274.png', '08149.png', '17283.png', '22977.png', '03893.png', '14873.png', '18885.png', '07139.png', '20048.png', '21752.png', '06838.png', '13486.png', '11817.png', '15611.png', '24871.png', '00981.png', '12740.png', '21313.png', '19908.png', '24621.png', '16333.png', '20447.png', '04793.png', '12719.png', '23198.png', '04022.png', '15751.png', '12327.png', '23759.png', '10113.png', '12492.png', '23612.png', '24694.png', '13868.png', '08896.png', '12918.png', '20064.png', '00212.png', '07669.png', '05414.png', '17061.png', '01327.png', '01178.png', '02924.png', '00252.png', '00254.png', '15313.png', '04675.png', '23814.png', '01205.png', '23659.png', '06592.png', '08232.png', '10333.png', '15529.png', '15083.png', '08577.png', '01474.png', '16921.png', '06611.png', '15635.png', '09482.png', '07250.png', '11566.png', '05463.png', '15680.png', '02310.png', '06087.png', '05849.png', '23576.png', '08355.png', '10054.png', '16065.png', '24493.png', '00006.png', '16428.png', '18899.png', '18060.png', '17036.png', '10115.png', '05809.png', '06308.png', '16877.png', '18217.png', '07762.png', '01460.png', '12012.png', '21179.png', '01773.png', '00891.png', '24558.png', '07157.png', '06645.png', '02328.png', '22162.png', '11176.png', '04319.png', '02274.png', '23716.png', '12775.png', '01746.png', '04145.png', '05974.png', '12856.png', '21127.png', '21034.png', '13275.png', '23108.png', '02069.png', '10677.png', '04256.png', '15798.png', '01422.png', '22649.png', '14230.png', '17644.png', '13576.png', '20761.png', '02286.png', '03409.png', '21625.png', '14184.png', '21841.png', '01402.png', '03405.png', '06362.png', '12490.png', '09697.png', '00078.png', '09517.png', '14622.png', '21996.png', '22614.png', '06981.png', '05953.png', '08110.png', '08400.png', '10816.png', '13271.png', '22037.png', '18083.png', '07167.png', '02380.png', '01420.png', '03371.png', '08270.png', '22360.png', '02670.png', '17817.png', '10858.png', '07413.png', '04151.png', '15684.png', '04830.png', '20842.png', '16284.png', '16116.png', '08722.png', '04816.png', '22755.png', '13437.png', '09167.png', '19471.png', '17588.png', '11292.png', '23340.png', '20901.png', '13889.png', '04286.png', '08874.png', '05333.png', '21347.png', '11087.png', '06814.png', '24683.png', '00275.png', '03672.png', '24699.png', '10228.png', '02972.png', '15311.png', '20242.png', '10243.png', '04016.png', '14433.png', '11594.png', '24831.png', '19805.png', '24084.png', '18165.png', '08074.png', '20277.png', '05308.png', '17568.png', '15051.png', '01802.png', '16867.png', '15598.png', '21329.png', '11856.png', '14196.png', '14954.png', '09109.png', '16858.png', '07156.png', '09983.png', '23833.png', '09619.png', '24261.png', '17661.png', '19441.png', '18755.png', '01116.png', '19103.png', '06629.png', '01811.png', '10715.png', '06617.png', '10519.png', '24654.png', '08398.png', '14460.png', '13671.png', '15331.png', '06230.png', '18833.png', '07626.png', '14300.png', '06775.png', '02917.png', '04664.png', '21850.png', '09001.png', '12102.png', '13145.png', '16381.png', '10638.png', '24431.png', '12817.png', '10366.png', '11818.png', '24772.png', '01918.png', '17823.png', '16148.png', '05237.png', '14368.png', '14385.png', '16435.png', '00045.png', '01414.png', '24740.png', '02162.png', '24329.png', '00326.png', '04477.png', '20479.png', '07142.png', '03482.png', '10107.png', '13087.png', '12132.png', '08224.png', '23816.png', '00318.png', '17762.png', '14566.png', '02760.png', '17327.png', '00877.png', '13635.png', '12791.png', '11158.png', '13870.png', '15515.png', '21331.png', '03249.png', '08573.png', '02399.png', '03345.png', '06411.png', '08205.png', '02213.png', '06104.png', '13062.png', '10977.png', '21093.png', '00163.png', '22009.png', '02125.png', '13556.png', '04691.png', '04117.png', '16823.png', '07215.png', '19067.png', '10088.png', '19810.png', '18835.png', '19036.png', '06208.png', '10881.png', '22269.png', '08883.png', '01288.png', '06929.png', '09780.png', '10999.png', '01361.png', '18431.png', '12273.png', '20683.png', '20397.png', '05412.png', '18761.png', '19998.png', '20465.png', '12075.png', '10566.png', '24544.png', '03736.png', '16291.png', '13516.png', '05815.png', '00365.png', '20180.png', '17754.png', '09387.png', '10446.png', '24090.png', '14393.png', '04223.png', '11923.png', '07397.png', '15571.png', '00422.png', '07682.png', '16768.png', '08290.png', '08467.png', '18851.png', '24145.png', '00234.png', '24143.png', '23060.png', '19168.png', '13044.png', '22331.png', '11096.png', '07857.png', '01646.png', '22077.png', '22650.png', '14502.png', '14204.png', '18430.png', '21664.png', '22828.png', '19510.png', '13153.png', '16130.png', '21243.png', '07064.png', '04864.png', '13041.png', '08434.png', '21923.png', '09649.png', '23242.png', '14770.png', '23202.png', '02400.png', '02202.png', '13679.png', '01375.png', '05062.png', '09262.png', '19725.png', '24579.png', '06746.png', '16467.png', '12467.png', '01349.png', '16371.png', '23347.png', '24881.png', '24701.png', '06920.png', '19489.png', '14665.png', '13732.png', '11182.png', '21902.png', '02534.png', '06679.png', '00059.png', '05287.png', '04231.png', '00481.png', '05834.png', '01199.png', '11385.png', '01964.png', '19799.png', '02074.png', '15188.png', '11853.png', '06903.png', '19876.png', '07048.png', '14900.png', '03912.png', '00685.png', '17600.png', '13659.png', '18323.png', '05711.png', '20873.png', '10669.png', '09277.png', '10893.png', '22808.png', '17578.png', '07608.png', '10617.png', '22342.png', '13823.png', '13443.png', '23004.png', '04306.png', '07854.png', '13758.png', '06911.png', '16568.png', '02216.png', '10582.png', '03178.png', '15091.png', '10128.png', '19187.png', '02584.png', '00576.png', '22245.png', '00838.png', '21785.png', '21966.png', '02820.png', '17799.png', '11485.png', '08765.png', '14952.png', '22940.png', '15407.png', '15195.png', '15429.png', '09814.png', '01160.png', '19678.png', '08613.png', '14750.png', '22737.png', '01861.png', '02782.png', '08810.png', '07321.png', '23344.png', '05418.png', '18947.png', '05518.png', '08537.png', '15302.png', '19217.png', '09090.png', '12449.png', '18089.png', '05132.png', '12282.png', '19566.png', '18714.png', '06632.png', '05175.png', '01012.png', '24189.png', '17275.png', '04235.png', '07503.png', '13423.png', '10688.png', '09060.png', '03364.png', '04500.png', '14631.png', '21202.png', '04997.png', '17793.png', '15223.png', '23116.png', '10574.png', '15263.png', '07610.png', '24612.png', '09042.png', '19590.png', '23976.png', '02089.png', '24619.png', '21325.png', '20773.png', '12824.png', '11633.png', '00225.png', '13358.png', '09518.png', '18197.png', '21224.png', '09285.png', '14844.png', '12134.png', '22865.png', '23317.png', '24693.png', '24280.png', '07300.png', '02818.png', '09756.png', '01996.png', '23069.png', '18892.png', '17892.png', '02155.png', '21982.png', '04467.png', '11937.png', '13531.png', '12107.png', '14266.png', '20351.png', '17436.png', '03078.png', '21610.png', '12464.png', '23329.png', '15522.png', '10142.png', '21320.png', '19164.png', '15045.png', '08684.png', '17695.png', '20988.png', '15327.png', '19961.png', '22553.png', '15512.png', '01050.png', '23206.png', '06236.png', '04988.png', '07971.png', '01011.png', '24936.png', '23828.png', '01674.png', '19637.png', '15679.png', '03823.png', '23751.png', '01079.png', '15021.png', '08373.png', '01959.png', '12230.png', '11588.png', '03875.png', '13475.png', '07232.png', '10702.png', '18737.png', '16202.png', '00902.png', '00367.png', '00589.png', '21344.png', '09008.png', '22326.png', '11194.png', '15009.png', '24939.png', '05861.png', '04492.png', '23771.png', '18052.png', '21555.png', '13506.png', '13104.png', '08549.png', '11187.png', '07196.png', '17247.png', '00627.png', '05668.png', '04850.png', '14869.png', '19142.png', '21453.png', '16367.png', '12633.png', '09910.png', '14978.png', '18192.png', '16546.png', '23594.png', '08865.png', '11454.png', '18759.png', '04121.png', '03353.png', '21563.png', '12305.png', '18663.png', '07152.png', '12723.png', '09175.png', '03488.png', '14156.png', '08431.png', '08831.png', '12377.png', '04442.png', '09061.png', '04313.png', '12980.png', '07299.png', '06349.png', '24927.png', '06313.png', '21787.png', '15549.png', '22419.png', '17543.png', '02728.png', '09440.png', '20307.png', '20713.png', '11840.png', '08975.png', '01194.png', '08259.png', '06584.png', '01916.png', '12129.png', '00693.png', '24803.png', '07132.png', '11324.png', '06989.png', '08044.png', '17687.png', '13146.png', '06275.png', '20201.png', '09607.png', '17155.png', '01220.png', '04135.png', '15278.png', '10997.png', '12711.png', '12015.png', '13342.png', '20019.png', '02077.png', '11344.png', '05829.png', '10222.png', '04586.png', '05386.png', '20250.png', '23683.png', '05863.png', '18168.png', '13310.png', '20236.png', '19373.png', '12306.png', '04020.png', '19115.png', '17150.png', '09634.png', '15033.png', '10016.png', '02701.png', '23273.png', '09313.png', '01221.png', '24158.png', '23906.png', '08320.png', '11703.png', '07910.png', '01883.png', '22015.png', '06355.png', '00146.png', '24421.png', '10716.png', '00099.png', '17559.png', '17290.png', '16118.png', '03760.png', '11415.png', '14715.png', '14519.png', '13236.png', '22509.png', '08324.png', '02735.png', '01830.png', '16803.png', '02926.png', '07733.png', '16129.png', '20954.png', '18218.png', '00271.png', '06506.png', '10696.png', '10380.png', '20142.png', '15102.png', '05896.png', '03215.png', '08454.png', '19195.png', '02169.png', '01211.png', '12868.png', '15562.png', '14751.png', '09127.png', '11775.png', '22773.png', '19287.png', '21717.png', '08786.png', '19348.png', '13464.png', '03269.png', '17678.png', '14352.png', '16161.png', '02798.png', '21538.png', '24344.png', '02620.png', '08757.png', '00906.png', '07337.png', '23925.png', '02252.png', '05605.png', '16941.png', '09438.png', '12519.png', '02394.png', '05793.png', '19167.png', '08984.png', '15730.png', '10611.png', '11323.png', '21040.png', '08141.png', '11220.png', '06871.png', '09612.png', '20530.png', '23515.png', '21368.png', '17976.png', '08397.png', '03308.png', '19797.png', '22291.png', '17148.png', '10524.png', '06272.png', '18244.png', '04683.png', '09421.png', '12302.png', '05403.png', '09863.png', '04139.png', '10009.png', '18773.png', '03917.png', '11863.png', '10627.png', '20897.png', '03271.png', '17181.png', '11513.png', '07017.png', '14822.png', '13586.png', '04914.png', '10230.png', '13819.png', '12082.png', '11836.png', '03845.png', '10200.png', '08691.png', '03811.png', '21437.png', '19376.png', '03436.png', '06704.png', '23988.png', '00096.png', '18090.png', '08109.png', '03033.png', '06156.png', '05917.png', '17096.png', '09616.png', '16444.png', '16547.png', '09138.png', '23879.png', '09301.png', '02515.png', '05415.png', '17169.png', '07899.png', '15161.png', '06932.png', '06255.png', '19082.png', '07811.png', '17804.png', '22967.png', '21358.png', '04746.png', '06079.png', '03284.png', '18838.png', '06129.png', '14542.png', '13034.png', '08022.png', '16750.png', '12408.png', '24018.png', '20072.png', '00999.png', '06950.png', '05762.png', '19027.png', '05928.png', '03577.png', '12571.png', '08579.png', '06625.png', '22412.png', '00654.png', '01293.png', '22407.png', '04823.png', '23412.png', '22715.png', '14716.png', '01322.png', '01702.png', '12958.png', '07765.png', '15723.png', '20984.png', '07622.png', '14337.png', '16486.png', '09222.png', '08837.png', '05079.png', '16955.png', '24673.png', '09837.png', '01219.png', '13077.png', '08430.png', '03003.png', '03780.png', '03079.png', '21429.png', '03166.png', '11127.png', '23876.png', '17727.png', '12887.png', '13837.png', '15871.png', '18904.png', '13500.png', '17256.png', '23604.png', '24198.png', '05207.png', '09896.png', '19037.png', '19344.png', '20391.png', '16719.png', '17807.png', '11615.png', '17686.png', '17776.png', '16517.png', '22117.png', '08340.png', '05034.png', '05634.png', '08532.png', '10783.png', '05661.png', '15870.png', '21824.png', '02669.png', '14527.png', '11316.png', '01540.png', '10249.png', '03859.png', '20751.png', '01952.png', '13982.png', '12961.png', '08991.png', '13465.png', '00830.png', '03363.png', '04808.png', '03288.png', '19086.png', '11001.png', '13493.png', '18682.png', '00507.png', '11733.png', '06969.png', '20056.png', '16721.png', '03307.png', '02967.png', '13385.png', '11855.png', '01258.png', '13091.png', '12117.png', '12800.png', '15016.png', '07341.png', '06278.png', '18760.png', '17983.png', '03182.png', '01046.png', '22141.png', '09282.png', '15316.png', '01023.png', '06724.png', '02928.png', '14143.png', '07471.png', '12835.png', '18530.png', '06759.png', '12790.png', '00750.png', '15066.png', '21360.png', '07173.png', '19058.png', '23979.png', '06953.png', '00582.png', '15032.png', '09037.png', '15166.png', '19700.png', '16054.png', '16332.png', '16192.png', '02832.png', '00599.png', '13963.png', '22475.png', '18324.png', '12315.png', '09503.png', '07234.png', '21829.png', '23119.png', '12881.png', '02877.png', '15452.png', '14348.png', '22212.png', '03613.png', '15834.png', '21041.png', '23905.png', '20738.png', '00117.png', '22131.png', '22305.png', '07966.png', '03175.png', '19756.png', '13185.png', '21514.png', '15653.png', '12836.png', '17297.png', '21448.png', '21082.png', '13131.png', '06091.png', '00933.png', '15388.png', '08002.png', '08493.png', '01747.png', '14010.png', '07781.png', '20218.png', '03268.png', '15181.png', '11428.png', '08176.png', '23391.png', '00680.png', '01281.png', '10654.png', '09055.png', '07381.png', '03953.png', '04301.png', '06618.png', '09566.png', '16424.png', '03214.png', '06240.png', '02358.png', '02948.png', '12474.png', '11699.png', '19098.png', '11412.png', '05348.png', '20332.png', '06467.png', '23896.png', '21377.png', '06758.png', '00630.png', '17822.png', '02501.png', '09403.png', '02349.png', '02128.png', '12350.png', '02115.png', '06415.png', '15191.png', '00506.png', '22982.png', '18254.png', '16246.png', '06221.png', '09165.png', '01561.png', '14549.png', '10849.png', '19551.png', '11557.png', '24003.png', '08011.png', '06286.png', '17735.png', '21793.png', '11304.png', '01721.png', '03941.png', '19623.png', '01692.png', '17032.png', '19145.png', '05869.png', '20166.png', '23147.png', '22933.png', '02053.png', '17484.png', '15885.png', '13477.png', '14025.png', '19118.png', '19630.png', '10836.png', '03747.png', '04658.png', '21907.png', '24726.png', '01343.png', '11876.png', '08123.png', '18182.png', '07289.png', '18381.png', '12657.png', '17286.png', '01725.png', '23686.png', '23567.png', '13977.png', '20559.png', '02612.png', '12034.png', '24930.png', '15375.png', '06083.png', '24779.png', '14094.png', '04293.png', '24882.png', '09221.png', '05580.png', '00947.png', '10905.png', '11452.png', '00542.png', '11079.png', '22616.png', '21810.png', '16436.png', '11582.png', '06425.png', '08525.png', '20883.png', '05977.png', '05240.png', '03974.png', '06455.png', '06987.png', '17288.png', '15145.png', '10108.png', '16972.png', '12059.png', '11340.png', '03636.png', '10646.png', '13339.png', '23311.png', '11120.png', '06134.png', '10242.png', '20872.png', '14907.png', '14882.png', '16315.png', '08826.png', '15906.png', '02535.png', '06145.png', '02241.png', '13672.png', '13747.png', '09655.png', '18717.png', '22959.png', '24479.png', '07044.png', '01201.png', '07267.png', '12372.png', '02969.png', '16364.png', '17223.png', '06120.png', '07795.png', '23993.png', '22472.png', '00403.png', '11300.png', '04188.png', '12339.png', '06706.png', '20305.png', '20928.png', '04716.png', '11048.png', '24299.png', '12793.png', '07330.png', '12151.png', '08526.png', '04859.png', '09154.png', '21677.png', '24171.png', '08312.png', '08558.png', '20876.png', '08665.png', '06452.png', '14020.png', '05860.png', '13553.png', '15464.png', '17253.png', '01451.png', '01049.png', '03319.png', '20320.png', '04648.png', '04951.png', '05390.png', '13505.png', '14127.png', '14401.png', '08723.png', '22102.png', '05934.png', '11536.png', '19449.png', '12530.png', '06192.png', '10288.png', '11386.png', '17319.png', '01008.png', '21486.png', '12223.png', '02642.png', '03655.png', '15029.png', '04106.png', '06781.png', '18805.png', '23442.png', '17301.png', '23006.png', '19378.png', '17409.png', '19747.png', '01595.png', '19033.png', '03159.png', '23711.png', '18876.png', '18584.png', '23631.png', '16163.png', '07709.png', '03987.png', '00325.png', '07480.png', '13367.png', '14463.png', '16562.png', '19445.png', '09669.png', '16915.png', '09818.png', '24641.png', '16578.png', '11418.png', '11030.png', '06069.png', '10125.png', '05594.png', '15754.png', '03924.png', '14600.png', '01266.png', '18894.png', '05316.png', '21964.png', '01000.png', '13274.png', '21867.png', '24568.png', '22059.png', '21407.png', '00539.png', '00284.png', '14775.png', '01261.png', '04428.png', '04474.png', '01072.png', '06534.png', '04271.png', '16348.png', '05823.png', '04351.png', '24051.png', '15114.png', '20669.png', '17755.png', '20441.png', '01607.png', '10579.png', '08354.png', '11617.png', '16817.png', '05528.png', '22589.png', '16994.png', '13273.png', '13204.png', '00574.png', '02389.png', '01653.png', '14732.png', '22284.png', '02878.png', '16560.png', '00968.png', '23966.png', '08402.png', '22513.png', '17933.png', '09904.png', '12259.png', '07105.png', '09551.png', '17935.png', '07846.png', '16481.png', '03750.png', '19229.png', '07862.png', '01824.png', '06947.png', '14884.png', '04726.png', '19155.png', '01146.png', '12691.png', '24706.png', '15804.png', '13452.png', '03836.png', '12409.png', '05633.png', '19939.png', '15836.png', '13618.png', '16403.png', '13094.png', '24357.png', '02321.png', '14621.png', '15504.png', '10951.png', '22357.png', '16402.png', '13365.png', '08907.png', '14221.png', '18362.png', '10884.png', '18915.png', '21655.png', '04018.png', '22994.png', '03478.png', '09215.png', '04328.png', '20942.png', '05636.png', '02533.png', '13796.png', '22411.png', '20849.png', '24272.png', '03123.png', '21593.png', '06836.png', '06764.png', '02995.png', '11544.png', '11643.png', '00022.png', '17987.png', '20906.png', '03496.png', '22938.png', '24257.png', '06299.png', '10757.png', '06027.png', '24512.png', '09401.png', '12445.png', '06958.png', '24175.png', '03463.png', '09912.png', '09674.png', '09310.png', '23413.png', '18926.png', '04013.png', '04024.png', '22528.png', '21973.png', '16545.png', '18589.png', '12886.png', '18740.png', '01872.png', '15761.png', '05534.png', '00728.png', '24425.png', '08197.png', '01796.png', '19783.png', '23083.png', '23916.png', '09273.png', '10318.png', '11246.png', '05995.png', '11026.png', '09733.png', '08413.png', '19934.png', '02309.png', '05239.png', '16998.png', '07934.png', '19110.png', '02120.png', '01845.png', '18583.png', '18973.png', '18732.png', '21645.png', '15168.png', '09080.png', '00778.png', '16765.png', '01902.png', '13322.png', '24216.png', '22728.png', '11985.png', '14476.png', '17587.png', '03418.png', '05578.png', '04419.png', '24702.png', '19882.png', '04827.png', '18134.png', '18311.png', '10804.png', '15712.png', '03261.png', '05611.png', '15660.png', '10520.png', '05000.png', '23247.png', '21250.png', '17427.png', '18853.png', '07486.png', '09325.png', '11053.png', '01823.png', '04172.png', '11852.png', '09164.png', '10224.png', '06106.png', '07869.png', '19485.png', '05958.png', '14557.png', '17149.png', '22832.png', '02418.png', '23672.png', '02267.png', '19220.png', '07271.png', '23183.png', '07949.png', '22118.png', '05136.png', '02047.png', '11002.png', '15408.png', '17692.png', '10859.png', '02403.png', '21843.png', '22214.png', '06175.png', '21472.png', '18800.png', '08864.png', '16331.png', '02672.png', '03825.png', '18571.png', '22205.png', '06238.png', '14888.png', '17447.png', '01586.png', '12630.png', '15733.png', '24944.png', '20312.png', '08446.png', '18972.png', '23321.png', '18859.png', '18667.png', '20081.png', '14397.png', '01233.png', '15436.png', '04901.png', '02247.png', '05843.png', '14473.png', '17889.png', '20880.png', '14279.png', '00276.png', '14228.png', '24182.png', '10549.png', '17774.png', '10406.png', '17960.png', '22919.png', '17294.png', '06507.png', '18359.png', '22905.png', '13496.png', '20850.png', '24210.png', '09573.png', '14534.png', '04553.png', '08178.png', '11811.png', '17403.png', '00569.png', '02287.png', '15193.png', '04705.png', '24077.png', '05168.png', '15991.png', '06861.png', '18560.png', '06633.png', '01406.png', '17037.png', '19901.png', '11640.png', '07677.png', '04566.png', '17829.png', '10385.png', '21749.png', '01058.png', '17202.png', '24134.png', '15061.png', '09265.png', '11878.png', '07175.png', '06095.png', '13447.png', '05847.png', '11955.png', '09023.png', '21485.png', '07591.png', '06316.png', '16248.png', '21411.png', '21921.png', '04454.png', '12903.png', '16760.png', '04130.png', '11155.png', '15164.png', '18434.png', '01166.png', '00237.png', '09364.png', '12351.png', '12834.png', '19819.png', '07086.png', '07085.png', '14033.png', '04487.png', '21581.png', '02272.png', '17783.png', '13390.png', '22500.png', '23865.png', '03218.png', '17227.png', '20573.png', '03684.png', '16979.png', '13118.png', '00003.png', '24031.png', '03513.png', '09608.png', '03777.png', '09545.png', '23075.png', '02537.png', '24170.png', '03242.png', '04656.png', '15176.png', '18874.png', '24918.png', '03500.png', '15209.png', '10067.png', '12293.png', '11402.png', '02514.png', '09637.png', '12960.png', '20494.png', '01463.png', '24178.png', '03378.png', '11210.png', '11792.png', '17453.png', '18170.png', '19749.png', '23249.png', '21452.png', '07344.png', '03054.png', '05598.png', '08302.png', '09883.png', '24653.png', '04593.png', '10690.png', '04646.png', '19305.png', '08934.png', '06863.png', '07543.png', '20993.png', '21640.png', '01428.png', '18707.png', '07975.png', '02255.png', '18872.png', '13658.png', '20200.png', '00832.png', '09020.png', '04323.png', '07643.png', '19429.png', '22775.png', '08491.png', '18995.png', '08128.png', '09636.png', '18370.png', '05715.png', '20380.png', '15125.png', '20345.png', '06420.png', '12636.png', '17071.png', '20259.png', '12051.png', '21819.png', '13417.png', '18666.png', '16296.png', '10747.png', '10536.png', '17465.png', '13947.png', '08753.png', '14181.png', '04530.png', '19925.png', '03543.png', '07004.png', '17262.png', '05341.png', '23980.png', '20759.png', '15741.png', '23268.png', '09463.png', '01801.png', '21415.png', '15365.png', '14961.png', '04190.png', '15998.png', '03152.png', '02609.png', '14048.png', '24676.png', '13471.png', '18335.png', '23293.png', '00290.png', '15560.png', '22128.png', '24021.png', '16515.png', '13442.png', '01877.png', '19184.png', '18147.png', '09973.png', '15606.png', '24787.png', '17857.png', '07732.png', '04706.png', '10723.png', '21559.png', '06567.png', '08712.png', '04717.png', '06697.png', '07148.png', '16259.png', '11757.png', '17535.png', '11286.png', '19636.png', '09911.png', '13803.png', '21763.png', '05717.png', '21616.png', '13841.png', '08566.png', '01370.png', '23504.png', '22453.png', '22889.png', '15419.png', '01554.png', '15376.png', '09446.png', '19279.png', '14706.png', '04316.png', '12945.png', '02336.png', '02428.png', '12565.png', '11986.png', '21790.png', '03662.png', '06667.png', '15743.png', '23904.png', '01993.png', '05481.png', '22692.png', '16071.png', '01458.png', '22653.png', '14772.png', '21460.png', '00530.png', '24614.png', '01548.png', '06047.png', '01657.png', '12054.png', '08428.png', '23565.png', '16109.png', '20679.png', '05272.png', '11603.png', '13474.png', '00350.png', '02885.png', '16516.png', '09475.png', '07055.png', '01594.png', '08645.png', '24277.png', '15748.png', '03027.png', '08839.png', '17138.png', '21333.png', '17348.png', '04792.png', '17293.png', '01645.png', '07309.png', '19729.png', '21707.png', '03352.png', '12136.png', '10237.png', '13115.png', '07663.png', '10427.png', '04991.png', '05646.png', '23194.png', '02449.png', '01487.png', '23618.png', '08595.png', '11822.png', '14912.png', '24957.png', '23390.png', '05816.png', '04701.png', '22275.png', '23761.png', '04133.png', '01947.png', '08209.png', '15978.png', '15303.png', '07560.png', '10670.png', '08580.png', '01600.png', '18144.png', '14820.png', '16124.png', '07956.png', '07602.png', '10608.png', '07510.png', '12314.png', '10488.png', '00063.png', '12933.png', '01282.png', '20925.png', '18186.png', '15675.png', '23401.png', '05480.png', '01756.png', '03208.png', '01076.png', '07384.png', '00924.png', '10575.png', '15266.png', '03498.png', '24864.png', '06239.png', '06744.png', '05346.png', '05643.png', '06345.png', '15096.png', '10689.png', '04155.png', '07379.png', '22054.png', '16527.png', '20340.png', '23630.png', '23400.png', '10792.png', '20861.png', '10604.png', '18909.png', '15692.png', '12183.png', '02633.png', '18815.png', '01770.png', '21557.png', '16140.png', '11265.png', '21450.png', '22300.png', '23417.png', '05913.png', '07197.png', '03076.png', '13923.png', '09448.png', '00216.png', '03898.png', '23702.png', '11637.png', '00786.png', '11165.png', '06303.png', '13008.png', '02295.png', '08758.png', '08718.png', '07007.png', '01345.png', '05437.png', '21211.png', '17602.png', '08655.png', '16909.png', '07838.png', '22234.png', '02797.png', '13126.png', '07298.png', '14305.png', '21884.png', '00383.png', '02641.png', '23213.png', '04329.png', '00825.png', '24462.png', '05436.png', '14257.png', '09537.png', '09150.png', '01961.png', '03201.png', '18952.png', '18160.png', '21007.png', '14343.png', '04165.png', '09509.png', '05355.png', '09189.png', '08299.png', '06256.png', '19410.png', '11144.png', '24764.png', '22603.png', '20003.png', '09894.png', '19241.png', '19192.png', '01735.png', '08774.png', '10866.png', '15886.png', '00744.png', '22697.png', '11108.png', '12250.png', '19089.png', '09949.png', '13239.png', '22743.png', '09069.png', '13691.png', '19055.png', '12419.png', '21978.png', '04556.png', '21071.png', '17632.png', '01616.png', '01469.png', '20043.png', '20854.png', '11884.png', '15308.png', '01161.png', '20579.png', '18346.png', '23558.png', '17552.png', '10006.png', '08390.png', '03304.png', '00921.png', '15595.png', '21675.png', '21849.png', '13302.png', '04093.png', '03332.png', '22476.png', '16005.png', '24886.png', '11156.png', '04198.png', '04608.png', '15385.png', '09276.png', '11968.png', '13329.png', '03403.png', '20832.png', '22619.png', '01387.png', '00754.png', '06514.png', '15884.png', '00668.png', '00594.png', '13191.png', '22174.png', '13853.png', '11641.png', '07045.png', '08827.png', '18171.png', '13011.png', '20094.png', '21166.png', '01568.png', '13897.png', '11709.png', '06183.png', '03391.png', '19337.png', '02552.png', '17607.png', '08997.png', '18002.png', '08667.png', '14957.png', '20435.png', '11245.png', '00970.png', '03874.png', '12993.png', '04326.png', '07276.png', '16069.png', '16991.png', '23608.png', '03372.png', '00736.png', '06518.png', '22350.png', '09819.png', '23135.png', '01785.png', '05728.png', '00817.png', '21471.png', '24260.png', '15579.png', '24639.png', '21786.png', '16978.png', '04637.png', '08699.png', '21901.png', '19622.png', '23917.png', '00373.png', '18941.png', '13668.png', '19392.png', '10252.png', '22203.png', '23387.png', '03408.png', '04910.png', '04142.png', '20723.png', '09258.png', '03610.png', '14811.png', '08416.png', '13355.png', '22343.png', '10839.png', '21156.png', '04579.png', '07772.png', '21102.png', '08578.png', '00521.png', '10212.png', '00300.png', '23834.png', '03395.png', '10820.png', '12109.png', '16155.png', '11538.png', '15883.png', '15681.png', '06300.png', '02474.png', '03579.png', '06495.png', '20349.png', '20934.png', '01022.png', '17670.png', '00343.png', '13299.png', '13300.png', '21647.png', '08470.png', '02103.png', '11934.png', '12341.png', '20817.png', '10246.png', '01570.png', '12745.png', '07287.png', '18654.png', '14479.png', '15268.png', '18905.png', '17520.png', '05330.png', '08444.png', '17328.png', '14269.png', '19276.png', '07359.png', '07062.png', '18615.png', '16653.png', '24855.png', '05909.png', '04027.png', '06440.png', '07687.png', '24863.png', '08484.png', '10270.png', '18486.png', '02822.png', '08274.png', '19141.png', '00593.png', '22024.png', '05775.png', '06417.png', '09134.png', '13249.png', '24587.png', '21817.png', '09851.png', '22864.png', '11471.png', '04534.png', '00942.png', '15355.png', '16812.png', '21225.png', '07736.png', '19299.png', '13970.png', '02088.png', '12736.png', '01289.png', '24221.png', '24422.png', '04694.png', '21746.png', '01018.png', '21546.png', '07587.png', '09768.png', '24454.png', '10502.png', '24453.png', '14777.png', '17561.png', '24620.png', '02925.png', '13328.png', '13591.png', '10944.png', '23690.png', '13306.png', '21556.png', '20913.png', '23818.png', '20627.png', '08471.png', '06887.png', '22708.png', '08357.png', '22701.png', '02630.png', '17512.png', '13862.png', '12982.png', '03103.png', '21302.png', '18241.png', '08100.png', '07383.png', '23572.png', '06919.png', '14921.png', '19918.png', '23928.png', '07613.png', '03165.png', '16511.png', '22222.png', '00967.png', '01068.png', '14522.png', '21002.png', '06391.png', '08863.png', '07476.png', '21461.png', '10697.png', '06063.png', '20402.png', '01805.png', '09302.png', '24721.png', '19571.png', '09083.png', '11523.png', '10295.png', '20159.png', '17627.png', '23669.png', '12616.png', '01484.png', '18845.png', '09583.png', '06483.png', '15725.png', '14738.png', '03454.png', '12498.png', '12005.png', '17724.png', '01509.png', '15309.png', '09284.png', '22000.png', '24359.png', '03668.png', '08410.png', '06329.png', '06456.png', '23684.png', '17445.png', '15373.png', '19326.png', '14347.png', '24615.png', '08219.png', '15530.png', '19407.png', '01351.png', '12658.png', '08887.png', '13173.png', '10227.png', '05632.png', '23752.png', '03290.png', '11381.png', '07072.png', '20706.png', '17157.png', '18226.png', '21898.png', '09725.png', '04994.png', '22302.png', '16334.png', '03158.png', '14043.png', '08252.png', '03399.png', '09148.png', '12415.png', '21569.png', '00088.png', '01627.png', '04835.png', '07806.png', '13383.png', '03050.png', '15362.png', '10745.png', '05955.png', '19826.png', '01521.png', '19979.png', '08634.png', '16977.png', '13048.png', '18837.png', '00362.png', '23356.png', '23709.png', '18257.png', '10758.png', '01378.png', '18579.png', '19711.png', '09452.png', '15177.png', '13582.png', '09968.png', '02958.png', '07502.png', '13965.png', '03765.png', '07985.png', '00616.png', '16555.png', '00061.png', '01518.png', '21782.png', '13096.png', '07646.png', '06957.png', '23875.png', '24457.png', '24262.png', '22208.png', '22813.png', '18382.png', '01047.png', '05304.png', '10870.png', '15062.png', '02707.png', '11979.png', '19533.png', '17314.png', '11920.png', '02593.png', '10596.png', '15842.png', '18109.png', '17115.png', '15153.png', '08033.png', '19021.png', '10321.png', '10665.png', '15764.png', '01241.png', '15228.png', '08293.png', '00356.png', '21322.png', '06143.png', '10298.png', '10603.png', '02869.png', '24956.png', '07406.png', '22647.png', '21596.png', '18569.png', '13957.png', '00774.png', '15858.png', '09183.png', '19216.png', '20421.png', '11970.png', '19063.png', '04975.png', '21428.png', '07662.png', '06938.png', '14955.png', '17861.png', '04660.png', '24161.png', '00274.png', '10419.png', '01368.png', '06515.png', '03471.png', '17353.png', '06561.png', '22441.png', '00008.png', '17268.png', '06498.png', '10258.png', '21153.png', '01612.png', '15172.png', '05232.png', '02736.png', '22091.png', '12152.png', '00261.png', '05312.png', '24766.png', '05406.png', '17654.png', '12400.png', '11965.png', '03603.png', '18824.png', '09382.png', '17583.png', '00136.png', '19425.png', '21969.png', '06718.png', '15357.png', '20054.png', '14440.png', '05521.png', '06460.png', '16393.png', '17741.png', '23212.png', '23175.png', '19116.png', '11718.png', '22384.png', '07507.png', '08007.png', '13778.png', '05785.png', '01224.png', '24294.png', '02200.png', '24572.png', '07240.png', '11839.png', '19530.png', '02623.png', '07473.png', '22046.png', '16901.png', '21804.png', '04434.png', '12670.png', '21014.png', '08075.png', '02804.png', '15878.png', '00079.png', '04962.png', '05694.png', '17770.png', '11969.png', '14147.png', '08624.png', '02997.png', '24001.png', '18608.png', '03194.png', '06270.png', '08303.png', '16738.png', '00609.png', '21084.png', '08495.png', '14573.png', '06877.png', '15449.png', '03052.png', '18336.png', '19864.png', '19767.png', '12214.png', '10008.png', '05503.png', '14705.png', '13051.png', '10857.png', '05413.png', '23393.png', '21897.png', '17134.png', '09225.png', '03992.png', '14936.png', '07277.png', '24518.png', '19196.png', '12585.png', '20581.png', '01691.png', '16758.png', '02073.png', '22878.png', '05978.png', '09737.png', '23049.png', '04926.png', '12375.png', '10093.png', '14112.png', '06815.png', '07219.png', '05427.png', '05996.png', '15169.png', '11838.png', '12104.png', '24567.png', '20052.png', '11552.png', '00169.png', '02492.png', '05089.png', '07761.png', '14392.png', '06962.png', '11466.png', '23101.png', '08861.png', '07911.png', '16292.png', '06447.png', '10908.png', '08707.png', '21025.png', '03331.png', '10244.png', '06217.png', '13998.png', '17390.png', '12150.png', '18143.png', '09435.png', '19038.png', '00495.png', '01730.png', '03601.png', '14061.png', '09707.png', '17489.png', '12627.png', '17888.png', '10045.png', '11570.png', '21148.png', '22034.png', '06454.png', '09269.png', '20541.png', '05511.png', '10014.png', '01577.png', '10968.png', '11927.png', '19963.png', '17809.png', '16193.png', '11907.png', '02402.png', '13735.png', '12469.png', '20195.png', '22359.png', '00309.png', '20710.png', '01764.png', '02093.png', '06338.png', '23931.png', '02558.png', '11793.png', '10806.png', '15364.png', '20477.png', '23810.png', '18971.png', '20702.png', '17419.png', '12466.png', '22479.png', '14024.png', '23063.png', '15439.png', '19531.png', '21055.png', '17024.png', '08248.png', '01810.png', '10807.png', '12008.png', '23872.png', '17357.png', '23935.png', '02970.png', '12852.png', '07841.png', '09046.png', '06203.png', '16089.png', '02693.png', '12389.png', '15133.png', '21881.png', '05680.png', '13924.png', '07577.png', '08150.png', '03435.png', '23477.png', '18911.png', '19679.png', '08494.png', '20597.png', '21423.png', '07458.png', '16580.png', '15685.png', '00833.png', '02475.png', '00414.png', '15040.png', '09044.png', '11224.png', '11139.png', '15485.png', '09275.png', '12370.png', '08381.png', '21115.png', '15514.png', '08572.png', '07994.png', '22229.png', '23440.png', '09521.png', '07160.png', '18984.png', '10052.png', '21414.png', '13430.png', '05438.png', '01481.png', '01939.png', '01677.png', '15332.png', '21794.png', '03202.png', '06013.png', '05763.png', '09012.png', '18618.png', '24104.png', '01359.png', '19190.png', '23861.png', '20787.png', '05344.png', '02963.png', '00465.png', '04213.png', '22413.png', '24114.png', '04662.png', '08448.png', '12689.png', '11399.png', '02071.png', '17361.png', '19539.png', '22168.png', '22188.png', '05093.png', '11031.png', '04603.png', '05830.png', '16136.png', '21965.png', '01838.png', '02435.png', '16052.png', '16920.png', '09336.png', '20550.png', '02907.png', '02487.png', '12901.png', '20018.png', '18839.png', '02572.png', '18225.png', '18195.png', '13028.png', '07155.png', '02908.png', '05247.png', '08105.png', '17953.png', '07290.png', '04998.png', '04537.png', '14784.png', '15894.png', '12618.png', '00121.png', '10232.png', '06674.png', '14717.png', '04882.png', '14643.png', '21611.png', '16409.png', '08421.png', '07842.png', '03864.png', '04843.png', '23775.png', '16582.png', '02715.png', '24056.png', '17387.png', '10571.png', '10960.png', '15519.png', '18875.png', '23540.png', '15607.png', '09461.png', '23339.png', '02163.png', '07590.png', '07770.png', '06808.png', '12712.png', '01606.png', '16630.png', '15645.png', '00798.png', '15006.png', '24798.png', '20118.png', '07778.png', '04869.png', '01218.png', '08020.png', '10769.png', '24945.png', '12639.png', '12665.png', '12424.png', '13880.png', '24049.png', '17701.png', '10597.png', '19550.png', '01226.png', '16030.png', '22675.png', '21533.png', '18794.png', '22415.png', '10492.png', '06779.png', '24472.png', '23376.png', '19751.png', '14788.png', '24217.png', '04969.png', '14172.png', '05195.png', '03646.png', '06181.png', '07453.png', '00812.png', '22019.png', '09642.png', '15210.png', '12049.png', '19264.png', '20884.png', '07067.png', '07050.png', '07448.png', '14289.png', '21918.png', '11247.png', '12237.png', '08247.png', '24319.png', '15119.png', '13242.png', '08812.png', '00598.png', '18893.png', '15736.png', '08987.png', '19658.png', '11919.png', '23617.png', '01253.png', '09555.png', '17813.png', '17826.png', '15812.png', '17875.png', '00664.png', '17679.png', '21750.png', '15330.png', '17893.png', '13956.png', '05449.png', '06653.png', '23015.png', '23029.png', '13313.png', '11608.png', '13023.png', '22104.png', '09293.png', '17304.png', '06620.png', '23269.png', '15574.png', '16191.png', '12967.png', '21809.png', '04511.png', '06894.png', '12435.png', '01669.png', '23543.png', '10472.png', '20080.png', '07112.png', '05500.png', '03758.png', '02507.png', '16647.png', '07311.png', '21968.png', '01508.png', '21442.png', '22404.png', '18832.png', '00286.png', '21711.png', '11770.png', '07395.png', '20525.png', '13877.png', '08326.png', '23538.png', '20044.png', '09462.png', '21152.png', '22325.png', '23680.png', '00623.png', '23898.png', '00529.png', '11503.png', '02770.png', '21318.png', '02431.png', '02004.png', '03223.png', '24013.png', '03931.png', '13661.png', '05115.png', '20223.png', '16362.png', '06193.png', '08107.png', '06112.png', '11819.png', '08281.png', '09502.png', '10595.png', '17997.png', '01873.png', '10203.png', '08550.png', '20827.png', '15911.png', '01184.png', '18204.png', '23397.png', '00982.png', '14283.png', '19812.png', '18516.png', '02204.png', '02708.png', '08925.png', '20208.png', '09180.png', '10738.png', '20689.png', '09613.png', '16265.png', '04146.png', '16942.png', '13928.png', '11239.png', '12916.png', '13545.png', '18394.png', '18942.png', '10542.png', '20743.png', '21367.png', '19135.png', '21856.png', '11660.png', '14903.png', '14765.png', '18069.png', '12688.png', '03853.png', '00186.png', '00631.png', '21913.png', '03321.png', '23220.png', '05337.png', '14054.png', '02457.png', '09219.png', '18055.png', '03960.png', '01566.png', '12422.png', '07047.png', '19222.png', '13776.png', '21035.png', '22887.png', '13609.png', '17270.png', '01894.png', '07190.png', '04721.png', '12991.png', '17352.png', '02689.png', '23292.png', '05194.png', '17598.png', '22984.png', '05698.png', '09812.png', '12225.png', '18797.png', '03935.png', '03508.png', '15023.png', '11940.png', '09742.png', '04513.png', '13489.png', '03667.png', '22952.png', '15851.png', '08792.png', '24372.png', '08116.png', '17906.png', '04669.png', '05510.png', '04966.png', '07686.png', '10352.png', '17693.png', '04252.png', '16945.png', '12246.png', '20401.png', '11689.png', '08441.png', '17880.png', '01066.png', '02468.png', '01524.png', '02096.png', '08872.png', '24305.png', '21009.png', '22928.png', '20290.png', '01144.png', '18371.png', '18567.png', '19661.png', '05285.png', '00488.png', '16341.png', '01940.png', '18580.png', '21779.png', '23970.png', '16354.png', '24384.png', '11877.png', '04622.png', '05750.png', '22624.png', '06546.png', '07705.png', '22796.png', '06331.png', '14006.png', '12297.png', '21433.png', '15273.png', '12805.png', '23805.png', '06608.png', '11255.png', '11272.png', '05603.png', '04509.png', '10381.png', '21024.png', '02624.png', '06963.png', '16406.png', '18513.png', '06397.png', '15265.png', '18278.png', '04714.png', '08026.png', '07559.png', '11481.png', '05216.png', '24174.png', '00496.png', '07143.png', '09351.png', '24398.png', '23290.png', '15399.png', '13141.png', '21699.png', '17582.png', '03939.png', '23196.png', '08503.png', '23586.png', '23765.png', '18923.png', '13154.png', '03334.png', '24681.png', '00794.png', '03491.png', '09161.png', '02048.png', '23100.png', '02123.png', '24617.png', '04031.png', '12013.png', '08096.png', '24502.png', '06130.png', '05273.png', '21043.png', '20730.png', '00795.png', '19144.png', '13284.png', '04904.png', '15954.png', '17210.png', '22434.png', '24811.png', '05148.png', '11562.png', '12456.png', '03445.png', '02127.png', '16283.png', '11702.png', '20923.png', '22825.png', '07881.png', '05939.png', '16914.png', '23960.png', '01492.png', '10134.png', '19394.png', '02371.png', '15526.png', '21199.png', '11121.png', '01362.png', '05701.png', '23951.png', '09123.png', '14639.png', '24440.png', '04403.png', '05540.png', '17209.png', '13195.png', '00841.png', '19273.png', '17621.png', '19457.png', '11786.png', '10062.png', '07941.png', '09053.png', '12536.png', '08393.png', '06128.png', '12899.png', '05652.png', '03162.png', '04153.png', '02944.png', '14078.png', '24677.png', '15649.png', '19426.png', '14735.png', '08278.png', '07823.png', '08746.png', '24735.png', '02335.png', '11567.png', '21526.png', '00040.png', '12843.png', '05984.png', '08591.png', '24083.png', '16763.png', '03648.png', '02727.png', '12036.png', '11027.png', '04090.png', '07552.png', '21584.png', '24893.png', '13083.png', '04216.png', '06729.png', '19953.png', '09188.png', '10562.png', '07728.png', '16596.png', '20705.png', '06513.png', '02411.png', '15205.png', '04131.png', '07788.png', '05447.png', '20864.png', '16123.png', '06994.png', '22917.png', '06418.png', '00095.png', '24177.png', '20383.png', '05227.png', '19880.png', '17130.png', '09823.png', '02808.png', '03921.png', '00371.png', '11382.png', '19567.png', '23192.png', '05367.png', '07671.png', '13431.png', '08664.png', '01621.png', '11398.png', '12505.png', '24336.png', '18070.png', '08748.png', '20002.png', '23052.png', '13610.png', '03541.png', '13396.png', '14634.png', '03009.png', '07767.png', '13468.png', '24142.png', '07070.png', '13316.png', '09126.png', '24525.png', '23632.png', '24381.png', '09481.png', '02608.png', '05964.png', '17560.png', '14109.png', '10013.png', '15693.png', '19322.png', '19360.png', '22968.png', '12261.png', '19003.png', '13491.png', '15134.png', '24547.png', '18442.png', '02892.png', '08979.png', '07305.png', '23903.png', '04280.png', '19230.png', '22050.png', '12148.png', '16484.png', '21896.png', '00941.png', '00606.png', '24816.png', '05365.png', '17833.png', '22318.png', '00235.png', '05382.png', '10521.png', '22279.png', '08794.png', '14250.png', '21816.png', '13544.png', '13395.png', '14370.png', '17468.png', '07903.png', '11590.png', '05781.png', '12603.png', '16953.png', '17119.png', '23271.png', '05020.png', '14482.png', '19848.png', '01505.png', '14599.png', '13030.png', '04037.png', '24227.png', '12190.png', '01981.png', '05236.png', '11764.png', '15486.png', '04161.png', '09841.png', '19736.png', '06016.png', '01571.png', '22084.png', '14063.png', '15208.png', '22646.png', '03548.png', '18856.png', '09653.png', '00730.png', '18357.png', '16049.png', '04388.png', '09701.png', '11338.png', '22806.png', '15719.png', '22431.png', '05005.png', '03973.png', '11296.png', '22396.png', '07229.png', '10199.png', '08158.png', '23189.png', '13488.png', '08030.png', '24700.png', '05868.png', '09394.png', '20664.png', '06970.png', '22231.png', '19494.png', '07680.png', '01745.png', '16691.png', '12025.png', '08731.png', '07540.png', '07136.png', '04613.png', '04457.png', '05107.png', '10396.png', '16454.png', '10610.png', '11760.png', '14696.png', '19609.png', '06096.png', '22481.png', '09585.png', '00400.png', '18351.png', '20867.png', '06015.png', '01109.png', '10443.png', '13693.png', '21783.png', '00424.png', '13134.png', '10100.png', '23911.png', '00159.png', '21848.png', '20190.png', '07021.png', '08239.png', '11685.png', '05442.png', '06410.png', '07184.png', '18743.png', '06933.png', '01760.png', '12821.png', '11352.png', '16587.png', '06849.png', '03629.png', '01876.png', '17808.png', '12215.png', '15319.png', '10119.png', '00689.png', '05128.png', '13910.png', '13215.png', '00353.png', '03051.png', '04385.png', '16160.png', '13875.png', '19817.png', '23553.png', '24671.png', '07718.png', '08159.png', '03607.png', '07206.png', '11410.png', '18734.png', '11671.png', '10988.png', '05810.png', '15271.png', '17927.png', '07347.png', '18132.png', '21541.png', '15482.png', '18237.png', '20293.png', '03487.png', '19697.png', '10936.png', '03388.png', '06710.png', '11283.png', '13323.png', '18249.png', '17354.png', '16255.png', '18948.png', '12990.png', '11516.png', '22748.png', '22702.png', '14923.png', '20377.png', '16110.png', '09299.png', '11913.png', '22788.png', '11663.png', '16872.png', '11051.png', '00423.png', '04996.png', '01485.png', '16702.png', '19656.png', '22780.png', '11929.png', '18100.png', '13286.png', '01763.png', '09525.png', '24712.png', '22684.png', '06776.png', '01131.png', '07786.png', '24854.png', '05043.png', '22299.png', '12590.png', '24059.png', '05267.png', '04257.png', '07270.png', '00308.png', '16418.png', '17277.png', '24790.png', '21190.png', '02987.png', '07624.png', '14458.png', '19987.png', '06922.png', '11730.png', '19830.png', '01950.png', '13580.png', '22586.png', '24962.png', '00180.png', '01150.png', '10798.png', '04008.png', '14510.png', '08982.png', '24444.png', '05769.png', '06807.png', '10071.png', '21327.png', '07481.png', '14486.png', '01125.png', '14833.png', '20824.png', '22583.png', '11169.png', '23332.png', '10576.png', '23720.png', '18271.png', '23721.png', '10650.png', '08359.png', '22661.png', '04868.png', '16090.png', '04770.png', '09468.png', '17950.png', '10308.png', '04802.png', '08466.png', '09825.png', '16737.png', '13152.png', '10285.png', '01397.png', '12720.png', '11591.png', '11088.png', '06451.png', '03080.png', '00231.png', '20518.png', '17299.png', '13315.png', '08058.png', '12589.png', '14056.png', '10974.png', '07880.png', '16936.png', '04704.png', '13000.png', '21205.png', '16405.png', '00512.png', '24659.png', '21132.png', '20141.png', '01245.png', '19029.png', '13781.png', '04757.png', '15813.png', '18251.png', '00181.png', '22211.png', '23151.png', '22707.png', '22941.png', '07919.png', '10803.png', '20852.png', '15568.png', '00273.png', '12617.png', '07870.png', '15903.png', '12985.png', '11695.png', '16345.png', '22578.png', '02585.png', '18538.png', '17410.png', '20227.png', '05068.png', '14173.png', '13025.png', '16057.png', '10057.png', '07497.png', '04922.png', '22099.png', '12222.png', '10584.png', '19740.png', '11097.png', '21352.png', '15071.png', '07313.png', '20522.png', '15783.png', '04561.png', '00316.png', '14114.png', '16534.png', '13237.png', '12804.png', '01852.png', '21839.png', '15720.png', '05832.png', '20130.png', '16563.png', '21959.png', '10309.png', '21042.png', '06470.png', '08407.png', '14937.png', '19757.png', '09975.png', '23501.png', '05735.png', '08637.png', '07580.png', '09548.png', '00865.png', '16772.png', '17866.png', '12694.png', '04934.png', '12173.png', '11644.png', '24481.png', '23327.png', '20086.png', '24835.png', '21121.png', '12693.png', '15964.png', '02495.png', '09900.png', '22916.png', '05524.png', '23148.png', '03923.png', '09367.png', '01037.png', '19814.png', '16825.png', '19388.png', '00626.png', '22094.png', '15825.png', '05585.png', '21168.png', '12763.png', '12875.png', '13815.png', '15929.png', '20160.png', '08207.png', '19462.png', '16933.png', '10323.png', '24183.png', '19606.png', '24750.png', '02864.png', '02315.png', '12913.png', '21365.png', '18004.png', '00887.png', '11266.png', '20104.png', '06113.png', '09464.png', '04298.png', '16518.png', '03238.png', '20039.png', '08956.png', '13783.png', '03247.png', '22005.png', '10005.png', '08952.png', '23251.png', '17736.png', '18813.png', '01385.png', '00923.png', '20012.png', '11892.png', '05877.png', '09034.png', '13637.png', '02810.png', '11549.png', '06402.png', '08087.png', '15840.png', '10981.png', '17978.png', '05903.png', '16143.png', '05807.png', '24515.png', '23579.png', '00128.png', '12111.png', '17696.png', '06881.png', '18889.png', '08093.png', '06138.png', '17623.png', '21690.png', '16739.png', '20341.png', '10751.png', '01073.png', '06058.png', '21159.png', '10883.png', '07698.png', '03781.png', '08038.png', '14773.png', '04666.png', '04196.png', '11821.png', '09137.png', '00493.png', '24912.png', '24635.png', '09065.png', '02271.png']
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
        
        if i_iter == 217101:
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
