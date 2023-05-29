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
    a = ['00171.png', '04051.png', '09929.png', '11708.png', '16075.png', '05821.png', '20407.png', '03942.png', '12409.png', '02830.png', '11242.png', '09262.png', '04102.png', '03343.png', '14229.png', '02550.png', '08391.png', '22499.png', '08897.png', '18726.png', '16234.png', '11475.png', '02642.png', '17830.png', '14751.png', '01170.png', '14928.png', '16961.png', '18492.png', '03991.png', '04343.png', '12626.png', '09024.png', '01751.png', '08819.png', '23973.png', '08122.png', '24196.png', '04144.png', '24688.png', '17764.png', '21479.png', '10163.png', '22925.png', '14444.png', '20007.png', '21926.png', '10861.png', '14574.png', '14531.png', '05007.png', '03348.png', '03042.png', '06471.png', '07178.png', '07241.png', '04582.png', '09904.png', '03753.png', '24213.png', '04568.png', '00879.png', '18053.png', '10263.png', '11184.png', '14250.png', '22447.png', '00960.png', '08546.png', '23025.png', '21738.png', '10477.png', '15460.png', '13896.png', '17546.png', '16103.png', '05143.png', '18280.png', '08991.png', '20383.png', '23114.png', '23563.png', '18107.png', '21829.png', '23770.png', '20868.png', '14793.png', '10421.png', '08375.png', '13636.png', '10588.png', '13898.png', '12386.png', '07933.png', '23902.png', '01729.png', '11252.png', '09140.png', '08207.png', '16364.png', '02314.png', '10859.png', '02347.png', '09906.png', '12965.png', '23213.png', '08292.png', '01363.png', '17611.png', '06027.png', '11172.png', '13222.png', '19038.png', '04552.png', '17500.png', '06199.png', '08117.png', '03809.png', '08854.png', '19155.png', '00071.png', '19766.png', '07761.png', '16268.png', '05928.png', '17372.png', '00048.png', '21951.png', '01045.png', '13522.png', '08640.png', '18104.png', '10750.png', '16427.png', '15980.png', '09455.png', '08705.png', '14725.png', '05271.png', '19007.png', '07772.png', '14385.png', '09457.png', '13975.png', '03495.png', '23585.png', '05029.png', '12738.png', '20699.png', '09431.png', '01920.png', '10560.png', '04751.png', '18388.png', '00535.png', '22469.png', '22956.png', '09782.png', '22039.png', '17626.png', '07456.png', '23286.png', '13747.png', '23647.png', '04685.png', '04536.png', '24259.png', '00003.png', '05597.png', '05419.png', '11387.png', '05250.png', '09911.png', '07551.png', '13301.png', '07396.png', '14815.png', '08392.png', '18594.png', '07214.png', '17803.png', '19223.png', '01566.png', '02350.png', '03454.png', '19161.png', '18651.png', '22441.png', '07290.png', '00957.png', '20229.png', '19980.png', '09470.png', '10171.png', '06763.png', '14359.png', '22492.png', '22436.png', '21721.png', '20707.png', '11378.png', '08608.png', '23564.png', '07359.png', '22794.png', '20889.png', '20029.png', '02507.png', '16678.png', '17987.png', '14036.png', '01935.png', '11507.png', '19435.png', '02897.png', '16862.png', '03727.png', '03231.png', '22443.png', '04997.png', '15666.png', '10653.png', '16008.png', '07331.png', '15196.png', '02997.png', '14004.png', '01874.png', '23995.png', '14930.png', '09176.png', '21719.png', '07174.png', '03126.png', '16202.png', '08560.png', '19818.png', '05260.png', '12887.png', '13503.png', '23675.png', '11334.png', '14399.png', '14571.png', '06866.png', '14424.png', '01083.png', '04326.png', '09795.png', '08833.png', '12299.png', '04185.png', '00222.png', '04898.png', '06899.png', '04034.png', '13631.png', '11447.png', '13953.png', '03703.png', '04339.png', '00424.png', '04079.png', '21048.png', '14097.png', '19425.png', '07003.png', '09953.png', '21873.png', '07380.png', '18696.png', '12028.png', '21943.png', '05245.png', '24297.png', '18279.png', '05892.png', '23736.png', '05741.png', '21349.png', '15003.png', '23250.png', '03904.png', '21300.png', '23235.png', '09065.png', '04378.png', '18235.png', '07578.png', '19275.png', '04754.png', '11934.png', '05680.png', '04334.png', '08462.png', '07762.png', '13112.png', '09525.png', '05145.png', '23556.png', '21861.png', '11648.png', '22985.png', '03935.png', '22815.png', '09903.png', '15405.png', '16213.png', '02820.png', '08557.png', '14568.png', '20665.png', '00846.png', '20420.png', '01027.png', '04152.png', '18985.png', '16156.png', '16511.png', '13071.png', '24069.png', '20644.png', '02680.png', '10241.png', '04891.png', '14510.png', '22680.png', '21743.png', '21627.png', '12559.png', '00265.png', '16643.png', '11487.png', '05523.png', '22218.png', '09976.png', '15963.png', '20528.png', '24548.png', '16701.png', '06950.png', '09617.png', '18175.png', '18812.png', '06425.png', '14802.png', '17730.png', '15761.png', '05032.png', '14594.png', '20135.png', '06382.png', '00287.png', '20860.png', '15619.png', '04713.png', '02275.png', '10558.png', '20898.png', '00721.png', '18877.png', '03233.png', '04920.png', '04089.png', '06652.png', '12276.png', '21067.png', '03175.png', '19906.png', '23447.png', '15393.png', '00874.png', '09983.png', '00746.png', '24654.png', '18605.png', '22128.png', '05566.png', '06243.png', '11554.png', '02947.png', '13646.png', '13321.png', '01563.png', '08140.png', '11925.png', '09402.png', '21886.png', '24449.png', '17582.png', '20126.png', '13243.png', '24511.png', '12195.png', '14052.png', '04655.png', '08342.png', '13788.png', '22368.png', '18932.png', '01753.png', '08141.png', '07028.png', '14301.png', '09732.png', '02417.png', '05588.png', '11498.png', '05695.png', '07079.png', '16094.png', '08443.png', '17352.png', '09163.png', '14575.png', '06792.png', '19866.png', '24509.png', '24737.png', '10506.png', '08201.png', '04638.png', '24101.png', '10680.png', '17558.png', '08607.png', '13937.png', '05558.png', '05027.png', '05690.png', '22329.png', '04970.png', '12904.png', '09913.png', '00097.png', '15834.png', '17959.png', '06481.png', '03932.png', '18695.png', '11515.png', '16631.png', '02391.png', '19237.png', '21215.png', '02799.png', '04330.png', '19438.png', '19876.png', '08248.png', '00533.png', '15716.png', '10075.png', '23105.png', '08821.png', '24403.png', '17734.png', '23938.png', '19144.png', '06560.png', '15079.png', '24690.png', '13995.png', '05514.png', '04484.png', '19193.png', '13881.png', '01707.png', '01105.png', '16165.png', '10487.png', '11374.png', '17027.png', '14046.png', '01064.png', '21100.png', '01817.png', '10733.png', '16372.png', '19433.png', '01712.png', '05636.png', '05147.png', '05174.png', '22520.png', '04546.png', '22635.png', '02489.png', '19616.png', '10509.png', '16868.png', '12959.png', '06014.png', '09891.png', '01150.png', '23045.png', '09583.png', '10165.png', '04954.png', '19254.png', '10468.png', '23863.png', '14374.png', '08731.png', '01589.png', '16957.png', '18870.png', '12550.png', '08161.png', '15012.png', '06820.png', '16553.png', '15607.png', '12336.png', '22848.png', '09997.png', '21092.png', '20062.png', '22971.png', '17051.png', '09930.png', '06938.png', '13654.png', '21047.png', '03314.png', '03945.png', '14910.png', '24933.png', '04628.png', '19998.png', '17789.png', '21020.png', '15779.png', '02025.png', '13266.png', '15103.png', '03925.png', '22009.png', '14324.png', '23878.png', '11065.png', '08630.png', '18778.png', '18911.png', '14476.png', '03754.png', '02509.png', '09712.png', '20922.png', '19741.png', '23612.png', '09800.png', '04191.png', '02449.png', '00262.png', '14075.png', '00793.png', '09971.png', '12663.png', '04477.png', '03279.png', '13312.png', '12520.png', '18048.png', '05622.png', '02192.png', '22205.png', '02167.png', '13160.png', '05604.png', '09972.png', '10702.png', '03907.png', '23111.png', '21214.png', '00670.png', '21431.png', '18010.png', '11959.png', '23273.png', '15383.png', '09327.png', '11774.png', '02226.png', '23860.png', '04008.png', '01674.png', '24272.png', '15298.png', '18407.png', '13551.png', '12500.png', '24733.png', '12742.png', '16772.png', '18251.png', '16475.png', '01252.png', '18998.png', '00333.png', '08818.png', '05714.png', '14738.png', '24538.png', '23847.png', '03272.png', '17190.png', '13132.png', '12855.png', '00586.png', '07834.png', '07171.png', '06182.png', '13063.png', '16728.png', '19291.png', '18202.png', '17940.png', '10248.png', '01807.png', '06930.png', '18442.png', '03268.png', '07206.png', '02849.png', '11468.png', '13128.png', '18787.png', '13348.png', '21765.png', '20263.png', '00940.png', '11109.png', '21154.png', '23012.png', '18451.png', '09943.png', '16623.png', '21312.png', '17781.png', '20130.png', '22386.png', '21607.png', '18720.png', '15017.png', '18078.png', '24206.png', '07639.png', '20095.png', '18293.png', '16035.png', '24524.png', '11506.png', '03729.png', '20945.png', '07224.png', '16603.png', '21980.png', '04022.png', '18974.png', '11735.png', '19824.png', '15595.png', '00270.png', '02931.png', '03835.png', '11692.png', '24247.png', '03099.png', '00244.png', '14502.png', '00275.png', '22766.png', '10433.png', '09340.png', '03661.png', '11220.png', '16577.png', '01736.png', '00259.png', '20341.png', '01861.png', '19861.png', '22049.png', '12666.png', '07746.png', '08056.png', '07407.png', '21986.png', '09220.png', '10644.png', '17396.png', '18813.png', '12317.png', '20475.png', '05614.png', '10501.png', '02294.png', '03436.png', '03186.png', '02807.png', '19359.png', '02062.png', '24408.png', '16412.png', '08896.png', '21967.png', '13720.png', '11043.png', '15291.png', '24575.png', '01229.png', '08087.png', '17294.png', '24885.png', '03584.png', '18463.png', '18586.png', '18269.png', '01472.png', '11743.png', '15535.png', '03533.png', '00873.png', '01324.png', '15046.png', '16251.png', '19967.png', '01482.png', '23483.png', '05130.png', '19008.png', '05631.png', '02127.png', '06640.png', '04226.png', '09204.png', '03160.png', '19242.png', '13099.png', '24055.png', '08279.png', '01824.png', '02424.png', '17342.png', '14637.png', '02404.png', '07903.png', '03479.png', '08310.png', '00907.png', '22123.png', '18736.png', '18409.png', '11602.png', '03786.png', '13754.png', '17135.png', '24222.png', '13281.png', '00835.png', '10624.png', '17535.png', '00153.png', '01360.png', '24419.png', '03598.png', '17220.png', '10781.png', '06386.png', '08942.png', '03489.png', '09819.png', '15611.png', '17737.png', '13199.png', '15664.png', '02767.png', '04798.png', '08021.png', '14188.png', '24028.png', '21975.png', '16366.png', '03666.png', '02210.png', '03868.png', '19414.png', '11060.png', '17906.png', '17201.png', '18818.png', '05660.png', '10711.png', '14132.png', '00903.png', '24347.png', '06940.png', '04753.png', '10520.png', '18527.png', '22736.png', '09756.png', '01059.png', '09598.png', '19925.png', '16933.png', '17820.png', '00407.png', '12580.png', '01225.png', '01818.png', '22986.png', '22253.png', '22752.png', '00942.png', '01848.png', '10668.png', '22885.png', '06583.png', '15846.png', '08756.png', '13740.png', '14829.png', '21036.png', '06844.png', '03063.png', '12149.png', '02055.png', '24032.png', '21496.png', '24784.png', '04399.png', '14253.png', '03252.png', '07122.png', '22345.png', '09319.png', '24337.png', '09053.png', '15863.png', '07416.png', '13724.png', '20648.png', '21369.png', '09000.png', '18609.png', '02535.png', '15953.png', '01463.png', '15640.png', '07340.png', '12123.png', '21953.png', '22775.png', '03387.png', '23197.png', '00204.png', '14493.png', '03270.png', '11778.png', '01714.png', '12824.png', '09641.png', '02915.png', '01142.png', '10007.png', '07088.png', '02715.png', '16726.png', '11722.png', '15975.png', '16435.png', '15024.png', '07101.png', '00815.png', '22617.png', '14580.png', '06788.png', '08352.png', '11154.png', '22154.png', '21893.png', '17061.png', '21849.png', '24537.png', '08179.png', '23783.png', '21930.png', '17221.png', '11360.png', '07328.png', '17283.png', '19086.png', '17092.png', '23321.png', '03887.png', '16549.png', '19796.png', '23796.png', '04600.png', '17949.png', '18322.png', '19619.png', '11633.png', '00386.png', '03152.png', '17308.png', '12223.png', '00437.png', '06150.png', '21098.png', '08907.png', '12294.png', '11891.png', '02036.png', '13240.png', '22945.png', '10992.png', '00020.png', '06969.png', '16480.png', '10538.png', '11024.png', '10739.png', '03445.png', '02172.png', '17606.png', '13459.png', '00273.png', '16415.png', '10598.png', '23972.png', '19369.png', '16259.png', '12228.png', '02601.png', '03430.png', '11726.png', '20518.png', '06072.png', '15556.png', '23968.png', '12770.png', '09866.png', '17208.png', '18348.png', '05844.png', '09511.png', '13103.png', '07312.png', '24880.png', '09385.png', '10811.png', '03631.png', '03246.png', '11137.png', '02635.png', '08495.png', '12905.png', '09788.png', '20405.png', '05521.png', '18535.png', '14772.png', '14333.png', '03146.png', '02016.png', '17434.png', '05360.png', '22194.png', '01813.png', '24578.png', '19532.png', '04911.png', '23005.png', '15168.png', '23013.png', '14044.png', '01156.png', '14550.png', '16217.png', '01444.png', '24916.png', '05427.png', '02265.png', '19165.png', '11138.png', '01073.png', '24831.png', '00764.png', '11314.png', '14215.png', '01557.png', '01641.png', '17391.png', '14564.png', '05577.png', '22595.png', '14488.png', '10218.png', '24717.png', '10167.png', '14783.png', '13658.png', '23772.png', '18664.png', '12474.png', '06445.png', '18296.png', '23432.png', '24335.png', '15740.png', '15214.png', '24463.png', '01539.png', '08778.png', '14743.png', '04154.png', '03154.png', '20588.png', '06316.png', '15304.png', '11872.png', '15654.png', '20631.png', '19146.png', '24754.png', '20521.png', '22357.png', '10023.png', '24724.png', '13387.png', '11306.png', '11201.png', '10804.png', '21170.png', '24943.png', '10527.png', '10409.png', '04894.png', '08055.png', '15057.png', '03876.png', '06204.png', '21243.png', '13591.png', '08684.png', '08038.png', '15866.png', '08002.png', '22336.png', '02185.png', '13053.png', '03769.png', '10786.png', '18971.png', '01438.png', '06087.png', '12355.png', '06895.png', '02351.png', '18277.png', '13460.png', '23698.png', '06003.png', '11839.png', '22202.png', '04257.png', '21079.png', '03044.png', '11038.png', '21373.png', '16521.png', '15812.png', '18862.png', '02599.png', '20789.png', '01907.png', '01516.png', '06746.png', '01474.png', '04491.png', '11185.png', '05028.png', '04730.png', '00028.png', '24617.png', '09540.png', '19103.png', '15725.png', '11938.png', '13166.png', '21810.png', '19474.png', '14886.png', '12628.png', '17131.png', '23946.png', '03737.png', '06766.png', '15578.png', '18012.png', '07200.png', '21895.png', '22498.png', '09168.png', '12691.png', '03286.png', '00007.png', '10534.png', '24076.png', '12084.png', '07469.png', '07168.png', '15820.png', '13038.png', '09944.png', '13052.png', '15133.png', '10511.png', '24107.png', '16826.png', '08349.png', '19378.png', '05486.png', '07056.png', '13149.png', '23533.png', '08882.png', '15828.png', '18084.png', '03750.png', '01293.png', '19579.png', '01137.png', '02443.png', '22384.png', '15084.png', '05199.png', '05346.png', '15310.png', '04722.png', '07524.png', '23668.png', '09218.png', '02101.png', '24398.png', '11559.png', '21260.png', '15560.png', '05319.png', '09290.png', '17113.png', '09135.png', '08840.png', '22231.png', '10270.png', '04716.png', '09297.png', '18464.png', '24396.png', '17984.png', '09050.png', '11972.png', '20268.png', '18581.png', '10848.png', '23681.png', '13227.png', '18639.png', '24136.png', '13890.png', '07251.png', '04981.png', '14249.png', '17121.png', '08622.png', '09367.png', '01206.png', '13538.png', '00952.png', '15720.png', '22444.png', '00573.png', '08745.png', '06191.png', '01879.png', '22339.png', '18795.png', '06184.png', '02310.png', '01061.png', '19715.png', '10934.png', '18615.png', '08586.png', '08285.png', '20190.png', '08145.png', '07951.png', '13276.png', '02390.png', '00456.png', '19162.png', '16837.png', '01245.png', '11903.png', '01199.png', '03266.png', '11089.png', '04742.png', '13868.png', '21383.png', '04992.png', '02399.png', '23883.png', '15239.png', '10040.png', '02446.png', '13663.png', '13010.png', '02630.png', '18373.png', '21988.png', '00803.png', '21529.png', '21338.png', '21567.png', '18167.png', '02317.png', '17852.png', '20497.png', '20719.png', '15220.png', '18842.png', '20781.png', '18742.png', '20783.png', '19817.png', '21974.png', '20109.png', '09196.png', '01235.png', '22405.png', '23827.png', '03657.png', '02864.png', '11086.png', '24849.png', '21337.png', '12774.png', '17826.png', '18122.png', '00227.png', '23188.png', '24804.png', '05318.png', '10484.png', '09320.png', '23231.png', '24767.png', '19081.png', '04749.png', '22642.png', '16191.png', '19509.png', '20232.png', '22967.png', '09787.png', '12012.png', '15543.png', '16194.png', '04117.png', '02487.png', '19884.png', '23016.png', '12387.png', '00716.png', '19136.png', '06983.png', '07566.png', '19638.png', '18573.png', '22789.png', '23856.png', '11724.png', '11344.png', '24038.png', '15342.png', '17538.png', '00320.png', '09915.png', '10244.png', '24691.png', '00239.png', '08430.png', '23769.png', '09830.png', '04790.png', '13554.png', '03358.png', '15523.png', '04863.png', '14955.png', '02442.png', '09400.png', '02070.png', '07599.png', '12424.png', '09608.png', '08707.png', '05798.png', '13519.png', '11818.png', '14649.png', '05045.png', '08933.png', '01560.png', '13877.png', '10919.png', '17001.png', '13504.png', '10853.png', '14679.png', '18986.png', '20566.png', '20101.png', '16797.png', '22984.png', '22045.png', '22016.png', '17537.png', '07278.png', '00023.png', '05823.png', '22369.png', '16254.png', '05124.png', '17988.png', '07873.png', '18441.png', '12802.png', '04155.png', '24598.png', '10243.png', '06475.png', '24942.png', '19560.png', '23049.png', '03394.png', '14768.png', '21595.png', '01691.png', '09985.png', '08698.png', '20131.png', '03354.png', '14872.png', '21901.png', '16065.png', '05882.png', '04317.png', '19411.png', '20902.png', '10056.png', '16578.png', '07389.png', '03591.png', '05249.png', '23754.png', '13826.png', '12584.png', '07052.png', '07240.png', '23942.png', '07182.png', '12669.png', '12836.png', '14323.png', '13083.png', '08914.png', '08506.png', '21032.png', '14363.png', '19960.png', '07665.png', '09546.png', '02541.png', '18343.png', '12800.png', '22145.png', '03818.png', '04427.png', '15606.png', '12637.png', '02766.png', '19640.png', '13633.png', '15713.png', '00349.png', '09897.png', '19942.png', '16448.png', '16976.png', '18476.png', '10280.png', '15138.png', '14122.png', '01649.png', '06144.png', '16619.png', '20137.png', '15365.png', '15343.png', '23617.png', '13206.png', '18820.png', '23499.png', '24901.png', '12535.png', '06734.png', '24835.png', '04024.png', '00674.png', '10652.png', '03857.png', '10518.png', '19269.png', '20081.png', '09005.png', '23857.png', '14471.png', '02338.png', '03435.png', '01099.png', '22917.png', '08324.png', '01192.png', '15732.png', '08621.png', '16737.png', '03846.png', '15465.png', '18494.png', '16466.png', '11798.png', '12796.png', '01732.png', '12558.png', '01146.png', '17455.png', '16091.png', '03327.png', '15464.png', '22927.png', '10958.png', '02018.png', '06948.png', '12319.png', '08695.png', '14727.png', '19000.png', '19761.png', '14908.png', '11503.png', '10177.png', '07680.png', '16283.png', '14655.png', '21287.png', '06462.png', '02571.png', '22432.png', '24474.png', '08332.png', '16070.png', '03701.png', '18517.png', '21817.png', '01593.png', '20575.png', '08841.png', '14171.png', '23764.png', '18683.png', '16838.png', '07243.png', '00688.png', '24453.png', '21009.png', '07910.png', '17937.png', '13385.png', '08687.png', '05308.png', '03181.png', '04630.png', '13830.png', '14878.png', '07988.png', '18582.png', '19988.png', '02288.png', '03561.png', '00760.png', '04622.png', '07995.png', '14364.png', '22758.png', '16792.png', '08112.png', '23569.png', '12431.png', '07111.png', '16517.png', '18868.png', '14820.png', '05764.png', '05432.png', '15215.png', '05127.png', '16177.png', '23590.png', '05664.png', '12060.png', '12308.png', '22232.png', '18040.png', '18365.png', '16231.png', '02201.png', '18356.png', '11294.png', '24554.png', '24019.png', '06005.png', '12998.png', '11842.png', '16557.png', '03274.png', '08105.png', '12967.png', '24037.png', '18566.png', '01450.png', '02707.png', '02575.png', '20034.png', '20111.png', '16490.png', '18499.png', '12858.png', '17495.png', '15898.png', '10687.png', '21924.png', '11153.png', '24301.png', '09590.png', '08082.png', '00857.png', '04104.png', '05396.png', '06048.png', '01290.png', '23723.png', '20140.png', '11801.png', '21673.png', '02111.png', '04020.png', '10110.png', '09873.png', '17108.png', '17382.png', '24096.png', '15014.png', '12473.png', '22522.png', '06028.png', '23007.png', '08290.png', '08526.png', '00639.png', '17456.png', '22262.png', '00049.png', '23051.png', '22509.png', '02322.png', '16780.png', '07393.png', '18290.png', '17498.png', '12165.png', '06035.png', '03636.png', '18685.png', '23266.png', '06727.png', '05800.png', '24226.png', '13008.png', '08046.png', '23285.png', '02981.png', '10456.png', '01089.png', '19567.png', '07843.png', '23078.png', '08835.png', '06101.png', '19497.png', '04698.png', '18772.png', '23906.png', '19327.png', '20733.png', '24420.png', '06451.png', '23261.png', '11589.png', '10957.png', '03324.png', '16923.png', '02519.png', '09931.png', '05923.png', '19642.png', '14867.png', '12402.png', '16440.png', '01713.png', '24277.png', '09445.png', '23575.png', '15811.png', '24360.png', '10065.png', '18378.png', '19632.png', '09403.png', '18303.png', '07508.png', '08235.png', '10539.png', '13338.png', '12184.png', '12411.png', '11962.png', '00299.png', '10460.png', '17298.png', '03397.png', '16426.png', '18881.png', '11580.png', '12970.png', '11915.png', '24195.png', '06935.png', '10314.png', '08093.png', '19669.png', '23549.png', '23519.png', '18693.png', '11408.png', '18027.png', '17269.png', '06495.png', '11939.png', '09926.png', '24192.png', '14275.png', '04241.png', '19860.png', '23626.png', '14537.png', '24564.png', '02227.png', '18455.png', '20124.png', '17282.png', '14731.png', '17818.png', '10720.png', '05971.png', '06725.png', '12827.png', '24949.png', '01458.png', '01332.png', '12756.png', '05215.png', '19338.png', '24497.png', '13402.png', '20330.png', '06639.png', '02097.png', '00109.png', '12129.png', '14369.png', '01005.png', '11508.png', '19803.png', '11540.png', '24377.png', '06542.png', '21548.png', '19367.png', '03444.png', '06904.png', '19292.png', '09689.png', '18241.png', '07633.png', '20771.png', '15301.png', '20396.png', '10991.png', '23460.png', '13106.png', '21443.png', '08468.png', '14864.png', '14939.png', '08368.png', '10209.png', '17660.png', '22465.png', '18238.png', '20206.png', '05745.png', '24640.png', '22622.png', '23426.png', '12211.png', '06789.png', '11984.png', '17026.png', '05739.png', '04760.png', '10651.png', '17608.png', '03006.png', '06268.png', '04272.png', '21811.png', '19067.png', '08796.png', '01840.png', '01995.png', '11158.png', '24608.png', '21937.png', '04737.png', '07078.png', '16042.png', '12729.png', '03662.png', '17729.png', '06487.png', '00453.png', '19767.png', '12760.png', '00037.png', '12557.png', '10356.png', '05686.png', '19424.png', '07483.png', '03342.png', '24707.png', '01184.png', '22131.png', '05227.png', '21577.png', '02596.png', '11998.png', '13003.png', '19603.png', '17749.png', '20218.png', '03779.png', '18918.png', '20037.png', '00093.png', '19654.png', '18978.png', '14235.png', '19447.png', '08004.png', '00620.png', '24850.png', '24533.png', '20930.png', '18130.png', '14191.png', '12470.png', '05447.png', '13137.png', '09746.png', '14500.png', '08576.png', '18847.png', '04490.png', '22305.png', '01653.png', '14189.png', '06406.png', '22023.png', '24774.png', '13780.png', '17963.png', '24622.png', '03912.png', '13490.png', '11333.png', '00987.png', '03491.png', '16534.png', '07809.png', '14661.png', '18230.png', '08382.png', '20614.png', '02861.png', '10014.png', '18895.png', '03309.png', '15293.png', '12788.png', '06389.png', '10410.png', '03346.png', '03052.png', '10302.png', '12532.png', '11477.png', '11701.png', '04642.png', '09310.png', '05251.png', '09169.png', '12510.png', '20811.png', '17519.png', '16193.png', '18669.png', '18015.png', '22983.png', '02846.png', '04721.png', '24585.png', '11450.png', '11616.png', '11982.png', '17004.png', '11545.png', '09189.png', '19213.png', '22430.png', '22943.png', '17896.png', '23184.png', '05613.png', '19826.png', '12128.png', '18103.png', '20113.png', '02455.png', '21091.png', '23246.png', '15672.png', '22942.png', '22889.png', '06318.png', '20758.png', '17550.png', '24042.png', '12942.png', '20087.png', '20072.png', '04219.png', '09086.png', '07042.png', '12685.png', '23587.png', '01680.png', '19294.png', '00998.png', '00306.png', '08602.png', '01947.png', '12612.png', '04816.png', '10297.png', '23638.png', '01131.png', '00374.png', '20800.png', '12627.png', '11105.png', '21389.png', '16055.png', '18070.png', '09741.png', '10141.png', '10415.png', '24362.png', '13119.png', '19777.png', '11565.png', '15527.png', '07913.png', '14231.png', '14771.png', '17556.png', '02457.png', '14013.png', '22364.png', '24321.png', '12620.png', '02531.png', '00690.png', '21804.png', '18961.png', '18973.png', '23965.png', '20157.png', '23648.png', '00357.png', '24627.png', '17946.png', '17900.png', '00508.png', '21180.png', '10882.png', '19340.png', '18883.png', '11928.png', '16370.png', '15081.png', '12567.png', '23806.png', '04899.png', '18372.png', '22148.png', '21329.png', '24004.png', '14030.png', '20496.png', '18115.png', '02906.png', '08806.png', '09430.png', '24499.png', '07982.png', '02922.png', '23947.png', '10845.png', '00205.png', '19566.png', '14737.png', '10561.png', '12947.png', '00441.png', '21266.png', '24314.png', '06511.png', '16166.png', '23645.png', '24383.png', '14952.png', '00117.png', '15987.png', '00129.png', '02204.png', '08287.png', '05682.png', '15498.png', '13088.png', '11096.png', '23416.png', '05760.png', '05967.png', '16818.png', '03616.png', '17151.png', '19228.png', '15902.png', '08830.png', '10764.png', '00138.png', '13161.png', '11229.png', '13484.png', '09801.png', '08266.png', '03964.png', '05818.png', '15829.png', '12141.png', '20612.png', '01106.png', '19323.png', '02150.png', '09353.png', '22628.png', '01812.png', '24540.png', '22374.png', '17243.png', '24081.png', '24492.png', '08227.png', '21752.png', '24366.png', '01296.png', '08049.png', '11400.png', '04864.png', '09464.png', '13594.png', '07460.png', '08800.png', '15947.png', '19462.png', '09112.png', '16582.png', '24574.png', '13885.png', '14152.png', '03476.png', '00376.png', '13319.png', '08456.png', '24562.png', '00978.png', '07459.png', '10889.png', '17379.png', '15869.png', '11370.png', '22534.png', '20447.png', '12934.png', '00172.png', '23448.png', '12956.png', '17084.png', '21372.png', '15418.png', '22006.png', '02787.png', '20055.png', '01592.png', '15982.png', '07314.png', '06432.png', '09558.png', '13802.png', '15426.png', '17516.png', '07113.png', '12094.png', '04589.png', '17829.png', '15719.png', '15676.png', '02803.png', '22211.png', '19041.png', '19222.png', '10149.png', '08968.png', '20534.png', '14062.png', '23820.png', '21806.png', '01511.png', '22277.png', '03241.png', '22065.png', '14871.png', '14666.png', '23864.png', '16487.png', '22712.png', '12070.png', '12143.png', '13480.png', '10799.png', '14598.png', '17185.png', '11088.png', '16241.png', '04982.png', '12640.png', '15194.png', '23372.png', '00126.png', '02151.png', '08975.png', '21436.png', '10788.png', '15399.png', '24532.png', '09282.png', '18056.png', '05819.png', '03441.png', '11965.png', '05630.png', '22378.png', '12803.png', '12530.png', '20959.png', '01402.png', '11192.png', '22140.png', '15263.png', '22152.png', '13549.png', '17388.png', '18460.png', '04971.png', '07260.png', '24134.png', '21676.png', '18975.png', '06429.png', '06407.png', '14816.png', '01875.png', '09734.png', '24718.png', '01338.png', '04723.png', '00993.png', '05895.png', '16006.png', '05333.png', '05449.png', '04963.png', '16848.png', '04972.png', '15539.png', '01376.png', '12290.png', '17769.png', '24592.png', '04924.png', '07647.png', '12334.png', '00916.png', '06787.png', '16636.png', '00915.png', '23936.png', '14272.png', '21929.png', '15287.png', '21463.png', '02855.png', '11420.png', '16139.png', '23999.png', '02891.png', '04454.png', '18923.png', '07307.png', '17040.png', '19865.png', '03738.png', '10517.png', '16847.png', '05392.png', '11275.png', '22495.png', '11262.png', '04756.png', '14797.png', '20045.png', '22563.png', '09504.png', '00618.png', '22472.png', '16959.png', '23719.png', '21268.png', '04983.png', '16618.png', '17030.png', '09378.png', '07939.png', '16854.png', '01080.png', '15222.png', '17405.png', '12082.png', '01661.png', '20274.png', '05945.png', '12875.png', '12491.png', '07563.png', '09321.png', '05535.png', '05914.png', '23461.png', '06031.png', '15891.png', '06953.png', '03888.png', '06641.png', '05131.png', '17653.png', '08195.png', '09743.png', '24190.png', '17227.png', '12725.png', '12278.png', '12897.png', '07275.png', '12671.png', '01828.png', '06258.png', '00050.png', '06103.png', '20671.png', '00696.png', '19448.png', '16832.png', '21378.png', '12840.png', '02898.png', '15753.png', '17460.png', '16387.png', '19908.png', '12301.png', '00095.png', '01942.png', '19575.png', '22047.png', '17512.png', '15776.png', '11103.png', '21265.png', '19635.png', '12226.png', '14516.png', '19181.png', '01081.png', '13033.png', '14373.png', '16565.png', '02729.png', '04413.png', '09729.png', '03988.png', '16265.png', '16849.png', '20038.png', '09292.png', '07399.png', '01802.png', '15637.png', '21227.png', '02659.png', '14298.png', '04877.png', '06669.png', '05084.png', '19090.png', '15106.png', '10847.png', '18671.png', '01247.png', '11005.png', '07348.png', '18681.png', '10082.png', '11762.png', '22596.png', '08940.png', '20484.png', '21262.png', '08474.png', '01797.png', '23577.png', '02935.png', '11012.png', '19533.png', '12275.png', '10949.png', '02631.png', '23518.png', '10441.png', '15316.png', '15429.png', '13410.png', '21148.png', '13620.png', '12180.png', '09823.png', '14739.png', '11451.png', '20523.png', '15490.png', '13075.png', '14907.png', '01011.png', '06030.png', '22121.png', '17260.png', '03092.png', '03948.png', '19380.png', '06815.png', '04058.png', '19052.png', '10096.png', '01954.png', '10319.png', '04212.png', '12134.png', '00834.png', '07594.png', '08848.png', '00191.png', '03996.png', '05064.png', '18396.png', '13210.png', '12261.png', '18656.png', '13131.png', '22555.png', '15738.png', '04876.png', '21543.png', '16318.png', '10201.png', '17599.png', '06928.png', '07329.png', '08692.png', '14553.png', '10825.png', '07865.png', '18124.png', '05772.png', '08875.png', '16593.png', '17750.png', '24664.png', '19830.png', '02394.png', '03244.png', '21200.png', '05482.png', '10583.png', '05534.png', '13330.png', '05394.png', '05830.png', '19130.png', '03316.png', '04827.png', '05494.png', '13091.png', '17047.png', '16169.png', '01927.png', '08986.png', '18850.png', '21184.png', '14167.png', '14242.png', '00067.png', '00818.png', '04467.png', '13791.png', '19314.png', '19918.png', '19225.png', '07651.png', '19692.png', '15936.png', '15242.png', '21716.png', '00013.png', '14607.png', '21270.png', '18857.png', '13716.png', '22343.png', '01461.png', '13451.png', '04680.png', '12539.png', '00880.png', '12433.png', '09539.png', '09702.png', '11022.png', '10715.png', '21415.png', '20127.png', '19956.png', '21602.png', '12595.png', '12185.png', '21025.png', '10705.png', '15967.png', '02108.png', '17240.png', '09182.png', '20729.png', '23790.png', '02309.png', '17280.png', '18731.png', '22873.png', '02970.png', '01177.png', '07048.png', '00449.png', '17678.png', '18094.png', '01418.png', '23435.png', '23442.png', '06630.png', '00293.png', '07535.png', '12540.png', '17367.png', '19409.png', '14879.png', '07705.png', '07657.png', '09515.png', '07941.png', '02662.png', '07673.png', '20297.png', '13336.png', '17562.png', '04221.png', '23987.png', '23747.png', '19543.png', '06755.png', '15124.png', '18780.png', '22839.png', '19371.png', '03965.png', '04626.png', '10159.png', '01263.png', '11967.png', '03201.png', '04116.png', '20163.png', '00962.png', '11878.png', '07796.png', '05150.png', '04233.png', '23132.png', '12849.png', '13024.png', '09363.png', '06540.png', '09245.png', '05288.png', '04151.png', '17424.png', '17228.png', '05342.png', '12560.png', '01294.png', '18467.png', '08944.png', '12831.png', '06339.png', '09326.png', '17072.png', '22223.png', '01936.png', '13584.png', '15586.png', '24413.png', '11974.png', '23588.png', '02105.png', '01517.png', '06261.png', '17000.png', '02393.png', '19701.png', '00567.png', '04070.png', '13263.png', '22033.png', '07971.png', '12772.png', '24712.png', '01378.png', '24175.png', '24573.png', '10913.png', '03110.png', '01353.png', '16912.png', '19420.png', '13314.png', '10664.png', '12071.png', '20980.png', '04068.png', '24311.png', '08530.png', '12706.png', '20537.png', '20969.png', '20590.png', '24155.png', '05389.png', '03530.png', '14289.png', '16078.png', '03872.png', '19724.png', '01846.png', '15878.png', '05811.png', '12812.png', '21908.png', '11191.png', '15857.png', '03824.png', '20193.png', '10844.png', '21899.png', '24941.png', '00537.png', '22646.png', '18264.png', '16908.png', '01136.png', '23343.png', '12022.png', '05794.png', '20799.png', '14149.png', '10707.png', '14035.png', '09824.png', '06284.png', '05706.png', '03026.png', '05810.png', '07813.png', '12687.png', '10169.png', '04806.png', '07938.png', '09077.png', '23693.png', '09248.png', '22139.png', '03724.png', '05015.png', '10722.png', '00595.png', '24266.png', '14401.png', '05279.png', '08631.png', '06598.png', '21651.png', '23406.png', '10951.png', '08754.png', '15642.png', '21626.png', '05410.png', '05366.png', '07896.png', '21196.png', '13291.png', '10994.png', '18268.png', '06892.png', '07932.png', '06905.png', '19156.png', '21072.png', '20287.png', '11882.png', '04632.png', '06986.png', '17276.png', '01973.png', '11764.png', '23513.png', '16060.png', '15028.png', '18380.png', '01829.png', '17147.png', '22256.png', '05952.png', '16052.png', '10330.png', '07401.png', '06026.png', '11916.png', '03239.png', '09704.png', '09060.png', '17696.png', '16036.png', '12743.png', '09351.png', '21305.png', '03276.png', '05609.png', '05194.png', '11119.png', '10085.png', '09348.png', '06557.png', '12394.png', '17771.png', '07530.png', '06609.png', '21608.png', '13875.png', '23781.png', '20311.png', '24203.png', '22240.png', '16182.png', '01159.png', '11614.png', '12327.png', '05889.png', '15233.png', '01727.png', '21613.png', '21380.png', '17817.png', '15848.png', '20893.png', '09157.png', '01226.png', '04937.png', '17022.png', '16510.png', '00315.png', '10978.png', '18600.png', '14909.png', '24348.png', '09439.png', '16246.png', '14948.png', '19587.png', '13345.png', '01392.png', '01038.png', '02862.png', '24489.png', '10641.png', '04856.png', '08544.png', '01847.png', '15417.png', '09468.png', '21239.png', '24444.png', '17603.png', '16062.png', '10612.png', '07031.png', '14234.png', '03736.png', '20539.png', '15608.png', '15639.png', '08915.png', '18176.png', '15276.png', '07075.png', '18116.png', '07258.png', '03687.png', '20397.png', '12001.png', '03684.png', '20905.png', '21427.png', '07378.png', '05621.png', '09755.png', '02152.png', '00026.png', '14972.png', '11781.png', '19440.png', '01386.png', '13269.png', '04236.png', '01825.png', '01197.png', '13157.png', '18016.png', '08189.png', '21452.png', '24476.png', '18431.png', '01222.png', '11797.png', '18700.png', '04715.png', '00743.png', '08466.png', '06485.png', '03627.png', '02083.png', '15727.png', '12447.png', '07372.png', '23500.png', '17485.png', '16980.png', '03586.png', '10257.png', '12808.png', '24661.png', '22860.png', '06862.png', '01406.png', '23208.png', '00137.png', '15068.png', '20782.png', '23366.png', '10528.png', '10864.png', '08836.png', '16590.png', '22828.png', '07089.png', '09296.png', '04870.png', '07718.png', '18335.png', '06019.png', '23548.png', '05226.png', '11670.png', '20514.png', '16079.png', '17212.png', '21702.png', '03509.png', '16845.png', '15454.png', '12785.png', '23836.png', '05415.png', '20362.png', '09224.png', '14409.png', '03761.png', '05610.png', '01111.png', '13444.png', '20382.png', '06333.png', '11597.png', '08273.png', '04752.png', '21612.png', '18604.png', '05099.png', '14919.png', '24687.png', '21711.png', '17992.png', '24908.png', '10637.png', '07351.png', '01286.png', '15624.png', '07151.png', '11006.png', '24115.png', '20855.png', '14331.png', '08469.png', '00365.png', '22014.png', '19877.png', '17195.png', '12127.png', '01467.png', '02579.png', '06703.png', '05727.png', '02602.png', '03410.png', '06809.png', '06760.png', '13924.png', '19121.png', '09652.png', '03773.png', '23371.png', '04458.png', '07504.png', '15351.png', '15213.png', '18080.png', '13142.png', '14651.png', '20685.png', '01762.png', '06196.png', '01977.png', '11505.png', '16950.png', '23176.png', '15020.png', '02188.png', '06768.png', '00633.png', '07024.png', '15665.png', '15583.png', '13982.png', '23990.png', '19088.png', '14452.png', '21256.png', '10712.png', '13729.png', '06049.png', '18281.png', '06071.png', '05899.png', '20533.png', '24100.png', '13965.png', '02195.png', '24429.png', '13919.png', '11261.png', '08676.png', '15302.png', '19699.png', '04424.png', '03303.png', '20290.png', '21805.png', '20320.png', '03595.png', '16138.png', '22446.png', '20628.png', '10931.png', '00906.png', '06424.png', '05269.png', '17590.png', '12383.png', '00757.png', '12005.png', '24496.png', '04397.png', '13611.png', '23283.png', '13702.png', '14125.png', '11993.png', '23182.png', '04710.png', '20836.png', '19565.png', '24340.png', '12797.png', '11146.png', '04711.png', '00448.png', '22912.png', '07672.png', '15369.png', '15382.png', '17897.png', '20016.png', '01133.png', '03698.png', '05808.png', '20351.png', '03525.png', '18585.png', '04513.png', '17161.png', '11221.png', '12868.png', '13220.png', '22406.png', '05518.png', '00338.png', '18516.png', '15013.png', '06795.png', '09810.png', '14989.png', '13959.png', '00919.png', '06437.png', '14585.png', '24446.png', '10344.png', '04652.png', '23194.png', '19429.png', '22278.png', '03633.png', '04415.png', '19882.png', '06169.png', '17138.png', '01922.png', '11016.png', '04206.png', '07333.png', '21576.png', '22550.png', '23703.png', '22519.png', '04941.png', '22689.png', '11048.png', '23843.png', '19894.png', '15836.png', '15933.png', '17932.png', '10718.png', '03187.png', '18897.png', '10912.png', '02588.png', '16204.png', '22562.png', '10213.png', '06697.png', '16673.png', '14175.png', '07957.png', '04975.png', '20617.png', '10594.png', '15329.png', '22455.png', '22695.png', '17385.png', '11663.png', '12328.png', '11460.png', '05635.png', '17381.png', '22470.png', '08374.png', '22420.png', '09672.png', '01125.png', '21508.png', '07684.png', '03493.png', '06057.png', '11257.png', '00198.png', '06338.png', '04686.png', '17645.png', '14406.png', '09551.png', '16051.png', '06750.png', '15029.png', '19185.png', '18398.png', '08527.png', '19485.png', '22059.png', '05509.png', '15388.png', '15915.png', '05224.png', '22293.png', '14912.png', '00466.png', '23493.png', '17917.png', '06961.png', '02818.png', '20047.png', '00862.png', '05332.png', '20502.png', '05677.png', '21922.png', '12579.png', '15559.png', '06348.png', '19556.png', '18482.png', '04677.png', '15900.png', '10160.png', '16757.png', '24706.png', '05747.png', '05565.png', '20791.png', '02824.png', '14139.png', '23849.png', '10238.png', '06799.png', '19774.png', '20756.png', '09076.png', '21371.png', '07000.png', '12494.png', '22440.png', '16344.png', '17038.png', '23073.png', '05218.png', '12151.png', '15770.png', '16855.png', '09568.png', '07244.png', '05508.png', '00575.png', '21392.png', '24480.png', '02929.png', '16698.png', '13936.png', '15661.png', '03439.png', '24874.png', '01997.png', '07539.png', '17111.png', '16375.png', '22634.png', '13432.png', '16969.png', '22048.png', '02497.png', '23092.png', '02264.png', '06714.png', '10182.png', '22783.png', '13044.png', '17006.png', '02162.png', '10907.png', '22111.png', '00381.png', '15620.png', '08453.png', '18076.png', '00585.png', '01381.png', '13182.png', '11840.png', '17410.png', '09705.png', '02838.png', '13211.png', '23108.png', '17445.png', '22138.png', '12779.png', '03978.png', '23520.png', '15946.png', '02059.png', '05595.png', '13712.png', '11995.png', '21945.png', '19495.png', '15494.png', '14547.png', '04942.png', '01792.png', '07755.png', '09566.png', '11707.png', '24433.png', '00899.png', '16277.png', '08938.png', '16676.png', '03385.png', '13413.png', '24527.png', '13375.png', '13245.png', '11307.png', '09796.png', '21593.png', '16741.png', '03963.png', '21353.png', '21768.png', '19403.png', '01975.png', '09932.png', '08620.png', '19947.png', '00294.png', '16352.png', '23749.png', '08771.png', '16647.png', '16995.png', '08531.png', '03608.png', '17362.png', '11041.png', '05175.png', '04294.png', '20643.png', '01957.png', '24164.png', '22421.png', '14429.png', '21453.png', '13906.png', '20975.png', '08223.png', '04067.png', '16242.png', '13271.png', '04692.png', '10661.png', '03911.png', '16236.png', '13967.png', '18931.png', '13796.png', '07418.png', '17491.png', '18315.png', '04426.png', '07510.png', '13761.png', '04762.png', '15935.png', '03974.png', '08407.png', '15941.png', '17473.png', '09769.png', '23773.png', '21742.png', '03055.png', '05975.png', '11130.png', '11931.png', '18997.png', '04255.png', '11833.png', '09054.png', '00428.png', '02750.png', '20444.png', '05543.png', '16824.png', '21022.png', '03113.png', '02921.png', '19252.png', '07928.png', '19070.png', '24714.png', '17463.png', '00195.png', '22531.png', '16279.png', '23289.png', '08162.png', '18145.png', '20398.png', '17801.png', '14750.png', '05982.png', '15193.png', '09115.png', '18799.png', '07482.png', '23562.png', '18588.png', '00468.png', '17306.png', '22638.png', '14497.png', '16230.png', '12467.png', '18406.png', '18698.png', '10464.png', '11266.png', '11933.png', '09577.png', '11848.png', '04492.png', '03555.png', '17787.png', '10552.png', '07683.png', '18957.png', '22773.png', '08579.png', '16952.png', '05840.png', '11283.png', '15371.png', '01424.png', '09041.png', '11270.png', '10916.png', '15149.png', '05919.png', '21179.png', '20466.png', '13507.png', '07386.png', '21757.png', '08962.png', '18459.png', '20576.png', '20331.png', '04193.png', '12975.png', '21286.png', '20571.png', '04287.png', '17509.png', '11825.png', '09461.png', '11539.png', '20208.png', '22528.png', '05112.png', '04222.png', '00250.png', '10350.png', '03064.png', '14763.png', '20790.png', '02058.png', '06235.png', '14632.png', '15211.png', '02860.png', '16109.png', '20022.png', '15186.png', '20788.png', '14713.png', '21129.png', '11905.png', '02720.png', '17740.png', '12181.png', '20547.png', '04609.png', '03882.png', '07668.png', '08438.png', '22685.png', '15899.png', '01970.png', '23993.png', '16388.png', '13915.png', '00574.png', '00203.png', '07296.png', '07788.png', '16675.png', '10674.png', '22688.png', '16067.png', '18801.png', '21725.png', '04502.png', '15319.png', '04449.png', '23855.png', '24649.png', '21996.png', '17876.png', '18628.png', '19343.png', '20556.png', '03543.png', '05671.png', '02425.png', '20899.png', '05700.png', '00506.png', '10525.png', '02131.png', '03344.png', '17407.png', '14410.png', '19853.png', '04860.png', '19964.png', '16679.png', '15758.png', '01938.png', '02841.png', '23249.png', '23558.png', '13092.png', '13485.png', '14982.png', '17324.png', '24071.png', '23216.png', '14958.png', '24333.png', '22799.png', '22620.png', '08808.png', '10955.png', '17178.png', '24753.png', '18025.png', '20277.png', '20096.png', '21763.png', '16718.png', '20972.png', '07722.png', '09091.png', '18054.png', '23803.png', '15116.png', '16340.png', '24112.png', '15773.png', '06409.png', '02030.png', '05805.png', '02280.png', '04679.png', '10320.png', '17430.png', '22565.png', '04824.png', '08774.png', '22952.png', '17975.png', '06082.png', '24636.png', '00902.png', '08091.png', '18936.png', '23665.png', '12108.png', '02693.png', '15931.png', '16409.png', '05561.png', '22250.png', '24851.png', '09346.png', '16418.png', '12455.png', '07706.png', '20517.png', '15646.png', '19714.png', '06315.png', '03743.png', '20705.png', '09497.png', '04501.png', '12878.png', '08887.png', '01421.png', '21948.png', '03075.png', '00824.png', '14091.png', '21232.png', '03564.png', '02492.png', '09026.png', '09337.png', '10636.png', '14032.png', '21093.png', '04852.png', '07778.png', '10714.png', '06468.png', '20368.png', '16621.png', '09668.png', '15255.png', '23328.png', '14490.png', '09587.png', '12105.png', '19432.png', '22718.png', '09768.png', '23926.png', '24197.png', '10869.png', '14877.png', '07438.png', '23278.png', '01710.png', '22095.png', '06347.png', '12682.png', '20235.png', '10504.png', '22193.png', '04851.png', '04802.png', '23347.png', '19062.png', '08618.png', '10904.png', '10943.png', '09721.png', '08034.png', '12153.png', '07236.png', '01442.png', '08255.png', '09748.png', '17274.png', '23325.png', '01552.png', '23458.png', '15483.png', '23379.png', '09499.png', '21812.png', '12696.png', '02161.png', '14172.png', '03037.png', '06154.png', '13809.png', '01640.png', '09492.png', '16206.png', '00525.png', '06965.png', '08678.png', '19364.png', '08426.png', '14458.png', '07218.png', '24152.png', '07153.png', '09524.png', '21276.png', '10545.png', '12187.png', '04866.png', '00785.png', '24077.png', '16613.png', '14582.png', '11348.png', '16046.png', '21610.png', '14222.png', '13415.png', '13540.png', '01538.png', '08079.png', '05958.png', '11336.png', '02797.png', '15710.png', '19938.png', '18865.png', '21664.png', '22156.png', '22490.png', '16800.png', '20053.png', '11363.png', '07934.png', '10434.png', '07301.png', '22255.png', '20780.png', '00108.png', '00111.png', '14115.png', '19120.png', '01356.png', '02565.png', '09325.png', '11462.png', '19382.png', '20730.png', '10514.png', '18275.png', '09722.png', '13705.png', '21783.png', '23532.png', '06690.png', '22613.png', '21767.png', '09146.png', '23639.png', '22247.png', '22344.png', '24512.png', '12695.png', '21568.png', '23482.png', '07606.png', '12487.png', '08173.png', '08267.png', '14107.png', '11466.png', '14789.png', '00231.png', '01240.png', '18449.png', '04184.png', '19878.png', '19129.png', '02889.png', '15307.png', '06041.png', '15791.png', '02912.png', '10970.png', '10665.png', '18481.png', '06615.png', '08284.png', '15247.png', '02257.png', '16185.png', '03417.png', '08538.png', '03678.png', '05420.png', '19416.png', '20760.png', '08934.png', '04362.png', '02518.png', '21066.png', '04060.png', '04996.png', '09022.png', '10779.png', '21750.png', '05105.png', '20527.png', '02078.png', '10401.png', '01203.png', '12106.png', '23854.png', '01774.png', '12150.png', '20499.png', '06998.png', '21135.png', '16216.png', '01084.png', '16851.png', '17088.png', '18522.png', '19723.png', '05503.png', '24799.png', '07899.png', '01166.png', '17363.png', '04474.png', '20565.png', '17264.png', '01189.png', '00436.png', '01047.png', '02395.png', '03082.png', '15203.png', '15232.png', '11143.png', '04253.png', '01007.png', '09618.png', '09629.png', '08479.png', '17150.png', '24900.png', '01665.png', '10145.png', '01694.png', '17794.png', '12722.png', '16888.png', '02851.png', '08427.png', '11207.png', '15472.png', '14676.png', '17068.png', '13617.png', '02199.png', '17400.png', '01921.png', '23609.png', '23268.png', '15979.png', '22947.png', '23315.png', '17119.png', '21335.png', '00219.png', '10054.png', '12773.png', '04057.png', '06794.png', '22083.png', '01460.png', '06996.png', '20806.png', '07708.png', '20017.png', '11815.png', '01121.png', '02144.png', '10272.png', '06949.png', '11796.png', '01860.png', '07538.png', '00446.png', '11608.png', '08855.png', '12488.png', '24697.png', '00136.png', '18503.png', '00881.png', '17659.png', '05190.png', '16353.png', '12449.png', '17112.png', '13649.png', '17255.png', '14921.png', '02307.png', '19028.png', '19392.png', '13371.png', '19180.png', '10354.png', '19554.png', '09720.png', '24036.png', '05138.png', '21965.png', '13713.png', '15943.png', '17933.png', '11169.png', '01572.png', '04526.png', '04107.png', '15870.png', '24911.png', '03728.png', '08434.png', '13407.png', '15793.png', '17347.png', '01340.png', '22280.png', '20548.png', '22865.png', '17496.png', '13032.png', '11179.png', '11689.png', '12432.png', '02211.png', '13961.png', '20154.png', '07660.png', '21451.png', '20385.png', '01810.png', '12186.png', '10937.png', '06022.png', '18198.png', '21963.png', '11868.png', '09507.png', '10220.png', '10732.png', '07980.png', '24788.png', '12775.png', '09565.png', '04583.png', '03202.png', '05762.png', '17844.png', '11907.png', '15386.png', '12192.png', '03972.png', '05314.png', '09197.png', '14892.png', '15777.png', '04423.png', '10692.png', '18226.png', '10130.png', '12076.png', '14019.png', '00469.png', '17529.png', '03429.png', '17751.png', '06358.png', '02110.png', '17231.png', '21663.png', '02616.png', '24212.png', '17471.png', '02233.png', '06518.png', '06829.png', '20364.png', '06890.png', '24025.png', '17234.png', '08397.png', '15964.png', '04311.png', '17505.png', '19750.png', '00564.png', '15614.png', '08606.png', '10170.png', '24254.png', '15345.png', '02153.png', '17303.png', '01741.png', '20955.png', '20973.png', '02923.png', '10411.png', '03069.png', '16090.png', '00414.png', '17901.png', '04991.png', '02133.png', '12787.png', '02295.png', '20216.png', '11331.png', '21579.png', '24269.png', '23207.png', '22515.png', '09116.png', '13681.png', '13635.png', '20507.png', '02353.png', '09280.png', '20862.png', '14001.png', '04591.png', '21168.png', '19783.png', '23120.png', '13278.png', '09744.png', '01871.png', '00018.png', '21879.png', '08959.png', '02761.png', '23537.png', '02027.png', '13602.png', '19401.png', '02334.png', '03195.png', '20701.png', '23895.png', '21231.png', '08164.png', '23058.png', '16885.png', '01480.png', '21520.png', '00779.png', '20419.png', '21094.png', '11081.png', '02648.png', '14929.png', '19303.png', '07842.png', '11403.png', '16616.png', '14813.png', '06077.png', '04340.png', '09767.png', '17711.png', '21923.png', '19981.png', '04188.png', '13704.png', '18289.png', '24118.png', '09561.png', '03339.png', '04821.png', '16346.png', '23064.png', '24373.png', '14818.png', '24407.png', '07500.png', '09207.png', '06314.png', '24628.png', '10895.png', '00280.png', '16696.png', '10717.png', '03117.png', '18718.png', '02050.png', '08578.png', '17095.png', '07148.png', '03467.png', '12549.png', '19909.png', '15582.png', '23998.png', '09597.png', '15334.png', '06287.png', '01584.png', '24086.png', '22403.png', '19820.png', '04199.png', '02043.png', '17256.png', '18717.png', '19184.png', '21235.png', '13524.png', '04216.png', '19176.png', '12439.png', '19362.png', '07895.png', '06430.png', '00411.png', '20090.png', '13204.png', '15845.png', '08781.png', '02372.png', '05667.png', '06791.png', '16059.png', '07713.png', '01664.png', '16494.png', '06964.png', '06050.png', '16399.png', '17206.png', '17870.png', '19561.png', '19123.png', '20338.png', '20381.png', '20877.png', '12884.png', '24490.png', '20869.png', '02755.png', '21530.png', '19283.png', '22810.png', '20076.png', '11004.png', '18456.png', '01459.png', '01629.png', '09079.png', '21425.png', '04574.png', '19002.png', '07050.png', '03176.png', '24072.png', '23331.png', '00356.png', '12054.png', '20912.png', '08905.png', '15380.png', '00434.png', '11359.png', '11373.png', '00368.png', '23535.png', '22521.png', '02135.png', '08736.png', '03139.png', '10838.png', '19207.png', '13004.png', '18802.png', '12969.png', '08709.png', '21870.png', '08592.png', '05300.png', '04805.png', '02056.png', '02136.png', '08328.png', '09624.png', '02367.png', '13080.png', '07720.png', '20261.png', '08651.png', '16181.png', '10563.png', '16551.png', '18319.png', '20454.png', '01232.png', '21002.png', '03596.png', '15258.png', '10352.png', '24739.png', '04204.png', '11070.png', '14847.png', '08121.png', '07499.png', '12994.png', '12368.png', '10479.png', '13948.png', '20000.png', '06341.png', '24700.png', '02775.png', '01931.png', '20110.png', '05738.png', '04733.png', '01213.png', '11787.png', '23792.png', '06248.png', '22573.png', '18210.png', '14542.png', '06532.png', '00417.png', '04390.png', '16982.png', '13389.png', '20377.png', '11502.png', '01034.png', '18256.png', '13764.png', '20272.png', '04807.png', '05206.png', '11755.png', '09276.png', '03757.png', '09289.png', '10348.png', '12169.png', '02612.png', '00781.png', '04763.png', '20075.png', '17077.png', '21493.png', '24006.png', '13377.png', '00606.png', '04141.png', '11719.png', '15320.png', '04442.png', '22136.png', '05012.png', '23290.png', '01670.png', '00626.png', '00378.png', '13295.png', '19507.png', '11765.png', '16703.png', '21318.png', '18773.png', '02173.png', '19992.png', '07208.png', '20132.png', '18770.png', '19097.png', '10364.png', '05338.png', '14579.png', '13127.png', '17545.png', '10175.png', '03369.png', '01281.png', '15682.png', '05833.png', '16709.png', '01964.png', '17981.png', '11496.png', '10449.png', '24601.png', '24235.png', '16170.png', '11741.png', '16197.png', '09040.png', '24665.png', '22706.png', '02094.png', '11763.png', '03929.png', '14239.png', '06281.png', '24205.png', '07176.png', '14554.png', '11702.png', '15147.png', '08181.png', '24922.png', '13316.png', '18058.png', '22310.png', '15181.png', '20097.png', '01898.png', '02804.png', '22316.png', '11742.png', '19449.png', '01012.png', '13697.png', '00326.png', '05114.png', '06200.png', '08660.png', '21619.png', '04611.png', '12607.png', '16542.png', '24198.png', '15243.png', '10173.png', '06739.png', '20960.png', '22042.png', '04044.png', '08035.png', '20146.png', '21824.png', '08558.png', '11223.png', '09980.png', '14785.png', '04025.png', '12014.png', '06493.png', '18083.png', '17133.png', '15350.png', '11643.png', '12139.png', '00969.png', '00948.png', '07229.png', '18415.png', '07963.png', '17525.png', '04555.png', '13797.png', '12915.png', '12850.png', '04817.png', '09237.png', '23358.png', '15303.png', '10476.png', '18330.png', '11546.png', '04625.png', '16724.png', '01561.png', '17972.png', '19977.png', '10403.png', '18488.png', '05763.png', '21342.png', '12951.png', '01951.png', '22731.png', '01110.png', '11189.png', '14915.png', '08583.png', '05797.png', '06627.png', '15136.png', '14047.png', '15402.png', '01178.png', '03752.png', '21302.png', '19423.png', '08646.png', '07688.png', '19072.png', '14023.png', '08218.png', '16087.png', '13077.png', '24859.png', '16747.png', '06096.png', '13154.png', '15585.png', '14203.png', '05965.png', '01183.png', '00301.png', '00653.png', '00318.png', '06289.png', '09896.png', '17120.png', '02839.png', '24534.png', '21405.png', '23198.png', '07637.png', '21364.png', '16292.png', '10981.png', '07600.png', '00105.png', '23370.png', '02553.png', '08889.png', '03024.png', '02475.png', '24558.png', '02177.png', '21737.png', '03034.png', '20317.png', '18950.png', '12222.png', '19937.png', '11538.png', '03140.png', '17944.png', '00395.png', '15545.png', '08365.png', '07976.png', '03250.png', '21298.png', '17415.png', '09654.png', '14611.png', '01343.png', '08724.png', '05435.png', '20230.png', '22861.png', '13089.png', '17880.png', '00782.png', '02927.png', '20294.png', '01999.png', '06587.png', '10210.png', '21667.png', '21848.png', '08130.png', '13639.png', '08039.png', '20441.png', '09313.png', '02089.png', '15695.png', '05017.png', '06531.png', '04182.png', '06625.png', '09933.png', '10611.png', '01960.png', '18266.png', '20937.png', '22954.png', '12931.png', '12217.png', '11435.png', '09600.png', '19744.png', '10286.png', '04132.png', '23099.png', '13907.png', '09886.png', '08856.png', '19287.png', '23596.png', '01550.png', '15835.png', '08828.png', '10513.png', '12110.png', '16829.png', '08307.png', '17633.png', '12416.png', '17782.png', '02178.png', '01250.png', '12832.png', '06981.png', '09667.png', '21784.png', '14855.png', '11330.png', '19839.png', '04250.png', '21160.png', '21825.png', '19658.png', '07039.png', '10917.png', '06088.png', '06777.png', '02397.png', '17855.png', '24596.png', '19373.png', '24910.png', '14517.png', '01978.png', '08858.png', '06238.png', '23610.png', '14232.png', '09171.png', '19169.png', '09113.png', '13619.png', '16945.png', '11668.png', '24010.png', '09238.png', '12492.png', '15404.png', '08449.png', '02780.png', '01676.png', '20924.png', '03089.png', '19517.png', '09371.png', '19848.png', '05907.png', '03981.png', '01932.png', '10635.png', '00371.png', '04643.png', '06779.png', '11197.png', '20271.png', '20357.png', '12984.png', '05098.png', '18190.png', '10807.png', '21395.png', '10725.png', '17629.png', '11957.png', '20019.png', '19726.png', '12873.png', '09995.png', '07453.png', '23724.png', '11098.png', '06806.png', '07110.png', '05890.png', '02291.png', '19011.png', '24414.png', '22078.png', '03695.png', '15715.png', '07853.png', '02323.png', '18285.png', '14179.png', '10626.png', '02961.png', '18658.png', '06850.png', '10542.png', '19641.png', '02740.png', '21614.png', '18465.png', '21441.png', '06508.png', '11410.png', '01543.png', '00141.png', '14906.png', '08989.png', '04262.png', '09860.png', '19602.png', '00408.png', '05382.png', '18362.png', '10731.png', '20735.png', '18608.png', '04090.png', '10151.png', '00150.png', '13000.png', '09589.png', '16489.png', '00031.png', '03145.png', '15246.png', '00221.png', '09315.png', '18908.png', '07043.png', '15237.png', '07192.png', '10375.png', '14780.png', '02138.png', '23766.png', '08239.png', '05601.png', '17163.png', '18243.png', '10659.png', '10206.png', '19322.png', '18832.png', '12842.png', '07498.png', '08755.png', '22449.png', '02933.png', '01730.png', '04168.png', '14105.png', '02168.png', '18761.png', '02484.png', '16683.png', '14754.png', '18929.png', '07556.png', '16734.png', '14951.png', '24057.png', '01723.png', '24879.png', '22467.png', '00016.png', '06547.png', '13573.png', '13748.png', '08489.png', '00078.png', '08192.png', '20998.png', '11212.png', '09885.png', '20106.png', '11397.png', '23762.png', '07268.png', '07313.png', '09520.png', '22177.png', '24953.png', '07105.png', '22529.png', '05228.png', '24515.png', '13489.png', '24954.png', '05837.png', '14835.png', '05884.png', '01776.png', '16624.png', '06762.png', '21547.png', '01115.png', '02776.png', '22864.png', '00796.png', '19594.png', '18045.png', '14744.png', '09009.png', '11490.png', '21242.png', '19280.png', '21798.png', '18385.png', '00342.png', '16406.png', '02529.png', '16132.png', '13014.png', '18095.png', '07724.png', '04781.png', '01708.png', '00806.png', '05377.png', '16625.png', '19604.png', '03994.png', '00959.png', '07481.png', '19214.png', '04167.png', '02035.png', '04938.png', '07590.png', '23966.png', '10686.png', '01268.png', '09871.png', '12482.png', '15923.png', '20112.png', '22973.png', '01777.png', '05691.png', '02844.png', '05368.png', '02473.png', '11351.png', '18342.png', '17619.png', '12751.png', '18519.png', '02069.png', '16102.png', '02028.png', '21506.png', '02071.png', '14491.png', '11773.png', '08345.png', '10061.png', '23817.png', '03816.png', '08214.png', '19523.png', '20191.png', '00732.png', '19929.png', '16395.png', '04514.png', '13462.png', '09438.png', '16312.png', '02371.png', '21035.png', '00192.png', '14415.png', '06404.png', '11574.png', '14072.png', '03814.png', '24742.png', '02485.png', '19093.png', '22287.png', '05751.png', '20915.png', '14361.png', '15463.png', '11947.png', '17757.png', '14092.png', '05252.png', '03691.png', '02306.png', '13186.png', '08263.png', '03950.png', '21771.png', '01412.png', '16599.png', '06131.png', '10489.png', '00933.png', '07086.png', '23071.png', '22919.png', '05846.png', '02788.png', '17810.png', '23877.png', '08196.png', '24684.png', '07820.png', '14248.png', '09287.png', '03673.png', '17024.png', '02339.png', '05222.png', '01770.png', '18024.png', '10601.png', '00487.png', '19538.png', '11532.png', '02318.png', '14918.png', '00687.png', '08302.png', '10886.png', '04640.png', '23881.png', '21977.png', '00596.png', '08837.png', '08031.png', '12678.png', '10360.png', '12359.png', '12768.png', '11768.png', '09365.png', '04560.png', '06452.png', '10616.png', '22994.png', '10134.png', '05498.png', '24762.png', '11936.png', '11394.png', '08597.png', '13775.png', '12055.png', '12545.png', '07357.png', '18894.png', '07094.png', '16001.png', '05256.png', '20690.png', '14971.png', '18079.png', '24931.png', '17174.png', '18841.png', '14595.png', '14187.png', '03534.png', '03706.png', '24855.png', '14650.png', '07209.png', '13356.png', '12898.png', '20926.png', '17299.png', '13197.png', '02779.png', '04440.png', '19520.png', '16503.png', '17390.png', '04660.png', '15260.png', '07037.png', '16374.png', '00561.png', '20725.png', '01055.png', '18532.png', '04746.png', '22125.png', '11419.png', '22182.png', '18487.png', '19650.png', '19751.png', '07905.png', '01200.png', '18026.png', '24358.png', '23239.png', '04787.png', '00340.png', '04794.png', '09562.png', '10950.png', '01801.png', '05078.png', '13273.png', '03162.png', '12987.png', '19857.png', '21566.png', '02913.png', '15598.png', '24026.png', '02469.png', '00254.png', '02957.png', '11454.png', '05122.png', '15050.png', '08325.png', '07016.png', '21490.png', '09193.png', '02868.png', '04934.png', '04707.png', '08000.png', '21688.png', '08471.png', '08144.png', '03755.png', '11747.png', '04197.png', '04799.png', '09907.png', '06044.png', '24959.png', '10282.png', '24287.png', '18627.png', '09842.png', '13628.png', '23125.png', '03299.png', '15792.png', '01404.png', '21621.png', '20845.png', '07009.png', '15356.png', '02621.png', '21696.png', '01894.png', '03801.png', '23529.png', '23784.png', '17982.png', '03305.png', '01161.png', '13986.png', '05909.png', '13357.png', '04181.png', '09415.png', '15192.png', '23260.png', '15847.png', '02880.png', '09494.png', '09643.png', '05679.png', '12059.png', '21656.png', '20838.png', '02576.png', '04930.png', '13209.png', '05579.png', '22949.png', '20379.png', '07892.png', '04906.png', '10222.png', '10784.png', '21782.png', '14943.png', '03861.png', '14380.png', '24623.png', '02193.png', '15269.png', '07008.png', '23679.png', '06127.png', '08701.png', '10490.png', '04800.png', '13158.png', '11194.png', '06411.png', '09498.png', '07501.png', '17132.png', '05350.png', '20406.png', '04007.png', '00179.png', '01017.png', '20873.png', '03447.png', '08562.png', '09893.png', '06973.png', '17349.png', '21942.png', '21082.png', '18355.png', '14987.png', '19427.png', '24919.png', '03992.png', '22538.png', '12348.png', '20974.png', '24555.png', '11053.png', '15141.png', '06134.png', '11269.png', '10258.png', '15209.png', '24518.png', '14318.png', '19711.png', '05091.png', '03802.png', '07870.png', '24841.png', '07306.png', '20961.png', '17957.png', '04843.png', '22570.png', '24246.png', '15849.png', '19216.png', '19730.png', '16220.png', '18853.png', '00745.png', '11948.png', '22394.png', '03163.png', '11885.png', '07371.png', '01020.png', '08060.png', '03958.png', '14678.png', '12152.png', '03985.png', '23378.png', '16834.png', '01212.png', '00072.png', '00416.png', '04789.png', '10051.png', '23155.png', '11129.png', '00473.png', '10758.png', '14981.png', '11277.png', '04203.png', '10593.png', '13792.png', '19047.png', '16371.png', '17290.png', '14775.png', '02464.png', '15721.png', '08644.png', '21657.png', '00809.png', '10416.png', '23925.png', '21863.png', '05116.png', '14752.png', '03557.png', '13005.png', '21138.png', '15536.png', '02047.png', '22970.png', '19609.png', '12617.png', '20270.png', '06803.png', '13621.png', '15566.png', '21068.png', '15734.png', '21495.png', '02090.png', '06400.png', '07685.png', '00348.png', '11926.png', '07851.png', '19460.png', '14278.png', '02608.png', '16015.png', '03431.png', '14077.png', '12436.png', '17109.png', '07222.png', '19177.png', '16598.png', '08948.png', '06164.png', '10515.png', '21840.png', '03428.png', '17990.png', '20856.png', '04635.png', '05311.png', '02428.png', '20880.png', '02876.png', '16918.png', '12162.png', '06782.png', '05902.png', '08542.png', '20706.png', '10581.png', '01658.png', '09263.png', '08147.png', '24584.png', '19976.png', '15533.png', '11937.png', '24053.png', '07698.png', '07420.png', '19151.png', '21865.png', '09388.png', '16642.png', '22301.png', '07854.png', '16743.png', '18394.png', '10632.png', '02996.png', '01621.png', '10897.png', '11235.png', '19994.png', '17031.png', '17712.png', '00264.png', '08327.png', '16239.png', '19235.png', '01672.png', '09751.png', '13252.png', '14900.png', '16899.png', '16528.png', '04432.png', '05932.png', '21902.png', '13553.png', '15912.png', '17254.png', '09840.png', '23547.png', '05388.png', '03008.png', '14697.png', '08839.png', '10047.png', '10979.png', '21880.png', '13436.png', '18219.png', '21074.png', '00888.png', '16706.png', '15572.png', '07646.png', '01132.png', '12493.png', '10197.png', '19919.png', '18584.png', '08107.png', '08094.png', '15784.png', '08208.png', '01330.png', '10673.png', '14578.png', '23599.png', '16192.png', '18697.png', '03919.png', '15476.png', '14390.png', '09895.png', '00412.png', '16481.png', '00327.png', '02690.png', '13134.png', '22271.png', '16765.png', '15809.png', '11224.png', '13400.png', '22192.png', '07202.png', '04093.png', '09408.png', '15156.png', '14593.png', '06704.png', '08666.png', '07297.png', '20256.png', '21356.png', '15235.png', '00282.png', '12452.png', '20069.png', '16303.png', '15558.png', '22938.png', '10984.png', '23752.png', '06130.png', '08559.png', '22999.png', '13048.png', '02521.png', '05293.png', '02547.png', '03560.png', '19021.png', '09921.png', '22968.png', '22037.png', '12918.png', '20171.png', '07727.png', '16884.png', '12135.png', '00457.png', '10472.png', '06877.png', '01273.png', '05184.png', '16566.png', '24119.png', '09610.png', '21308.png', '05282.png', '08379.png', '00759.png', '18912.png', '03090.png', '04908.png', '18316.png', '22435.png', '03105.png', '16005.png', '10252.png', '18422.png', '19050.png', '14461.png', '14251.png', '13155.png', '14636.png', '18064.png', '02582.png', '18863.png', '06047.png', '23395.png', '17318.png', '09212.png', '22004.png', '22572.png', '00019.png', '07588.png', '11888.png', '23026.png', '19742.png', '14708.png', '17736.png', '04645.png', '19238.png', '18707.png', '15338.png', '14976.png', '24033.png', '22610.png', '21733.png', '03742.png', '01102.png', '05948.png', '21438.png', '24224.png', '04256.png', '24132.png', '08813.png', '07952.png', '00220.png', '24626.png', '10264.png', '12148.png', '01120.png', '16082.png', '16351.png', '14357.png', '22671.png', '14583.png', '19390.png', '17985.png', '10982.png', '08611.png', '00290.png', '00955.png', '04511.png', '02422.png', '22442.png', '05160.png', '12263.png', '15373.png', '20993.png', '03819.png', '08383.png', '01699.png', '01259.png', '16422.png', '23057.png', '01074.png', '07376.png', '13009.png', '02958.png', '21749.png', '16210.png', '10860.png', '17969.png', '13002.png', '06744.png', '14017.png', '07081.png', '11115.png', '16615.png', '19220.png', '06600.png', '09019.png', '20796.png', '10158.png', '10395.png', '14712.png', '19078.png', '23145.png', '08096.png', '13789.png', '06962.png', '19762.png', '15289.png', '14291.png', '05771.png', '03705.png', '00383.png', '07334.png', '21689.png', '20321.png', '01689.png', '19166.png', '07825.png', '11748.png', '17760.png', '10920.png', '18333.png', '19274.png', '09968.png', '03341.png', '11452.png', '18902.png', '06245.png', '05589.png', '16907.png', '13406.png', '21026.png', '03529.png', '16003.png', '02645.png', '21997.png', '00897.png', '22466.png', '24618.png', '22941.png', '01793.png', '02809.png', '04936.png', '15886.png', '12414.png', '15182.png', '10502.png', '12730.png', '00257.png', '16514.png', '04939.png', '05014.png', '12210.png', '20767.png', '07978.png', '23607.png', '21638.png', '24566.png', '03178.png', '01771.png', '02903.png', '13998.png', '04927.png', '22090.png', '21936.png', '09482.png', '16977.png', '24275.png', '16441.png', '12531.png', '08937.png', '19945.png', '04376.png', '13499.png', '12732.png', '06033.png', '11171.png', '08373.png', '08441.png', '20751.png', '01093.png', '03182.png', '05281.png', '02037.png', '10857.png', '14529.png', '10084.png', '05253.png', '08588.png', '04231.png', '09828.png', '02895.png', '14099.png', '05162.png', '11493.png', '02980.png', '06987.png', '11843.png', '09465.png', '08268.png', '19562.png', '05937.png', '24349.png', '24631.png', '12090.png', '13015.png', '14048.png', '14866.png', '20340.png', '21652.png', '23352.png', '09266.png', '09586.png', '14718.png', '24124.png', '21561.png', '03883.png', '06177.png', '10034.png', '14874.png', '06879.png', '04548.png', '24255.png', '06512.png', '00838.png', '20179.png', '21797.png', '01261.png', '14313.png', '24177.png', '22835.png', '08603.png', '14660.png', '00811.png', '01830.png', '06249.png', '14480.png', '07280.png', '21835.png', '12248.png', '04581.png', '23711.png', '18687.png', '01053.png', '17571.png', '14749.png', '10766.png', '15581.png', '09144.png', '21220.png', '10683.png', '22460.png', '13664.png', '00433.png', '11983.png', '20749.png', '20895.png', '15179.png', '09538.png', '17671.png', '18262.png', '07861.png', '20748.png', '04460.png', '21121.png', '12069.png', '23958.png', '06638.png', '24084.png', '18118.png', '20057.png', '09027.png', '12371.png', '17520.png', '14264.png', '12844.png', '06933.png', '20240.png', '02000.png', '16122.png', '03756.png', '20201.png', '18838.png', '20651.png', '09950.png', '18239.png', '16272.png', '18472.png', '17649.png', '12676.png', '02901.png', '10053.png', '20342.png', '00840.png', '21317.png', '20999.png', '10115.png', '19738.png', '14402.png', '03367.png', '09459.png', '21694.png', '00963.png', '01842.png', '17467.png', '18524.png', '23118.png', '03366.png', '07542.png', '00088.png', '08187.png', '09092.png', '04667.png', '03177.png', '07463.png', '04593.png', '16135.png', '13960.png', '15998.png', '16712.png', '05075.png', '03166.png', '09500.png', '08804.png', '12461.png', '08425.png', '06976.png', '03836.png', '21228.png', '05506.png', '17075.png', '07966.png', '04042.png', '23173.png', '06731.png', '12221.png', '00347.png', '04731.png', '14241.png', '18192.png', '18093.png', '12599.png', '10892.png', '06567.png', '16828.png', '10242.png', '04110.png', '16421.png', '22759.png', '09864.png', '06882.png', '14113.png', '02255.png', '04580.png', '03624.png', '01060.png', '18113.png', '22402.png', '23505.png', '04694.png', '00778.png', '17339.png', '17009.png', '16764.png', '10968.png', '17570.png', '01274.png', '09582.png', '20138.png', '16739.png', '08166.png', '05378.png', '10378.png', '11278.png', '07445.png', '22489.png', '06663.png', '03223.png', '21690.png', '24798.png', '11069.png', '00539.png', '08670.png', '04275.png', '14110.png', '17105.png', '12700.png', '24169.png', '04010.png', '09848.png', '24495.png', '11112.png', '14495.png', '21237.png', '18599.png', '18057.png', '13023.png', '12610.png', '10587.png', '09010.png', '15005.png', '20848.png', '10045.png', '14673.png', '20269.png', '04842.png', '00311.png', '15622.png', '23829.png', '12242.png', '14969.png', '09392.png', '18826.png', '05335.png', '05545.png', '02229.png', '07889.png', '17672.png', '07038.png', '07614.png', '12051.png', '23542.png', '10303.png', '04401.png', '09125.png', '07807.png', '11195.png', '08344.png', '23901.png', '08259.png', '22225.png', '05718.png', '22705.png', '14851.png', '09236.png', '01781.png', '01697.png', '07695.png', '23305.png', '21774.png', '13626.png', '02990.png', '18474.png', '09016.png', '00133.png', '07004.png', '01929.png', '10137.png', '19212.png', '11831.png', '08129.png', '24229.png', '11140.png', '23456.png', '13603.png', '08062.png', '17501.png', '10080.png', '16856.png', '21537.png', '03957.png', '09223.png', '01505.png', '17958.png', '02749.png', '22834.png', '22594.png', '13659.png', '22114.png', '15769.png', '12588.png', '04780.png', '15421.png', '14127.png', '20215.png', '20204.png', '11575.png', '02597.png', '10337.png', '12665.png', '07316.png', '23174.png', '19944.png', '15130.png', '01435.png', '10507.png', '17967.png', '00370.png', '04872.png', '11208.png', '07785.png', '07974.png', '08984.png', '18074.png', '21938.png', '24852.png', '08146.png', '00419.png', '19277.png', '15938.png', '21910.png', '01348.png', '13121.png', '08972.png', '22787.png', '03225.png', '11106.png', '04928.png', '02202.png', '01204.png', '01271.png', '22992.png', '23756.png', '24232.png', '05983.png', '22356.png', '12285.png', '07839.png', '05037.png', '16860.png', '22422.png', '23298.png', '13267.png', '18825.png', '14822.png', '07392.png', '24884.png', '09988.png', '08866.png', '13933.png', '03585.png', '05235.png', '18675.png', '10642.png', '13344.png', '24561.png', '05412.png', '19816.png', '12497.png', '03931.png', '12309.png', '11193.png', '16486.png', '02095.png', '06229.png', '04527.png', '08236.png', '22551.png', '11020.png', '00065.png', '24378.png', '18148.png', '06148.png', '19491.png', '24836.png', '10016.png', '24180.png', '09023.png', '23797.png', '20483.png', '09056.png', '24434.png', '05526.png', '02963.png', '02205.png', '04189.png', '06597.png', '11594.png', '00087.png', '17265.png', '17010.png', '17821.png', '01138.png', '09771.png', '24834.png', '11023.png', '17158.png', '00791.png', '23393.png', '06578.png', '18876.png', '03930.png', '19686.png', '14064.png', '22829.png', '23034.png', '16380.png', '18759.png', '19077.png', '01236.png', '02669.png', '24869.png', '24432.png', '08681.png', '06546.png', '23457.png', '09706.png', '22951.png', '00149.png', '17213.png', '13839.png', '13050.png', '19459.png', '22149.png', '13660.png', '03012.png', '06994.png', '09420.png', '16606.png', '23697.png', '24105.png', '06004.png', '21039.png', '16509.png', '17293.png', '06902.png', '15142.png', '11932.png', '21615.png', '20682.png', '14061.png', '14394.png', '03528.png', '08318.png', '20656.png', '23072.png', '10862.png', '17667.png', '01307.png', '22236.png', '01502.png', '00747.png', '15165.png', '01127.png', '06045.png', '11239.png', '15986.png', '22814.png', '06786.png', '20426.png', '21586.png', '19756.png', '13899.png', '04965.png', '04019.png', '20886.png', '20871.png', '21236.png', '07486.png', '07703.png', '07643.png', '21115.png', '03156.png', '03921.png', '05721.png', '00053.png', '09790.png', '20309.png', '23465.png', '07557.png', '14413.png', '10427.png', '08794.png', '21005.png', '02733.png', '21480.png', '18898.png', '04826.png', '20992.png', '16413.png', '23767.png', '22619.png', '09839.png', '07347.png', '19441.png', '16403.png', '21310.png', '02354.png', '05391.png', '24565.png', '19764.png', '13370.png', '19031.png', '17934.png', '06302.png', '07867.png', '07946.png', '00990.png', '23981.png', '17914.png', '09614.png', '09772.png', '23423.png', '10246.png', '10547.png', '00754.png', '18168.png', '06536.png', '17247.png', '21692.png', '03496.png', '19521.png', '22939.png', '22071.png', '05964.png', '04895.png', '19735.png', '17065.png', '00621.png', '13808.png', '11231.png', '09356.png', '16586.png', '13192.png', '01370.png', '18397.png', '05066.png', '14704.png', '12161.png', '02118.png', '09087.png', '05658.png', '10216.png', '13634.png', '12489.png', '21685.png', '04032.png', '03106.png', '06066.png', '22802.png', '23713.png', '14857.png', '03923.png', '06572.png', '24571.png', '20421.png', '19526.png', '07568.png', '17114.png', '18552.png', '20371.png', '19736.png', '15480.png', '11527.png', '10233.png', '00321.png', '19226.png', '20219.png', '00627.png', '00307.png', '20611.png', '13181.png', '24080.png', '15602.png', '20681.png', '11296.png', '08170.png', '11500.png', '07410.png', '10265.png', '08788.png', '19454.png', '00009.png', '03668.png', '23525.png', '11416.png', '21713.png', '11691.png', '13213.png', '05273.png', '20482.png', '22188.png', '16071.png', '01654.png', '09996.png', '13290.png', '05437.png', '12109.png', '18670.png', '24167.png', '14045.png', '23193.png', '01023.png', '12376.png', '09164.png', '07019.png', '17679.png', '12726.png', '08782.png', '05304.png', '19276.png', '24151.png', '19649.png', '19127.png', '19834.png', '12354.png', '19244.png', '23124.png', '02629.png', '10747.png', '02836.png', '14021.png', '17521.png', '22259.png', '14962.png', '02877.png', '20568.png', '18571.png', '17920.png', '14040.png', '24826.png', '06303.png', '05274.png', '12771.png', '24170.png', '11600.png', '19597.png', '14425.png', '24221.png', '14506.png', '06061.png', '20079.png', '16360.png', '24526.png', '20779.png', '01545.png', '01843.png', '13288.png', '21816.png', '04862.png', '15688.png', '16898.png', '12041.png', '19347.png', '16567.png', '22209.png', '14791.png', '01107.png', '17676.png', '01029.png', '11091.png', '24034.png', '14931.png', '20329.png', '11323.png', '05285.png', '13963.png', '09463.png', '14787.png', '15968.png', '19999.png', '17805.png', '18302.png', '18808.png', '04435.png', '02381.png', '21187.png', '08650.png', '10881.png', '24471.png', '12507.png', '21636.png', '16808.png', '03025.png', '18539.png', '24409.png', '02691.png', '17685.png', '16850.png', '15370.png', '01114.png', '02180.png', '09250.png', '16756.png', '10087.png', '24201.png', '02107.png', '18822.png', '05608.png', '20589.png', '17258.png', '06921.png', '11264.png', '14905.png', '17268.png', '08234.png', '11759.png', '09268.png', '11039.png', '07144.png', '15757.png', '14950.png', '19089.png', '14609.png', '21122.png', '07509.png', '22362.png', '08018.png', '07421.png', '10430.png', '12945.png', '05665.png', '23896.png', '15171.png', '10389.png', '07065.png', '18069.png', '21727.png', '13171.png', '07303.png', '08237.png', '06374.png', '23554.png', '19806.png', '19625.png', '15751.png', '22517.png', '17057.png', '24330.png', '01032.png', '24129.png', '14747.png', '06226.png', '20942.png', '18725.png', '21106.png', '19659.png', '24002.png', '10226.png', '00794.png', '09984.png', '17429.png', '13493.png', '23976.png', '21263.png', '23201.png', '22904.png', '06749.png', '12632.png', '16529.png', '17568.png', '16768.png', '01119.png', '16016.png', '14687.png', '22838.png', '02308.png', '18655.png', '24299.png', '18234.png', '05010.png', '00656.png', '19739.png', '23566.png', '24035.png', '16782.png', '02792.png', '09231.png', '01909.png', '12957.png', '21830.png', '12305.png', '09673.png', '01405.png', '00140.png', '10716.png', '07106.png', '19324.png', '02878.png', '01737.png', '15677.png', '06208.png', '10481.png', '00769.png', '12709.png', '05910.png', '10818.png', '09107.png', '19251.png', '06440.png', '01569.png', '16695.png', '03184.png', '15415.png', '13897.png', '08740.png', '19606.png', '11829.png', '22624.png', '07904.png', '24829.png', '13029.png', '08274.png', '19525.png', '03851.png', '17399.png', '20288.png', '10999.png', '16227.png', '04962.png', '09899.png', '16358.png', '22015.png', '19713.png', '17427.png', '01628.png', '08445.png', '03722.png', '09739.png', '19198.png', '02408.png', '14426.png', '18272.png', '01946.png', '20988.png', '01546.png', '24159.png', '23810.png', '05524.png', '06484.png', '02810.png', '12164.png', '18777.png', '09937.png', '06153.png', '19422.png', '24787.png', '12191.png', '24862.png', '21326.png', '19465.png', '19295.png', '15831.png', '17614.png', '02292.png', '24428.png', '21961.png', '15375.png', '01904.png', '18187.png', '15236.png', '19342.png', '18204.png', '24099.png', '08625.png', '17085.png', '24961.png', '07583.png', '22459.png', '16175.png', '09359.png', '08749.png', '15244.png', '11170.png', '23921.png', '17458.png', '03483.png', '19765.png', '08920.png', '23021.png', '03938.png', '22655.png', '07177.png', '01254.png', '00339.png', '23954.png', '02419.png', '10691.png', '20608.png', '02692.png', '19539.png', '02184.png', '06812.png', '06974.png', '17358.png', '14076.png', '08733.png', '00849.png', '24806.png', '02977.png', '12765.png', '12177.png', '14732.png', '17449.png', '24696.png', '01157.png', '09573.png', '17885.png', '08417.png', '16114.png', '15511.png', '16471.png', '10328.png', '18255.png', '06078.png', '03218.png', '15015.png', '14261.png', '12113.png', '13575.png', '19781.png', '10888.png', '05544.png', '04621.png', '11862.png', '15750.png', '22932.png', '24300.png', '09102.png', '09535.png', '08155.png', '07878.png', '19680.png', '07691.png', '02785.png', '16142.png', '08229.png', '03693.png', '00863.png', '20458.png', '12125.png', '00587.png', '21869.png', '17187.png', '05941.png', '11032.png', '22908.png', '07836.png', '21512.png', '15032.png', '03655.png', '14121.png', '21111.png', '02567.png', '17110.png', '24346.png', '03085.png', '13411.png', '09621.png', '09216.png', '24189.png', '04129.png', '05264.png', '05063.png', '16099.png', '06759.png', '24779.png', '06886.png', '22210.png', '12566.png', '04418.png', '22200.png', '16770.png', '11055.png', '05286.png', '11786.png', '14570.png', '01481.png', '13001.png', '13017.png', '19940.png', '04213.png', '10095.png', '06884.png', '01576.png', '24372.png', '24747.png', '21139.png', '06873.png', '05501.png', '18917.png', '12759.png', '23491.png', '05627.png', '01234.png', '23716.png', '00081.png', '06584.png', '07355.png', '03138.png', '22808.png', '13343.png', '01787.png', '17643.png', '19480.png', '00730.png', '20916.png', '24637.png', '03685.png', '11071.png', '16209.png', '14088.png', '08613.png', '10153.png', '11286.png', '20115.png', '06501.png', '20065.png', '08387.png', '12714.png', '19452.png', '00331.png', '21222.png', '12029.png', '03766.png', '00568.png', '01742.png', '15518.png', '15419.png', '10944.png', '21868.png', '09029.png', '00052.png', '00831.png', '19437.png', '22453.png', '00910.png', '22426.png', '00735.png', '16545.png', '09483.png', '20266.png', '06837.png', '24305.png', '01391.png', '01850.png', '10703.png', '14903.png', '07876.png', '09444.png', '09785.png', '05668.png', '21103.png', '02914.png', '16048.png', '22106.png', '13450.png', '03936.png', '20593.png', '01049.png', '02400.png', '01591.png', '01551.png', '17908.png', '22326.png', '14532.png', '07877.png', '01210.png', '01237.png', '09443.png', '05302.png', '21085.png', '09488.png', '19737.png', '03822.png', '22606.png', '11159.png', '01385.png', '09440.png', '21050.png', '18525.png', '04047.png', '02186.png', '22796.png', '05633.png', '18621.png', '18871.png', '15887.png', '16325.png', '24854.png', '04260.png', '13978.png', '06497.png', '05033.png', '22599.png', '07624.png', '11366.png', '04796.png', '13913.png', '11364.png', '02756.png', '02007.png', '15854.png', '23684.png', '00080.png', '00369.png', '06770.png', '17073.png', '01288.png', '19407.png', '01794.png', '02986.png', '13310.png', '01953.png', '06963.png', '03392.png', '22274.png', '22321.png', '05644.png', '08341.png', '23254.png', '13758.png', '13583.png', '12476.png', '06827.png', '10858.png', '01509.png', '10625.png', '11464.png', '10010.png', '12196.png', '11746.png', '11375.png', '02641.png', '00739.png', '08838.png', '07558.png', '09487.png', '13040.png', '17357.png', '23620.png', '13020.png', '17853.png', '16626.png', '03416.png', '02722.png', '14631.png', '04917.png', '17875.png', '13055.png', '14381.png', '03336.png', '10559.png', '24609.png', '22933.png', '23095.png', '08410.png', '18300.png', '12544.png', '15685.png', '19551.png', '21822.png', '18810.png', '20487.png', '13678.png', '06158.png', '05484.png', '11876.png', '00738.png', '20901.png', '09364.png', '04483.png', '12045.png', '20839.png', '03051.png', '18819.png', '04922.png', '19271.png', '01537.png', '14762.png', '19219.png', '14563.png', '12066.png', '15110.png', '22659.png', '04654.png', '21112.png', '20900.png', '10496.png', '22376.png', '22142.png', '07219.png', '14699.png', '20646.png', '01396.png', '01555.png', '17312.png', '18220.png', '04557.png', '11133.png', '04037.png', '10974.png', '06736.png', '12636.png', '17275.png', '20015.png', '06737.png', '09034.png', '05069.png', '12081.png', '02032.png', '09075.png', '03377.png', '15901.png', '05517.png', '22955.png', '24331.png', '24710.png', '22960.png', '24945.png', '03207.png', '03617.png', '02727.png', '14665.png', '21435.png', '09170.png', '08405.png', '15021.png', '10188.png', '22793.png', '10278.png', '10126.png', '12980.png', '07846.png', '03014.png', '09209.png', '01789.png', '03129.png', '18809.png', '08120.png', '23579.png', '03159.png', '08194.png', '17845.png', '04505.png', '20982.png', '04130.png', '09657.png', '01262.png', '20172.png', '22548.png', '19116.png', '04664.png', '01432.png', '01490.png', '07593.png', '14974.png', '04053.png', '16459.png', '19588.png', '11492.png', '06086.png', '07569.png', '24418.png', '19091.png', '13153.png', '21063.png', '13471.png', '19678.png', '21089.png', '07135.png', '24270.png', '22589.png', '11110.png', '11619.png', '11478.png', '00289.png', '16905.png', '12271.png', '13647.png', '18987.png', '17776.png', '17989.png', '12974.png', '18489.png', '24864.png', '12625.png', '08598.png', '16886.png', '12997.png', '22101.png', '01971.png', '04728.png', '16257.png', '21693.png', '18798.png', '04126.png', '05689.png', '02139.png', '18031.png', '13616.png', '18254.png', '19926.png', '13196.png', '08182.png', '23152.png', '24723.png', '00180.png', '09910.png', '09982.png', '12744.png', '13164.png', '04174.png', '20890.png', '14140.png', '05861.png', '22525.png', '11066.png', '21030.png', '01149.png', '12104.png', '06643.png', '15569.png', '13042.png', '08435.png', '13920.png', '21786.png', '11530.png', '00143.png', '23080.png', '17565.png', '12413.png', '13662.png', '07286.png', '07669.png', '06250.png', '08024.png', '09101.png', '07337.png', '02637.png', '24223.png', '18537.png', '15918.png', '21204.png', '05239.png', '18278.png', '02813.png', '15879.png', '23410.png', '04105.png', '19688.png', '16092.png', '14932.png', '18274.png', '11025.png', '21819.png', '15075.png', '21935.png', '16867.png', '02486.png', '00685.png', '19804.png', '01553.png', '18222.png', '21424.png', '14375.png', '23451.png', '00106.png', '11049.png', '21295.png', '24720.png', '19696.png', '19354.png', '21458.png', '01164.png', '04540.png', '04363.png', '05729.png', '07881.png', '08575.png', '11917.png', '08465.png', '06420.png', '17432.png', '17139.png', '20774.png', '23910.png', '04488.png', '19633.png', '04831.png', '04629.png', '12015.png', '19656.png', '09375.png', '18301.png', '24751.png', '14648.png', '07492.png', '15525.png', '13828.png', '11135.png', '22926.png', '14144.png', '01162.png', '19645.png', '19268.png', '16026.png', '15548.png', '05225.png', '11622.png', '14876.png', '06311.png', '07602.png', '15112.png', '15484.png', '04959.png', '23259.png', '15983.png', '02867.png', '00580.png', '14656.png', '21839.png', '24795.png', '16432.png', '10789.png', '20236.png', '11315.png', '23441.png', '04318.png', '23983.png', '12200.png', '18283.png', '24376.png', '03423.png', '04175.png', '15893.png', '11107.png', '05352.png', '02332.png', '13742.png', '01650.png', '23930.png', '16347.png', '17023.png', '23126.png', '16248.png', '12122.png', '15729.png', '07066.png', '24286.png', '04988.png', '09730.png', '14807.png', '22666.png', '07203.png', '14445.png', '19590.png', '20134.png', '07795.png', '19681.png', '11305.png', '12403.png', '00848.png', '10898.png', '10104.png', '14337.png', '08084.png', '12650.png', '11845.png', '14217.png', '02649.png', '11174.png', '02086.png', '24964.png', '24761.png', '03038.png', '09200.png', '24966.png', '19583.png', '03100.png', '00317.png', '12244.png', '05167.png', '16141.png', '22567.png', '04054.png', '10060.png', '23211.png', '24389.png', '22067.png', '00475.png', '03446.png', '08157.png', '22676.png', '05499.png', '07967.png', '01657.png', '14174.png', '01117.png', '24957.png', '09889.png', '22625.png', '24388.png', '21224.png', '16199.png', '03806.png', '03313.png', '16533.png', '01926.png', '06526.png', '18880.png', '02216.png', '01223.png', '11483.png', '01822.png', '11365.png', '24210.png', '09655.png', '19033.png', '02212.png', '08834.png', '07212.png', '04948.png', '06340.png', '06715.png', '04320.png', '20505.png', '01219.png', '16587.png', '23694.png', '22650.png', '19221.png', '10891.png', '13842.png', '16639.png', '23592.png', '03952.png', '01754.png', '06622.png', '19061.png', '10783.png', '24111.png', '11737.png', '16496.png', '24061.png', '23892.png', '18622.png', '17861.png', '14705.png', '23167.png', '11923.png', '03265.png', '16687.png', '18760.png', '24450.png', '24944.png', '24116.png', '02569.png', '23550.png', '17129.png', '06958.png', '10174.png', '19883.png', '19614.png', '23312.png', '18796.png', '03680.png', '05353.png', '13446.png', '13122.png', '14000.png', '01667.png', '01617.png', '23691.png', '08500.png', '09270.png', '24792.png', '10088.png', '14256.png', '05547.png', '06042.png', '01377.png', '18135.png', '22473.png', '09185.png', '15326.png', '22354.png', '05488.png', '00164.png', '04517.png', '13228.png', '10670.png', '06228.png', '07789.png', '24149.png', '15284.png', '06364.png', '24193.png', '13457.png', '19743.png', '21080.png', '09312.png', '07362.png', '11690.png', '22375.png', '20761.png', '15311.png', '20678.png', '16168.png', '06137.png', '20744.png', '24381.png', '21747.png', '13556.png', '19006.png', '10183.png', '15683.png', '05133.png', '24702.png', '10922.png', '04724.png', '16821.png', '05809.png', '06463.png', '01644.png', '19410.png', '11187.png', '15592.png', '12136.png', '14667.png', '17439.png', '17481.png', '06999.png', '07898.png', '24650.png', '12597.png', '15906.png', '18188.png', '01991.png', '18657.png', '03135.png', '07254.png', '11260.png', '24126.png', '15690.png', '02155.png', '22744.png', '08679.png', '00363.png', '08277.png', '07940.png', '20054.png', '24102.png', '10543.png', '10628.png', '04063.png', '14008.png', '14386.png', '15616.png', '12653.png', '15121.png', '01181.png', '21691.png', '09396.png', '16120.png', '24424.png', '11356.png', '21203.png', '21559.png', '08390.png', '00094.png', '05142.png', '07537.png', '23244.png', '05879.png', '13930.png', '08015.png', '06436.png', '04951.png', '03217.png', '12338.png', '12182.png', '12380.png', '14128.png', '19985.png', '06858.png', '05272.png', '02240.png', '23666.png', '23912.png', '10842.png', '10315.png', '03466.png', '05758.png', '11564.png', '08798.png', '08688.png', '21149.png', '13403.png', '07710.png', '17999.png', '14706.png', '00245.png', '21037.png', '13888.png', '02237.png', '04278.png', '01682.png', '06267.png', '02552.png', '13683.png', '06728.png', '04337.png', '19618.png', '04612.png', '13799.png', '21422.png', '05527.png', '13589.png', '03308.png', '23453.png', '24422.png', '07304.png', '12731.png', '19542.png', '00193.png', '01744.png', '06095.png', '13673.png', '00854.png', '05992.png', '01371.png', '03542.png', '05845.png', '21406.png', '03890.png', '24593.png', '22670.png', '20660.png', '10709.png', '21440.png', '11889.png', '12583.png', '15425.png', '22450.png', '00677.png', '19903.png', '23591.png', '17721.png', '11123.png', '06198.png', '15615.png', '00482.png', '06717.png', '21274.png', '05571.png', '13078.png', '00398.png', '06893.png', '21345.png', '05502.png', '16431.png', '03400.png', '24265.png', '09591.png', '17902.png', '21360.png', '00401.png', '14118.png', '21381.png', '15531.png', '07414.png', '15903.png', '08206.png', '09572.png', '11068.png', '19387.png', '23209.png', '24183.png', '07555.png', '12570.png', '04578.png', '15205.png', '23392.png', '10586.png', '07844.png', '00917.png', '03612.png', '17564.png', '23486.png', '12647.png', '20908.png', '15632.png', '20373.png', '07433.png', '03712.png', '20897.png', '19689.png', '03788.png', '19358.png', '07432.png', '18237.png', '07956.png', '16904.png', '09278.png', '22911.png', '24059.png', '19710.png', '21858.png', '06457.png', '10009.png', '16539.png', '04002.png', '11198.png', '23220.png', '09255.png', '04747.png', '11712.png', '22019.png', '03748.png', '23900.png', '07195.png', '07641.png', '08514.png', '02471.png', '09883.png', '06392.png', '00229.png', '00837.png', '17034.png', '13329.png', '08498.png', '10876.png', '04148.png', '10990.png', '22234.png', '04984.png', '08663.png', '12938.png', '09167.png', '20948.png', '15990.png', '12353.png', '13672.png', '22464.png', '10275.png', '21186.png', '13835.png', '23838.png', '17922.png', '20716.png', '21153.png', '24701.png', '01815.png', '10363.png', '09096.png', '07068.png', '09837.png', '12783.png', '09894.png', '07302.png', '08913.png', '00499.png', '02821.png', '08125.png', '21101.png', '08083.png', '18955.png', '18480.png', '19758.png', '16069.png', '08655.png', '01185.png', '03628.png', '18247.png', '12091.png', '16544.png', '19114.png', '04194.png', '00169.png', '21352.png', '04714.png', '03732.png', '01483.png', '08751.png', '15504.png', '21049.png', '17341.png', '23017.png', '03845.png', '12943.png', '18689.png', '03571.png', '16118.png', '03473.png', '16106.png', '00243.png', '10267.png', '11127.png', '13144.png', '18437.png', '22264.png', '07226.png', '20911.png', '06380.png', '22483.png', '17128.png', '10585.png', '17348.png', '13974.png', '14824.png', '14135.png', '00989.png', '18443.png', '13390.png', '21683.png', '04309.png', '14615.png', '10772.png', '00695.png', '02636.png', '10123.png', '07366.png', '12406.png', '04904.png', '21336.png', '15871.png', '14169.png', '08633.png', '10168.png', '17461.png', '18205.png', '15384.png', '03328.png', '24621.png', '00006.png', '16960.png', '00603.png', '14277.png', '04371.png', '22672.png', '23891.png', '19101.png', '12480.png', '15187.png', '20339.png', '10987.png', '15210.png', '10355.png', '05683.png', '16881.png', '23826.png', '23425.png', '24797.png', '16773.png', '02522.png', '13342.png', '22244.png', '18703.png', '18273.png', '10480.png', '18916.png', '02044.png', '05390.png', '08330.png', '05875.png', '08879.png', '00842.png', '18564.png', '24298.png', '01266.png', '18793.png', '18128.png', '24867.png', '03031.png', '05294.png', '04853.png', '23676.png', '10322.png', '10812.png', '24291.png', '14978.png', '21989.png', '08439.png', '08017.png', '13857.png', '01962.png', '15795.png', '18110.png', '12392.png', '23992.png', '23036.png', '17915.png', '15052.png', '18391.png', '22027.png', '06498.png', '23504.png', '06459.png', '24415.png', '06654.png', '18505.png', '00651.png', '00485.png', '17566.png', '19522.png', '11476.png', '20164.png', '20380.png', '23775.png', '09215.png', '02234.png', '10967.png', '09350.png', '12249.png', '01477.png', '07780.png', '01155.png', '08917.png', '21625.png', '18426.png', '10435.png', '16537.png', '05021.png', '00341.png', '21730.png', '13862.png', '06209.png', '15226.png', '12508.png', '00472.png', '20704.png', '07943.png', '18948.png', '10854.png', '20365.png', '04529.png', '05101.png', '20550.png', '02704.png', '17923.png', '17717.png', '13841.png', '03944.png', '09956.png', '10076.png', '01768.png', '15111.png', '15760.png', '21888.png', '01243.png', '22477.png', '19044.png', '19934.png', '08081.png', '06712.png', '22922.png', '15292.png', '03783.png', '16056.png', '14083.png', '19892.png', '16080.png', '11986.png', '09656.png', '10239.png', '12344.png', '11621.png', '13170.png', '01969.png', '03194.png', '20278.png', '21780.png', '08674.png', '19836.png', '00698.png', '16978.png', '06628.png', '23923.png', '01510.png', '03810.png', '14854.png', '05694.png', '02564.png', '05639.png', '15731.png', '20008.png', '02811.png', '20310.png', '24070.png', '13183.png', '15748.png', '02888.png', '18139.png', '16474.png', '00228.png', '03892.png', '16264.png', '22901.png', '19298.png', '09665.png', '15440.png', '15627.png', '16027.png', '04299.png', '18098.png', '08051.png', '02008.png', '04916.png', '22017.png', '23342.png', '07595.png', '00645.png', '03977.png', '05989.png', '16742.png', '07997.png', '02113.png', '03414.png', '06336.png', '11121.png', '16699.png', '09854.png', '08945.png', '09639.png', '02247.png', '13945.png', '24208.png', '10657.png', '03226.png', '23262.png', '01646.png', '14588.png', '02081.png', '08528.png', '05984.png', '12397.png', '16245.png', '05046.png', '03315.png', '19153.png', '24020.png', '00937.png', '10726.png', '09905.png', '18781.png', '14145.png', '20670.png', '23819.png', '10469.png', '11176.png', '02737.png', '03550.png', '22730.png', '23624.png', '10610.png', '07754.png', '02590.png', '18485.png', '12442.png', '14808.png', '10884.png', '05442.png', '08884.png', '15934.png', '11720.png', '00403.png', '17903.png', '15459.png', '23572.png', '22514.png', '15278.png', '02403.png', '24493.png', '20284.png', '24143.png', '07832.png', '22360.png', '18207.png', '01518.png', '18375.png', '05915.png', '13613.png', '08996.png', '08253.png', '23227.png', '17107.png', '13787.png', '12590.png', '13440.png', '22524.png', '00855.png', '00883.png', '03419.png', '11499.png', '10063.png', '23369.png', '24162.png', '02822.png', '16208.png', '19881.png', '09918.png', '08556.png', '11357.png', '00814.png', '24494.png', '14421.png', '17583.png', '01090.png', '04237.png', '17862.png', '03310.png', '06294.png', '10908.png', '16063.png', '19639.png', '09844.png', '12002.png', '03588.png', '22698.png', '23580.png', '06853.png', '12399.png', '23809.png', '14420.png', '14994.png', '19719.png', '09052.png', '01500.png', '14778.png', '16491.png', '05398.png', '24263.png', '04579.png', '21818.png', '17704.png', '22950.png', '12940.png', '20979.png', '07479.png', '05405.png', '19317.png', '06990.png', '24128.png', '23004.png', '06984.png', '05522.png', '16526.png', '20128.png', '11246.png', '09509.png', '15647.png', '08295.png', '14639.png', '06155.png', '12662.png', '18052.png', '09874.png', '13598.png', '07955.png', '01721.png', '15503.png', '06605.png', '07625.png', '22726.png', '15340.png', '03018.png', '16720.png', '10101.png', '10301.png', '03115.png', '13268.png', '10648.png', '17668.png', '00870.png', '20711.png', '20066.png', '12321.png', '15882.png', '21034.png', '12479.png', '09201.png', '12776.png', '11439.png', '07012.png', '14475.png', '05201.png', '06274.png', '10424.png', '02299.png', '13116.png', '00410.png', '17336.png', '08177.png', '15016.png', '22018.png', '16988.png', '07285.png', '20488.png', '01256.png', '20267.png', '05918.png', '20709.png', '19486.png', '08054.png', '01532.png', '12077.png', '08281.png', '04178.png', '18830.png', '16125.png', '05005.png', '07493.png', '18899.png', '15135.png', '20102.png', '08699.png', '24818.png', '20786.png', '24503.png', '23386.png', '17617.png', '10086.png', '09162.png', '02342.png', '17961.png', '12049.png', '18515.png', '09622.png', '09274.png', '12688.png', '05092.png', '02950.png', '19157.png', '12477.png', '22549.png', '21952.png', '05141.png', '19194.png', '02019.png', '02683.png', '16617.png', '00238.png', '15804.png', '13401.png', '15337.png', '10748.png', '07437.png', '03850.png', '19024.png', '19655.png', '09851.png', '07490.png', '04931.png', '17540.png', '06446.png', '00667.png', '17654.png', '19139.png', '16688.png', '06832.png', '20620.png', '24022.png', '09822.png', '03790.png', '06117.png', '11470.png', '16159.png', '12806.png', '17198.png', '22749.png', '18384.png', '23041.png', '19571.png', '06461.png', '07560.png', '20509.png', '13172.png', '19142.png', '22578.png', '15656.png', '07247.png', '03484.png', '19282.png', '22146.png', '24334.png', '15689.png', '18983.png', '17186.png', '08371.png', '22580.png', '24544.png', '10863.png', '24278.png', '15095.png', '19264.png', '20306.png', '09817.png', '03224.png', '13491.png', '09252.png', '03713.png', '01715.png', '11248.png', '19304.png', '08159.png', '17028.png', '15702.png', '10933.png', '15385.png', '20511.png', '02024.png', '01341.png', '08737.png', '24745.png', '15692.png', '19469.png', '14325.png', '04043.png', '02706.png', '04508.png', '16025.png', '08690.png', '10470.png', '19476.png', '10743.png', '10801.png', '02377.png', '10031.png', '21950.png', '23082.png', '02140.png', '22417.png', '23991.png', '15395.png', '16334.png', '00344.png', '11253.png', '09307.png', '14285.png', '07667.png', '02951.png', '01075.png', '23436.png', '21279.png', '12606.png', '01158.png', '21715.png', '09329.png', '08926.png', '08532.png', '12917.png', '07848.png', '07172.png', '06059.png', '17887.png', '09966.png', '00024.png', '00104.png', '18072.png', '02498.png', '11101.png', '17067.png', '05316.png', '06458.png', '21785.png', '04417.png', '23377.png', '01634.png', '00614.png', '17259.png', '08748.png', '19066.png', '04359.png', '16309.png', '10727.png', '01345.png', '10630.png', '13988.png', '00773.png', '18250.png', '18077.png', '00291.png', '00927.png', '11777.png', '24160.png', '24230.png', '00184.png', '14263.png', '17472.png', '15271.png', '03188.png', '00226.png', '07382.png', '11912.png', '17189.png', '12724.png', '00185.png', '19393.png', '17191.png', '01359.png', '05582.png', '24168.png', '12233.png', '12694.png', '05212.png', '12767.png', '05125.png', '02754.png', '06856.png', '04308.png', '20276.png', '00367.png', '12179.png', '07823.png', '00464.png', '14049.png', '00421.png', '03644.png', '06594.png', '05039.png', '12594.png', '15985.png', '01724.png', '01095.png', '12277.png', '06381.png', '07644.png', '17199.png', '23389.png', '10880.png', '23538.png', '13848.png', '07937.png', '21678.png', '08979.png', '18153.png', '16339.png', '23916.png', '13034.png', '08636.png', '23565.png', '08805.png', '06322.png', '07161.png', '16882.png', '18900.png', '00519.png', '19666.png', '23086.png', '02073.png', '11335.png', '13925.png', '05265.png', '04268.png', '14262.png', '05838.png', '01965.png', '23640.png', '20233.png', '24009.png', '04352.png', '21334.png', '17790.png', '05022.png', '01734.png', '22029.png', '13270.png', '16478.png', '10816.png', '15922.png', '07234.png', '23074.png', '05570.png', '11988.png', '19189.png', '06543.png', '18229.png', '19705.png', '02463.png', '11586.png', '14282.png', '00288.png', '07374.png', '00894.png', '01276.png', '05935.png', '13159.png', '10027.png', '01891.png', '20952.png', '23475.png', '14093.png', '07511.png', '07824.png', '15281.png', '17323.png', '24279.png', '18562.png', '17137.png', '16499.png', '12428.png', '04443.png', '02769.png', '20892.png', '23375.png', '11624.png', '11572.png', '16697.png', '06702.png', '10964.png', '08534.png', '05240.png', '13771.png', '17474.png', '00810.png', '09160.png', '11456.png', '24274.png', '18851.png', '18087.png', '14441.png', '10426.png', '22024.png', '23481.png', '04341.png', '08638.png', '04518.png', '00062.png', '07448.png', '13790.png', '04389.png', '02748.png', '09084.png', '21461.png', '07804.png', '04586.png', '06450.png', '09674.png', '13036.png', '08612.png', '23006.png', '17747.png', '09516.png', '03393.png', '03840.png', '24827.png', '05985.png', '16361.png', '22805.png', '11240.png', '03086.png', '01091.png', '11820.png', '01716.png', '21161.png', '07205.png', '17523.png', '02917.png', '05052.png', '03784.png', '18267.png', '16313.png', '02594.png', '18490.png', '09859.png', '09749.png', '16997.png', '14728.png', '23611.png', '09711.png', '05916.png', '20645.png', '22221.png', '21760.png', '02143.png', '02932.png', '21382.png', '10694.png', '24557.png', '04358.png', '16702.png', '17117.png', '06747.png', '14840.png', '15341.png', '00394.png', '08029.png', '19231.png', '10696.png', '14147.png', '03485.png', '17942.png', '03849.png', '18765.png', '19610.png', '23272.png', '16766.png', '10064.png', '20687.png', '18977.png', '24109.png', '03418.png', '04811.png', '23884.png', '14393.png', '07480.png', '05126.png', '21366.png', '14177.png', '16083.png', '24817.png', '14643.png', '24051.png', '08555.png', '17726.png', '18252.png', '06566.png', '10580.png', '09940.png', '19248.png', '05905.png', '04839.png', '12552.png', '05100.png', '22974.png', '01134.png', '02250.png', '15328.png', '10672.png', '22199.png', '13572.png', '19328.png', '18690.png', '16393.png', '00801.png', '20957.png', '10180.png', '11232.png', '15131.png', '15060.png', '12939.png', '12517.png', '11969.png', '08304.png', '13354.png', '08103.png', '18328.png', '21722.png', '17998.png', '23876.png', '18941.png', '23043.png', '23473.png', '19703.png', '20259.png', '14082.png', '04313.png', '16420.png', '14764.png', '24926.png', '13653.png', '23859.png', '16021.png', '11303.png', '01529.png', '17919.png', '22859.png', '05074.png', '17188.png', '05466.png', '12061.png', '01961.png', '05756.png', '10571.png', '24840.png', '03102.png', '13777.png', '02384.png', '16207.png', '11997.png', '07310.png', '16306.png', '04776.png', '14868.png', '22781.png', '10202.png', '02194.png', '06002.png', '04822.png', '17165.png', '14784.png', '11237.png', '16532.png', '21086.png', '24925.png', '01530.png', '03363.png', '06516.png', '18636.png', '07520.png', '07291.png', '04745.png', '24451.png', '04186.png', '23115.png', '19754.png', '24594.png', '06533.png', '09917.png', '04179.png', '11902.png', '05770.png', '18458.png', '12115.png', '13987.png', '09404.png', '00930.png', '24268.png', '20028.png', '01000.png', '12193.png', '05351.png', '15071.png', '14348.png', '07045.png', '04810.png', '22913.png', '21241.png', '24658.png', '04837.png', '18969.png', '24652.png', '24017.png', '24448.png', '07649.png', '01153.png', '24894.png', '12472.png', '04248.png', '15190.png', '19053.png', '18992.png', '03827.png', '24813.png', '21720.png', '16796.png', '08910.png', '18990.png', '13379.png', '16948.png', '21123.png', '20921.png', '18312.png', '22770.png', '20353.png', '03179.png', '12791.png', '06295.png', '11566.png', '00215.png', '03820.png', '01118.png', '04459.png', '00640.png', '17842.png', '11886.png', '09861.png', '24759.png', '20246.png', '23831.png', '05030.png', '22645.png', '17966.png', '00700.png', '01495.png', '15022.png', '21658.png', '12445.png', '11149.png', '00022.png', '04683.png', '22751.png', '11432.png', '17945.png', '13095.png', '15657.png', '20677.png', '10902.png', '15010.png', '19810.png', '19337.png', '04979.png', '16569.png', '08767.png', '08871.png', '10116.png', '20433.png', '08978.png', '15874.png', '22882.png', '06159.png', '19760.png', '08680.png', '03003.png', '18108.png', '17224.png', '21489.png', '15139.png', '11913.png', '07739.png', '21565.png', '23189.png', '07450.png', '20663.png', '15774.png', '21278.png', '04624.png', '05384.png', '13690.png', '09490.png', '18806.png', '10899.png', '08138.png', '22716.png', '10962.png', '07488.png', '15377.png', '19952.png', '07790.png', '21773.png', '03019.png', '19528.png', '21834.png', '20431.png', '18081.png', '09774.png', '23625.png', '14150.png', '08190.png', '12481.png', '09254.png', '02937.png', '12154.png', '07900.png', '21176.png', '17665.png', '00833.png', '00032.png', '02304.png', '20347.png', '11989.png', '19623.png', '02029.png', '16470.png', '14142.png', '22115.png', '06162.png', '08954.png', '21400.png', '10396.png', '01681.png', '21634.png', '05180.png', '12166.png', '14946.png', '01196.png', '06903.png', '24323.png', '17345.png', '06544.png', '20558.png', '14284.png', '00737.png', '05008.png', '01410.png', '22296.png', '14403.png', '19054.png', '03547.png', '10921.png', '14508.png', '09043.png', '13726.png', '18901.png', '12769.png', '17774.png', '12879.png', '12035.png', '17557.png', '13721.png', '17177.png', '06282.png', '06262.png', '09478.png', '10796.png', '20684.png', '18224.png', '19778.png', '20718.png', '22903.png', '19511.png', '23232.png', '02203.png', '20522.png', '01242.png', '09585.png', '03976.png', '05927.png', '05580.png', '06951.png', '03989.png', '12362.png', '09536.png', '00979.png', '23302.png', '22319.png', '19505.png', '15571.png', '13308.png', '02383.png', '18928.png', '15742.png', '20563.png', '23383.png', '21877.png', '00947.png', '18111.png', '24445.png', '12638.png', '04823.png', '24816.png', '17553.png', '07167.png', '13931.png', '09514.png', '01524.png', '19117.png', '19753.png', '12798.png', '22260.png', '11256.png', '01612.png', '15837.png', '12573.png', '11850.png', '00350.png', '20552.png', '05792.png', '17772.png', '21024.png', '12140.png', '04610.png', '13884.png', '24749.png', '02816.png', '12341.png', '00720.png', '13133.png', '24236.png', '13865.png', '08358.png', '05653.png', '19134.png', '08027.png', '23861.png', '16187.png', '12723.png', '21197.png', '05917.png', '06388.png', '07872.png', '06474.png', '01832.png', '10952.png', '21707.png', '05361.png', '14269.png', '06132.png', '16171.png', '03087.png', '20596.png', '06593.png', '21990.png', '18840.png', '00932.png', '24121.png', '12440.png', '23613.png', '17935.png', '16645.png', '10000.png', '21675.png', '09959.png', '05280.png', '23889.png', '05152.png', '09257.png', '12044.png', '13294.png', '07653.png', '06373.png', '14317.png', '19928.png', '11860.png', '02705.png', '24510.png', '21846.png', '02460.png', '11836.png', '19379.png', '11381.png', '21299.png', '05413.png', '01865.png', '22263.png', '23527.png', '24686.png', '06711.png', '21704.png', '21226.png', '17892.png', '13871.png', '21629.png', '07283.png', '02219.png', '16941.png', '15318.png', '13904.png', '15799.png', '16555.png', '03551.png', '11059.png', '21754.png', '12142.png', '19939.png', '13733.png', '06093.png', '13533.png', '16368.png', '17615.png', '05793.png', '20429.png', '07084.png', '22981.png', '15822.png', '04662.png', '10003.png', '23937.png', '15408.png', '01097.png', '00034.png', '19471.png', '14674.png', '14483.png', '07561.png', '21058.png', '13431.png', '14259.png', '18972.png', '24382.png', '19557.png', '02114.png', '11573.png', '02214.png', '08116.png', '15959.png', '12011.png', '21052.png', '03677.png', '02971.png', '08950.png', '06871.png', '02568.png', '11832.png', '03381.png', '05400.png', '10036.png', '06439.png', '03398.png', '12458.png', '20634.png', '09233.png', '24464.png', '18209.png', '08932.png', '18157.png', '20083.png', '22583.png', '24293.png', '05043.png', '04310.png']
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
        
        if i_iter == 142751:
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
