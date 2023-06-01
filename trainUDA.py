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
    a = ['20164.png', '16276.png', '24062.png', '09747.png', '22787.png', '10094.png', '14967.png', '20011.png', '10154.png', '11651.png', '21587.png', '07809.png', '21508.png', '15825.png', '22321.png', '23740.png', '03772.png', '10012.png', '16391.png', '10702.png', '04423.png', '24049.png', '24913.png', '14576.png', '06680.png', '22648.png', '00519.png', '24358.png', '11584.png', '00881.png', '24379.png', '06136.png', '03260.png', '04515.png', '15521.png', '01953.png', '16054.png', '02118.png', '21091.png', '07610.png', '18633.png', '16379.png', '21124.png', '23075.png', '05913.png', '17648.png', '11574.png', '12978.png', '08686.png', '15559.png', '16310.png', '15767.png', '02643.png', '14461.png', '20610.png', '12073.png', '06213.png', '13784.png', '22082.png', '08921.png', '20946.png', '05035.png', '23231.png', '04538.png', '15624.png', '08363.png', '21266.png', '02890.png', '07889.png', '16247.png', '02172.png', '18326.png', '15243.png', '21723.png', '07445.png', '11573.png', '22684.png', '23444.png', '16413.png', '21224.png', '20048.png', '20080.png', '06362.png', '23084.png', '09174.png', '15615.png', '00781.png', '20917.png', '10914.png', '17728.png', '20056.png', '16542.png', '21003.png', '17367.png', '17597.png', '07722.png', '06848.png', '13242.png', '16117.png', '05448.png', '20861.png', '14695.png', '19721.png', '15535.png', '12980.png', '10884.png', '09759.png', '14924.png', '12903.png', '13036.png', '08841.png', '08136.png', '18652.png', '16869.png', '20122.png', '00257.png', '22238.png', '15360.png', '10866.png', '10770.png', '14056.png', '13940.png', '21459.png', '10844.png', '03582.png', '18394.png', '04329.png', '05906.png', '01783.png', '15212.png', '01982.png', '15190.png', '00716.png', '18508.png', '08183.png', '05724.png', '14974.png', '12361.png', '23241.png', '20681.png', '11283.png', '07696.png', '02612.png', '11826.png', '11117.png', '01405.png', '23862.png', '10999.png', '11836.png', '17309.png', '06965.png', '01178.png', '07950.png', '11890.png', '20542.png', '24101.png', '03619.png', '19742.png', '01613.png', '13474.png', '13544.png', '07257.png', '06036.png', '10989.png', '07699.png', '08884.png', '12823.png', '07222.png', '01212.png', '23300.png', '22186.png', '15523.png', '13910.png', '01415.png', '22458.png', '00044.png', '15126.png', '22382.png', '20331.png', '03500.png', '23095.png', '09821.png', '05190.png', '10804.png', '03671.png', '17908.png', '11885.png', '07367.png', '04574.png', '17058.png', '12211.png', '23776.png', '14431.png', '13836.png', '03867.png', '14031.png', '24485.png', '08657.png', '15754.png', '00277.png', '12405.png', '01911.png', '02901.png', '06995.png', '18018.png', '11320.png', '03208.png', '10470.png', '17383.png', '24064.png', '20496.png', '07724.png', '03600.png', '10313.png', '15452.png', '15749.png', '24856.png', '06120.png', '04964.png', '00195.png', '12974.png', '23818.png', '15595.png', '10918.png', '14800.png', '00003.png', '13453.png', '21424.png', '16472.png', '15092.png', '21935.png', '15807.png', '07824.png', '10695.png', '19582.png', '23463.png', '03418.png', '03042.png', '16393.png', '18562.png', '15011.png', '13095.png', '08173.png', '17340.png', '19146.png', '02604.png', '23751.png', '24900.png', '19587.png', '10097.png', '12456.png', '19492.png', '10118.png', '04534.png', '06947.png', '23443.png', '20736.png', '20349.png', '14953.png', '16073.png', '10297.png', '12292.png', '10745.png', '10420.png', '17542.png', '01459.png', '03830.png', '09262.png', '12221.png', '12500.png', '14215.png', '15411.png', '20471.png', '17642.png', '24755.png', '24297.png', '21937.png', '14973.png', '02293.png', '10909.png', '02553.png', '20541.png', '22512.png', '06722.png', '07657.png', '24192.png', '16654.png', '00509.png', '24350.png', '16602.png', '07011.png', '03196.png', '06726.png', '09685.png', '17747.png', '10160.png', '00595.png', '01799.png', '08720.png', '08409.png', '14736.png', '00290.png', '14237.png', '07349.png', '07497.png', '15900.png', '17697.png', '01879.png', '15198.png', '01133.png', '17373.png', '24467.png', '07198.png', '11662.png', '02923.png', '00812.png', '22106.png', '08369.png', '01931.png', '23972.png', '13195.png', '00698.png', '21447.png', '03253.png', '14007.png', '24136.png', '10810.png', '15215.png', '17827.png', '05870.png', '08713.png', '02803.png', '04874.png', '19298.png', '01955.png', '01229.png', '19430.png', '12199.png', '11674.png', '23049.png', '10566.png', '06282.png', '21493.png', '09106.png', '01247.png', '18333.png', '21148.png', '12177.png', '12990.png', '05819.png', '21928.png', '03322.png', '11844.png', '09073.png', '21563.png', '17484.png', '08238.png', '02891.png', '08327.png', '17476.png', '08779.png', '14710.png', '10579.png', '02938.png', '13381.png', '02412.png', '23267.png', '00497.png', '19268.png', '08095.png', '18683.png', '21369.png', '23295.png', '07823.png', '11014.png', '00212.png', '19719.png', '17111.png', '13774.png', '13739.png', '16079.png', '23625.png', '23512.png', '15672.png', '18312.png', '22378.png', '10132.png', '10758.png', '05999.png', '03827.png', '20828.png', '07066.png', '20293.png', '24879.png', '23891.png', '17049.png', '11305.png', '04287.png', '20608.png', '15719.png', '12587.png', '04182.png', '00333.png', '18646.png', '21526.png', '15724.png', '08328.png', '16142.png', '05421.png', '18151.png', '09364.png', '23503.png', '10239.png', '06369.png', '12997.png', '06964.png', '13354.png', '10006.png', '10050.png', '11987.png', '03579.png', '22062.png', '21231.png', '01777.png', '22443.png', '09647.png', '18452.png', '03511.png', '07386.png', '00305.png', '14334.png', '01416.png', '04277.png', '07754.png', '04597.png', '07015.png', '18839.png', '13076.png', '22367.png', '14156.png', '14244.png', '18532.png', '00717.png', '05619.png', '24830.png', '23996.png', '02594.png', '11598.png', '07268.png', '24447.png', '08212.png', '21400.png', '15306.png', '24598.png', '06116.png', '11639.png', '07209.png', '16515.png', '10930.png', '10627.png', '05961.png', '10302.png', '22560.png', '00600.png', '11127.png', '17550.png', '05173.png', '17155.png', '11744.png', '06363.png', '02949.png', '24483.png', '20835.png', '09716.png', '20808.png', '07285.png', '03390.png', '15669.png', '18181.png', '21854.png', '15864.png', '18244.png', '02964.png', '08049.png', '04084.png', '02270.png', '23893.png', '10996.png', '05233.png', '15752.png', '07044.png', '14091.png', '00196.png', '08250.png', '01866.png', '12590.png', '18602.png', '14814.png', '07108.png', '04103.png', '17517.png', '13014.png', '19915.png', '02772.png', '04549.png', '12071.png', '08852.png', '24675.png', '11898.png', '05174.png', '04867.png', '15763.png', '12565.png', '12800.png', '16643.png', '00080.png', '07822.png', '03431.png', '05917.png', '24130.png', '09878.png', '24054.png', '13162.png', '00943.png', '06780.png', '00976.png', '05677.png', '06488.png', '18379.png', '09471.png', '09312.png', '08189.png', '14262.png', '21654.png', '05492.png', '22457.png', '09814.png', '04222.png', '03963.png', '00164.png', '09531.png', '13671.png', '09922.png', '14268.png', '00347.png', '04395.png', '19053.png', '22118.png', '19572.png', '17606.png', '23732.png', '11973.png', '01049.png', '02957.png', '07174.png', '03286.png', '06597.png', '00715.png', '13763.png', '02451.png', '17582.png', '01549.png', '09619.png', '06630.png', '02308.png', '02063.png', '03490.png', '15851.png', '21593.png', '20084.png', '20369.png', '07566.png', '16242.png', '05115.png', '22988.png', '04066.png', '12714.png', '17999.png', '09898.png', '13255.png', '10281.png', '16777.png', '09828.png', '08001.png', '10971.png', '11909.png', '07825.png', '02056.png', '13991.png', '13272.png', '04512.png', '13499.png', '07317.png', '17019.png', '22989.png', '07483.png', '00535.png', '23863.png', '08384.png', '13690.png', '17396.png', '10675.png', '17750.png', '21834.png', '10711.png', '11352.png', '24770.png', '02110.png', '03375.png', '09778.png', '12163.png', '24248.png', '03291.png', '10314.png', '04791.png', '17140.png', '01780.png', '02247.png', '17944.png', '11814.png', '18418.png', '23519.png', '17575.png', '09320.png', '10453.png', '15242.png', '14511.png', '12912.png', '05519.png', '24238.png', '19436.png', '05676.png', '19478.png', '00896.png', '11269.png', '05437.png', '07561.png', '24395.png', '23281.png', '03222.png', '01021.png', '11956.png', '19178.png', '15698.png', '01618.png', '08443.png', '09220.png', '24429.png', '05369.png', '01211.png', '03837.png', '20622.png', '02524.png', '08872.png', '00515.png', '12668.png', '17068.png', '00143.png', '22500.png', '18967.png', '14326.png', '10968.png', '09763.png', '01395.png', '24180.png', '20135.png', '14979.png', '17769.png', '00726.png', '23962.png', '16439.png', '15808.png', '17062.png', '13964.png', '17346.png', '20428.png', '07614.png', '20109.png', '03982.png', '19962.png', '04473.png', '08558.png', '09937.png', '20360.png', '22142.png', '01170.png', '09590.png', '17814.png', '12542.png', '03289.png', '06331.png', '11123.png', '01414.png', '10986.png', '21258.png', '12703.png', '08246.png', '11407.png', '08762.png', '03974.png', '17873.png', '04387.png', '14964.png', '17128.png', '24066.png', '19032.png', '20210.png', '05150.png', '15622.png', '15930.png', '18223.png', '12882.png', '14909.png', '22911.png', '19596.png', '06093.png', '15475.png', '13058.png', '12956.png', '10886.png', '15612.png', '01216.png', '04907.png', '10368.png', '24518.png', '19577.png', '22251.png', '07122.png', '22777.png', '22026.png', '07725.png', '10596.png', '15583.png', '24220.png', '20395.png', '00572.png', '17379.png', '03157.png', '18268.png', '14359.png', '21216.png', '05364.png', '22498.png', '02648.png', '14488.png', '16903.png', '19233.png', '03284.png', '16492.png', '06107.png', '04887.png', '14386.png', '13261.png', '18720.png', '06656.png', '04569.png', '11092.png', '18259.png', '05098.png', '10738.png', '06858.png', '16892.png', '09299.png', '04717.png', '09920.png', '17537.png', '06543.png', '09967.png', '07149.png', '14444.png', '07513.png', '00031.png', '24965.png', '20381.png', '19294.png', '24090.png', '10553.png', '23577.png', '15373.png', '00135.png', '23318.png', '21703.png', '07945.png', '21340.png', '03719.png', '01703.png', '05363.png', '17943.png', '13101.png', '24063.png', '07295.png', '20762.png', '18882.png', '07652.png', '06283.png', '15689.png', '24085.png', '15324.png', '04883.png', '06830.png', '03020.png', '24594.png', '01708.png', '08286.png', '06420.png', '04559.png', '12055.png', '09357.png', '12552.png', '05464.png', '11188.png', '05976.png', '21547.png', '23276.png', '21134.png', '05645.png', '16332.png', '03551.png', '18841.png', '00360.png', '06612.png', '21838.png', '18295.png', '03895.png', '09157.png', '22112.png', '12091.png', '04797.png', '13716.png', '05842.png', '20353.png', '03859.png', '18958.png', '06876.png', '14914.png', '10287.png', '14320.png', '10979.png', '00223.png', '18556.png', '04139.png', '21362.png', '02640.png', '10284.png', '09180.png', '17335.png', '14823.png', '01672.png', '10624.png', '11435.png', '04965.png', '16899.png', '15015.png', '19397.png', '01521.png', '07062.png', '08317.png', '14045.png', '20950.png', '03607.png', '04377.png', '19099.png', '19325.png', '23812.png', '22699.png', '18108.png', '12068.png', '24326.png', '13012.png', '15089.png', '18497.png', '00550.png', '17503.png', '17608.png', '10512.png', '19218.png', '00227.png', '02703.png', '13127.png', '06130.png', '12494.png', '00677.png', '12034.png', '12850.png', '13966.png', '09955.png', '17611.png', '14965.png', '13582.png', '20933.png', '07670.png', '03617.png', '18505.png', '14002.png', '08252.png', '24476.png', '18355.png', '11052.png', '13567.png', '15266.png', '06629.png', '08393.png', '13023.png', '23846.png', '08153.png', '18344.png', '21670.png', '22067.png', '06390.png', '00915.png', '18301.png', '22959.png', '10303.png', '17255.png', '19383.png', '21954.png', '11361.png', '22357.png', '15157.png', '14274.png', '06837.png', '17047.png', '16655.png', '17970.png', '05289.png', '19813.png', '13653.png', '00554.png', '08498.png', '23147.png', '11967.png', '24012.png', '18998.png', '23909.png', '13250.png', '14373.png', '02530.png', '01217.png', '16360.png', '09280.png', '05170.png', '16579.png', '05317.png', '00754.png', '03897.png', '05231.png', '00954.png', '09286.png', '20034.png', '08707.png', '09570.png', '18362.png', '06031.png', '22303.png', '01084.png', '08412.png', '24240.png', '00712.png', '18246.png', '19090.png', '02324.png', '23364.png', '09785.png', '04683.png', '19994.png', '08121.png', '12178.png', '05822.png', '05306.png', '03805.png', '08590.png', '22447.png', '12333.png', '15663.png', '05536.png', '09126.png', '07358.png', '22664.png', '16588.png', '13164.png', '05237.png', '20448.png', '06415.png', '05809.png', '08832.png', '00919.png', '23879.png', '22584.png', '19951.png', '11473.png', '21892.png', '17901.png', '01384.png', '06365.png', '01169.png', '24507.png', '08756.png', '05909.png', '22641.png', '21478.png', '10294.png', '17504.png', '15984.png', '00373.png', '23680.png', '19348.png', '11803.png', '19787.png', '18331.png', '13037.png', '23979.png', '12502.png', '09219.png', '16496.png', '19304.png', '12488.png', '15410.png', '13103.png', '09107.png', '19329.png', '14239.png', '19676.png', '12674.png', '24156.png', '15726.png', '15911.png', '04952.png', '14765.png', '02548.png', '02137.png', '15687.png', '02587.png', '21145.png', '18171.png', '07584.png', '16107.png', '02711.png', '24924.png', '15250.png', '11718.png', '14313.png', '14301.png', '23819.png', '00416.png', '08301.png', '12623.png', '02399.png', '10796.png', '14808.png', '11128.png', '05063.png', '16568.png', '06118.png', '02626.png', '17969.png', '17421.png', '06040.png', '21261.png', '24778.png', '16444.png', '22038.png', '18588.png', '08944.png', '18408.png', '06370.png', '22545.png', '10471.png', '18734.png', '04677.png', '20139.png', '11773.png', '10025.png', '03777.png', '05493.png', '22929.png', '15746.png', '21016.png', '14199.png', '09063.png', '05471.png', '20171.png', '23975.png', '05264.png', '07361.png', '18769.png', '05569.png', '04373.png', '18323.png', '15286.png', '16226.png', '21634.png', '06690.png', '06623.png', '00778.png', '05953.png', '10693.png', '24554.png', '08027.png', '18335.png', '03373.png', '23345.png', '05995.png', '08295.png', '15936.png', '06843.png', '01232.png', '24787.png', '01309.png', '16886.png', '13355.png', '00944.png', '12375.png', '24803.png', '07055.png', '21264.png', '08792.png', '24474.png', '19853.png', '11252.png', '02838.png', '15273.png', '15393.png', '13274.png', '12403.png', '05860.png', '21397.png', '21525.png', '10651.png', '16868.png', '07506.png', '14760.png', '22692.png', '04814.png', '11478.png', '15511.png', '04567.png', '07528.png', '14847.png', '11279.png', '17435.png', '13818.png', '20204.png', '17870.png', '04368.png', '07534.png', '12665.png', '23980.png', '12574.png', '22505.png', '04042.png', '03040.png', '24215.png', '02817.png', '05754.png', '13546.png', '01303.png', '08496.png', '08584.png', '06025.png', '03469.png', '24519.png', '11650.png', '22424.png', '12127.png', '21986.png', '05695.png', '16452.png', '07351.png', '10605.png', '02113.png', '13744.png', '09134.png', '16036.png', '00791.png', '12165.png', '04180.png', '11933.png', '14224.png', '05601.png', '20834.png', '18403.png', '18957.png', '07893.png', '11739.png', '11514.png', '13434.png', '00950.png', '23071.png', '07042.png', '19302.png', '12288.png', '06633.png', '05630.png', '01870.png', '00747.png', '15666.png', '24473.png', '17086.png', '04562.png', '03517.png', '19916.png', '22363.png', '11774.png', '12461.png', '01897.png', '12586.png', '00613.png', '17293.png', '04391.png', '11445.png', '19759.png', '22877.png', '15636.png', '14535.png', '15466.png', '24175.png', '14377.png', '09259.png', '21000.png', '19052.png', '18558.png', '17693.png', '00771.png', '17868.png', '09250.png', '22319.png', '21871.png', '08889.png', '05702.png', '18761.png', '03085.png', '15601.png', '01601.png', '04017.png', '07546.png', '00467.png', '14868.png', '00801.png', '07309.png', '05713.png', '24013.png', '23215.png', '14194.png', '12052.png', '11026.png', '01696.png', '17425.png', '22991.png', '19025.png', '17308.png', '06152.png', '04161.png', '03451.png', '03730.png', '24551.png', '06514.png', '07830.png', '03976.png', '21401.png', '14883.png', '08576.png', '02946.png', '19273.png', '09582.png', '02407.png', '18995.png', '17381.png', '23326.png', '09994.png', '19910.png', '01596.png', '03829.png', '11941.png', '06411.png', '10980.png', '11173.png', '05562.png', '05102.png', '09761.png', '17360.png', '04680.png', '22355.png', '03991.png', '18230.png', '24892.png', '12719.png', '02130.png', '11029.png', '19378.png', '17087.png', '14347.png', '04319.png', '21407.png', '06570.png', '06410.png', '16145.png', '05542.png', '04962.png', '17402.png', '04610.png', '03946.png', '12647.png', '24817.png', '01080.png', '02520.png', '00268.png', '24791.png', '10261.png', '13519.png', '02598.png', '00939.png', '14022.png', '14016.png', '23735.png', '01124.png', '20561.png', '11588.png', '19120.png', '16338.png', '08309.png', '04581.png', '10173.png', '11545.png', '08870.png', '05965.png', '00057.png', '08618.png', '05637.png', '03565.png', '21717.png', '08289.png', '03459.png', '24586.png', '02468.png', '19632.png', '20143.png', '00757.png', '19856.png', '01453.png', '00478.png', '24023.png', '11629.png', '08776.png', '09429.png', '10371.png', '05296.png', '17710.png', '02781.png', '14723.png', '03510.png', '15923.png', '01222.png', '13266.png', '17032.png', '18063.png', '10120.png', '08727.png', '20953.png', '02685.png', '21549.png', '07007.png', '24607.png', '16158.png', '01085.png', '10856.png', '07162.png', '18953.png', '15664.png', '02455.png', '03819.png', '03255.png', '14258.png', '23358.png', '18011.png', '10954.png', '01857.png', '20660.png', '01265.png', '05918.png', '15531.png', '14525.png', '22822.png', '00667.png', '05073.png', '24015.png', '14159.png', '09940.png', '15694.png', '17654.png', '24723.png', '12730.png', '19239.png', '09484.png', '22710.png', '06772.png', '04207.png', '19869.png', '14030.png', '10066.png', '03381.png', '21506.png', '17314.png', '23126.png', '21989.png', '12425.png', '11523.png', '08896.png', '23480.png', '11658.png', '05413.png', '10128.png', '12625.png', '07254.png', '12313.png', '03949.png', '22115.png', '02303.png', '19516.png', '14739.png', '01833.png', '01254.png', '06188.png', '12249.png', '18265.png', '20199.png', '01088.png', '03019.png', '24033.png', '19145.png', '06112.png', '12784.png', '15364.png', '00885.png', '08606.png', '04571.png', '01462.png', '13929.png', '09859.png', '19468.png', '10828.png', '23073.png', '06381.png', '03628.png', '07745.png', '01651.png', '07473.png', '19650.png', '12588.png', '15826.png', '19969.png', '19227.png', '22413.png', '22851.png', '02622.png', '00073.png', '17652.png', '00066.png', '10196.png', '13619.png', '04723.png', '01505.png', '23844.png', '21782.png', '00818.png', '14403.png', '12884.png', '04932.png', '00650.png', '05140.png', '05442.png', '06277.png', '05162.png', '22829.png', '19313.png', '23783.png', '05251.png', '07960.png', '10987.png', '24293.png', '12548.png', '02419.png', '16060.png', '07340.png', '15056.png', '15848.png', '14028.png', '16792.png', '07738.png', '20500.png', '16825.png', '23019.png', '08541.png', '24575.png', '11578.png', '17096.png', '08750.png', '05180.png', '22134.png', '02771.png', '12031.png', '03090.png', '23959.png', '03457.png', '05315.png', '18378.png', '01062.png', '06654.png', '04679.png', '20274.png', '10923.png', '19653.png', '24651.png', '01238.png', '19750.png', '08455.png', '16244.png', '13970.png', '05167.png', '24595.png', '02482.png', '09466.png', '19584.png', '08186.png', '03801.png', '08385.png', '21057.png', '06460.png', '12732.png', '13438.png', '06914.png', '01734.png', '23882.png', '02294.png', '23913.png', '06715.png', '22766.png', '12371.png', '08617.png', '10501.png', '10459.png', '02299.png', '14350.png', '05052.png', '07177.png', '18925.png', '16898.png', '18576.png', '07518.png', '14264.png', '14499.png', '17773.png', '07380.png', '20887.png', '07662.png', '02033.png', '24233.png', '08643.png', '21072.png', '02842.png', '07943.png', '20155.png', '10574.png', '01120.png', '06615.png', '14114.png', '19979.png', '12157.png', '09256.png', '15252.png', '18608.png', '22793.png', '19643.png', '15613.png', '10762.png', '08279.png', '20563.png', '16116.png', '07408.png', '13206.png', '08152.png', '09797.png', '24866.png', '07085.png', '06545.png', '04096.png', '14602.png', '12024.png', '01625.png', '10463.png', '01739.png', '05656.png', '06800.png', '04702.png', '00461.png', '21945.png', '15405.png', '05931.png', '04289.png', '00398.png', '17737.png', '15046.png', '03655.png', '18848.png', '16269.png', '21845.png', '15931.png', '21435.png', '13098.png', '00075.png', '24152.png', '06023.png', '13209.png', '06427.png', '12019.png', '10211.png', '05444.png', '03725.png', '12640.png', '02974.png', '08634.png', '04437.png', '02248.png', '16845.png', '14766.png', '18469.png', '16572.png', '02840.png', '08169.png', '16726.png', '14742.png', '12967.png', '02988.png', '19413.png', '19387.png', '09535.png', '23196.png', '16159.png', '22643.png', '22536.png', '21615.png', '24172.png', '11075.png', '21104.png', '09757.png', '11685.png', '04720.png', '16021.png', '04362.png', '11231.png', '13925.png', '12397.png', '04531.png', '01873.png', '08116.png', '21735.png', '19129.png', '04273.png', '05717.png', '15747.png', '06086.png', '04686.png', '23973.png', '08652.png', '11764.png', '17927.png', '03744.png', '18809.png', '01044.png', '17853.png', '05984.png', '15197.png', '00021.png', '01900.png', '24843.png', '17355.png', '14826.png', '17195.png', '12563.png', '02543.png', '17121.png', '24182.png', '14043.png', '05777.png', '18545.png', '17337.png', '22495.png', '10216.png', '20872.png', '08682.png', '18860.png', '24783.png', '22203.png', '06486.png', '06043.png', '05764.png', '18876.png', '14553.png', '22426.png', '01945.png', '12484.png', '09239.png', '03694.png', '02417.png', '24271.png', '22320.png', '13800.png', '07080.png', '22868.png', '14251.png', '18681.png', '18959.png', '10623.png', '22511.png', '12697.png', '04170.png', '19770.png', '01924.png', '10146.png', '10965.png', '06315.png', '20359.png', '11535.png', '00732.png', '06241.png', '09993.png', '00441.png', '12787.png', '07035.png', '22509.png', '16459.png', '01242.png', '09210.png', '16722.png', '09607.png', '20290.png', '06822.png', '08571.png', '21906.png', '06312.png', '24543.png', '08368.png', '08690.png', '03501.png', '13236.png', '12538.png', '05480.png', '10279.png', '10661.png', '23279.png', '16979.png', '05670.png', '18550.png', '06902.png', '08695.png', '00857.png', '21333.png', '18867.png', '22009.png', '17387.png', '18942.png', '19276.png', '18537.png', '21226.png', '07138.png', '20258.png', '10789.png', '07190.png', '23559.png', '04295.png', '22566.png', '21820.png', '18818.png', '14802.png', '01471.png', '18166.png', '06943.png', '23490.png', '07353.png', '23649.png', '20485.png', '00198.png', '15727.png', '17328.png', '16902.png', '23796.png', '14562.png', '06174.png', '17415.png', '11777.png', '18094.png', '19739.png', '20874.png', '06908.png', '15078.png', '23585.png', '04949.png', '19995.png', '08782.png', '14083.png', '08110.png', '09568.png', '22182.png', '10615.png', '03190.png', '07167.png', '07703.png', '19453.png', '00310.png', '03241.png', '13614.png', '22267.png', '20020.png', '01347.png', '05151.png', '10538.png', '22896.png', '15788.png', '01029.png', '20724.png', '09187.png', '17897.png', '02945.png', '05399.png', '02116.png', '01480.png', '07187.png', '04598.png', '24828.png', '02290.png', '04544.png', '23854.png', '20593.png', '14563.png', '00897.png', '06245.png', '08494.png', '14240.png', '20169.png', '06105.png', '23589.png', '03640.png', '23461.png', '18870.png', '23852.png', '01747.png', '14925.png', '12376.png', '03996.png', '12604.png', '15239.png', '10717.png', '21064.png', '08278.png', '05494.png', '04876.png', '20862.png', '11790.png', '22387.png', '17287.png', '20855.png', '13801.png', '19455.png', '14618.png', '15163.png', '01419.png', '05693.png', '15230.png', '01206.png', '24592.png', '01680.png', '11270.png', '14812.png', '06376.png', '16037.png', '22416.png', '12025.png', '09588.png', '06865.png', '12677.png', '06866.png', '08406.png', '05138.png', '17077.png', '12184.png', '01506.png', '03163.png', '09454.png', '22655.png', '00050.png', '11389.png', '19556.png', '08216.png', '20396.png', '00556.png', '20900.png', '00603.png', '02601.png', '21779.png', '03349.png', '03355.png', '16285.png', '13244.png', '23643.png', '00106.png', '24050.png', '12769.png', '16214.png', '02580.png', '22441.png', '11087.png', '13696.png', '07873.png', '21077.png', '07909.png', '01840.png', '01560.png', '02238.png', '20301.png', '23104.png', '10867.png', '15334.png', '09012.png', '16126.png', '14797.png', '16844.png', '00743.png', '06332.png', '09502.png', '01059.png', '24210.png', '23054.png', '04735.png', '12993.png', '04737.png', '18547.png', '17261.png', '04181.png', '07736.png', '22151.png', '13196.png', '24760.png', '00884.png', '03158.png', '00114.png', '21530.png', '24409.png', '15643.png', '16474.png', '19022.png', '00351.png', '23889.png', '11521.png', '02910.png', '19257.png', '20379.png', '00126.png', '01943.png', '18966.png', '03623.png', '22557.png', '02514.png', '24140.png', '06716.png', '20487.png', '21991.png', '13028.png', '03629.png', '09913.png', '07564.png', '18552.png', '23614.png', '18748.png', '04189.png', '11596.png', '03370.png', '01371.png', '19433.png', '06889.png', '14317.png', '00334.png', '23699.png', '01000.png', '10515.png', '02128.png', '08500.png', '02147.png', '15261.png', '10638.png', '11684.png', '22348.png', '19571.png', '08326.png', '14982.png', '01767.png', '23207.png', '01119.png', '17002.png', '02852.png', '22256.png', '01594.png', '09885.png', '24138.png', '09247.png', '20147.png', '17352.png', '03374.png', '06160.png', '12401.png', '04142.png', '06033.png', '21183.png', '05950.png', '07477.png', '21228.png', '12051.png', '08428.png', '08902.png', '03324.png', '24336.png', '01364.png', '07360.png', '10110.png', '06565.png', '21318.png', '17743.png', '12352.png', '17172.png', '07142.png', '04906.png', '16477.png', '22349.png', '19223.png', '02425.png', '06192.png', '01859.png', '07485.png', '03017.png', '07434.png', '19773.png', '19680.png', '04931.png', '14136.png', '19532.png', '20976.png', '14026.png', '19461.png', '02814.png', '11643.png', '14165.png', '05849.png', '21853.png', '07357.png', '03541.png', '20820.png', '01848.png', '13235.png', '24596.png', '05916.png', '18791.png', '17893.png', '09643.png', '18678.png', '11197.png', '14820.png', '02448.png', '10002.png', '02035.png', '06556.png', '20697.png', '23958.png', '21594.png', '04945.png', '12022.png', '05086.png', '12011.png', '12557.png', '04296.png', '11544.png', '18541.png', '13199.png', '09237.png', '21162.png', '12780.png', '06248.png', '02762.png', '10268.png', '16296.png', '18650.png', '16141.png', '08863.png', '17034.png', '12584.png', '23263.png', '18033.png', '16796.png', '20203.png', '01668.png', '21151.png', '05563.png', '08043.png', '07049.png', '01570.png', '21880.png', '08171.png', '07412.png', '22801.png', '14866.png', '18062.png', '17158.png', '08100.png', '21110.png', '07019.png', '14200.png', '03516.png', '01257.png', '20645.png', '01527.png', '17417.png', '17584.png', '21156.png', '07714.png', '20242.png', '18465.png', '24262.png', '14324.png', '20893.png', '09775.png', '22890.png', '22628.png', '08163.png', '11679.png', '01475.png', '19173.png', '14878.png', '06026.png', '11738.png', '23910.png', '01282.png', '02486.png', '15991.png', '04253.png', '18621.png', '23238.png', '08637.png', '12004.png', '01423.png', '18030.png', '14079.png', '22228.png', '19380.png', '04241.png', '22287.png', '21771.png', '24468.png', '07693.png', '16344.png', '08257.png', '17804.png', '08767.png', '03248.png', '24535.png', '01562.png', '00445.png', '19880.png', '16599.png', '03961.png', '19080.png', '01156.png', '21672.png', '07641.png', '00255.png', '12201.png', '01644.png', '00986.png', '11775.png', '16446.png', '07723.png', '21404.png', '00880.png', '02912.png', '20571.png', '06408.png', '19712.png', '19153.png', '20838.png', '08042.png', '20205.png', '13779.png', '01324.png', '10824.png', '14085.png', '02732.png', '22135.png', '16574.png', '22039.png', '12448.png', '24148.png', '02509.png', '00163.png', '24478.png', '04098.png', '11813.png', '17860.png', '12881.png', '16013.png', '02863.png', '17615.png', '00926.png', '02075.png', '21857.png', '19243.png', '05163.png', '15339.png', '11192.png', '03234.png', '02802.png', '22723.png', '19638.png', '05460.png', '17700.png', '17558.png', '15600.png', '18910.png', '11711.png', '12438.png', '23718.png', '17033.png', '11057.png', '00190.png', '16165.png', '15852.png', '13845.png', '16757.png', '11974.png', '19372.png', '19828.png', '24390.png', '16202.png', '11001.png', '22501.png', '09804.png', '03839.png', '10015.png', '06014.png', '07718.png', '09294.png', '17551.png', '18567.png', '21329.png', '00110.png', '18421.png', '21370.png', '10379.png', '16059.png', '12808.png', '09272.png', '12664.png', '07214.png', '24005.png', '13386.png', '02693.png', '13591.png', '07252.png', '01148.png', '14893.png', '07859.png', '07421.png', '10318.png', '08423.png', '08572.png', '09630.png', '14809.png', '10771.png', '20749.png', '09987.png', '07543.png', '21870.png', '01436.png', '10998.png', '23277.png', '02660.png', '15800.png', '23043.png', '20121.png', '10573.png', '07448.png', '24432.png', '00653.png', '07536.png', '22879.png', '05435.png', '15479.png', '14529.png', '11296.png', '07493.png', '22386.png', '10906.png', '18212.png', '20112.png', '17903.png', '04000.png', '18425.png', '22611.png', '08274.png', '14035.png', '13137.png', '01730.png', '23116.png', '10899.png', '24354.png', '10895.png', '11635.png', '12107.png', '00324.png', '14566.png', '07601.png', '22838.png', '11601.png', '00284.png', '02823.png', '00514.png', '02040.png', '04346.png', '04624.png', '15451.png', '04355.png', '10642.png', '09190.png', '01796.png', '20382.png', '10692.png', '04166.png', '19626.png', '08945.png', '02499.png', '15111.png', '23606.png', '09027.png', '23664.png', '22329.png', '18843.png', '12781.png', '19050.png', '06206.png', '17182.png', '00414.png', '20873.png', '21040.png', '14517.png', '16993.png', '06588.png', '12366.png', '02257.png', '20137.png', '20817.png', '01834.png', '02590.png', '22836.png', '04336.png', '22947.png', '12813.png', '13626.png', '16738.png', '22704.png', '20142.png', '06734.png', '10382.png', '10741.png', '09687.png', '18032.png', '11339.png', '20271.png', '10410.png', '01837.png', '05910.png', '11893.png', '15211.png', '03129.png', '24190.png', '24426.png', '20982.png', '20777.png', '06881.png', '04816.png', '21095.png', '24393.png', '07743.png', '00682.png', '21256.png', '12898.png', '18423.png', '17618.png', '24221.png', '17625.png', '05130.png', '05590.png', '19827.png', '20569.png', '18123.png', '05346.png', '02156.png', '14885.png', '16890.png', '21178.png', '15847.png', '19271.png', '01614.png', '16092.png', '10289.png', '10616.png', '15863.png', '10772.png', '21158.png', '06643.png', '20796.png', '07577.png', '17718.png', '24127.png', '20737.png', '01430.png', '18644.png', '11255.png', '15259.png', '24893.png', '16419.png', '04681.png', '12134.png', '24187.png', '02186.png', '12479.png', '01698.png', '07914.png', '02533.png', '17218.png', '22552.png', '13931.png', '04613.png', '21466.png', '04234.png', '09517.png', '22055.png', '06318.png', '19318.png', '14708.png', '09601.png', '13430.png', '24753.png', '14615.png', '10835.png', '23945.png', '20510.png', '24026.png', '24258.png', '21622.png', '04803.png', '08697.png', '09164.png', '03959.png', '20072.png', '09260.png', '21704.png', '18755.png', '07145.png', '10405.png', '13288.png', '16798.png', '19691.png', '19832.png', '06502.png', '21302.png', '22465.png', '10136.png', '13559.png', '05674.png', '15974.png', '12859.png', '20890.png', '07409.png', '24157.png', '09525.png', '21908.png', '16028.png', '02922.png', '22145.png', '20596.png', '02965.png', '09808.png', '21521.png', '05763.png', '04699.png', '20195.png', '02443.png', '13246.png', '12136.png', '11842.png', '01526.png', '04626.png', '06797.png', '23507.png', '03476.png', '02951.png', '14498.png', '09322.png', '06760.png', '21348.png', '20097.png', '08538.png', '12477.png', '20747.png', '23802.png', '05837.png', '03475.png', '07221.png', '19641.png', '23820.png', '23492.png', '03105.png', '03679.png', '07631.png', '07368.png', '20239.png', '08656.png', '14097.png', '19630.png', '23737.png', '05857.png', '08980.png', '12238.png', '20162.png', '07710.png', '01692.png', '14613.png', '00746.png', '13892.png', '13852.png', '22195.png', '11079.png', '05968.png', '23290.png', '03249.png', '02421.png', '16270.png', '02268.png', '23258.png', '14109.png', '09495.png', '15419.png', '12750.png', '11266.png', '00590.png', '19597.png', '01326.png', '15184.png', '07827.png', '05012.png', '20489.png', '09684.png', '04947.png', '10387.png', '07912.png', '21657.png', '07486.png', '04386.png', '13795.png', '23174.png', '15433.png', '16770.png', '14520.png', '07712.png', '17949.png', '10175.png', '07052.png', '19609.png', '09441.png', '23227.png', '02834.png', '16849.png', '14591.png', '06287.png', '07776.png', '00899.png', '06526.png', '21271.png', '03641.png', '14772.png', '08661.png', '13287.png', '02555.png', '10995.png', '10549.png', '17562.png', '21428.png', '05126.png', '22716.png', '02022.png', '19215.png', '18043.png', '18424.png', '05506.png', '08687.png', '17187.png', '22781.png', '01173.png', '24957.png', '22042.png', '14542.png', '02557.png', '21582.png', '14656.png', '22679.png', '14226.png', '18687.png', '19902.png', '17691.png', '10671.png', '24696.png', '09233.png', '11301.png', '12977.png', '20818.png', '09082.png', '05622.png', '07865.png', '19231.png', '03205.png', '15919.png', '18159.png', '14110.png', '13776.png', '00730.png', '04147.png', '03133.png', '07322.png', '07492.png', '10340.png', '14368.png', '12751.png', '22782.png', '18260.png', '04080.png', '07365.png', '22196.png', '11019.png', '21872.png', '09151.png', '11063.png', '21181.png', '11284.png', '09146.png', '01052.png', '11287.png', '03472.png', '18707.png', '12968.png', '01140.png', '00399.png', '07707.png', '17069.png', '23689.png', '09273.png', '00400.png', '12518.png', '19602.png', '12875.png', '10502.png', '14410.png', '07683.png', '21278.png', '14501.png', '21781.png', '04059.png', '04767.png', '01200.png', '11793.png', '09196.png', '07527.png', '15633.png', '24123.png', '23632.png', '05886.png', '22100.png', '00702.png', '23327.png', '11618.png', '13574.png', '05958.png', '01632.png', '16837.png', '13842.png', '16235.png', '06462.png', '07772.png', '06319.png', '09101.png', '23866.png', '22635.png', '17580.png', '14132.png', '02274.png', '13525.png', '17974.png', '24680.png', '19358.png', '12101.png', '14619.png', '17116.png', '05633.png', '15700.png', '19801.png', '24445.png', '24608.png', '10740.png', '11828.png', '07760.png', '16201.png', '10214.png', '20234.png', '20336.png', '13870.png', '09692.png', '12149.png', '16140.png', '15149.png', '07201.png', '13366.png', '11285.png', '11921.png', '19027.png', '03021.png', '09621.png', '18850.png', '09739.png', '02064.png', '10348.png', '01038.png', '05034.png', '19169.png', '09537.png', '24556.png', '06794.png', '02769.png', '10485.png', '17586.png', '24137.png', '19332.png', '02195.png', '19262.png', '13988.png', '05985.png', '23813.png', '05923.png', '14101.png', '14554.png', '18377.png', '04976.png', '00909.png', '03919.png', '08540.png', '16365.png', '17911.png', '01681.png', '11993.png', '05419.png', '24067.png', '24284.png', '02899.png', '16243.png', '15488.png', '24349.png', '09166.png', '10981.png', '16397.png', '05729.png', '20916.png', '24870.png', '24480.png', '24186.png', '20536.png', '15477.png', '04999.png', '10030.png', '19167.png', '00335.png', '00562.png', '10948.png', '05434.png', '02633.png', '01726.png', '21924.png', '01711.png', '17407.png', '21353.png', '22573.png', '09301.png', '10938.png', '20727.png', '02512.png', '01351.png', '13083.png', '23676.png', '10877.png', '16952.png', '17229.png', '21291.png', '21734.png', '04740.png', '12793.png', '24106.png', '21313.png', '04335.png', '16964.png', '02181.png', '00742.png', '06258.png', '02469.png', '22430.png', '23028.png', '19604.png', '14971.png', '14599.png', '11479.png', '12558.png', '13396.png', '12653.png', '07241.png', '00868.png', '00430.png', '01319.png', '09403.png', '14434.png', '16842.png', '18000.png', '13954.png', '05812.png', '00350.png', '05046.png', '24208.png', '01408.png', '01850.png', '24941.png', '13017.png', '06564.png', '01113.png', '14532.png', '06109.png', '02651.png', '01050.png', '05966.png', '02969.png', '01342.png', '06020.png', '15142.png', '05287.png', '01790.png', '07553.png', '24627.png', '03097.png', '14036.png', '13365.png', '01822.png', '11771.png', '01115.png', '01981.png', '15237.png', '24764.png', '09050.png', '04121.png', '03080.png', '12908.png', '08264.png', '01131.png', '06647.png', '12343.png', '22912.png', '08477.png', '20219.png', '16544.png', '04003.png', '10113.png', '08053.png', '13689.png', '15027.png', '00193.png', '19342.png', '17397.png', '12657.png', '07397.png', '13011.png', '11616.png', '24891.png', '15083.png', '08610.png', '06825.png', '15786.png', '22815.png', '22510.png', '13092.png', '18788.png', '11018.png', '15327.png', '16350.png', '13105.png', '17098.png', '20425.png', '18462.png', '12962.png', '10912.png', '23317.png', '11379.png', '08360.png', '09714.png', '00545.png', '09500.png', '21008.png', '03136.png', '02968.png', '04782.png', '02807.png', '17010.png', '21576.png', '02741.png', '00326.png', '16888.png', '15738.png', '06089.png', '17984.png', '13177.png', '16646.png', '09555.png', '15413.png', '13692.png', '16636.png', '19252.png', '22575.png', '16094.png', '00177.png', '13557.png', '00661.png', '03130.png', '24038.png', '18842.png', '03038.png', '08058.png', '13974.png', '10212.png', '02967.png', '12766.png', '19960.png', '04921.png', '12526.png', '18499.png', '12841.png', '16002.png', '18725.png', '09845.png', '00971.png', '07976.png', '23298.png', '12736.png', '24548.png', '23268.png', '00245.png', '10085.png', '23407.png', '16546.png', '17440.png', '20384.png', '19997.png', '14663.png', '17215.png', '12244.png', '19636.png', '06090.png', '18348.png', '08184.png', '09229.png', '16499.png', '22091.png', '00184.png', '07576.png', '21757.png', '02883.png', '11604.png', '16699.png', '18128.png', '12958.png', '05244.png', '18067.png', '18322.png', '03084.png', '09178.png', '05427.png', '08847.png', '09773.png', '11408.png', '21863.png', '20413.png', '08396.png', '02919.png', '06156.png', '21030.png', '14570.png', '13888.png', '08293.png', '05187.png', '03152.png', '18986.png', '16791.png', '06097.png', '14349.png', '06798.png', '11012.png', '15392.png', '11794.png', '22957.png', '00123.png', '13329.png', '17995.png', '04929.png', '19822.png', '06879.png', '22719.png', '11765.png', '00208.png', '06875.png', '24332.png', '22696.png', '00192.png', '05573.png', '00494.png', '02403.png', '12805.png', '20555.png', '14441.png', '12003.png', '01426.png', '10802.png', '00841.png', '07749.png', '23630.png', '14984.png', '21667.png', '13372.png', '07762.png', '12459.png', '03587.png', '03645.png', '10000.png', '08764.png', '23406.png', '12243.png', '06095.png', '24231.png', '24936.png', '21058.png', '01999.png', '23457.png', '20227.png', '00869.png', '00548.png', '15839.png', '13268.png', '19588.png', '01836.png', '01967.png', '12110.png', '04286.png', '07796.png', '04563.png', '12133.png', '20868.png', '02728.png', '03978.png', '01973.png', '15024.png', '15026.png', '22549.png', '13605.png', '10764.png', '23542.png', '20042.png', '02614.png', '23121.png', '00327.png', '21518.png', '22167.png', '05152.png', '13021.png', '19486.png', '17021.png', '00498.png', '16138.png', '05816.png', '19488.png', '09083.png', '17132.png', '12953.png', '08931.png', '08410.png', '11471.png', '08987.png', '24676.png', '06642.png', '11189.png', '06836.png', '02140.png', '18449.png', '19483.png', '10220.png', '04284.png', '07874.png', '18202.png', '21756.png', '09831.png', '01725.png', '21577.png', '17109.png', '04215.png', '23949.png', '22459.png', '11454.png', '03773.png', '12247.png', '20408.png', '02970.png', '20776.png', '10233.png', '13264.png', '04804.png', '24842.png', '18038.png', '06320.png', '17286.png', '16948.png', '09595.png', '23462.png', '21209.png', '01285.png', '24081.png', '11289.png', '12225.png', '13536.png', '06606.png', '19192.png', '22626.png', '19976.png', '02414.png', '13359.png', '06412.png', '12407.png', '04618.png', '24261.png', '04392.png', '11894.png', '12559.png', '01937.png', '02331.png', '06735.png', '20418.png', '11144.png', '23383.png', '16883.png', '11849.png', '19020.png', '08093.png', '20715.png', '05328.png', '04442.png', '17351.png', '15386.png', '03815.png', '21732.png', '15287.png', '10165.png', '09912.png', '11390.png', '05160.png', '12824.png', '21099.png', '15177.png', '11597.png', '11009.png', '12960.png', '23598.png', '16342.png', '15957.png', '05490.png', '13703.png', '16790.png', '04759.png', '00934.png', '08936.png', '07499.png', '10928.png', '08239.png', '20960.png', '20423.png', '24411.png', '22821.png', '00750.png', '14792.png', '07608.png', '03960.png', '01540.png', '17799.png', '13466.png', '21208.png', '00344.png', '15519.png', '20482.png', '07664.png', '12312.png', '07476.png', '15322.png', '00502.png', '14214.png', '11949.png', '10435.png', '00522.png', '11465.png', '09995.png', '19188.png', '02672.png', '22402.png', '23506.png', '21046.png', '19629.png', '06256.png', '16156.png', '09057.png', '19275.png', '13389.png', '12753.png', '22210.png', '10686.png', '13955.png', '01154.png', '05422.png', '03850.png', '10518.png', '13078.png', '16058.png', '05731.png', '17933.png', '19873.png', '09341.png', '07810.png', '19284.png', '09002.png', '04445.png', '12374.png', '12399.png', '16370.png', '09990.png', '02066.png', '07407.png', '22950.png', '11516.png', '21948.png', '10069.png', '05529.png', '06662.png', '24287.png', '09764.png', '24896.png', '21831.png', '00758.png', '24649.png', '16382.png', '04085.png', '15434.png', '07545.png', '10729.png', '23533.png', '19919.png', '06968.png', '01793.png', '10985.png', '05867.png', '04673.png', '05226.png', '17460.png', '05508.png', '08909.png', '00001.png', '18741.png', '18279.png', '07288.png', '20512.png', '10833.png', '18521.png', '23944.png', '16275.png', '17722.png', '10475.png', '12103.png', '20186.png', '05212.png', '21694.png', '21679.png', '08233.png', '24108.png', '04110.png', '23273.png', '16609.png', '04925.png', '21573.png', '13380.png', '14356.png', '15748.png', '06550.png', '20324.png', '06359.png', '24725.png', '21334.png', '13601.png', '00877.png', '11007.png', '17097.png', '02461.png', '23961.png', '22464.png', '11083.png', '14538.png', '12633.png', '00821.png', '20221.png', '15372.png', '01318.png', '08928.png', '21026.png', '18672.png', '19512.png', '16359.png', '17456.png', '12797.png', '21932.png', '22567.png', '09360.png', '07118.png', '10421.png', '24950.png', '21032.png', '03445.png', '00905.png', '24605.png', '19793.png', '13145.png', '05790.png', '11217.png', '22632.png', '22250.png', '17006.png', '10907.png', '11178.png', '06501.png', '06009.png', '23558.png', '16679.png', '08207.png', '17273.png', '03650.png', '09877.png', '24923.png', '11872.png', '16873.png', '21338.png', '03037.png', '23193.png', '03321.png', '09028.png', '08253.png', '00842.png', '08726.png', '01485.png', '15201.png', '12303.png', '16056.png', '20505.png', '03135.png', '21814.png', '12078.png', '22588.png', '08387.png', '20823.png', '13616.png', '16564.png', '09074.png', '20077.png', '16265.png', '16559.png', '14900.png', '05504.png', '18248.png', '13812.png', '06540.png', '10024.png', '22939.png', '11035.png', '19320.png', '10973.png', '06689.png', '00611.png', '23582.png', '14286.png', '18292.png', '18990.png', '13915.png', '02742.png', '05532.png', '02178.png', '20307.png', '12067.png', '15108.png', '05088.png', '16064.png', '09468.png', '04064.png', '22054.png', '23704.png', '11653.png', '09467.png', '11005.png', '05331.png', '16212.png', '21464.png', '18142.png', '19010.png', '02882.png', '00369.png', '15922.png', '19238.png', '05243.png', '18200.png', '04483.png', '19114.png', '04521.png', '10680.png', '22713.png', '19519.png', '15538.png', '18288.png', '14076.png', '09382.png', '19480.png', '24793.png', '13158.png', '20819.png', '03320.png', '08254.png', '05461.png', '23985.png', '20328.png', '03697.png', '00638.png', '18084.png', '03792.png', '07728.png', '21794.png', '17676.png', '06710.png', '04642.png', '11926.png', '20132.png', '10081.png', '11101.png', '18743.png', '08751.png', '23146.png', '13617.png', '04847.png', '16383.png', '10286.png', '11646.png', '08291.png', '14198.png', '17764.png', '11905.png', '16251.png', '15446.png', '10170.png', '02297.png', '20460.png', '06580.png', '20641.png', '15545.png', '08210.png', '10437.png', '15495.png', '14354.png', '21697.png', '06431.png', '23369.png', '20741.png', '12294.png', '17389.png', '18617.png', '20163.png', '01648.png', '17844.png', '03116.png', '15766.png', '02101.png', '20572.png', '16680.png', '00275.png', '21308.png', '11150.png', '16695.png', '09623.png', '21847.png', '24707.png', '00886.png', '07283.png', '05778.png', '19095.png', '11634.png', '24273.png', '04494.png', '19473.png', '24733.png', '24522.png', '08365.png', '16484.png', '13223.png', '23581.png', '13461.png', '10941.png', '23665.png', '09847.png', '03811.png', '21741.png', '23474.png', '15869.png', '20935.png', '13831.png', '18146.png', '20441.png', '16118.png', '16596.png', '08795.png', '08801.png', '17026.png', '17980.png', '24931.png', '15058.png', '13693.png', '00968.png', '21715.png', '12420.png', '01083.png', '05799.png', '14338.png', '21714.png', '00739.png', '06240.png', '21436.png', '22806.png', '13283.png', '09745.png', '01815.png', '18471.png', '14990.png', '20421.png', '21940.png', '05144.png', '10263.png', '09005.png', '01186.png', '15820.png', '18553.png', '00471.png', '22098.png', '07229.png', '10929.png', '07903.png', '23448.png', '01443.png', '11250.png', '00240.png', '00945.png', '02531.png', '15703.png', '09102.png', '00843.png', '17863.png', '05862.png', '22468.png', '16351.png', '19671.png', '23694.png', '05145.png', '07671.png', '18485.png', '16585.png', '23302.png', '14773.png', '08636.png', '24178.png', '18679.png', '07842.png', '20182.png', '05408.png', '15441.png', '03179.png', '16624.png', '03441.png', '12334.png', '05811.png', '23157.png', '05245.png', '21373.png', '14966.png', '14292.png', '18917.png', '19279.png', '09676.png', '08040.png', '05977.png', '05395.png', '24319.png', '08068.png', '18251.png', '19971.png', '06717.png', '12422.png', '17178.png', '09608.png', '13686.png', '02165.png', '15055.png', '08321.png', '07253.png', '04846.png', '14624.png', '15117.png', '01019.png', '03914.png', '02749.png', '22127.png', '21802.png', '04004.png', '18877.png', '20786.png', '05650.png', '12891.png', '09880.png', '07195.png', '16589.png', '20623.png', '00854.png', '05820.png', '23003.png', '24590.png', '14768.png', '10893.png', '16237.png', '05424.png', '21888.png', '12373.png', '06200.png', '12645.png', '12350.png', '08124.png', '01995.png', '01702.png', '09636.png', '14049.png', '11862.png', '06919.png', '01335.png', '23736.png', '14963.png', '04762.png', '04705.png', '09573.png', '04186.png', '10271.png', '20033.png', '09185.png', '14154.png', '13520.png', '10246.png', '17152.png', '23825.png', '15846.png', '08361.png', '21023.png', '19048.png', '06487.png', '15001.png', '12785.png', '13070.png', '17297.png', '07299.png', '19529.png', '21160.png', '03489.png', '03917.png', '08865.png', '07040.png', '17978.png', '13554.png', '15867.png', '15140.png', '03589.png', '01374.png', '00936.png', '10103.png', '03800.png', '13297.png', '08616.png', '14055.png', '06944.png', '17315.png', '08804.png', '06219.png', '10312.png', '23634.png', '21205.png', '08341.png', '23721.png', '17060.png', '20503.png', '09463.png', '06688.png', '01550.png', '05447.png', '04753.png', '12989.png', '15792.png', '21628.png', '05925.png', '01536.png', '03427.png', '19085.png', '15337.png', '08149.png', '15803.png', '14573.png', '00421.png', '24652.png', '21776.png', '22286.png', '03221.png', '06569.png', '01344.png', '14325.png', '17774.png', '06835.png', '21232.png', '24315.png', '06243.png', '10039.png', '22122.png', '00923.png', '16067.png', '16684.png', '10795.png', '18083.png', '18227.png', '23739.png', '12347.png', '04074.png', '15758.png', '16831.png', '16605.png', '07442.png', '10174.png', '00751.png', '08669.png', '23409.png', '15247.png', '23291.png', '09175.png', '13620.png', '08712.png', '21929.png', '08296.png', '24815.png', '16284.png', '14999.png', '16003.png', '03702.png', '02474.png', '24345.png', '14053.png', '01763.png', '03894.png', '13537.png', '07807.png', '21912.png', '02298.png', '21306.png', '01225.png', '08349.png', '09622.png', '24499.png', '06874.png', '12798.png', '14008.png', '14975.png', '00673.png', '22181.png', '14032.png', '04599.png', '04330.png', '19575.png', '20303.png', '01939.png', '08075.png', '07220.png', '02327.png', '02870.png', '21532.png', '20692.png', '16861.png', '12213.png', '07028.png', '11737.png', '14433.png', '20422.png', '24069.png', '11638.png', '12402.png', '20750.png', '21710.png', '02996.png', '07806.png', '16639.png', '01445.png', '00622.png', '20095.png', '14202.png', '24912.png', '16390.png', '05877.png', '03544.png', '16273.png', '11209.png', '08148.png', '00380.png', '20673.png', '20577.png', '13513.png', '07648.png', '04156.png', '12620.png', '24713.png', '09817.png', '07008.png', '12700.png', '23772.png', '06263.png', '06886.png', '18122.png', '01253.png', '24792.png', '20913.png', '24294.png', '09091.png', '12906.png', '02578.png', '23252.png', '20507.png', '15619.png', '11936.png', '04960.png', '05784.png', '19950.png', '01795.png', '11347.png', '22987.png', '22857.png', '09268.png', '08277.png', '11360.png', '23858.png', '11428.png', '01476.png', '12282.png', '14151.png', '20722.png', '05826.png', '00925.png', '20430.png', '23063.png', '22480.png', '23132.png', '04021.png', '11482.png', '13218.png', '06936.png', '24253.png', '13303.png', '23061.png', '16210.png', '18583.png', '08809.png', '08195.png', '02208.png', '20733.png', '23098.png', '17994.png', '08785.png', '19266.png', '15912.png', '24469.png', '23836.png', '18907.png', '19803.png', '18758.png', '17089.png', '07415.png', '09181.png', '18776.png', '06027.png', '17345.png', '20076.png', '09226.png', '03412.png', '01135.png', '05462.png', '23435.png', '21213.png', '08906.png', '00202.png', '18523.png', '05605.png', '22111.png', '01055.png', '00203.png', '03139.png', '05982.png', '19697.png', '22850.png', '10014.png', '09221.png', '06430.png', '23223.png', '19870.png', '00722.png', '24321.png', '18535.png', '10016.png', '19054.png', '02623.png', '18226.png', '07488.png', '03397.png', '23667.png', '16085.png', '20021.png', '06948.png', '07675.png', '04710.png', '06561.png', '05628.png', '02768.png', '12749.png', '14516.png', '15101.png', '19982.png', '17643.png', '12229.png', '11211.png', '09810.png', '13295.png', '08408.png', '01732.png', '01920.png', '00081.png', '10377.png', '18330.png', '19879.png', '18140.png', '04866.png', '05518.png', '11626.png', '00989.png', '09116.png', '07438.png', '22677.png', '19166.png', '15529.png', '21793.png', '07871.png', '24292.png', '17296.png', '14917.png', '09098.png', '06652.png', '06631.png', '14333.png', '05739.png', '18086.png', '20527.png', '14821.png', '04508.png', '17469.png', '09077.png', '03407.png', '16516.png', '22598.png', '00298.png', '19579.png', '19800.png', '06110.png', '01831.png', '11583.png', '12654.png', '11184.png', '11082.png', '00659.png', '19043.png', '03937.png', '12501.png', '17115.png', '20415.png', '06255.png', '02132.png', '06077.png', '17590.png', '01761.png', '06978.png', '24364.png', '23422.png', '20278.png', '11621.png', '11059.png', '12196.png', '13088.png', '09095.png', '21309.png', '05556.png', '19672.png', '19530.png', '07626.png', '22264.png', '12564.png', '23595.png', '20166.png', '20192.png', '23763.png', '18972.png', '15146.png', '04946.png', '08554.png', '11043.png', '11971.png', '07227.png', '21282.png', '12533.png', '20191.png', '06941.png', '13046.png', '20549.png', '20701.png', '14352.png', '02259.png', '20081.png', '09939.png', '03699.png', '01736.png', '02071.png', '03060.png', '05477.png', '10905.png', '17055.png', '14006.png', '01508.png', '16366.png', '10075.png', '21636.png', '15019.png', '23144.png', '05293.png', '00684.png', '05800.png', '09613.png', '03546.png', '24555.png', '14997.png', '15210.png', '14661.png', '00483.png', '08850.png', '09504.png', '17568.png', '05818.png', '09358.png', '20973.png', '08308.png', '20552.png', '23287.png', '10388.png', '15675.png', '12075.png', '11543.png', '20344.png', '22189.png', '07327.png', '20399.png', '21034.png', '01960.png', '06291.png', '12062.png', '18480.png', '08471.png', '00606.png', '18388.png', '03966.png', '11726.png', '08483.png', '20706.png', '09774.png', '22290.png', '12779.png', '08592.png', '03611.png', '19250.png', '00251.png', '05358.png', '15468.png', '01785.png', '08886.png', '00584.png', '16841.png', '09244.png', '16014.png', '09979.png', '12379.png', '09309.png', '20308.png', '19536.png', '18461.png', '22218.png', '08808.png', '04498.png', '18021.png', '23696.png', '11258.png', '06696.png', '15420.png', '03460.png', '05771.png', '22773.png', '21157.png', '04061.png', '23767.png', '06445.png', '18274.png', '11994.png', '22379.png', '18768.png', '22214.png', '13198.png', '07256.png', '11022.png', '00904.png', '07334.png', '06108.png', '19917.png', '16912.png', '24016.png', '01026.png', '05400.png', '20543.png', '01695.png', '22226.png', '04041.png', '18375.png', '15673.png', '00500.png', '00972.png', '23615.png', '19351.png', '13699.png', '20663.png', '04402.png', '02311.png', '22259.png', '04608.png', '21981.png', '20879.png', '12212.png', '07232.png', '14804.png', '01072.png', '23110.png', '08903.png', '00991.png', '11359.png', '23501.png', '06001.png', '14769.png', '05988.png', '06778.png', '20294.png', '02550.png', '03580.png', '15458.png', '19996.png', '17452.png', '05623.png', '03742.png', '09371.png', '23808.png', '09064.png', '01740.png', '19235.png', '20902.png', '07991.png', '18695.png', '05618.png', '02743.png', '16664.png', '18222.png', '01884.png', '08214.png', '19354.png', '16612.png', '14171.png', '01639.png', '04075.png', '18149.png', '11307.png', '07248.png', '16693.png', '16510.png', '20928.png', '23703.png', '10316.png', '01369.png', '05730.png', '01977.png', '05446.png', '13150.png', '07244.png', '18673.png', '21243.png', '22556.png', '12314.png', '17082.png', '07892.png', '18207.png', '17937.png', '11282.png', '17371.png', '24257.png', '03035.png', '06744.png', '05893.png', '05377.png', '24964.png', '11860.png', '24132.png', '07023.png', '08010.png', '08445.png', '22057.png', '08952.png', '24276.png', '10781.png', '03191.png', '23941.png', '18578.png', '07396.png', '24762.png', '09600.png', '13868.png', '19944.png', '00281.png', '04148.png', '10581.png', '01745.png', '21828.png', '22790.png', '02462.png', '06659.png', '19847.png', '24414.png', '02583.png', '20252.png', '05686.png', '06010.png', '23400.png', '09206.png', '15450.png', '05659.png', '10801.png', '00358.png', '10080.png', '03448.png', '11226.png', '21139.png', '04298.png', '19804.png', '17350.png', '02031.png', '01682.png', '03183.png', '13930.png', '18964.png', '03012.png', '01230.png', '04853.png', '14872.png', '14168.png', '15256.png', '16795.png', '22434.png', '20698.png', '21375.png', '24470.png', '16317.png', '00903.png', '09683.png', '13593.png', '10819.png', '20113.png', '24677.png', '08742.png', '07056.png', '04105.png', '23405.png', '08143.png', '07899.png', '11917.png', '04973.png', '04760.png', '01092.png', '11815.png', '12098.png', '18885.png', '16441.png', '13790.png', '01576.png', '00686.png', '10735.png', '15910.png', '21905.png', '14681.png', '24517.png', '23622.png', '13654.png', '15303.png', '16741.png', '19835.png', '06958.png', '10133.png', '07594.png', '21300.png', '18498.png', '24443.png', '05455.png', '16812.png', '15344.png', '22414.png', '02345.png', '03537.png', '16506.png', '10123.png', '21150.png', '04201.png', '08342.png', '12016.png', '16832.png', '04566.png', '09768.png', '22073.png', '06940.png', '00117.png', '01899.png', '00238.png', '11896.png', '16762.png', '22444.png', '03557.png', '05112.png', '06144.png', '04224.png', '21763.png', '22701.png', '23082.png', '10498.png', '00790.png', '20746.png', '13707.png', '08011.png', '12626.png', '07623.png', '11476.png', '13314.png', '06336.png', '03746.png', '07247.png', '09026.png', '21612.png', '17225.png', '09675.png', '02096.png', '21540.png', '02575.png', '10082.png', '07282.png', '11721.png', '12913.png', '23343.png', '20291.png', '15625.png', '02830.png', '11170.png', '03059.png', '06894.png', '20731.png', '13174.png', '02519.png', '07575.png', '24291.png', '03265.png', '19012.png', '00889.png', '06051.png', '18297.png', '16272.png', '19352.png', '17671.png', '07455.png', '02990.png', '08873.png', '09943.png', '19061.png', '04452.png', '14289.png', '15877.png', '20257.png', '21021.png', '01504.png', '23936.png', '22041.png', '19477.png', '20798.png', '10327.png', '24351.png', '11268.png', '05903.png', '03050.png', '22362.png', '14686.png', '09462.png', '18934.png', '15980.png', '18490.png', '01281.png', '08333.png', '11858.png', '12659.png', '14705.png', '22638.png', '20983.png', '08031.png', '13502.png', '20914.png', '14588.png', '00690.png', '18236.png', '02906.png', '14712.png', '15842.png', '13284.png', '17455.png', '04920.png', '05256.png', '10723.png', '13294.png', '06864.png', '03802.png', '17237.png', '10719.png', '03616.png', '07844.png', '05203.png', '15095.png', '22995.png', '15050.png', '06768.png', '15685.png', '16280.png', '24640.png', '15711.png', '07097.png', '15582.png', '13440.png', '11115.png', '11452.png', '07416.png', '13652.png', '16703.png', '24876.png', '12187.png', '04465.png', '17084.png', '20854.png', '03482.png', '14288.png', '15791.png', '22576.png', '03521.png', '02678.png', '11948.png', '03578.png', '20576.png', '11397.png', '09414.png', '04510.png', '12801.png', '06708.png', '13292.png', '01393.png', '18978.png', '14880.png', '23401.png', '09823.png', '12995.png', '06831.png', '05943.png', '21323.png', '05389.png', '18036.png', '04890.png', '08735.png', '20371.png', '16575.png', '16361.png', '16410.png', '16209.png', '00130.png', '04554.png', '17610.png', '07716.png', '16482.png', '01811.png', '23610.png', '01400.png', '02747.png', '16737.png', '19108.png', '05726.png', '04541.png', '13639.png', '08609.png', '00308.png', '21788.png', '10854.png', '02084.png', '16019.png', '00378.png', '22396.png', '20398.png', '23469.png', '15093.png', '13368.png', '17738.png', '19901.png', '02755.png', '20759.png', '05989.png', '09330.png', '14954.png', '21255.png', '02034.png', '04843.png', '24502.png', '10650.png', '00583.png', '10667.png', '15546.png', '18161.png', '03776.png', '00552.png', '10483.png', '23576.png', '10218.png', '12460.png', '14086.png', '22921.png', '04865.png', '03485.png', '09023.png', '01807.png', '02963.png', '02793.png', '03070.png', '00189.png', '00995.png', '19547.png', '18328.png', '01693.png', '11498.png', '16987.png', '10428.png', '14629.png', '20225.png', '21597.png', '19057.png', '07224.png', '13731.png', '16047.png', '20826.png', '03077.png', '02446.png', '11204.png', '00894.png', '08336.png', '12175.png', '06425.png', '04623.png', '19147.png', '01756.png', '19920.png', '24718.png', '19747.png', '23657.png', '18905.png', '08659.png', '23442.png', '19248.png', '01909.png', '02320.png', '16984.png', '20735.png', '08766.png', '03416.png', '24475.png', '10845.png', '19682.png', '07456.png', '18316.png', '23065.png', '10249.png', '13358.png', '10076.png', '01022.png', '22636.png', '23536.png', '13291.png', '20597.png', '16186.png', '18409.png', '06088.png', '04805.png', '07292.png', '01871.png', '00826.png', '09786.png', '19118.png', '15395.png', '07882.png', '16918.png', '16150.png', '13340.png', '13512.png', '09718.png', '14684.png', '01441.png', '14575.png', '14827.png', '24247.png', '09658.png', '11072.png', '12986.png', '01678.png', '03145.png', '01279.png', '19424.png', '22761.png', '00288.png', '20635.png', '06070.png', '11437.png', '17241.png', '24330.png', '11039.png', '09827.png', '20069.png', '10191.png', '06491.png', '04304.png', '06309.png', '07502.png', '06433.png', '03847.png', '07783.png', '03750.png', '00723.png', '04801.png', '22642.png', '17942.png', '03662.png', '20521.png', '12811.png', '05511.png', '15977.png', '16322.png', '08353.png', '04428.png', '12617.png', '04013.png', '12965.png', '06418.png', '22768.png', '13129.png', '23122.png', '07791.png', '01585.png', '14621.png', '14270.png', '06334.png', '14282.png', '18187.png', '15941.png', '16011.png', '06127.png', '23517.png', '21345.png', '00546.png', '22446.png', '22463.png', '13238.png', '00823.png', '14875.png', '21438.png', '05756.png', '14898.png', '15564.png', '06394.png', '11259.png', '19865.png', '09602.png', '14522.png', '08901.png', '01356.png', '13899.png', '16152.png', '24353.png', '12959.png', '04537.png', '22622.png', '24853.png', '23178.png', '19362.png', '22993.png', '15508.png', '02860.png', '06340.png', '13097.png', '08793.png', '22311.png', '00299.png', '07067.png', '20429.png', '06685.png', '22170.png', '08178.png', '11652.png', '14533.png', '12690.png', '00191.png', '08752.png', '24425.png', '12597.png', '24850.png', '24740.png', '10407.png', '06268.png', '05732.png', '14920.png', '02304.png', '00536.png', '12497.png', '05082.png', '21177.png', '20491.png', '02721.png', '04706.png', '11067.png', '06774.png', '10749.png', '03863.png', '05250.png', '08066.png', '20726.png', '10436.png', '21648.png', '13104.png', '19036.png', '20062.png', '12646.png', '06532.png', '14574.png', '06429.png', '12866.png', '05297.png', '18983.png', '06270.png', '04150.png', '20333.png', '05525.png', '02636.png', '05111.png', '23896.png', '07596.png', '19594.png', '11740.png', '00619.png', '02434.png', '21101.png', '23815.png', '16627.png', '04454.png', '20954.png', '23548.png', '00296.png', '21398.png', '02909.png', '11430.png', '02045.png', '08599.png', '13360.png', '08201.png', '18933.png', '23983.png', '15725.png', '10780.png', '04421.png', '21873.png', '17748.png', '04471.png', '17105.png', '04775.png', '18817.png', '17876.png', '11687.png', '01825.png', '18058.png', '19644.png', '14530.png', '24533.png', '00388.png', '06757.png', '14660.png', '23952.png', '04385.png', '01542.png', '05169.png', '21298.png', '23775.png', '12099.png', '19601.png', '09039.png', '04986.png', '06738.png', '02677.png', '19442.png', '12216.png', '04316.png', '12848.png', '07557.png', '09445.png', '05417.png', '22842.png', '08421.png', '13550.png', '17074.png', '17964.png', '01349.png', '07079.png', '13132.png', '02120.png', '04527.png', '22538.png', '21443.png', '17242.png', '23164.png', '00491.png', '11215.png', '09270.png', '12838.png', '13758.png', '01643.png', '01404.png', '09527.png', '02342.png', '12132.png', '07498.png', '08518.png', '08628.png', '12230.png', '14173.png', '12963.png', '09142.png', '04028.png', '02615.png', '22828.png', '24226.png', '06664.png', '01375.png', '23912.png', '23090.png', '06602.png', '18282.png', '11728.png', '05792.png', '03955.png', '22025.png', '14039.png', '15955.png', '03169.png', '13256.png', '07713.png', '12194.png', '09777.png', '01993.png', '11503.png', '19954.png', '04985.png', '12388.png', '15811.png', '23148.png', '10157.png', '05392.png', '16991.png', '14771.png', '22901.png', '21190.png', '24399.png', '06536.png', '08524.png', '17317.png', '10257.png', '07428.png', '02167.png', '04640.png', '07574.png', '11140.png', '19203.png', '22376.png', '11753.png', '09887.png', '21523.png', '21445.png', '14205.png', '12880.png', '13727.png', '05744.png', '05861.png', '17665.png', '23710.png', '11641.png', '18664.png', '15292.png', '01411.png', '19008.png', '17100.png', '13597.png', '18105.png', '02148.png', '13260.png', '08993.png', '17973.png', '14626.png', '22978.png', '06870.png', '01293.png', '01333.png', '07573.png', '21878.png', '15193.png', '18052.png', '08265.png', '03948.png', '03148.png', '14196.png', '22726.png', '20791.png', '22381.png', '07249.png', '11507.png', '24662.png', '17280.png', '10919.png', '18643.png', '04481.png', '24785.png', '12555.png', '04311.png', '18229.png', '18635.png', '07275.png', '12599.png', '08953.png', '04187.png', '04622.png', '13176.png', '03245.png', '21315.png', '17465.png', '09840.png', '07185.png', '24043.png', '17365.png', '11984.png', '09016.png', '23385.png', '21144.png', '16761.png', '02992.png', '07767.png', '08193.png', '02785.png', '20392.png', '18984.png', '24392.png', '18294.png', '02867.png', '11427.png', '08504.png', '20103.png', '15606.png', '18511.png', '02821.png', '14798.png', '14970.png', '19185.png', '09423.png', '08719.png', '09693.png', '03064.png', '01907.png', '12287.png', '13018.png', '00159.png', '10225.png', '03225.png', '06651.png', '04448.png', '09970.png', '19474.png', '14033.png', '00438.png', '15983.png', '23981.png', '13307.png', '08144.png', '06432.png', '02687.png', '22732.png', '20438.png', '07503.png', '16196.png', '17632.png', '10121.png', '00387.png', '04091.png', '06991.png', '09356.png', '22759.png', '19392.png', '16800.png', '16820.png', '10766.png', '23359.png', '23351.png', '16533.png', '17996.png', '23594.png', '23372.png', '13459.png', '09110.png', '02464.png', '02710.png', '16104.png', '15926.png', '12228.png', '23733.png', '14890.png', '16519.png', '19003.png', '01003.png', '24505.png', '01033.png', '02839.png', '07440.png', '09782.png', '12807.png', '14220.png', '07462.png', '03915.png', '23811.png', '22469.png', '17524.png', '20686.png', '11038.png', '14414.png', '07432.png', '13962.png', '16220.png', '06364.png', '21041.png', '10551.png', '01328.png', '18869.png', '22224.png', '14729.png', '09333.png', '12300.png', '08641.png', '12609.png', '15227.png', '10722.png', '17501.png', '04873.png', '02927.png', '08447.png', '18456.png', '10599.png', '07280.png', '05755.png', '06986.png', '05996.png', '13573.png', '03398.png', '23728.png', '17090.png', '13485.png', '20995.png', '17894.png', '05498.png', '19898.png', '24412.png', '07480.png', '15342.png', '23096.png', '23554.png', '15676.png', '23123.png', '09667.png', '00481.png', '01818.png', '23971.png', '16303.png', '10855.png', '02144.png', '00121.png', '19897.png', '04374.png', '03406.png', '06087.png', '18487.png', '12378.png', '20768.png', '02266.png', '09569.png', '11275.png', '06259.png', '07326.png', '09130.png', '07507.png', '19913.png', '03926.png', '15409.png', '13565.png', '15858.png', '20929.png', '13558.png', '01181.png', '11867.png', '19415.png', '14801.png', '17438.png', '11567.png', '13847.png', '17059.png', '12834.png', '07753.png', '00025.png', '11321.png', '08685.png', '16027.png', '20212.png', '05345.png', '14597.png', '23410.png', '07923.png', '22731.png', '05841.png', '23299.png', '09639.png', '00593.png', '10865.png', '24316.png', '09410.png', '07500.png', '19955.png', '16623.png', '15638.png', '15394.png', '19947.png', '14874.png', '00235.png', '18404.png', '06905.png', '09115.png', '02406.png', '13407.png', '10109.png', '14361.png', '23727.png', '15414.png', '11207.png', '07595.png', '05099.png', '13237.png', '00489.png', '24631.png', '02472.png', '22979.png', '07649.png', '08940.png', '13035.png', '06713.png', '19518.png', '06054.png', '00368.png', '12860.png', '20337.png', '22083.png', '20412.png', '20041.png', '19469.png', '02911.png', '02629.png', '07514.png', '10251.png', '00379.png', '11533.png', '10166.png', '01025.png', '11854.png', '21497.png', '14469.png', '00668.png', '00618.png', '20300.png', '06150.png', '11100.png', '24041.png', '21140.png', '18386.png', '16399.png', '00813.png', '12947.png', '20058.png', '09113.png', '03411.png', '13019.png', '10853.png', '20124.png', '03933.png', '20433.png', '22212.png', '15297.png', '16020.png', '04800.png', '19230.png', '19523.png', '23560.png', '09712.png', '17516.png', '14556.png', '15608.png', '03480.png', '22308.png', '15737.png', '09709.png', '22016.png', '05994.png', '15249.png', '01413.png', '17609.png', '17653.png', '05357.png', '19957.png', '06611.png', '15908.png', '23631.png', '06069.png', '23516.png', '06657.png', '21047.png', '21137.png', '07184.png', '16512.png', '11736.png', '22952.png', '09876.png', '10585.png', '23473.png', '04635.png', '23538.png', '22022.png', '22824.png', '23861.png', '04213.png', '12556.png', '06892.png', '02929.png', '18744.png', '20402.png', '07323.png', '01885.png', '12755.png', '04612.png', '08418.png', '23481.png', '15548.png', '01261.png', '19015.png', '09049.png', '05194.png', '04372.png', '15191.png', '05552.png', '22299.png', '16735.png', '14876.png', '03560.png', '06677.png', '04757.png', '22268.png', '07255.png', '19786.png', '07551.png', '18179.png', '04449.png', '23066.png', '18780.png', '21706.png', '19496.png', '06484.png', '21384.png', '23124.png', '17322.png', '24695.png', '10500.png', '13110.png', '18837.png', '11951.png', '19088.png', '19849.png', '12439.png', '08440.png', '21189.png', '17705.png', '13337.png', '21529.png', '22220.png', '17193.png', '03872.png', '09459.png', '24401.png', '04989.png', '14382.png', '10977.png', '08359.png', '09176.png', '24959.png', '09585.png', '00614.png', '15950.png', '03962.png', '24047.png', '15103.png', '03656.png', '17277.png', '07093.png', '15051.png', '18463.png', '09980.png', '19105.png', '03386.png', '03173.png', '06057.png', '02043.png', '00566.png', '07172.png', '02556.png', '13736.png', '07424.png', '05036.png', '08334.png', '04424.png', '19510.png', '02684.png', '09152.png', '08162.png', '24060.png', '13861.png', '14201.png', '23399.png', '11694.png', '24812.png', '02737.png', '18712.png', '08479.png', '02980.png', '04978.png', '23527.png', '03423.png', '13410.png', '17989.png', '10221.png', '24185.png', '18760.png', '06034.png', '14367.png', '23278.png', '15203.png', '11094.png', '17294.png', '12490.png', '20453.png', '13658.png', '07125.png', '05216.png', '06131.png', '24593.png', '13409.png', '15640.png', '15369.png', '18663.png', '16267.png', '18794.png', '12231.png', '09409.png', '03700.png', '24437.png', '15597.png', '09129.png', '08392.png', '16033.png', '20809.png', '23482.png', '07778.png', '14670.png', '00714.png', '15231.png', '02897.png', '04073.png', '00248.png', '20264.png', '17919.png', '05459.png', '14959.png', '11053.png', '18269.png', '15579.png', '04858.png', '00314.png', '13760.png', '10460.png', '18879.png', '03624.png', '20638.png', '24532.png', '04633.png', '22163.png', '23472.png', '02077.png', '11276.png', '05635.png', '00125.png', '05738.png', '19237.png', '18213.png', '20435.png', '14855.png', '23711.png', '24606.png', '24710.png', '04417.png', '07747.png', '23853.png', '22651.png', '16017.png', '16042.png', '20181.png', '05441.png', '01377.png', '12005.png', '00724.png', '02746.png', '13755.png', '12369.png', '14568.png', '11422.png', '18692.png', '07674.png', '06002.png', '08544.png', '13179.png', '04086.png', '17530.png', '19568.png', '17045.png', '10610.png', '11725.png', '23371.png', '06440.png', '10891.png', '23264.png', '03596.png', '08041.png', '13549.png', '09720.png', '15556.png', '10056.png', '01259.png', '02554.png', '21606.png', '21018.png', '21161.png', '01299.png', '12832.png', '16245.png', '00772.png', '01276.png', '22705.png', '11243.png', '08442.png', '17852.png', '22572.png', '21118.png', '04545.png', '07660.png', '13290.png', '05805.png', '02236.png', '02413.png', '07962.png', '12486.png', '20712.png', '21517.png', '23804.png', '19112.png', '03606.png', '00557.png', '18118.png', '03986.png', '08067.png', '16259.png', '03166.png', '18890.png', '03938.png', '21222.png', '02465.png', '12341.png', '10593.png', '05843.png', '18308.png', '20856.png', '13293.png', '15350.png', '08230.png', '21273.png', '20665.png', '07678.png', '07934.png', '20601.png', '16687.png', '21538.png', '19143.png', '12569.png', '00149.png', '09803.png', '18097.png', '02569.png', '00544.png', '11442.png', '23636.png', '21036.png', '12689.png', '09412.png', '05502.png', '18932.png', '11494.png', '20833.png', '08404.png', '22527.png', '24869.png', '22173.png', '01880.png', '01690.png', '05681.png', '18845.png', '13421.png', '15985.png', '09379.png', '21957.png', '00417.png', '06701.png', '07785.png', '04939.png', '07296.png', '11142.png', '09183.png', '07126.png', '21122.png', '16718.png', '16032.png', '23087.png', '23350.png', '04818.png', '08898.png', '15347.png', '05894.png', '16177.png', '02774.png', '19845.png', '01258.png', '24952.png', '20007.png', '07369.png', '19379.png', '17634.png', '02091.png', '12883.png', '19213.png', '18825.png', '02209.png', '19561.png', '04991.png', '10924.png', '17686.png', '05844.png', '03305.png', '07563.png', '03051.png', '09890.png', '22019.png', '04564.png', '18855.png', '21293.png', '21061.png', '12236.png', '05259.png', '02722.png', '08468.png', '05119.png', '04461.png', '02987.png', '02376.png', '02764.png', '00874.png', '18332.png', '05505.png', '11617.png', '11688.png', '10446.png', '18701.png', '23154.png', '15831.png', '05213.png', '03651.png', '00264.png', '16170.png', '24314.png', '06671.png', '23485.png', '20194.png', '21645.png', '06509.png', '14064.png', '01210.png', '09044.png', '08165.png', '13797.png', '22499.png', '16591.png', '13045.png', '18357.png', '04020.png', '19631.png', '18488.png', '22412.png', '19323.png', '21108.png', '03391.png', '22554.png', '14986.png', '04656.png', '21708.png', '10852.png']
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
        
        if i_iter == 328626:
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
