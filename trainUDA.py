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
    a = ['19807.png', '14332.png', '10164.png', '00162.png', '24027.png', '14802.png', '08345.png', '07065.png', '22266.png', '04284.png', '02921.png', '10901.png', '19185.png', '10374.png', '12465.png', '15060.png', '21716.png', '08092.png', '11988.png', '00985.png', '18225.png', '01907.png', '14933.png', '06481.png', '02591.png', '16309.png', '01922.png', '10098.png', '13483.png', '13197.png', '11774.png', '04355.png', '15577.png', '20947.png', '12716.png', '03653.png', '23268.png', '19872.png', '11207.png', '06193.png', '11107.png', '11795.png', '12726.png', '13222.png', '22667.png', '17309.png', '16901.png', '17647.png', '11030.png', '14380.png', '03997.png', '11455.png', '16805.png', '14098.png', '13012.png', '10998.png', '01846.png', '11802.png', '08138.png', '22070.png', '23431.png', '14067.png', '20136.png', '09904.png', '22957.png', '11904.png', '10703.png', '11358.png', '15660.png', '19523.png', '02990.png', '23613.png', '06298.png', '07103.png', '12855.png', '01087.png', '19311.png', '01839.png', '14448.png', '10549.png', '21862.png', '03007.png', '09355.png', '18223.png', '23499.png', '24704.png', '01640.png', '07411.png', '19130.png', '16958.png', '22019.png', '01577.png', '04592.png', '04345.png', '19565.png', '06738.png', '12934.png', '23595.png', '14725.png', '00418.png', '23754.png', '20385.png', '23680.png', '04520.png', '16893.png', '19602.png', '05909.png', '15498.png', '03815.png', '15395.png', '03379.png', '07129.png', '12615.png', '16627.png', '18888.png', '12155.png', '20932.png', '03049.png', '14352.png', '14206.png', '12663.png', '00055.png', '06682.png', '20215.png', '02703.png', '03217.png', '06629.png', '21653.png', '03151.png', '16926.png', '13011.png', '02801.png', '00891.png', '01223.png', '17606.png', '23685.png', '16024.png', '21578.png', '18293.png', '24601.png', '19861.png', '08258.png', '03276.png', '07564.png', '20634.png', '14210.png', '03933.png', '21054.png', '03871.png', '04002.png', '20024.png', '04433.png', '19032.png', '01499.png', '04380.png', '08689.png', '02453.png', '03747.png', '20774.png', '14309.png', '11242.png', '05367.png', '06935.png', '08219.png', '16565.png', '15533.png', '06259.png', '11510.png', '18718.png', '08276.png', '14688.png', '13629.png', '01380.png', '16941.png', '05484.png', '15101.png', '07953.png', '17508.png', '17210.png', '11306.png', '09523.png', '21563.png', '21586.png', '02964.png', '12731.png', '21536.png', '00575.png', '21474.png', '17284.png', '24557.png', '06528.png', '06000.png', '19917.png', '23580.png', '04483.png', '02336.png', '23971.png', '23287.png', '13333.png', '06336.png', '09993.png', '12684.png', '08985.png', '03813.png', '15838.png', '15513.png', '03439.png', '00777.png', '01502.png', '09714.png', '01568.png', '03600.png', '18988.png', '17457.png', '20815.png', '13424.png', '09652.png', '11803.png', '15621.png', '19075.png', '16043.png', '08058.png', '24570.png', '22375.png', '03784.png', '23823.png', '14869.png', '22857.png', '00502.png', '03327.png', '06404.png', '10096.png', '19923.png', '13260.png', '17470.png', '24957.png', '00868.png', '23242.png', '15434.png', '10183.png', '02078.png', '24587.png', '01261.png', '11013.png', '01987.png', '02833.png', '08646.png', '12616.png', '20746.png', '03951.png', '03168.png', '19795.png', '23121.png', '18023.png', '05774.png', '11964.png', '04084.png', '00849.png', '07217.png', '08563.png', '09096.png', '19319.png', '06376.png', '15913.png', '23060.png', '15074.png', '16386.png', '23425.png', '00975.png', '16753.png', '13199.png', '22744.png', '16770.png', '14544.png', '14884.png', '18745.png', '01745.png', '14313.png', '05054.png', '24790.png', '17288.png', '18722.png', '10724.png', '24776.png', '08327.png', '19110.png', '17163.png', '21999.png', '02434.png', '04258.png', '16198.png', '03746.png', '23179.png', '23027.png', '01601.png', '15242.png', '00043.png', '22569.png', '16678.png', '23995.png', '09907.png', '13874.png', '19183.png', '19104.png', '02483.png', '23444.png', '03638.png', '19979.png', '24371.png', '15747.png', '11945.png', '10186.png', '23651.png', '10595.png', '00988.png', '18290.png', '03889.png', '03339.png', '17282.png', '15322.png', '08905.png', '15162.png', '20690.png', '05811.png', '00544.png', '13037.png', '17436.png', '13513.png', '08489.png', '07609.png', '24019.png', '22525.png', '07423.png', '08767.png', '16628.png', '05709.png', '10454.png', '05464.png', '14157.png', '05924.png', '06788.png', '07909.png', '07780.png', '11851.png', '14601.png', '04285.png', '07778.png', '18952.png', '08102.png', '21106.png', '18637.png', '21896.png', '13448.png', '02951.png', '18401.png', '24054.png', '24716.png', '17466.png', '01711.png', '04431.png', '04568.png', '01272.png', '00634.png', '14207.png', '21461.png', '04648.png', '15303.png', '01817.png', '22864.png', '00760.png', '06505.png', '20513.png', '07234.png', '02880.png', '18204.png', '05897.png', '11652.png', '23122.png', '04928.png', '24586.png', '13761.png', '01754.png', '06289.png', '19744.png', '08621.png', '05281.png', '17485.png', '18726.png', '02806.png', '20258.png', '00064.png', '19772.png', '06593.png', '23125.png', '09007.png', '17678.png', '08554.png', '18800.png', '06374.png', '04921.png', '12344.png', '15783.png', '06735.png', '08944.png', '17763.png', '01354.png', '18868.png', '14135.png', '12263.png', '09258.png', '21003.png', '10528.png', '23080.png', '16474.png', '07773.png', '08076.png', '07993.png', '15128.png', '04184.png', '10911.png', '08363.png', '06247.png', '18575.png', '16528.png', '12129.png', '21633.png', '17885.png', '01112.png', '06101.png', '09194.png', '18620.png', '01668.png', '07662.png', '16398.png', '02864.png', '11400.png', '07464.png', '17633.png', '11705.png', '00681.png', '14457.png', '03618.png', '17310.png', '14858.png', '23829.png', '17991.png', '24595.png', '01309.png', '24143.png', '09117.png', '12477.png', '12162.png', '14255.png', '10868.png', '00668.png', '03437.png', '10105.png', '21543.png', '16709.png', '20606.png', '09024.png', '00248.png', '09922.png', '15567.png', '00651.png', '18709.png', '22182.png', '24062.png', '00445.png', '23573.png', '13315.png', '16364.png', '01706.png', '12114.png', '10585.png', '12345.png', '01471.png', '19174.png', '10159.png', '17478.png', '03262.png', '09153.png', '19545.png', '00884.png', '07036.png', '21140.png', '09510.png', '15349.png', '23473.png', '11882.png', '23089.png', '09136.png', '05315.png', '21946.png', '16040.png', '12063.png', '05833.png', '09079.png', '14749.png', '09176.png', '22127.png', '05859.png', '21362.png', '05674.png', '01293.png', '21895.png', '17958.png', '13455.png', '06350.png', '04561.png', '02424.png', '14173.png', '21854.png', '24655.png', '05783.png', '21541.png', '13919.png', '15969.png', '17033.png', '08647.png', '07002.png', '13807.png', '13237.png', '12930.png', '10959.png', '07787.png', '12644.png', '18840.png', '23563.png', '18402.png', '13548.png', '23581.png', '23508.png', '08064.png', '18497.png', '12754.png', '22943.png', '08297.png', '07112.png', '24759.png', '03193.png', '03802.png', '23944.png', '12486.png', '00087.png', '08974.png', '13180.png', '07832.png', '09407.png', '00373.png', '05410.png', '04885.png', '13341.png', '01631.png', '13575.png', '01837.png', '13168.png', '21888.png', '19881.png', '05051.png', '18244.png', '07072.png', '16438.png', '20816.png', '00075.png', '18796.png', '22339.png', '15075.png', '10165.png', '20833.png', '15400.png', '21065.png', '12506.png', '24946.png', '13220.png', '05598.png', '08099.png', '00913.png', '20365.png', '22219.png', '16092.png', '10856.png', '22170.png', '21344.png', '21460.png', '00342.png', '14236.png', '08046.png', '24325.png', '23935.png', '14235.png', '16120.png', '08529.png', '22123.png', '06989.png', '02796.png', '00159.png', '01206.png', '24830.png', '06696.png', '02084.png', '20605.png', '09607.png', '06822.png', '23813.png', '04287.png', '17582.png', '06597.png', '06381.png', '14411.png', '01216.png', '00934.png', '01580.png', '09854.png', '23163.png', '07897.png', '17142.png', '18563.png', '07651.png', '04077.png', '10360.png', '13098.png', '03329.png', '22780.png', '22109.png', '17522.png', '03880.png', '20406.png', '22244.png', '22007.png', '00979.png', '07158.png', '12967.png', '24263.png', '01886.png', '01818.png', '22510.png', '12287.png', '09506.png', '10288.png', '22779.png', '13602.png', '21014.png', '04979.png', '21011.png', '17766.png', '10336.png', '05314.png', '09428.png', '12921.png', '02940.png', '20151.png', '12704.png', '05629.png', '14378.png', '21748.png', '10120.png', '01358.png', '10849.png', '04805.png', '07549.png', '14930.png', '02942.png', '06558.png', '05636.png', '18766.png', '17107.png', '20374.png', '22305.png', '01859.png', '10610.png', '11589.png', '07859.png', '09597.png', '00510.png', '00725.png', '18094.png', '17955.png', '03588.png', '05018.png', '11377.png', '23418.png', '03689.png', '14598.png', '20176.png', '15885.png', '00438.png', '22335.png', '21914.png', '22615.png', '22224.png', '17740.png', '22872.png', '23586.png', '21735.png', '04610.png', '06059.png', '04008.png', '21912.png', '04704.png', '22995.png', '10195.png', '19518.png', '00609.png', '11257.png', '08981.png', '13686.png', '10472.png', '05289.png', '23987.png', '03320.png', '03401.png', '05892.png', '05454.png', '14414.png', '06166.png', '17349.png', '23422.png', '00124.png', '05470.png', '10840.png', '19107.png', '01014.png', '08949.png', '18949.png', '13667.png', '23423.png', '20858.png', '23590.png', '14119.png', '23984.png', '18136.png', '10972.png', '16113.png', '23892.png', '20302.png', '02555.png', '11701.png', '06985.png', '01695.png', '18222.png', '03730.png', '14398.png', '04236.png', '00936.png', '24502.png', '20765.png', '20637.png', '05824.png', '18470.png', '00847.png', '12168.png', '13976.png', '23107.png', '01465.png', '17563.png', '14126.png', '10796.png', '08751.png', '02441.png', '11273.png', '09417.png', '01205.png', '17039.png', '02729.png', '05953.png', '17124.png', '00918.png', '13836.png', '09906.png', '04854.png', '09951.png', '15289.png', '09174.png', '15287.png', '16045.png', '01218.png', '09941.png', '02992.png', '23550.png', '06930.png', '13343.png', '06291.png', '09530.png', '17483.png', '13627.png', '15453.png', '10332.png', '04945.png', '06199.png', '22014.png', '23655.png', '19334.png', '20789.png', '09926.png', '16612.png', '22699.png', '17031.png', '13030.png', '20453.png', '07709.png', '02893.png', '04462.png', '06753.png', '08779.png', '07681.png', '00551.png', '07631.png', '15282.png', '24734.png', '22808.png', '21100.png', '08678.png', '17636.png', '01992.png', '24436.png', '03874.png', '14153.png', '20188.png', '14177.png', '04096.png', '12600.png', '07119.png', '16572.png', '08225.png', '03835.png', '19952.png', '21623.png', '07088.png', '23779.png', '11989.png', '02234.png', '22518.png', '08561.png', '03692.png', '02869.png', '15908.png', '11786.png', '11552.png', '09288.png', '17458.png', '09010.png', '23602.png', '06669.png', '04497.png', '03185.png', '15401.png', '05003.png', '14831.png', '12851.png', '13634.png', '24466.png', '02978.png', '19771.png', '05316.png', '17559.png', '13163.png', '16758.png', '08815.png', '14547.png', '12990.png', '11831.png', '00250.png', '22688.png', '06439.png', '17891.png', '21353.png', '02231.png', '11158.png', '21516.png', '04860.png', '07479.png', '16157.png', '15262.png', '01647.png', '22920.png', '16522.png', '17591.png', '04594.png', '10279.png', '00255.png', '24934.png', '14605.png', '12964.png', '10617.png', '02406.png', '00392.png', '11442.png', '16783.png', '17211.png', '01285.png', '01854.png', '09960.png', '02840.png', '23553.png', '19216.png', '17726.png', '08859.png', '22029.png', '05610.png', '19662.png', '01769.png', '11508.png', '24873.png', '08319.png', '19983.png', '05452.png', '09689.png', '23450.png', '00831.png', '07428.png', '03669.png', '12337.png', '11071.png', '17684.png', '23682.png', '06569.png', '05810.png', '13987.png', '10198.png', '09574.png', '09403.png', '08851.png', '22420.png', '07061.png', '15007.png', '00384.png', '15222.png', '12115.png', '15803.png', '19777.png', '09620.png', '02159.png', '24435.png', '02952.png', '06321.png', '02028.png', '04783.png', '08155.png', '06104.png', '13387.png', '03568.png', '17149.png', '15916.png', '15997.png', '17777.png', '18405.png', '24166.png', '14960.png', '12384.png', '21702.png', '09596.png', '10057.png', '19838.png', '06322.png', '02943.png', '23408.png', '04680.png', '08629.png', '02426.png', '11019.png', '22326.png', '09615.png', '06716.png', '11332.png', '23281.png', '06601.png', '12011.png', '10479.png', '23712.png', '22186.png', '24428.png', '07759.png', '02910.png', '21966.png', '23152.png', '05600.png', '17267.png', '04891.png', '00238.png', '11312.png', '24461.png', '20941.png', '14937.png', '18395.png', '20757.png', '01700.png', '19400.png', '05601.png', '11141.png', '04876.png', '04835.png', '21321.png', '01297.png', '14696.png', '01652.png', '19445.png', '24553.png', '15471.png', '05607.png', '01056.png', '13055.png', '24694.png', '02183.png', '23140.png', '13952.png', '05749.png', '19988.png', '00856.png', '21995.png', '02622.png', '07896.png', '22942.png', '20602.png', '19286.png', '24292.png', '10158.png', '10605.png', '05929.png', '15523.png', '19858.png', '11444.png', '07396.png', '17473.png', '06550.png', '23524.png', '01372.png', '11538.png', '08566.png', '12216.png', '17994.png', '19245.png', '15018.png', '09264.png', '05429.png', '22412.png', '00652.png', '00400.png', '19528.png', '20866.png', '22628.png', '24846.png', '19333.png', '00490.png', '01248.png', '13393.png', '04560.png', '18885.png', '07838.png', '24335.png', '11282.png', '10147.png', '20150.png', '10442.png', '17712.png', '13719.png', '15165.png', '06833.png', '19871.png', '11808.png', '03819.png', '07183.png', '16731.png', '07275.png', '16501.png', '13911.png', '20952.png', '09695.png', '12649.png', '03065.png', '22243.png', '05688.png', '19689.png', '12352.png', '08124.png', '04772.png', '14304.png', '08338.png', '03447.png', '20267.png', '12279.png', '17974.png', '11673.png', '22661.png', '07928.png', '08167.png', '16623.png', '21898.png', '12834.png', '04028.png', '20137.png', '06124.png', '03658.png', '23803.png', '22889.png', '15203.png', '04709.png', '16943.png', '13746.png', '21334.png', '06933.png', '22431.png', '03683.png', '18898.png', '13114.png', '06269.png', '21513.png', '05427.png', '17754.png', '24722.png', '24824.png', '20473.png', '07318.png', '00202.png', '10494.png', '15291.png', '24395.png', '08303.png', '01802.png', '08045.png', '20063.png', '22283.png', '00615.png', '20995.png', '04404.png', '00577.png', '07949.png', '02023.png', '17329.png', '16124.png', '16302.png', '10690.png', '22234.png', '17056.png', '21455.png', '19405.png', '01741.png', '10074.png', '05728.png', '08868.png', '07467.png', '05559.png', '15200.png', '18365.png', '18100.png', '09231.png', '02463.png', '20138.png', '20275.png', '19853.png', '08894.png', '17837.png', '09826.png', '09990.png', '00735.png', '22891.png', '24931.png', '03691.png', '11436.png', '15415.png', '11153.png', '19568.png', '08971.png', '05896.png', '12351.png', '04974.png', '13771.png', '20798.png', '14458.png', '07937.png', '06635.png', '02368.png', '22522.png', '10940.png', '18008.png', '03632.png', '03909.png', '19798.png', '05696.png', '09497.png', '23146.png', '01943.png', '10020.png', '07121.png', '02923.png', '14018.png', '24633.png', '18973.png', '10619.png', '14358.png', '09072.png', '04889.png', '06063.png', '24095.png', '12503.png', '21485.png', '13800.png', '21440.png', '24556.png', '17241.png', '00613.png', '00746.png', '23078.png', '18096.png', '10795.png', '22841.png', '21030.png', '14732.png', '12680.png', '16457.png', '20458.png', '15026.png', '17603.png', '00234.png', '13898.png', '23073.png', '08464.png', '20005.png', '14838.png', '12989.png', '00269.png', '21426.png', '01984.png', '06234.png', '07274.png', '01258.png', '21760.png', '12778.png', '19513.png', '07823.png', '10134.png', '21131.png', '07045.png', '01866.png', '15405.png', '22521.png', '13253.png', '00076.png', '11762.png', '18975.png', '00557.png', '04918.png', '18461.png', '13853.png', '09726.png', '23835.png', '24206.png', '21049.png', '23566.png', '12996.png', '03476.png', '10935.png', '15360.png', '23667.png', '22408.png', '05997.png', '24542.png', '17168.png', '16751.png', '23928.png', '17188.png', '15760.png', '11672.png', '14563.png', '05060.png', '15006.png', '15073.png', '02578.png', '13568.png', '23730.png', '22075.png', '11610.png', '01827.png', '24346.png', '19369.png', '10792.png', '04427.png', '08313.png', '16976.png', '02455.png', '05587.png', '06163.png', '16008.png', '20071.png', '11956.png', '20903.png', '17597.png', '20853.png', '06719.png', '12327.png', '18969.png', '24899.png', '04878.png', '01816.png', '20484.png', '08243.png', '09283.png', '23805.png', '08385.png', '05123.png', '09775.png', '08510.png', '08580.png', '05792.png', '04252.png', '22225.png', '23144.png', '04832.png', '20426.png', '15440.png', '21430.png', '01413.png', '23209.png', '00742.png', '08764.png', '03761.png', '12985.png', '13125.png', '00277.png', '18073.png', '24385.png', '14169.png', '00070.png', '04625.png', '11714.png', '23288.png', '01920.png', '10763.png', '17765.png', '22907.png', '14271.png', '00200.png', '03645.png', '01490.png', '17269.png', '02494.png', '17082.png', '18778.png', '00521.png', '01462.png', '06720.png', '11839.png', '00649.png', '08854.png', '12953.png', '08477.png', '23864.png', '05353.png', '03675.png', '05781.png', '18641.png', '16603.png', '05937.png', '06587.png', '07374.png', '19873.png', '08959.png', '08218.png', '02671.png', '00820.png', '02709.png', '01511.png', '19101.png', '18527.png', '23384.png', '07059.png', '00879.png', '00571.png', '07880.png', '00389.png', '05714.png', '20285.png', '13477.png', '03479.png', '23136.png', '17066.png', '09185.png', '09790.png', '08955.png', '11047.png', '07404.png', '15488.png', '04901.png', '16708.png', '10304.png', '03055.png', '08699.png', '14129.png', '08671.png', '10178.png', '02794.png', '16717.png', '03867.png', '21175.png', '17171.png', '14489.png', '18919.png', '19758.png', '11243.png', '17206.png', '22898.png', '20286.png', '02541.png', '04773.png', '00175.png', '09166.png', '08073.png', '03827.png', '16111.png', '15711.png', '16299.png', '13990.png', '17785.png', '06878.png', '17041.png', '19992.png', '09164.png', '15481.png', '04303.png', '04337.png', '21208.png', '14786.png', '06560.png', '05437.png', '22787.png', '20388.png', '24467.png', '11854.png', '13252.png', '18383.png', '14667.png', '18259.png', '06992.png', '07244.png', '11550.png', '11402.png', '14624.png', '04656.png', '14616.png', '17930.png', '16218.png', '00308.png', '13104.png', '24102.png', '12492.png', '11730.png', '23582.png', '00363.png', '10367.png', '18452.png', '10034.png', '18182.png', '13573.png', '15229.png', '08009.png', '17289.png', '24673.png', '06840.png', '14446.png', '04396.png', '11234.png', '11406.png', '09480.png', '02053.png', '09595.png', '03051.png', '08281.png', '01918.png', '20492.png', '06409.png', '02006.png', '20590.png', '10505.png', '05845.png', '14474.png', '16542.png', '03082.png', '18566.png', '13823.png', '05884.png', '07359.png', '05159.png', '14426.png', '20644.png', '20797.png', '05968.png', '13939.png', '12579.png', '24121.png', '03153.png', '03068.png', '10560.png', '18630.png', '01044.png', '00419.png', '12127.png', '21435.png', '22627.png', '17351.png', '10520.png', '24305.png', '00317.png', '15725.png', '01058.png', '23301.png', '15701.png', '17174.png', '19382.png', '07881.png', '17803.png', '23500.png', '09973.png', '19931.png', '06001.png', '17084.png', '08487.png', '05282.png', '11189.png', '04720.png', '00189.png', '18981.png', '17861.png', '23521.png', '10615.png', '19492.png', '23966.png', '19743.png', '21662.png', '12693.png', '03955.png', '19399.png', '02165.png', '21339.png', '05010.png', '03478.png', '08291.png', '15721.png', '17154.png', '24223.png', '17924.png', '09639.png', '19401.png', '12135.png', '02019.png', '23462.png', '02237.png', '12798.png', '21546.png', '20944.png', '20320.png', '17169.png', '02308.png', '24823.png', '16841.png', '04003.png', '23445.png', '11908.png', '13486.png', '12844.png', '17887.png', '18550.png', '06146.png', '04710.png', '24144.png', '02470.png', '19471.png', '10869.png', '00524.png', '16430.png', '08594.png', '10430.png', '06393.png', '20319.png', '02082.png', '03156.png', '07581.png', '03181.png', '12837.png', '04997.png', '14602.png', '08642.png', '20538.png', '00069.png', '01628.png', '01781.png', '16544.png', '20679.png', '23843.png', '13042.png', '11211.png', '21406.png', '13902.png', '08221.png', '02948.png', '19363.png', '22061.png', '13007.png', '03987.png', '00921.png', '24443.png', '18700.png', '01234.png', '11822.png', '04572.png', '18861.png', '10676.png', '18238.png', '09284.png', '05895.png', '11880.png', '21844.png', '10872.png', '02928.png', '05342.png', '16832.png', '04881.png', '08287.png', '01483.png', '13880.png', '19714.png', '11334.png', '02257.png', '13375.png', '17005.png', '06056.png', '16726.png', '17555.png', '15680.png', '21706.png', '19927.png', '23570.png', '02793.png', '02055.png', '15389.png', '24666.png', '04692.png', '07739.png', '09916.png', '13620.png', '13891.png', '06476.png', '17510.png', '07891.png', '16556.png', '21502.png', '24317.png', '17815.png', '09098.png', '15204.png', '04977.png', '21378.png', '17112.png', '12604.png', '12847.png', '19959.png', '10010.png', '19374.png', '03060.png', '15766.png', '24092.png', '14451.png', '13425.png', '21408.png', '07283.png', '10278.png', '01418.png', '13099.png', '00056.png', '15115.png', '09646.png', '23285.png', '23101.png', '15469.png', '02695.png', '05700.png', '04766.png', '00386.png', '17036.png', '12026.png', '21355.png', '09806.png', '12192.png', '08077.png', '24244.png', '14341.png', '09292.png', '01070.png', '03281.png', '13207.png', '11768.png', '11850.png', '17839.png', '23237.png', '23176.png', '13131.png', '14643.png', '19316.png', '18174.png', '10609.png', '00362.png', '17468.png', '07639.png', '01282.png', '07408.png', '19848.png', '07498.png', '11383.png', '02858.png', '16137.png', '08860.png', '21663.png', '19250.png', '05318.png', '11403.png', '04880.png', '04563.png', '15604.png', '05631.png', '01912.png', '23907.png', '09822.png', '14227.png', '07480.png', '21289.png', '03278.png', '11505.png', '14390.png', '01449.png', '13035.png', '24835.png', '16583.png', '05169.png', '00683.png', '00722.png', '13647.png', '21757.png', '01555.png', '08848.png', '12100.png', '04780.png', '18196.png', '08932.png', '16853.png', '21250.png', '23708.png', '22113.png', '19691.png', '18615.png', '16039.png', '18610.png', '01054.png', '15555.png', '22858.png', '24150.png', '04414.png', '01076.png', '13205.png', '16938.png', '19285.png', '04849.png', '24421.png', '12857.png', '15459.png', '01089.png', '01390.png', '04555.png', '24726.png', '15989.png', '03510.png', '22317.png', '05415.png', '22348.png', '14801.png', '17836.png', '20456.png', '03732.png', '07465.png', '05982.png', '00536.png', '04153.png', '08458.png', '09717.png', '18950.png', '04309.png', '17274.png', '04817.png', '14092.png', '09322.png', '21708.png', '03782.png', '05635.png', '09538.png', '02258.png', '10983.png', '02552.png', '11133.png', '10459.png', '13592.png', '02039.png', '11278.png', '09101.png', '09242.png', '00810.png', '12047.png', '00959.png', '23688.png', '10899.png', '05882.png', '08644.png', '09824.png', '15442.png', '00642.png', '16860.png', '01913.png', '01479.png', '24238.png', '10769.png', '06135.png', '19651.png', '08893.png', '19074.png', '17034.png', '16594.png', '15897.png', '03333.png', '21385.png', '05941.png', '05893.png', '00305.png', '18880.png', '12660.png', '16080.png', '07208.png', '04346.png', '07385.png', '17944.png', '18532.png', '07792.png', '23477.png', '24036.png', '24072.png', '04774.png', '03165.png', '13492.png', '09265.png', '00614.png', '24105.png', '02815.png', '05925.png', '09997.png', '09638.png', '00596.png', '16986.png', '13922.png', '20772.png', '14670.png', '16071.png', '22060.png', '03565.png', '17714.png', '15234.png', '24122.png', '13585.png', '08038.png', '07306.png', '09057.png', '19749.png', '13920.png', '00151.png', '12058.png', '07381.png', '11772.png', '15463.png', '17147.png', '05083.png', '10342.png', '14812.png', '03868.png', '07398.png', '16837.png', '18275.png', '23069.png', '13402.png', '17542.png', '05910.png', '05727.png', '21405.png', '17378.png', '19765.png', '03906.png', '04633.png', '00307.png', '17134.png', '19160.png', '00296.png', '23958.png', '05000.png', '11087.png', '07442.png', '06603.png', '12364.png', '19658.png', '01302.png', '04325.png', '08541.png', '22830.png', '11479.png', '18077.png', '22749.png', '05609.png', '00627.png', '04924.png', '14954.png', '23465.png', '18355.png', '00989.png', '07676.png', '17308.png', '11719.png', '15713.png', '22822.png', '18785.png', '16336.png', '09825.png', '13040.png', '06292.png', '24411.png', '18005.png', '04536.png', '13177.png', '15940.png', '06282.png', '16647.png', '24951.png', '04128.png', '23409.png', '18779.png', '19649.png', '20376.png', '01033.png', '18245.png', '04683.png', '05391.png', '13421.png', '11642.png', '02559.png', '20882.png', '03735.png', '22415.png', '14717.png', '10031.png', '01566.png', '04824.png', '14029.png', '00785.png', '24847.png', '07616.png', '22298.png', '24897.png', '22141.png', '02954.png', '10743.png', '05663.png', '12350.png', '11160.png', '17601.png', '18202.png', '14710.png', '10092.png', '23128.png', '20357.png', '22768.png', '03209.png', '23857.png', '19127.png', '14161.png', '11317.png', '23141.png', '17541.png', '22832.png', '23542.png', '23420.png', '06606.png', '04911.png', '15126.png', '12097.png', '10032.png', '17751.png', '22208.png', '10818.png', '21107.png', '02281.png', '00347.png', '09914.png', '07875.png', '08246.png', '16232.png', '16687.png', '08693.png', '06207.png', '11447.png', '09102.png', '17870.png', '24109.png', '18867.png', '17738.png', '21144.png', '08648.png', '11502.png', '03577.png', '24026.png', '24296.png', '11501.png', '09588.png', '21652.png', '07015.png', '18815.png', '05561.png', '19767.png', '11408.png', '04250.png', '06514.png', '21885.png', '15920.png', '15357.png', '06486.png', '22187.png', '04742.png', '02212.png', '16422.png', '09976.png', '24807.png', '03164.png', '02484.png', '23745.png', '24220.png', '07325.png', '20633.png', '17654.png', '01416.png', '20298.png', '12525.png', '02489.png', '20876.png', '16712.png', '06924.png', '12268.png', '23976.png', '12730.png', '09062.png', '19352.png', '20824.png', '22804.png', '16862.png', '22946.png', '16369.png', '09886.png', '09679.png', '13586.png', '13330.png', '06973.png', '24736.png', '18185.png', '19683.png', '14920.png', '12898.png', '22049.png', '09507.png', '10402.png', '14039.png', '08022.png', '18623.png', '23647.png', '20053.png', '10754.png', '18108.png', '07304.png', '15272.png', '23099.png', '16959.png', '07854.png', '15853.png', '18965.png', '13613.png', '14428.png', '11021.png', '09310.png', '09980.png', '11196.png', '03837.png', '19825.png', '14264.png', '00198.png', '16173.png', '22215.png', '20800.png', '03280.png', '20647.png', '05862.png', '12208.png', '21608.png', '16775.png', '24229.png', '18519.png', '11885.png', '24250.png', '00809.png', '16355.png', '05555.png', '04894.png', '08056.png', '16480.png', '09216.png', '01584.png', '13087.png', '02584.png', '17317.png', '06363.png', '21605.png', '08604.png', '09113.png', '11605.png', '12163.png', '21442.png', '14739.png', '19617.png', '00077.png', '02347.png', '11237.png', '03603.png', '18501.png', '18605.png', '17463.png', '24717.png', '16693.png', '05070.png', '03790.png', '05985.png', '05603.png', '15630.png', '01340.png', '12755.png', '16963.png', '18040.png', '09210.png', '15088.png', '08957.png', '06846.png', '06563.png', '21566.png', '01227.png', '13122.png', '20227.png', '12143.png', '15877.png', '00952.png', '06869.png', '20697.png', '18736.png', '05788.png', '17779.png', '12854.png', '05902.png', '09382.png', '10660.png', '16993.png', '03126.png', '02302.png', '09106.png', '22868.png', '16294.png', '24353.png', '12846.png', '01015.png', '01474.png', '16076.png', '09982.png', '21123.png', '13430.png', '13843.png', '15699.png', '12109.png', '00321.png', '04398.png', '00741.png', '08010.png', '19660.png', '18434.png', '12886.png', '18205.png', '24159.png', '24616.png', '04342.png', '06142.png', '06760.png', '18820.png', '04087.png', '23195.png', '23105.png', '01701.png', '03968.png', '01675.png', '16344.png', '07235.png', '02248.png', '23915.png', '15196.png', '12296.png', '03876.png', '13750.png', '04874.png', '04172.png', '05252.png', '19240.png', '24306.png', '02508.png', '06999.png', '22652.png', '10791.png', '14756.png', '02235.png', '21803.png', '15665.png', '06981.png', '03769.png', '07074.png', '01538.png', '19606.png', '09228.png', '19716.png', '21738.png', '10170.png', '16434.png', '16733.png', '02739.png', '10412.png', '17945.png', '01247.png', '11428.png', '11488.png', '01445.png', '13193.png', '22715.png', '08148.png', '20601.png', '06300.png', '08624.png', '22885.png', '09477.png', '07578.png', '17103.png', '19337.png', '22726.png', '22336.png', '04377.png', '20029.png', '06011.png', '09915.png', '22833.png', '20711.png', '01919.png', '14085.png', '19940.png', '00351.png', '17494.png', '22175.png', '01316.png', '04582.png', '12476.png', '17481.png', '01320.png', '21579.png', '01589.png', '07169.png', '10017.png', '11877.png', '13076.png', '05220.png', '00699.png', '00912.png', '24650.png', '19406.png', '11197.png', '01851.png', '22386.png', '21115.png', '07536.png', '11063.png', '23250.png', '17511.png', '12075.png', '02882.png', '23224.png', '19024.png', '12078.png', '08835.png', '14608.png', '02900.png', '00590.png', '22750.png', '01673.png', '23173.png', '19548.png', '20980.png', '02562.png', '11114.png', '13954.png', '19630.png', '01040.png', '17596.png', '16685.png', '06959.png', '20212.png', '07445.png', '24886.png', '10050.png', '16027.png', '22575.png', '23071.png', '17679.png', '19738.png', '15658.png', '09074.png', '09938.png', '11845.png', '21412.png', '20740.png', '22048.png', '02573.png', '19424.png', '00179.png', '23990.png', '15216.png', '01703.png', '17214.png', '07474.png', '00759.png', '17805.png', '02209.png', '20114.png', '20714.png', '08900.png', '12315.png', '19212.png', '24043.png', '01762.png', '11311.png', '12041.png', '04474.png', '21674.png', '06857.png', '13307.png', '07360.png', '23784.png', '22285.png', '03132.png', '03310.png', '19312.png', '04502.png', '16610.png', '10665.png', '05359.png', '06798.png', '09939.png', '23139.png', '08417.png', '12241.png', '08688.png', '22474.png', '10024.png', '04494.png', '11631.png', '05514.png', '09892.png', '08762.png', '04753.png', '18097.png', '19652.png', '12272.png', '14353.png', '04133.png', '13278.png', '06315.png', '21023.png', '03144.png', '17078.png', '02619.png', '21732.png', '13628.png', '04949.png', '18680.png', '22926.png', '19089.png', '08370.png', '04190.png', '01583.png', '13692.png', '16558.png', '09472.png', '02157.png', '03571.png', '17191.png', '06954.png', '11161.png', '21659.png', '03829.png', '21421.png', '24829.png', '19561.png', '08145.png', '16821.png', '17488.png', '15187.png', '02724.png', '03148.png', '13003.png', '04173.png', '23750.png', '03363.png', '07794.png', '24675.png', '03241.png', '19277.png', '03936.png', '10556.png', '01933.png', '09038.png', '04343.png', '18659.png', '06944.png', '04205.png', '11975.png', '02229.png', '09604.png', '12368.png', '18033.png', '19820.png', '06277.png', '14694.png', '22788.png', '10508.png', '16251.png', '05136.png', '23657.png', '05708.png', '00604.png', '10293.png', '02322.png', '14803.png', '21925.png', '22195.png', '20991.png', '12815.png', '24115.png', '05848.png', '24253.png', '05246.png', '20108.png', '13266.png', '18105.png', '12667.png', '10083.png', '21312.png', '08450.png', '22486.png', '17344.png', '09899.png', '20072.png', '04401.png', '15338.png', '06940.png', '00812.png', '05595.png', '13080.png', '06737.png', '13398.png', '13796.png', '21971.png', '15108.png', '08862.png', '20495.png', '08198.png', '15797.png', '17222.png', '01069.png', '07547.png', '10038.png', '19668.png', '09370.png', '19373.png', '08197.png', '15626.png', '05089.png', '02025.png', '04093.png', '16017.png', '03143.png', '20093.png', '14344.png', '10779.png', '12900.png', '21352.png', '14600.png', '00062.png', '15518.png', '10926.png', '07201.png', '00763.png', '24865.png', '24394.png', '10410.png', '00739.png', '22492.png', '11083.png', '19783.png', '09466.png', '06604.png', '20630.png', '23360.png', '04690.png', '22136.png', '18881.png', '05280.png', '06483.png', '04301.png', '11421.png', '01940.png', '21514.png', '10896.png', '01409.png', '00801.png', '24462.png', '03357.png', '07424.png', '06435.png', '04130.png', '11322.png', '09022.png', '14674.png', '21155.png', '21625.png', '08271.png', '04605.png', '04676.png', '07998.png', '23196.png', '19589.png', '19480.png', '17707.png', '06881.png', '07795.png', '20020.png', '05840.png', '10626.png', '18469.png', '13313.png', '13555.png', '10079.png', '17595.png', '08608.png', '00659.png', '12343.png', '09692.png', '15678.png', '21300.png', '23720.png', '05872.png', '00278.png', '02054.png', '04445.png', '02265.png', '05201.png', '20078.png', '12574.png', '00468.png', '08171.png', '10961.png', '07267.png', '16216.png', '07625.png', '02136.png', '16634.png', '10775.png', '11376.png', '13637.png', '05932.png', '19191.png', '02556.png', '17114.png', '16955.png', '12982.png', '16936.png', '17626.png', '23802.png', '18553.png', '09685.png', '23571.png', '24876.png', '17435.png', '06776.png', '19313.png', '23052.png', '01565.png', '24318.png', '05807.png', '11097.png', '16607.png', '18061.png', '19397.png', '11591.png', '19387.png', '04859.png', '09240.png', '08364.png', '16871.png', '22393.png', '16720.png', '22261.png', '09366.png', '15236.png', '18004.png', '12902.png', '14383.png', '07632.png', '01460.png', '15770.png', '22251.png', '01325.png', '03929.png', '04734.png', '24169.png', '23869.png', '03979.png', '07716.png', '05042.png', '00253.png', '07842.png', '04074.png', '06731.png', '06112.png', '09856.png', '21934.png', '17788.png', '13318.png', '14758.png', '03237.png', '24131.png', '17443.png', '23040.png', '01373.png', '09636.png', '21046.png', '09758.png', '24172.png', '09808.png', '02861.png', '14588.png', '16166.png', '22513.png', '19805.png', '06845.png', '05288.png', '13186.png', '20695.png', '02646.png', '08146.png', '02645.png', '18343.png', '05211.png', '02099.png', '16022.png', '14753.png', '23867.png', '13626.png', '23261.png', '20265.png', '18075.png', '05438.png', '24811.png', '00084.png', '24765.png', '21195.png', '21225.png', '24676.png', '09517.png', '20926.png', '11778.png', '24659.png', '02213.png', '19219.png', '07415.png', '09036.png', '24513.png', '11564.png', '19342.png', '06067.png', '14335.png', '00727.png', '15521.png', '03404.png', '15758.png', '17864.png', '20177.png', '07494.png', '22941.png', '16639.png', '09864.png', '17960.png', '16220.png', '19404.png', '09855.png', '11284.png', '05161.png', '12435.png', '19614.png', '04938.png', '13453.png', '03186.png', '09014.png', '09831.png', '23572.png', '07251.png', '02571.png', '16741.png', '18944.png', '22010.png', '09163.png', '04022.png', '06764.png', '17915.png', '16825.png', '00325.png', '09033.png', '20235.png', '21894.png', '04238.png', '09745.png', '20735.png', '23620.png', '16548.png', '21499.png', '19620.png', '13275.png', '23749.png', '02532.png', '24378.png', '23443.png', '23000.png', '02711.png', '18235.png', '00548.png', '12390.png', '12885.png', '23072.png', '09661.png', '11064.png', '02733.png', '08227.png', '03639.png', '05702.png', '02809.png', '16982.png', '06847.png', '05944.png', '13200.png', '21773.png', '09377.png', '05207.png', '15109.png', '19801.png', '09970.png', '04903.png', '03900.png', '08669.png', '02345.png', '14932.png', '08025.png', '18681.png', '17190.png', '14253.png', '17561.png', '15823.png', '02606.png', '04408.png', '08984.png', '13731.png', '12422.png', '17680.png', '09375.png', '21656.png', '15071.png', '11218.png', '18506.png', '24691.png', '12766.png', '22164.png', '15774.png', '08042.png', '19015.png', '02145.png', '09920.png', '12357.png', '07460.png', '23481.png', '01475.png', '18351.png', '05655.png', '15247.png', '07461.png', '16085.png', '19622.png', '22461.png', '21188.png', '02810.png', '17060.png', '20449.png', '21692.png', '17165.png', '02515.png', '14392.png', '03572.png', '06418.png', '16679.png', '21396.png', '02396.png', '16325.png', '01789.png', '18640.png', '06419.png', '20640.png', '23256.png', '23208.png', '24031.png', '21411.png', '05933.png', '05185.png', '15862.png', '08457.png', '00060.png', '00758.png', '07226.png', '23117.png', '20848.png', '06577.png', '20663.png', '21200.png', '14461.png', '00401.png', '13972.png', '21788.png', '07504.png', '00770.png', '04693.png', '23296.png', '18374.png', '16190.png', '18915.png', '06666.png', '17197.png', '18775.png', '24913.png', '15067.png', '22082.png', '14898.png', '08584.png', '10685.png', '24283.png', '13259.png', '09727.png', '16312.png', '03546.png', '08654.png', '03598.png', '22835.png', '12169.png', '03630.png', '15668.png', '07278.png', '17906.png', '00457.png', '24925.png', '10259.png', '15271.png', '18970.png', '04930.png', '06811.png', '07320.png', '16728.png', '07892.png', '11060.png', '20241.png', '07120.png', '12094.png', '24600.png', '17693.png', '11481.png', '20587.png', '20280.png', '23347.png', '01328.png', '08299.png', '02404.png', '11824.png', '22572.png', '05260.png', '00012.png', '23855.png', '02095.png', '24154.png', '17451.png', '08796.png', '19321.png', '18583.png', '10065.png', '02799.png', '24465.png', '05050.png', '18325.png', '16458.png', '08129.png', '10774.png', '20452.png', '19238.png', '23972.png', '13304.png', '11106.png', '10208.png', '11168.png', '22906.png', '03718.png', '22990.png', '11125.png', '20584.png', '16897.png', '01090.png', '18916.png', '09828.png', '18382.png', '18091.png', '08183.png', '04353.png', '21235.png', '05826.png', '08738.png', '10946.png', '23006.png', '22000.png', '06463.png', '05184.png', '07271.png', '07127.png', '02437.png', '11784.png', '05986.png', '19173.png', '08725.png', '04839.png', '05434.png', '15051.png', '13296.png', '07198.png', '13282.png', '17809.png', '22655.png', '05669.png', '11568.png', '21858.png', '06120.png', '22859.png', '19540.png', '01337.png', '05583.png', '13751.png', '10790.png', '04000.png', '19922.png', '16903.png', '10348.png', '08539.png', '04512.png', '10269.png', '21582.png', '00353.png', '05412.png', '17042.png', '06413.png', '22489.png', '08062.png', '01385.png', '21853.png', '00511.png', '05881.png', '02650.png', '18743.png', '02931.png', '06692.png', '04361.png', '11463.png', '08717.png', '12313.png', '09503.png', '10645.png', '10060.png', '06082.png', '21821.png', '00150.png', '11078.png', '15084.png', '18874.png', '05277.png', '14641.png', '10313.png', '05999.png', '03962.png', '15386.png', '01775.png', '04583.png', '10148.png', '08063.png', '01722.png', '12214.png', '11676.png', '23491.png', '01941.png', '23191.png', '13966.png', '09446.png', '15802.png', '04456.png', '04348.png', '01001.png', '08935.png', '17325.png', '24482.png', '11799.png', '10471.png', '01226.png', '13872.png', '07955.png', '09557.png', '01183.png', '04640.png', '15888.png', '19282.png', '01150.png', '16168.png', '23519.png', '16619.png', '06595.png', '14625.png', '13480.png', '10399.png', '01326.png', '01004.png', '06816.png', '18518.png', '19594.png', '14189.png', '21084.png', '07314.png', '22599.png', '09781.png', '21891.png', '24047.png', '15147.png', '17787.png', '15025.png', '15639.png', '16437.png', '23204.png', '12732.png', '15466.png', '23599.png', '12257.png', '18302.png', '13047.png', '17997.png', '07079.png', '14633.png', '06484.png', '15475.png', '14452.png', '10764.png', '20490.png', '15237.png', '13856.png', '19990.png', '03375.png', '18313.png', '21615.png', '16340.png', '07070.png', '24800.png', '16990.png', '13498.png', '13619.png', '15645.png', '00370.png', '16531.png', '21473.png', '09844.png', '19431.png', '15356.png', '06003.png', '01181.png', '24384.png', '16646.png', '06394.png', '11203.png', '11269.png', '13155.png', '00236.png', '21085.png', '09256.png', '04203.png', '19123.png', '16536.png', '12697.png', '01586.png', '10224.png', '06962.png', '01578.png', '15860.png', '20716.png', '12491.png', '20104.png', '08909.png', '18849.png', '06472.png', '20457.png', '15757.png', '17260.png', '22562.png', '10751.png', '03221.png', '13019.png', '14794.png', '20131.png', '12561.png', '24136.png', '00892.png', '21518.png', '20845.png', '18658.png', '19357.png', '11369.png', '18170.png', '17024.png', '05593.png', '01377.png', '20730.png', '11752.png', '15781.png', '05284.png', '05712.png', '05283.png', '00709.png', '09872.png', '05776.png', '11935.png', '12772.png', '09520.png', '13569.png', '00541.png', '17202.png', '10055.png', '03482.png', '15211.png', '06641.png', '08565.png', '02492.png', '20955.png', '17368.png', '00074.png', '20009.png', '22865.png', '07028.png', '23109.png', '21849.png', '14254.png', '18883.png', '17739.png', '08050.png', '00706.png', '15319.png', '22391.png', '03177.png', '05250.png', '07178.png', '12706.png', '14683.png', '23950.png', '14780.png', '18417.png', '11435.png', '12031.png', '05639.png', '05737.png', '16394.png', '17382.png', '04777.png', '23380.png', '11388.png', '21161.png', '00848.png', '03530.png', '23760.png', '00054.png', '12304.png', '02329.png', '12009.png', '12473.png', '06338.png', '07230.png', '02875.png', '17988.png', '16766.png', '18433.png', '22691.png', '18570.png', '15597.png', '15552.png', '17446.png', '14526.png', '15748.png', '14885.png', '03767.png', '06865.png', '00437.png', '15825.png', '12569.png', '13348.png', '09053.png', '21681.png', '19298.png', '04366.png', '17800.png', '14685.png', '07352.png', '06013.png', '14964.png', '17471.png', '13695.png', '10967.png', '18197.png', '09104.png', '24020.png', '05065.png', '24615.png', '02512.png', '01466.png', '10408.png', '14308.png', '15968.png', '07412.png', '17061.png', '14399.png', '10497.png', '08936.png', '10362.png', '08733.png', '15495.png', '06672.png', '12030.png', '11416.png', '11390.png', '02366.png', '19703.png', '04188.png', '16329.png', '11734.png', '01464.png', '14223.png', '04779.png', '14167.png', '17364.png', '00704.png', '22001.png', '11561.png', '13217.png', '05270.png', '01529.png', '03995.png', '04612.png', '17362.png', '24612.png', '14509.png', '15485.png', '10638.png', '16103.png', '03461.png', '13616.png', '17618.png', '18876.png', '13024.png', '07659.png', '15149.png', '06080.png', '11763.png', '24551.png', '02422.png', '09850.png', '10142.png', '20943.png', '15257.png', '19195.png', '15709.png', '14125.png', '10325.png', '07531.png', '11972.png', '23284.png', '18902.png', '10562.png', '02500.png', '08906.png', '20368.png', '03024.png', '03012.png', '05172.png', '22845.png', '20710.png', '00280.png', '16648.png', '03175.png', '11464.png', '00867.png', '09031.png', '03828.png', '13507.png', '14362.png', '08757.png', '00022.png', '09988.png', '12281.png', '18968.png', '20189.png', '17064.png', '19947.png', '21401.png', '21316.png', '09841.png', '17843.png', '16923.png', '05151.png', '18288.png', '01712.png', '00096.png', '03334.png', '16694.png', '11475.png', '24090.png', '05668.png', '08527.png', '09121.png', '19893.png', '00129.png', '04757.png', '04679.png', '08380.png', '23891.png', '23427.png', '07590.png', '04187.png', '05725.png', '12190.png', '23086.png', '22098.png', '02585.png', '16931.png', '23282.png', '20699.png', '19327.png', '15267.png', '07257.png', '11760.png', '09035.png', '02780.png', '10275.png', '08475.png', '02789.png', '21576.png', '06202.png', '19350.png', '07996.png', '22090.png', '20014.png', '07040.png', '18095.png', '05955.png', '10365.png', '17276.png', '22952.png', '23919.png', '24396.png', '24531.png', '13384.png', '09582.png', '10699.png', '20233.png', '19243.png', '10179.png', '18853.png', '11367.png', '03728.png', '07844.png', '12475.png', '12825.png', '12238.png', '01590.png', '12029.png', '21403.png', '07533.png', '06385.png', '18831.png', '10232.png', '20872.png', '17926.png', '19477.png', '22944.png', '00176.png', '21020.png', '04437.png', '15743.png', '16608.png', '17574.png', '02050.png', '10999.png', '04043.png', '04657.png', '18232.png', '04661.png', '17480.png', '12185.png', '05120.png', '21682.png', '19283.png', '24427.png', '01105.png', '05773.png', '18354.png', '02984.png', '14924.png', '09783.png', '21244.png', '24345.png', '05036.png', '05570.png', '06784.png', '01194.png', '21037.png', '18525.png', '08484.png', '02277.png', '16333.png', '19535.png', '17975.png', '23399.png', '20122.png', '15169.png', '18320.png', '15114.png', '24429.png', '05834.png', '23575.png', '07086.png', '03424.png', '21899.png', '14196.png', '12008.png', '09133.png', '01011.png', '00463.png', '00455.png', '17477.png', '08819.png', '14325.png', '10518.png', '18146.png', '18993.png', '05827.png', '05319.png', '02481.png', '11102.png', '21288.png', '04965.png', '24056.png', '04996.png', '20567.png', '17371.png', '13356.png', '03902.png', '10299.png', '04240.png', '16086.png', '05340.png', '05550.png', '13901.png', '05673.png', '14323.png', '08994.png', '22055.png', '22874.png', '03848.png', '20919.png', '13103.png', '11298.png', '09481.png', '19526.png', '22451.png', '10832.png', '06787.png', '00496.png', '00330.png', '20300.png', '17658.png', '18624.png', '14237.png', '00900.png', '24528.png', '16642.png', '07857.png', '19409.png', '11066.png', '17562.png', '20260.png', '03443.png', '15558.png', '00488.png', '14456.png', '07683.png', '09324.png', '17102.png', '02125.png', '11790.png', '15589.png', '10857.png', '23926.png', '14170.png', '00433.png', '12299.png', '16413.png', '18388.png', '15425.png', '01682.png', '00999.png', '22916.png', '10236.png', '24819.png', '13053.png', '14890.png', '21940.png', '23771.png', '04415.png', '06947.png', '16175.png', '21822.png', '02626.png', '12376.png', '23387.png', '11391.png', '08810.png', '02400.png', '07321.png', '04980.png', '18786.png', '16234.png', '17246.png', '14285.png', '12277.png', '01209.png', '01161.png', '12762.png', '00023.png', '10475.png', '13596.png', '08643.png', '00229.png', '02901.png', '04869.png', '23759.png', '22675.png', '01184.png', '22778.png', '20514.png', '06805.png', '20783.png', '02324.png', '02181.png', '08332.png', '09023.png', '21770.png', '06867.png', '20118.png', '20829.png', '02038.png', '01341.png', '14908.png', '09987.png', '12575.png', '10651.png', '12489.png', '21733.png', '09672.png', '03805.png', '23147.png', '13273.png', '02042.png', '12170.png', '20236.png', '05817.png', '20062.png', '09078.png', '02572.png', '08666.png', '12626.png', '00388.png', '13611.png', '20975.png', '16518.png', '14292.png', '02201.png', '18480.png', '17225.png', '12521.png', '03072.png', '05106.png', '18377.png', '18762.png', '17723.png', '23718.png', '12146.png', '16954.png', '22766.png', '04962.png', '09272.png', '12625.png', '02566.png', '05879.png', '11323.png', '04498.png', '17962.png', '08736.png', '16891.png', '07870.png', '17189.png', '20894.png', '09082.png', '19420.png', '13369.png', '06489.png', '20852.png', '24538.png', '17725.png', '09793.png', '03233.png', '24680.png', '22866.png', '23426.png', '09492.png', '10616.png', '18851.png', '12223.png', '21192.png', '12239.png', '17584.png', '23725.png', '06359.png', '03788.png', '02382.png', '17576.png', '24914.png', '24208.png', '19574.png', '11695.png', '19332.png', '24714.png', '11381.png', '02618.png', '16143.png', '09817.png', '21078.png', '24065.png', '05734.png', '15489.png', '00354.png', '13539.png', '15796.png', '06695.png', '22346.png', '00705.png', '03155.png', '22284.png', '02167.png', '03654.png', '09624.png', '02278.png', '20589.png', '04823.png', '24523.png', '10285.png', '13738.png', '23732.png', '19998.png', '17393.png', '10542.png', '01896.png', '17696.png', '08269.png', '08168.png', '16304.png', '01135.png', '05119.png', '10914.png', '05343.png', '24908.png', '22504.png', '14663.png', '05021.png', '13487.png', '09224.png', '21171.png', '08337.png', '09179.png', '17748.png', '01137.png', '17406.png', '18213.png', '05784.png', '13389.png', '11194.png', '07056.png', '07301.png', '02589.png', '02911.png', '12161.png', '03137.png', '00352.png', '06464.png', '23079.png', '22643.png', '21753.png', '02161.png', '03547.png', '05922.png', '23515.png', '16225.png', '21519.png', '23001.png', '23127.png', '11811.png', '07071.png', '10390.png', '06548.png', '14376.png', '11088.png', '15972.png', '03961.png', '05870.png', '08493.png', '00930.png', '20446.png', '13701.png', '21717.png', '12572.png', '08252.png', '14560.png', '19398.png', '10603.png', '14250.png', '18496.png', '07715.png', '17850.png', '10537.png', '21948.png', '24561.png', '12454.png', '14487.png', '20305.png', '20636.png', '10997.png', '09649.png', '16777.png', '08735.png', '16915.png', '08071.png', '05096.png', '02452.png', '22754.png', '02263.png', '05080.png', '24924.png', '07177.png', '23659.png', '16058.png', '02530.png', '19109.png', '06684.png', '05971.png', '22139.png', '07116.png', '14761.png', '23210.png', '10168.png', '07670.png', '04352.png', '12292.png', '01327.png', '12555.png', '04052.png', '04039.png', '01179.png', '12347.png', '02504.png', '17697.png', '22508.png', '08627.png', '03113.png', '02096.png', '07001.png', '01724.png', '04065.png', '20002.png', '22296.png', '16948.png', '21149.png', '21643.png', '15202.png', '10830.png', '17313.png', '13685.png', '19944.png', '05689.png', '00911.png', '13002.png', '03083.png', '04334.png', '16802.png', '09504.png', '10760.png', '14145.png', '07562.png', '06751.png', '13269.png', '06969.png', '06734.png', '12046.png', '16192.png', '00498.png', '15107.png', '14880.png', '04233.png', '21715.png', '01808.png', '17448.png', '11842.png', '17238.png', '02273.png', '09277.png', '12933.png', '08111.png', '02022.png', '23844.png', '09389.png', '08612.png', '23831.png', '15029.png', '13178.png', '00358.png', '17560.png', '17590.png', '13768.png', '24801.png', '22149.png', '11007.png', '02991.png', '13427.png', '06217.png', '05684.png', '11791.png', '05223.png', '18823.png', '05616.png', '16547.png', '06972.png', '10320.png', '13148.png', '21281.png', '12074.png', '12750.png', '12049.png', '17125.png', '21148.png', '16861.png', '00919.png', '15118.png', '20335.png', '20359.png', '18029.png', '22140.png', '12727.png', '02063.png', '13336.png', '02150.png', '11929.png', '06395.png', '01830.png', '03063.png', '21368.png', '00171.png', '24234.png', '19842.png', '14261.png', '24224.png', '15263.png', '24744.png', '19785.png', '16277.png', '15320.png', '07970.png', '11293.png', '05286.png', '10991.png', '00041.png', '00235.png', '06423.png', '10947.png', '15179.png', '24768.png', '03031.png', '23817.png', '03393.png', '00678.png', '07865.png', '12992.png', '06926.png', '17503.png', '14437.png', '06961.png', '24838.png', '21789.png', '09094.png', '04006.png', '00279.png', '17691.png', '21454.png', '17760.png', '15230.png', '03670.png', '20645.png', '09294.png', '24163.png', '08719.png', '05498.png', '17622.png', '17823.png', '21683.png', '12780.png', '00532.png', '09566.png', '09609.png', '16795.png', '07783.png', '08132.png', '13802.png', '01961.png', '19604.png', '11308.png', '01368.png', '01773.png', '12377.png', '10084.png', '17303.png', '20971.png', '10082.png', '00871.png', '06017.png', '12332.png', '10711.png', '01914.png', '11004.png', '24377.png', '09794.png', '18284.png', '04038.png', '11092.png', '01515.png', '03636.png', '02147.png', '21784.png', '09498.png', '13326.png', '15361.png', '09796.png', '13795.png', '16957.png', '15062.png', '12968.png', '02108.png', '11219.png', '03429.png', '05758.png', '21923.png', '19237.png', '04248.png', '23345.png', '12120.png', '24413.png', '00902.png', '04556.png', '17821.png', '04925.png', '23638.png', '15233.png', '24059.png', '22230.png', '12221.png', '14843.png', '01556.png', '01976.png', '21721.png', '04182.png', '01755.png', '00948.png', '00148.png', '14991.png', '13558.png', '10931.png', '07124.png', '22394.png', '11249.png', '07174.png', '00594.png', '12769.png', '23368.png', '21127.png', '09364.png', '08598.png', '10861.png', '07538.png', '17172.png', '08282.png', '23842.png', '07231.png', '19855.png', '14716.png', '03579.png', '11910.png', '06288.png', '23658.png', '10964.png', '10130.png', '13095.png', '11585.png', '20117.png', '04016.png', '03037.png', '23921.png', '03844.png', '03923.png', '17847.png', '08087.png', '00241.png', '20156.png', '15455.png', '05088.png', '21077.png', '12010.png', '23709.png', '16652.png', '08822.png', '09362.png', '14510.png', '21921.png', '07811.png', '06460.png', '00914.png', '06519.png', '17407.png', '21956.png', '24156.png', '00589.png', '11011.png', '18805.png', '09882.png', '15691.png', '23939.png', '22387.png', '03862.png', '11933.png', '03154.png', '03118.png', '00371.png', '08436.png', '02543.png', '14591.png', '07862.png', '24574.png', '17152.png', '22765.png', '21957.png', '17935.png', '14432.png', '02375.png', '02704.png', '06064.png', '20047.png', '14100.png', '02790.png', '02608.png', '19688.png', '12554.png', '10456.png', '12451.png', '10257.png', '16243.png', '24460.png', '19071.png', '15218.png', '20237.png', '11588.png', '19112.png', '10733.png', '08753.png', '06893.png', '24926.png', '12975.png', '06872.png', '20498.png', '06834.png', '01500.png', '15014.png', '17300.png', '15052.png', '19898.png', '13794.png', '17148.png', '14051.png', '22986.png', '15410.png', '10245.png', '05690.png', '17894.png', '08134.png', '06779.png', '24401.png', '03991.png', '13784.png', '20422.png', '04263.png', '03937.png', '20271.png', '17119.png', '07093.png', '07791.png', '20951.png', '02487.png', '19143.png', '01831.png', '09280.png', '03736.png', '05387.png', '00214.png', '01088.png', '01963.png', '04335.png', '06403.png', '06576.png', '20685.png', '03261.png', '12424.png', '14318.png', '20990.png', '22876.png', '21213.png', '23214.png', '06324.png', '15265.png', '14976.png', '23048.png', '05736.png', '13886.png', '15461.png', '11003.png', '19860.png', '21278.png', '22797.png', '23675.png', '09147.png', '14813.png', '01771.png', '04763.png', '04960.png', '14861.png', '07699.png', '00555.png', '12249.png', '15253.png', '17213.png', '02497.png', '10059.png', '05532.png', '04297.png', '23560.png', '10511.png', '09263.png', '04922.png', '11688.png', '07747.png', '10787.png', '01813.png', '22325.png', '17020.png', '06290.png', '23773.png', '03545.png', '05238.png', '22196.png', '23372.png', '22731.png', '05506.png', '22815.png', '17116.png', '12788.png', '13527.png', '24053.png', '22466.png', '09827.png', '00808.png', '08387.png', '19541.png', '22183.png', '08925.png', '06479.png', '20997.png', '18661.png', '08958.png', '02056.png', '01420.png', '11305.png', '00107.png', '00994.png', '22592.png', '05395.png', '08623.png', '14280.png', '01641.png', '11691.png', '00495.png', '11094.png', '01038.png', '17581.png', '13642.png', '08495.png', '21009.png', '01046.png', '17758.png', '22133.png', '10614.png', '18326.png', '18344.png', '12324.png', '15022.png', '05390.png', '13008.png', '17629.png', '03566.png', '11934.png', '12951.png', '12289.png', '24940.png', '20043.png', '00122.png', '03314.png', '23100.png', '24457.png', '19452.png', '21930.png', '09519.png', '09211.png', '03318.png', '21294.png', '24856.png', '09356.png', '21746.png', '07805.png', '04993.png', '17981.png', '02451.png', '20167.png', '17921.png', '11915.png', '05221.png', '18027.png', '12688.png', '17141.png', '22434.png', '01255.png', '12595.png', '23634.png', '22998.png', '20531.png', '24775.png', '21641.png', '14615.png', '01833.png', '12591.png', '05244.png', '06868.png', '12998.png', '22606.png', '23203.png', '03290.png', '12246.png', '10489.png', '17070.png', '08741.png', '19754.png', '02897.png', '02310.png', '19054.png', '22429.png', '05978.png', '15106.png', '05208.png', '18812.png', '11509.png', '03664.png', '11289.png', '24902.png', '16807.png', '03694.png', '16587.png', '05248.png', '13238.png', '18908.png', '18574.png', '13154.png', '17883.png', '16582.png', '04820.png', '05372.png', '04516.png', '01845.png', '19392.png', '17737.png', '17668.png', '10314.png', '23396.png', '24628.png', '22044.png', '05880.png', '01923.png', '14569.png', '11897.png', '20338.png', '12113.png', '01012.png', '17878.png', '06019.png', '13258.png', '12907.png', '08589.png', '18629.png', '16125.png', '10173.png', '15122.png', '00514.png', '10341.png', '03372.png', '22045.png', '17531.png', '21583.png', '08610.png', '09543.png', '22165.png', '20750.png', '17153.png', '20508.png', '04145.png', '01383.png', '10637.png', '21173.png', '12278.png', '06938.png', '08163.png', '19935.png', '07915.png', '15817.png', '15979.png', '24495.png', '24101.png', '08131.png', '01674.png', '09259.png', '13949.png', '03646.png', '23822.png', '22156.png', '16799.png', '07281.png', '15040.png', '23617.png', '16095.png', '23742.png', '00766.png', '21480.png', '15502.png', '11925.png', '09552.png', '13923.png', '13110.png', '00715.png', '24145.png', '00119.png', '24683.png', '17287.png', '22659.png', '24363.png', '23945.png', '23201.png', '22676.png', '04759.png', '24532.png', '04110.png', '09501.png', '09746.png', '12105.png', '19811.png', '05775.png', '11193.png', '10014.png', '13614.png', '10211.png', '10898.png', '11898.png', '08184.png', '03004.png', '13081.png', '08697.png', '03804.png', '15809.png', '21258.png', '19792.png', '23897.png', '04393.png', '07737.png', '18144.png', '24747.png', '00618.png', '13210.png', '03620.png', '18730.png', '20899.png', '01144.png', '14175.png', '03199.png', '13404.png', '01172.png', '08391.png', '19061.png', '11020.png', '03243.png', '17095.png', '10504.png', '15939.png', '21151.png', '17441.png', '07686.png', '06688.png', '14938.png', '23696.png', '21937.png', '17617.png', '15822.png', '14479.png', '11745.png', '09763.png', '01871.png', '08710.png', '05043.png', '19961.png', '15618.png', '01719.png', '10917.png', '20482.png', '11754.png', '03521.png', '09065.png', '19359.png', '02485.png', '01542.png', '23495.png', '04369.png', '10587.png', '00835.png', '15682.png', '16664.png', '06727.png', '04116.png', '10221.png', '18083.png', '14788.png', '02925.png', '17749.png', '09542.png', '16965.png', '09555.png', '07358.png', '16748.png', '13533.png', '09410.png', '06440.png', '12612.png', '12527.png', '11437.png', '07345.png', '15123.png', '10589.png', '02459.png', '14727.png', '12212.png', '01550.png', '05602.png', '24271.png', '10122.png', '24943.png', '20125.png', '02881.png', '22479.png', '07034.png', '01780.png', '03507.png', '23583.png', '20870.png', '17089.png', '07969.png', '04242.png', '08852.png', '04235.png', '00880.png', '15686.png', '22742.png', '10138.png', '05662.png', '11183.png', '06057.png', '18699.png', '01686.png', '20178.png', '14400.png', '13287.png', '05418.png', '09464.png', '05750.png', '00372.png', '21758.png', '01694.png', '04796.png', '04505.png', '03893.png', '22862.png', '19136.png', '09701.png', '14781.png', '02102.png', '09755.png', '03383.png', '08362.png', '04667.png', '12076.png', '13126.png', '19119.png', '19296.png', '07966.png', '24606.png', '11894.png', '05422.png', '20238.png', '15423.png', '24505.png', '17279.png', '02517.png', '24512.png', '00504.png', '19268.png', '23155.png', '03734.png', '08505.png', '04613.png', '07684.png', '16750.png', '18552.png', '16354.png', '11262.png', '22367.png', '05967.png', '14923.png', '10346.png', '07205.png', '01613.png', '22507.png', '13295.png', '24017.png', '13529.png', '08170.png', '18500.png', '05803.png', '08655.png', '11823.png', '07939.png', '19134.png', '01834.png', '11628.png', '13554.png', '09762.png', '15344.png', '12255.png', '09219.png', '15510.png', '11212.png', '02903.png', '16077.png', '06052.png', '01815.png', '08494.png', '18038.png', '12234.png', '15777.png', '02270.png', '06256.png', '14349.png', '15091.png', '10507.png', '14987.png', '18208.png', '24042.png', '11453.png', '06058.png', '03248.png', '17919.png', '04778.png', '16739.png', '01241.png', '23604.png', '20090.png', '07250.png', '11846.png', '21581.png', '16996.png', '14668.png', '07456.png', '05751.png', '00886.png', '19272.png', '06858.png', '19868.png', '16595.png', '23751.png', '03712.png', '01505.png', '14941.png', '04082.png', '21593.png', '17497.png', '10635.png', '17032.png', '17258.png', '10090.png', '08945.png', '14499.png', '09380.png', '19455.png', '06127.png', '16062.png', '07607.png', '13693.png', '17539.png', '05509.png', '21967.png', '09881.png', '01561.png', '05918.png', '09761.png', '09361.png', '23467.png', '02787.png', '20507.png', '16115.png', '07291.png', '05143.png', '03927.png', '05361.png', '07669.png', '03067.png', '11622.png', '13370.png', '08535.png', '17052.png', '22511.png', '07961.png', '14139.png', '15700.png', '22922.png', '24420.png', '22345.png', '24792.png', '06854.png', '09536.png', '09157.png', '20450.png', '11653.png', '18899.png', '07947.png', '16701.png', '23115.png', '08817.png', '19335.png', '02896.png', '23955.png', '15438.png', '17233.png', '03922.png', '18267.png', '22850.png', '08544.png', '18748.png', '02057.png', '21082.png', '23605.png', '04595.png', '14158.png', '20563.png', '01903.png', '07647.png', '05379.png', '19782.png', '13838.png', '00272.png', '20500.png', '02305.png', '20360.png', '05647.png', '02885.png', '17585.png', '19208.png', '01660.png', '19317.png', '11324.png', '24326.png', '04113.png', '23476.png', '21707.png', '14074.png', '12634.png', '06987.png', '15945.png', '20143.png', '23264.png', '04021.png', '16259.png', '11546.png', '04471.png', '02352.png', '21794.png', '20977.png', '07082.png', '00753.png', '23884.png', '03873.png', '01540.png', '10624.png', '06549.png', '05720.png', '24194.png', '06329.png', '17871.png', '03274.png', '20468.png', '16911.png', '22782.png', '15724.png', '04619.png', '14533.png', '23104.png', '08864.png', '05566.png', '21580.png', '06996.png', '22468.png', '12582.png', '17753.png', '20896.png', '15837.png', '15094.png', '14742.png', '19809.png', '19973.png', '15373.png', '22322.png', '05951.png', '00037.png', '18069.png', '09487.png', '07658.png', '03260.png', '14298.png', '04799.png', '14136.png', '22573.png', '14270.png', '05524.png', '06446.png', '23538.png', '17326.png', '03750.png', '04118.png', '08637.png', '15158.png', '05483.png', '19730.png', '06184.png', '21375.png', '08199.png', '22823.png', '11881.png', '04751.png', '15669.png', '08357.png', '08846.png', '14833.png', '06265.png', '11432.png', '11095.png', '02821.png', '19816.png', '03075.png', '02319.png', '06461.png', '01967.png', '10423.png', '21932.png', '02743.png', '13780.png', '24733.png', '11185.png', '20542.png', '07298.png', '10531.png', '21119.png', '11966.png', '19034.png', '19747.png', '20823.png', '15157.png', '15731.png', '15576.png', '17235.png', '07652.png', '14916.png', '07497.png', '00837.png', '16473.png', '24555.png', '22338.png', '04416.png', '09995.png', '02113.png', '02468.png', '19147.png', '08557.png', '03458.png', '18251.png', '12803.png', '16411.png', '03263.png', '14208.png', '17624.png', '20123.png', '08960.png', '15843.png', '03172.png', '06354.png', '12929.png', '23826.png', '12681.png', '20967.png', '18996.png', '11175.png', '08503.png', '15547.png', '12867.png', '20303.png', '20874.png', '22264.png', '14835.png', '16914.png', '20631.png', '21884.png', '14755.png', '08652.png', '19466.png', '20715.png', '23177.png', '24938.png', '15297.png', '06923.png', '20661.png', '16737.png', '04035.png', '12322.png', '23222.png', '23531.png', '02312.png', '04042.png', '12630.png', '24232.png', '09897.png', '22240.png', '24887.png', '17701.png', '22776.png', '18435.png', '11492.png', '19230.png', '01849.png', '21311.png', '12275.png', '08250.png', '10773.png', '06007.png', '07472.png', '15480.png', '08837.png', '17632.png', '20522.png', '17656.png', '20109.png', '17642.png', '11960.png', '20316.png', '22435.png', '07441.png', '17519.png', '06102.png', '01167.png', '15314.png', '17252.png', '13606.png', '07956.png', '17167.png', '01353.png', '00622.png', '08551.png', '08463.png', '00540.png', '06752.png', '08532.png', '06304.png', '12715.png', '03789.png', '16656.png', '15563.png', '20635.png', '01547.png', '24661.png', '07054.png', '18705.png', '07515.png', '07617.png', '22313.png', '23401.png', '10965.png', '12613.png', '11174.png', '11184.png', '03699.png', '09004.png', '14772.png', '13082.png', '17704.png', '17145.png', '04295.png', '18018.png', '03167.png', '16521.png', '02389.png', '20962.png', '07885.png', '22042.png', '16477.png', '07108.png', '14684.png', '07799.png', '12252.png', '18426.png', '08873.png', '06942.png', '15097.png', '00525.png', '21840.png', '23727.png', '10203.png', '24212.png', '12462.png', '03295.png', '07022.png', '24727.png', '15786.png', '14630.png', '22847.png', '21387.png', '19058.png', '05147.png', '21648.png', '06233.png', '12060.png', '03041.png', '09883.png', '13262.png', '13083.png', '17376.png', '10892.png', '19103.png', '04833.png', '12403.png', '21108.png', '18071.png', '05254.png', '14673.png', '21024.png', '16482.png', '23447.png', '08913.png', '18277.png', '24024.png', '20877.png', '19905.png', '19257.png', '16271.png', '04501.png', '06612.png', '21090.png', '14188.png', '07965.png', '22379.png', '06093.png', '05757.png', '12533.png', '11999.png', '14724.png', '18720.png', '10755.png', '04050.png', '20868.png', '18088.png', '05579.png', '01393.png', '24264.png', '03136.png', '18564.png', '05449.png', '24398.png', '10474.png', '18780.png', '08210.png', '07500.png', '01707.png', '00046.png', '18983.png', '09900.png', '24877.png', '06988.png', '18887.png', '15909.png', '13664.png', '13834.png', '01329.png', '17054.png', '18369.png', '23461.png', '15741.png', '22666.png', '05527.png', '03240.png', '23666.png', '17374.png', '11833.png', '15834.png', '02831.png', '11136.png', '20282.png', '19138.png', '14613.png', '02847.png', '20622.png', '14181.png', '22157.png', '14505.png', '20392.png', '12562.png', '17155.png', '04776.png', '21038.png', '16493.png', '08490.png', '23965.png', '00285.png', '23306.png', '16505.png', '20433.png', '17098.png', '23721.png', '05642.png', '15674.png', '20001.png', '06945.png', '23497.png', '16649.png', '16065.png', '10570.png', '22523.png', '01759.png', '09967.png', '03610.png', '19053.png', '03811.png', '09110.png', '19175.png', '17550.png', '23643.png', '10151.png', '16916.png', '02182.png', '06210.png', '14691.png', '19879.png', '05009.png', '07009.png', '24049.png', '12335.png', '07308.png', '13274.png', '12433.png', '20992.png', '04200.png', '24915.png', '17950.png', '07509.png', '17932.png', '01240.png', '23192.png', '12893.png', '14433.png', '03499.png', '18596.png', '20612.png', '22651.png', '06167.png', '08344.png', '01690.png', '21510.png', '09723.png', '03312.png', '06718.png', '18704.png', '11755.png', '21834.png', '20964.png', '03010.png', '01723.png', '19207.png', '05299.png', '15181.png', '21060.png', '19131.png', '18987.png', '15974.png', '20442.png', '19859.png', '06975.png', '19646.png', '09465.png', '20901.png', '20098.png', '21071.png', '07357.png', '02296.png', '13561.png', '02995.png', '01262.png', '07402.png', '21952.png', '02818.png', '12325.png', '12149.png', '19027.png', '13850.png', '10502.png', '15696.png', '12794.png', '12395.png', '20604.png', '09223.png', '03377.png', '08211.png', '06086.png', '00391.png', '21845.png', '05930.png', '20692.png', '09590.png', '19845.png', '05850.png', '23487.png', '12237.png', '02920.png', '13741.png', '18798.png', '13115.png', '05675.png', '08479.png', '08816.png', '01552.png', '04211.png', '17565.png', '01737.png', '23576.png', '19886.png', '04888.png', '04815.png', '15214.png', '20091.png', '03421.png', '11800.png', '03602.png', '11651.png', '09091.png', '23699.png', '00788.png', '20803.png', '06758.png', '16736.png', '01158.png', '05294.png', '24755.png', '03315.png', '00465.png', '13264.png', '13204.png', '02744.png', '05276.png', '20726.png', '11666.png', '13855.png', '00080.png', '00874.png', '03838.png', '02836.png', '10435.png', '14034.png', '16416.png', '12874.png', '19623.png', '12736.png', '04333.png', '05004.png', '04195.png', '04412.png', '09177.png', '15012.png', '04383.png', '15261.png', '12408.png', '17138.png', '02613.png', '16267.png', '19958.png', '07340.png', '11241.png', '05731.png', '20731.png', '17619.png', '04490.png', '22378.png', '17880.png', '12566.png', '05071.png', '05991.png', '24692.png', '01929.png', '01113.png', '22680.png', '11543.png', '16383.png', '00598.png', '10039.png', '00215.png', '15634.png', '16650.png', '23389.png', '22633.png', '23333.png', '17273.png', '12269.png', '24412.png', '01362.png', '17201.png', '20261.png', '03264.png', '21259.png', '21282.png', '02349.png', '22099.png', '07484.png', '05403.png', '08100.png', '01995.png', '09658.png', '24827.png', '18635.png', '16370.png', '09802.png', '21494.png', '02892.png', '10317.png', '15799.png', '20455.png', '04853.png', '07584.png', '16524.png', '09485.png', '06030.png', '23558.png', '12235.png', '00263.png', '05012.png', '21687.png', '08601.png', '05148.png', '16425.png', '13819.png', '23838.png', '17537.png', '07240.png', '21088.png', '02177.png', '13140.png', '12286.png', '05031.png', '21246.png', '19338.png', '03397.png', '09444.png', '09456.png', '14049.png', '09170.png', '01159.png', '01274.png', '02580.png', '13069.png', '03527.png', '03272.png', '24836.png', '15554.png', '19657.png', '07638.png', '20306.png', '23686.png', '16746.png', '19377.png', '15733.png', '17372.png', '21326.png', '09491.png', '04032.png', '18012.png', '21170.png', '19099.png', '11965.png', '24087.png', '20310.png', '20867.png', '01123.png', '08353.png', '05466.png', '21742.png', '09531.png', '19259.png', '22172.png', '03725.png', '02822.png', '10659.png', '02450.png', '03522.png', '04293.png', '07589.png', '01006.png', '05328.png', '23836.png', '08690.png', '15609.png', '16460.png', '22281.png', '06897.png', '22153.png', '06889.png', '10534.png', '15366.png', '11995.png', '22774.png', '01969.png', '02249.png', '03599.png', '07830.png', '03134.png', '09964.png', '08640.png', '04750.png', '08942.png', '17369.png', '20713.png', '07666.png', '07378.png', '16358.png', '22318.png', '18683.png', '23433.png', '09208.png', '10237.png', '20279.png', '19162.png', '23930.png', '11566.png', '04920.png', '14028.png', '15053.png', '14502.png', '06242.png', '10409.png', '21231.png', '10449.png', '23340.png', '02493.png', '11869.png', '03771.png', '01169.png', '11123.png', '18030.png', '23845.png', '16427.png', '20949.png', '09514.png', '14133.png', '07232.png', '24919.png', '09747.png', '11090.png', '03180.png', '21774.png', '23045.png', '20911.png', '23946.png', '00792.png', '15891.png', '04959.png', '22008.png', '18631.png', '16749.png', '16402.png', '15171.png', '09848.png', '11110.png', '21939.png', '02800.png', '04747.png', '20771.png', '06756.png', '05301.png', '05472.png', '20192.png', '10682.png', '03411.png', '15417.png', '24389.png', '01322.png', '08324.png', '21163.png', '02531.png', '00082.png', '16591.png', '05596.png', '07848.png', '03101.png', '22837.png', '00116.png', '11637.png', '03152.png', '00520.png', '22162.png', '23818.png', '19209.png', '23639.png', '21975.png', '21655.png', '14921.png', '18450.png', '20673.png', '23653.png', '05224.png', '08638.png', '23693.png', '11118.png', '00505.png', '05693.png', '08996.png', '09086.png', '02142.png', '05672.png', '21924.png', '06588.png', '19144.png', '17665.png', '04111.png', '19473.png', '03463.png', '16725.png', '16436.png', '06274.png', '14532.png', '14388.png', '09474.png', '20596.png', '03343.png', '04311.png', '24419.png', '21526.png', '00286.png', '21160.png', '15407.png', '06506.png', '20027.png', '15971.png', '11641.png', '07391.png', '14806.png', '17949.png', '16357.png', '10667.png', '19899.png', '10484.png', '11936.png', '14337.png', '04836.png', '06702.png', '08091.png', '09924.png', '22632.png', '01017.png', '04978.png', '07764.png', '17389.png', '01160.png', '18842.png', '22499.png', '07361.png', '15702.png', '15540.png', '09093.png', '10546.png', '19550.png', '06780.png', '07568.png', '15140.png', '05141.png', '02069.png', '05160.png', '12553.png', '19187.png', '05722.png', '02727.png', '08729.png', '03808.png', '09966.png', '20155.png', '24312.png', '24132.png', '08312.png', '08701.png', '07951.png', '07863.png', '22300.png', '22578.png', '09952.png', '15649.png', '23956.png', '14983.png', '06353.png', '04706.png', '18776.png', '21560.png', '18043.png', '03578.png', '21520.png', '19111.png', '12449.png', '11031.png', '08093.png', '16822.png', '22112.png', '17956.png', '20575.png', '24860.png', '10150.png', '23908.png', '20277.png', '22631.png', '15397.png', '19890.png', '07706.png', '14327.png', '14260.png', '07931.png', '08641.png', '23061.png', '18486.png', '09711.png', '06863.png', '22883.png', '20207.png', '10291.png', '04882.png', '07840.png', '17009.png', '07932.png', '21538.png', '04452.png', '19628.png', '06547.png', '06496.png', '06921.png', '17130.png', '16439.png', '20850.png', '21027.png', '18942.png', '10076.png', '19525.png', '21013.png', '24937.png', '07025.png', '21143.png', '19955.png', '19064.png', '15933.png', '21982.png', '07908.png', '02721.png', '10048.png', '15798.png', '07889.png', '20960.png', '04057.png', '21216.png', '09002.png', '24237.png', '09161.png', '20253.png', '13754.png', '07929.png', '23510.png', '15250.png', '08500.png', '22050.png', '11476.png', '10121.png', '15422.png', '19572.png', '06015.png', '20496.png', '24737.png', '05814.png', '24842.png', '14176.png', '07762.png', '10860.png', '20103.png', '18137.png', '07163.png', '21797.png', '03373.png', '08103.png', '11900.png', '21365.png', '01734.png', '16755.png', '18263.png', '19106.png', '03996.png', '19215.png', '12042.png', '15738.png', '05389.png', '03474.png', '09138.png', '03114.png', '13945.png', '09535.png', '17536.png', '19991.png', '14218.png', '14699.png', '11144.png', '05331.png', '16072.png', '21883.png', '07138.png', '18312.png', '05888.png', '16151.png', '23034.png', '19450.png', '01528.png', '00102.png', '03797.png', '06433.png', '12355.png', '01140.png', '17831.png', '20819.png', '09612.png', '00986.png', '14939.png', '05436.png', '01571.png', '08874.png', '17583.png', '20996.png', '07222.png', '09972.png', '05198.png', '12484.png', '19266.png', '11826.png', '06131.png', '09617.png', '17948.png', '17077.png', '02369.png', '22587.png', '01422.png', '10834.png', '17587.png', '09439.png', '13774.png', '24282.png', '12912.png', '08564.png', '09244.png', '14238.png', '15085.png', '20412.png', '06475.png', '15585.png', '09212.png', '04762.png', '09836.png', '24652.png', '14046.png', '22732.png', '02507.png', '01536.png', '19493.png', '18678.png', '00443.png', '17412.png', '05332.png', '07239.png', '09169.png', '19356.png', '08843.png', '11149.png', '08425.png', '19788.png', '05936.png', '04448.png', '18134.png', '19612.png', '00344.png', '22520.png', '02527.png', '17341.png', '14097.png', '00783.png', '24061.png', '18028.png', '08293.png', '01267.png', '09406.png', '19547.png', '13116.png', '12461.png', '05536.png', '03541.png', '06025.png', '14333.png', '06263.png', '20583.png', '07284.png', '12783.png', '02553.png', '02035.png', '05993.png', '20683.png', '14155.png', '04927.png', '20405.png', '18872.png', '24549.png', '03631.png', '07570.png', '03016.png', '02415.png', '19308.png', '14785.png', '07295.png', '04075.png', '23600.png', '09946.png', '01007.png', '07872.png', '24447.png', '01486.png', '21755.png', '14627.png', '13510.png', '18133.png', '07409.png', '05440.png', '11199.png', '14528.png', '22570.png', '04204.png', '16928.png', '20662.png', '09700.png', '07561.png', '17302.png', '16906.png', '24623.png', '00227.png', '19167.png', '23002.png', '02624.png', '09114.png', '04015.png', '06471.png', '23625.png', '09898.png', '15591.png', '05337.png', '04328.png', '14387.png', '19368.png', '14883.png', '10265.png', '02812.png', '08328.png', '22406.png', '05549.png', '07289.png', '09953.png', '20794.png', '13845.png', '16308.png', '15594.png', '11462.png', '08571.png', '07294.png', '11482.png', '21001.png', '10977.png', '23035.png', '15168.png', '05530.png', '06158.png', '02381.png', '07782.png', '02605.png', '10668.png', '04718.png', '10395.png', '21963.png', '15490.png', '12084.png', '23297.png', '01811.png', '09003.png', '24188.png', '13208.png', '20487.png', '01982.png', '03120.png', '11033.png', '22925.png', '16626.png', '04694.png', '23320.png', '07653.png', '02917.png', '07364.png', '18728.png', '11513.png', '09495.png', '14705.png', '18473.png', '07846.png', '00534.png', '09291.png', '17810.png', '02676.png', '09626.png', '17651.png', '08990.png', '02653.png', '10829.png', '00904.png', '10776.png', '01739.png', '08970.png', '12186.png', '12870.png', '11217.png', '10833.png', '18526.png', '20171.png', '08079.png', '08098.png', '14002.png', '07211.png', '04270.png', '20851.png', '18939.png', '00665.png', '24766.png', '15631.png', '24674.png', '10231.png', '04273.png', '19810.png', '08259.png', '24868.png', '18338.png', '07800.png', '13045.png', '06295.png', '18236.png', '19264.png', '23037.png', '09159.png', '20216.png', '06965.png', '13604.png', '16763.png', '01797.png', '04058.png', '21416.png', '20795.png', '12500.png', '12201.png', '04181.png', '18989.png', '02224.png', '18227.png', '11520.png', '06121.png', '03984.png', '08991.png', '13423.png', '00599.png', '22363.png', '16350.png', '16182.png', '04569.png', '21783.png', '19343.png', '07362.png', '07407.png', '09450.png', '17411.png', '00066.png', '06426.png', '02937.png', '23391.png', '17566.png', '14665.png', '13984.png', '13174.png', '08408.png', '23979.png', '04373.png', '12206.png', '10487.png', '13663.png', '08341.png', '21220.png', '16787.png', '14928.png', '07569.png', '15474.png', '02466.png', '03716.png', '05398.png', '19079.png', '14134.png', '07435.png', '18356.png', '23616.png', '02854.png', '11313.png', '12924.png', '00677.png', '20389.png', '04697.png', '21433.png', '04114.png', '24585.png', '17840.png', '09765.png', '22935.png', '21729.png', '14019.png', '07252.png', '00029.png', '05237.png', '18215.png', '02172.png', '00172.png', '09399.png', '09089.png', '16951.png', '15924.png', '20049.png', '02496.png', '24376.png', '05576.png', '05853.png', '15435.png', '22290.png', '18498.png', '15898.png', '02792.png', '19648.png', '14351.png', '19663.png', '00409.png', '24445.png', '05547.png', '20133.png', '07943.png', '20025.png', '05503.png', '12086.png', '14009.png', '09773.png', '15318.png', '09423.png', '17268.png', '06580.png', '24033.png', '19467.png', '19621.png', '21212.png', '18057.png', '21547.png', '19555.png', '04246.png', '22710.png', '22053.png', '05625.png', '06355.png', '10788.png', '16363.png', '23402.png', '05608.png', '11876.png', '08616.png', '11748.png', '05397.png', '14063.png', '15795.png', '08175.png', '03104.png', '17226.png', '10552.png', '20113.png', '00158.png', '20427.png', '01795.png', '00757.png', '04046.png', '13570.png', '06455.png', '12316.png', '02652.png', '02187.png', '24127.png', '17507.png', '03212.png', '12928.png', '15698.png', '16146.png', '17081.png', '07975.png', '10398.png', '01073.png', '02365.png', '13797.png', '09739.png', '18787.png', '10642.png', '02795.png', '17917.png', '18173.png', '07665.png', '09239.png', '13974.png', '10152.png', '00318.png', '12896.png', '15274.png', '14193.png', '01999.png', '13441.png', '09167.png', '13263.png', '13299.png', '16471.png', '10593.png', '16442.png', '04731.png', '16224.png', '01299.png', '04890.png', '16616.png', '13702.png', '08728.png', '07757.png', '21993.png', '10443.png', '03222.png', '21661.png', '23886.png', '09126.png', '14927.png', '12464.png', '16406.png', '22275.png', '07187.png', '08988.png', '00073.png', '07329.png', '15907.png', '23733.png', '11261.png', '00410.png', '24407.png', '01952.png', '23175.png', '14171.png', '20439.png', '03969.png', '02189.png', '05515.png', '11702.png', '05564.png', '14741.png', '19885.png', '21122.png', '12632.png', '02877.png', '23311.png', '20187.png', '14069.png', '04264.png', '13431.png', '21184.png', '17150.png', '09622.png', '04324.png', '10719.png', '22798.png', '00492.png', '09688.png', '10902.png', '04985.png', '01207.png', '01570.png', '21775.png', '08517.png', '06890.png', '00063.png', '04770.png', '01893.png', '10113.png', '24541.png', '21443.png', '21361.png', '14999.png', '13897.png', '02101.png', '04573.png', '05400.png', '12745.png', '04518.png', '20860.png', '23594.png', '07707.png', '07042.png', '06285.png', '23933.png', '05964.png', '21751.png', '21856.png', '22809.png', '04126.png', '07565.png', '00395.png', '12311.png', '05098.png', '10116.png', '00341.png', '17900.png', '08055.png', '11614.png', '16069.png', '12843.png', '02529.png', '23166.png', '17044.png', '12593.png', '11587.png', '23382.png', '21573.png', '07133.png', '20729.png', '12136.png', '22299.png', '13070.png', '24164.png', '03231.png', '16932.png', '03182.png', '01656.png', '22625.png', '01443.png', '23794.png', '03989.png', '01591.png', '16320.png', '21366.png', '01852.png', '13355.png', '00826.png', '04492.png', '16817.png', '10282.png', '19320.png', '06250.png', '09600.png', '06443.png', '00357.png', '06449.png', '16686.png', '03210.png', '04798.png', '15286.png', '19839.png', '23999.png', '20936.png', '13357.png', '00903.png', '19242.png', '20225.png', '08231.png', '15310.png', '08482.png', '04622.png', '22432.png', '17237.png', '12831.png', '10514.png', '01790.png', '21723.png', '16014.png', '13918.png', '00643.png', '23641.png', '22566.png', '09346.png', '09445.png', '13876.png', '07642.png', '04806.png', '24309.png', '23761.png', '13931.png', '01484.png', '16073.png', '12532.png', '17772.png', '11859.png', '20415.png', '24642.png', '24288.png', '16465.png', '23967.png', '24613.png', '16866.png', '02856.png', '00750.png', '24368.png', '00624.png', '21330.png', '16940.png', '04449.png', '18482.png', '11737.png', '24275.png', '22143.png', '20672.png', '11076.png', '22791.png', '20441.png', '04459.png', '09843.png', '11290.png', '07790.png', '22439.png', '05408.png', '16172.png', '16713.png', '07734.png', '05790.png', '05287.png', '22315.png', '08377.png', '10918.png', '16397.png', '18929.png', '01937.png', '05091.png', '03516.png', '13171.png', '05818.png', '01924.png', '14647.png', '13454.png', '08035.png', '04177.png', '11923.png', '08674.png', '14397.png', '22855.png', '08504.png', '08850.png', '16013.png', '01906.png', '13347.png', '04226.png', '18414.png', '09885.png', '11354.png', '15273.png', '18389.png', '22409.png', '20315.png', '10004.png', '07024.png', '17688.png', '10677.png', '23893.png', '22698.png', '10554.png', '09424.png', '14127.png', '14185.png', '18315.png', '18044.png', '04898.png', '24257.png', '07262.png', '05114.png', '09200.png', '14246.png', '23687.png', '03528.png', '14653.png', '03597.png', '16743.png', '02410.png', '18166.png', '00997.png', '23587.png', '15100.png', '22221.png', '22717.png', '08295.png', '11364.png', '20703.png', '22199.png', '19141.png', '03757.png', '13851.png', '03982.png', '15603.png', '16295.png', '13837.png', '24545.png', '13522.png', '16311.png', '01679.png', '10580.png', '15170.png', '21380.png', '23417.png', '07591.png', '08075.png', '04174.png', '18801.png', '08314.png', '15637.png', '18021.png', '24320.png', '16079.png', '21911.png', '15409.png', '22902.png', '23795.png', '03606.png', '01512.png', '05465.png', '09751.png', '10101.png', '23648.png', '06796.png', '22851.png', '19330.png', '11424.png', '10078.png', '13745.png', '06293.png', '06456.png', '18761.png', '20743.png', '12592.png', '24656.png', '04365.png', '20115.png', '09414.png', '14420.png', '08812.png', '17212.png', '18538.png', '21472.png', '15327.png', '03818.png', '13810.png', '19142.png', '22366.png', '04522.png', '23220.png', '19866.png', '13717.png', '23475.png', '13854.png', '14393.png', '02776.png', '04811.png', '03251.png', '15865.png', '01039.png', '11717.png', '06903.png', '13218.png', '20963.png', '19966.png', '18357.png', '15662.png', '14110.png', '24006.png', '07477.png', '01710.png', '14722.png', '00190.png', '16605.png', '23381.png', '18192.png', '07313.png', '00453.png', '23849.png', '10943.png', '19080.png', '07075.png', '03178.png', '12828.png', '12824.png', '02960.png', '14012.png', '01617.png', '11224.png', '17097.png', '07701.png', '00637.png', '16870.png', '17977.png', '11867.png', '20160.png', '11667.png', '15112.png', '09950.png', '20969.png', '24773.png', '17513.png', '21673.png', '05439.png', '19964.png', '11044.png', '04787.png', '08144.png', '03966.png', '23557.png', '23013.png', '22202.png', '20965.png', '03685.png', '01433.png', '14105.png', '08842.png', '03526.png', '21138.png', '01200.png', '05234.png', '04403.png', '00963.png', '16401.png', '04886.png', '14026.png', '11343.png', '07355.png', '04509.png', '02093.png', '02134.png', '22605.png', '05606.png', '01605.png', '22398.png', '20865.png', '13875.png', '15811.png', '01772.png', '05329.png', '16000.png', '02808.png', '04091.png', '07339.png', '18508.png', '12317.png', '06591.png', '10177.png', '17676.png', '06971.png', '15391.png', '19821.png', '23286.png', '05819.png', '13535.png', '23328.png', '08278.png', '02478.png', '23880.png', '05028.png', '09374.png', '01459.png', '07978.png', '14751.png', '20197.png', '08833.png', '05145.png', '04549.png', '23198.png', '00246.png', '13770.png', '08481.png', '06534.png', '07833.png', '19736.png', '22032.png', '14374.png', '14243.png', '21874.png', '17373.png', '03116.png', '05446.png', '14239.png', '03581.png', '10581.png', '12523.png', '05939.png', '24431.png', '08742.png', '16059.png', '03901.png', '24945.png', '19957.png', '11946.png', '02160.png', '17756.png', '12619.png', '17442.png', '18396.png', '22340.png', '11785.png', '07614.png', '08284.png', '06203.png', '22094.png', '03197.png', '20228.png', '11980.png', '02718.png', '04765.png', '22287.png', '04942.png', '20875.png', '16424.png', '14711.png', '03497.png', '13303.png', '15092.png', '13961.png', '10458.png', '11493.png', '06390.png', '21701.png', '12202.png', '09255.png', '23219.png', '15580.png', '01959.png', '14295.png', '04223.png', '12526.png', '08128.png', '03330.png', '18790.png', '23689.png', '15946.png', '14293.png', '13587.png', '18636.png', '09321.png', '08253.png', '21626.png', '09804.png', '03322.png', '11156.png', '21906.png', '01855.png', '17768.png', '00561.png', '01663.png', '12609.png', '09833.png', '16121.png', '24200.png', '19814.png', '10765.png', '06844.png', '23055.png', '01730.png', '15858.png', '06739.png', '06527.png', '01575.png', '17699.png', '16449.png', '20863.png', '14820.png', '02625.png', '01717.png', '09526.png', '19202.png', '23294.png', '20968.png', '14449.png', '08117.png', '24108.png', '11607.png', '02816.png', '02371.png', '00164.png', '16431.png', '21431.png', '09287.png', '07322.png', '11621.png', '05768.png', '07887.png', '03229.png', '11277.png', '07256.png', '02535.png', '12420.png', '02052.png', '01654.png', '14088.png', '08437.png', '23024.png', '08205.png', '04724.png', '20841.png', '21933.png', '24267.png', '02214.png', '11728.png', '19045.png', '00726.png', '18753.png', '08635.png', '09986.png', '13182.png', '18147.png', '01357.png', '09734.png', '15965.png', '18764.png', '08372.png', '15949.png', '15141.png', '18617.png', '15454.png', '06864.png', '18270.png', '24329.png', '13831.png', '12265.png', '23135.png', '18352.png', '21684.png', '11580.png', '16764.png', '07997.png', '08005.png', '06081.png', '24135.png', '23227.png', '11049.png', '11776.png', '20776.png', '24597.png', '04213.png', '03384.png', '07044.png', '22598.png', '18363.png', '07346.png', '00582.png', '13578.png', '15926.png', '15671.png', '23298.png', '02927.png', '02888.png', '22930.png', '18154.png', '18707.png', '10417.png', '04722.png', '11985.png', '15433.png', '24400.png', '22955.png', '12524.png', '23321.png', '21322.png', '11430.png', '02654.png', '18367.png', '15199.png', '03908.png', '14418.png', '14862.png', '21892.png', '02116.png', '19681.png', '03396.png', '12779.png', '05093.png', '00976.png', '09238.png', '00319.png', '22909.png', '22962.png', '18747.png', '06095.png', '04609.png', '22238.png', '05175.png', '07290.png', '06373.png', '15807.png', '05258.png', '18813.png', '12558.png', '13399.png', '03727.png', '14915.png', '09408.png', '06977.png', '21404.png', '16197.png', '17361.png', '21002.png', '17350.png', '09928.png', '06711.png', '19626.png', '18808.png', '15756.png', '21147.png', '01407.png', '22017.png', '05962.png', '02215.png', '18660.png', '24315.png', '21942.png', '20610.png', '19441.png', '04304.png', '12734.png', '14251.png', '05782.png', '18135.png', '09327.png', '14267.png', '18906.png', '18286.png', '24015.png', '17731.png', '08120.png', '24104.png', '16859.png', '11916.png', '08068.png', '08720.png', '09307.png', '24327.png', '00207.png', '19043.png', '23834.png', '22711.png', '02656.png', '10233.png', '19324.png', '07227.png', '05657.png', '19965.png', '12340.png', '18454.png', '07107.png', '23164.png', '03019.png', '15526.png', '13728.png', '10384.png', '17698.png', '14783.png', '14734.png', '20328.png', '08748.png', '12148.png', '18995.png', '04714.png', '24004.png', '06728.png', '10480.png', '15249.png', '03170.png', '03551.png', '03857.png', '06870.png', '17040.png', '21310.png', '18669.png', '14677.png', '14178.png', '01925.png', '01585.png', '24153.png', '01010.png', '14782.png', '23254.png', '10752.png', '14676.png', '02216.png', '00160.png', '22222.png', '08308.png', '01910.png', '09511.png', '13317.png', '01263.png', '04194.png', '15833.png', '22939.png', '13372.png', '22881.png', '14300.png', '00882.png', '21332.png', '08397.png', '04856.png', '11949.png', '07620.png', '15712.png', '22757.png', '04827.png', '13312.png', '06830.png', '08992.png', '08376.png', '12356.png', '17743.png', '11130.png', '08523.png', '10298.png', '00355.png', '13960.png', '17305.png', '11079.png', '12643.png', '08938.png', '19726.png', '06391.png', '11253.png', '10816.png', '16507.png', '19132.png', '12502.png', '19534.png', '24397.png', '00099.png', '16718.png', '00145.png', '19423.png', '23458.png', '14807.png', '03399.png', '11169.png', '21825.png', '13071.png', '18857.png', '24711.png', '02384.png', '11844.png', '24511.png', '05298.png', '04795.png', '03337.png', '13280.png', '02353.png', '13735.png', '10730.png', '02012.png', '05707.png', '14611.png', '00787.png', '16814.png', '24177.png', '09012.png', '01862.png', '21740.png', '07471.png', '06681.png', '20419.png', '17335.png', '14473.png', '13442.png', '23781.png', '20923.png', '22992.png', '09653.png', '16498.png', '06094.png', '22505.png', '04604.png', '15692.png', '05295.png', '24721.png', '23768.png', '04364.png', '00744.png', '10501.png', '06261.png', '10843.png', '15867.png', '12021.png', '13563.png', '07814.png', '14922.png', '06396.png', '12590.png', '11511.png', '03369.png', '06683.png', '11962.png', '01136.png', '18199.png', '16575.png', '02233.png', '16209.png', '12381.png', '24365.png', '02598.png', '20709.png', '18877.png', '21154.png', '09145.png', '10810.png', '04318.png', '22762.png', '07749.png', '19253.png', '15778.png', '01446.png', '10251.png', '10529.png', '12793.png', '09347.png', '08329.png', '14577.png', '13211.png', '07725.png', '15775.png', '00527.png', '23470.png', '10462.png', '00933.png', '16580.png', '11571.png', '22388.png', '15045.png', '09657.png', '21096.png', '13411.png', '16331.png', '17554.png', '18664.png', '11727.png', '06841.png', '17230.png', '05263.png', '04422.png', '11759.png', '23303.png', '22414.png', '15334.png', '08388.png', '04465.png', '15592.png', '18176.png', '15836.png', '16665.png', '16808.png', '07451.png', '22254.png', '04218.png', '12354.png', '13198.png', '18279.png', '01697.png', '24051.png', '01402.png', '16278.png', '09500.png', '00943.png', '15535.png', '14440.png', '07822.png', '12733.png', '24361.png', '02307.png', '07014.png', '20909.png', '05269.png', '13861.png', '23574.png', '10678.png', '19826.png', '22689.png', '01951.png', '09044.png', '09234.png', '04919.png', '18081.png', '09702.png', '23278.png', '24362.png', '13964.png', '10022.png', '19380.png', '12057.png', '01621.png', '05167.png', '23876.png', '23738.png', '08288.png', '05694.png', '05469.png', '14990.png', '16403.png', '00212.png', '05090.png', '01799.png', '22611.png', '15690.png', '13829.png', '03877.png', '13532.png', '00916.png', '02299.png', '00701.png', '17804.png', '00562.png', '19633.png', '16201.png', '17589.png', '22357.png', '17750.png', '07349.png', '14359.png', '17721.png', '05375.png', '11378.png', '13636.png', '06794.png', '18262.png', '21306.png', '19989.png', '04943.png', '09443.png', '10305.png', '24591.png', '14997.png', '19996.png', '14367.png', '18239.png', '13358.png', '15364.png', '05356.png', '10889.png', '07982.png', '08830.png', '23518.png', '04488.png', '16476.png', '10966.png', '00961.png', '23383.png', '04868.png', '19976.png', '05935.png', '10897.png', '23941.png', '19085.png', '12405.png', '10606.png', '20248.png', '16285.png', '01230.png', '17161.png', '09250.png', '24930.png', '03741.png', '03488.png', '19791.png', '02049.png', '00995.png', '01650.png', '19361.png', '16101.png', '19862.png', '10913.png', '05580.png', '11375.png', '03739.png', '17025.png', '12698.png', '13844.png', '24850.png', '13980.png', '21198.png', '02894.png', '00219.png', '05747.png', '00784.png', '24818.png', '05239.png', '10954.png', '15483.png', '05864.png', '09486.png', '15993.png', '24630.png', '12546.png', '06427.png', '15133.png', '07962.png', '21214.png', '14450.png', '13672.png', '23194.png', '07135.png', '24433.png', '05386.png', '05324.png', '01714.png', '24720.png', '04183.png', '15903.png', '10664.png', '00616.png', '12443.png', '08424.png', '15744.png', '05205.png', '10099.png', '14062.png', '10884.png', '22389.png', '07809.png', '15768.png', '00841.png', '16483.png', '03963.png', '02939.png', '04972.png', '05841.png', '14980.png', '13395.png', '05837.png', '15806.png', '21113.png', '10490.png', '04167.png', '04241.png', '24966.png', '05842.png', '02547.png', '07526.png', '15132.png', '01063.png', '05592.png', '15596.png', '18767.png', '17437.png', '07101.png', '20782.png', '10261.png', '11787.png', '22769.png', '07098.png', '07712.png', '12358.png', '22114.png', '10312.png', '06230.png', '08793.png', '17432.png', '05948.png', '07157.png', '06607.png', '03737.png', '01257.png', '12485.png', '04986.png', '04608.png', '04531.png', '21265.png', '21828.png', '17021.png', '06767.png', '00261.png', '15328.png', '00947.png', '19962.png', '13742.png', '16066.png', '07392.png', '05958.png', '06200.png', '23448.png', '24949.png', '12817.png', '12878.png', '23579.png', '22882.png', '02320.png', '16317.png', '01470.png', '05640.png', '19830.png', '07544.png', '03626.png', '01119.png', '13559.png', '08115.png', '01345.png', '08193.png', '16292.png', '24920.png', '11477.png', '10154.png', '10197.png', '14765.png', '07621.png', '02009.png', '02524.png', '14220.png', '15226.png', '22657.png', '09550.png', '16953.png', '14000.png', '17387.png', '15612.png', '02895.png', '04055.png', '05554.png', '24493.png', '04072.png', '22043.png', '24636.png', '24802.png', '20041.png', '23797.png', '13485.png', '08964.png', '11635.png', '09021.png', '16549.png', '03054.png', '11058.png', '13947.png', '05779.png', '00776.png', '06937.png', '00836.png', '13846.png', '20581.png', '24895.png', '17746.png', '05111.png', '06800.png', '21974.png', '15517.png', '15260.png', '24770.png', '20593.png', '00775.png', '23632.png', '02271.png', '21005.png', '05697.png', '23959.png', '19560.png', '19522.png', '17547.png', '03470.png', '22539.png', '23503.png', '13132.png', '23856.png', '09810.png', '23359.png', '07835.png', '04912.png', '18416.png', '15520.png', '12228.png', '02448.png', '02377.png', '19517.png', '08542.png', '05752.png', '10316.png', '20220.png', '06652.png', '09381.png', '14338.png', '05739.png', '13679.png', '00794.png', '14882.png', '04554.png', '05777.png', '21471.png', '10493.png', '14796.png', '14901.png', '08875.png', '14713.png', '03719.png', '08296.png', '19225.png', '12849.png', '22064.png', '13736.png', '09525.png', '01361.png', '10425.png', '02217.png', '01926.png', '02327.png', '04826.png', '10962.png', '04131.png', '09715.png', '11930.png', '07057.png', '05187.png', '13777.png', '07427.png', '15787.png', '04675.png', '16019.png', '15312.png', '04915.png', '13279.png', '00639.png', '23489.png', '09011.png', '11452.png', '05101.png', '10424.png', '17291.png', '19028.png', '20073.png', '24582.png', '05502.png', '00663.png', '21254.png', '04253.png', '11586.png', '00887.png', '06276.png', '20199.png', '16453.png', '24635.png', '13509.png', '13409.png', '04639.png', '19951.png', '10026.png', '24709.png', '07592.png', '18618.png', '18323.png', '13321.png', '11871.png', '23257.png', '04239.png', '21340.png', '21613.png', '14899.png', '01270.png', '16010.png', '01728.png', '14576.png', '04350.png', '21099.png', '21799.png', '19349.png', '06648.png', '17506.png', '14015.png', '19661.png', '04457.png', '12044.png', '15468.png', '23739.png', '23097.png', '06661.png', '22914.png', '23253.png', '14269.png', '11426.png', '09767.png', '11909.png', '19176.png', '10095.png', '03652.png', '00140.png', '14511.png', '21650.png', '09140.png', '10639.png', '17971.png', '12739.png', '18612.png', '24645.png', '09693.png', '14486.png', '12987.png', '11597.png', '06377.png', '12414.png', '23683.png', '17744.png', '08531.png', '04073.png', '08968.png', '19748.png', '05981.png', '04056.png', '11832.png', '22023.png', '15450.png', '04933.png', '06412.png', '16672.png', '18188.png', '14515.png', '22954.png', '02510.png', '23799.png', '20403.png', '14728.png', '06084.png', '11899.png', '04688.png', '03855.png', '18802.png', '04411.png', '00507.png', '00027.png', '15441.png', '19246.png', '21809.png', '00131.png', '01065.png', '16107.png', '15771.png', '15396.png', '14369.png', '00881.png', '24874.png', '17686.png', '17995.png', '10114.png', '17990.png', '17914.png', '06347.png', '19817.png', '17256.png', '04654.png', '23741.png', '10712.png', '18554.png', '22034.png', '15182.png', '15988.png', '20954.png', '16972.png', '24330.png', '19263.png', '16222.png', '20493.png', '19577.png', '07150.png', '08453.png', '05180.png', '22856.png', '21097.png', '18017.png', '22673.png', '24468.png', '08279.png', '18689.png', '07722.png', '19475.png', '13195.png', '01760.png', '19904.png', '02088.png', '06741.png', '19274.png', '21820.png', '19715.png', '11941.png', '05489.png', '06712.png', '06860.png', '16815.png', '10495.png', '19041.png', '22752.png', '00644.png', '21320.png', '24276.png', '00413.png', '00345.png', '23283.png', '12875.png', '12635.png', '06405.png', '13484.png', '01434.png', '23743.png', '20869.png', '14008.png', '02567.png', '17014.png', '08036.png', '05179.png', '03690.png', '21075.png', '17059.png', '13179.png', '13097.png', '23792.png', '24700.png', '22860.png', '00297.png', '03972.png', '10499.png', '21183.png', '12549.png', '11535.png', '11878.png', '10268.png', '14016.png', '23992.png', '14278.png', '14549.png', '17016.png', '05459.png', '16152.png', '00587.png', '02208.png', '22612.png', '24889.png', '17813.png', '03309.png', '02765.png', '09108.png', '08006.png', '10981.png', '17231.png', '09360.png', '00731.png', '02367.png', '18157.png', '13077.png', '18068.png', '04473.png', '10433.png', '03044.png', '10001.png', '06717.png', '05426.png', '10033.png', '17923.png', '18897.png', '21059.png', '24509.png', '02819.png', '14334.png', '17886.png', '04198.png', '06909.png', '01955.png', '04630.png', '23353.png', '19515.png', '12023.png', '13359.png', '16162.png', '22058.png', '13571.png', '21512.png', '09056.png', '15896.png', '21141.png', '09819.png', '03021.png', '19844.png', '14827.png', '01704.png', '21668.png', '09721.png', '13621.png', '24044.png', '06515.png', '00591.png', '07315.png', '07387.png', '13764.png', '12577.png', '15478.png', '02662.png', '14172.png', '07146.png', '14998.png', '22471.png', '00108.png', '17195.png', '01252.png', '19078.png', '14190.png', '21919.png', '10875.png', '21025.png', '07380.png', '01841.png', '09271.png', '08574.png', '15647.png', '00186.png', '17402.png', '12045.png', '13248.png', '13130.png', '00519.png', '17363.png', '14164.png', '13776.png', '14118.png', '16977.png', '07882.png', '06429.png', '10054.png', '09226.png', '16163.png', '24170.png', '17367.png', '00144.png', '21591.png', '00647.png', '16126.png', '17186.png', '09671.png', '08684.png', '00057.png', '03860.png', '22024.png', '17038.png', '14115.png', '04214.png', '22642.png', '14425.png', '14485.png', '01099.png', '16034.png', '18741.png', '08030.png', '18422.png', '01429.png', '11142.png', '02164.png', '09690.png', '08888.png', '18109.png', '17006.png', '01532.png', '10979.png', '03998.png', '17122.png', '14952.png', '22506.png', '14822.png', '13181.png', '12230.png', '18757.png', '22579.png', '06894.png', '02394.png', '00472.png', '19281.png', '05164.png', '08245.png', '04579.png', '10974.png', '23169.png', '21356.png', '14958.png', '08334.png', '21423.png', '06399.png', '11135.png', '01616.png', '09521.png', '07645.png', '06205.png', '13705.png', '17247.png', '20576.png', '21483.png', '20598.png', '02325.png', '06487.png', '05373.png', '15543.png', '18901.png', '03149.png', '21117.png', '22320.png', '16729.png', '12091.png', '13459.png', '07144.png', '21847.png', '05365.png', '10895.png', '00065.png', '13597.png', '06770.png', '01774.png', '12949.png', '16003.png', '22490.png', '01671.png', '15912.png', '14347.png', '11912.png', '21275.png', '05099.png', '04567.png', '23076.png', '17425.png', '21000.png', '19386.png', '14141.png', '19120.png', '17455.png', '04893.png', '03307.png', '22087.png', '11244.png', '16099.png', '06176.png', '16908.png', '00100.png', '21772.png', '00482.png', '00439.png', '20435.png', '17176.png', '00427.png', '24663.png', '20554.png', '07316.png', '18512.png', '04256.png', '05762.png', '01964.png', '11143.png', '15285.png', '19752.png', '07529.png', '18904.png', '12067.png', '14284.png', '17709.png', '06559.png', '05716.png', '05876.png', '21922.png', '14301.png', '16324.png', '12380.png', '17266.png', '24866.png', '08007.png', '13256.png', '04715.png', '21754.png', '11677.png', '23674.png', '01860.png', '05891.png', '24748.png', '06356.png', '11890.png', '18578.png', '22163.png', '14640.png', '13950.png', '11127.png', '14013.png', '09634.png', '08322.png', '11468.png', '09741.png', '21771.png', '10851.png', '19050.png', '12903.png', '22102.png', '21110.png', '06134.png', '05863.png', '13028.png', '00050.png', '05844.png', '14612.png', '07758.png', '17623.png', '14620.png', '18521.png', '03257.png', '21490.png', '10802.png', '11978.png', '10345.png', '21315.png', '13791.png', '23419.png', '22519.png', '11920.png', '16023.png', '11270.png', '12245.png', '07843.png', '24444.png', '13209.png', '24922.png', '14490.png', '03459.png', '07258.png', '13320.png', '11788.png', '24602.png', '15175.png', '05110.png', '20518.png', '21864.png', '00316.png', '02783.png', '20781.png', '10401.png', '04696.png', '08351.png', '13167.png', '17121.png', '23432.png', '00691.png', '13707.png', '11596.png', '19122.png', '23922.png', '02058.png', '13982.png', '23783.png', '03109.png', '00655.png', '10691.png', '17304.png', '08264.png', '02346.png', '23791.png', '08426.png', '05772.png', '22764.png', '06710.png', '24262.png', '13564.png', '13992.png', '13525.png', '16660.png', '20779.png', '02044.png', '15953.png', '14551.png', '10008.png', '22928.png', '09412.png', '10879.png', '02899.png', '15537.png', '13671.png', '24649.png', '11001.png', '13811.png', '19228.png', '22983.png', '00765.png', '22530.png', '19391.png', '16724.png', '16297.png', '05562.png', '05877.png', '05507.png', '23968.png', '23636.png', '18835.png', '18772.png', '12253.png', '09816.png', '12944.png', '02067.png', '20910.png', '19018.png', '03775.png', '13117.png', '11875.png', '17728.png', '00383.png', '01391.png', '12585.png', '22658.png', '02938.png', '09392.png', '04078.png', '24955.png', '14589.png', '06026.png', '06656.png', '01394.png', '00044.png', '09067.png', '11659.png', '05854.png', '01288.png', '20891.png', '05653.png', '21394.png', '15876.png', '15037.png', '07199.png', '08732.png', '24820.png', '11010.png', '17524.png', '02046.png', '17978.png', '17178.png', '01442.png', '17786.png', '22853.png', '10766.png', '16368.png', '14229.png', '18280.png', '07520.png', '12631.png', '08400.png', '11329.png', '14047.png', '02545.png', '07371.png', '14090.png', '15651.png', '02204.png', '03914.png', '15308.png', '08521.png', '08798.png', '10582.png', '23247.png', '06226.png', '00929.png', '19781.png', '13767.png', '14651.png', '10620.png', '16695.png', '24322.png', '24793.png', '23290.png', '19995.png', '00258.png', '09253.png', '02759.png', '08346.png', '13603.png', '06555.png', '00493.png', '15120.png', '07369.png', '15539.png', '03353.png', '00755.png', '01101.png', '24251.png', '05213.png', '11623.png', '14154.png', '16142.png', '24360.png', '06360.png', '24911.png', '01108.png', '24041.png', '15871.png', '02000.png', '07580.png', '19265.png', '02374.png', '16798.png', '04150.png', '18921.png', '08775.png', '07989.png', '16487.png', '07954.png', '07540.png', '07537.png', '15143.png', '20168.png', '11265.png', '17280.png', '11593.png', '07242.png', '16468.png', '22337.png', '05423.png', '23662.png', '02130.png', '13109.png', '13113.png', '16981.png', '08473.png', '14847.png', '13538.png', '24074.png', '23357.png', '07353.png', '11555.png', '17221.png', '05045.png', '03833.png', '07397.png', '15329.png', '23694.png', '24071.png', '16752.png', '13981.png', '24000.png', '18557.png', '11029.png', '06732.png', '00632.png', '07092.png', '20483.png', '06253.png', '09579.png', '00222.png', '11640.png', '05243.png', '01102.png', '11162.png', '11958.png', '21876.png', '05813.png', '02700.png', '20128.png', '11138.png', '17918.png', '23272.png', '20185.png', '24248.png', '06098.png', '19394.png', '18189.png', '13832.png', '05494.png', '13188.png', '15227.png', '18559.png', '12359.png', '03493.png', '09627.png', '22033.png', '23087.png', '24525.png', '07145.png', '19564.png', '12678.png', '01162.png', '15000.png', '03637.png', '07023.png', '00728.png', '20629.png', '02941.png', '11200.png', '11572.png', '05094.png', '00402.png', '12140.png', '23440.png', '10455.png', '22085.png', '12288.png', '08478.png', '03744.png', '01665.png', '04169.png', '09120.png', '06594.png', '03884.png', '20205.png', '13242.png', '02525.png', '23412.png', '24667.png', '07905.png', '08157.png', '13157.png', '01290.png', '10543.png', '08395.png', '05414.png', '06765.png', '00954.png', '08550.png', '23207.png', '09937.png', '18891.png', '09565.png', '11761.png', '19309.png', '14122.png', '21083.png', '03142.png', '16093.png', '01427.png', '06155.png', '14949.png', '17677.png', '16195.png', '11565.png', '03503.png', '00938.png', '09060.png', '21700.png', '07076.png', '16249.png', '15835.png', '15390.png', '02536.png', '16367.png', '02846.png', '04716.png', '05476.png', '13720.png', '19887.png', '03313.png', '07983.png', '11544.png', '00415.png', '23161.png', '10045.png', '08920.png', '09540.png', '10119.png', '06431.png', '01264.png', '20444.png', '14644.png', '02537.png', '21572.png', '03319.png', '01185.png', '20259.png', '21801.png', '00789.png', '09730.png', '14799.png', '21317.png', '19288.png', '18848.png', '08856.png', '01147.png', '24778.png', '08409.png', '19580.png', '15292.png', '02017.png', '00294.png', '13009.png', '08651.png', '21269.png', '17275.png', '05014.png', '09342.png', '07153.png', '24260.png', '03259.png', '12444.png', '12589.png', '08545.png', '06417.png', '18084.png', '02070.png', '07991.png', '23506.png', '04645.png', '07750.png', '15792.png', '15726.png', '15606.png', '06175.png', '13516.png', '14160.png', '14715.png', '04940.png', '15890.png', '22452.png', '02386.png', '11238.png', '23734.png', '09942.png', '20100.png', '21823.png', '16875.png', '20402.png', '17127.png', '10200.png', '20970.png', '01050.png', '18410.png', '17838.png', '13349.png', '22524.png', '02731.png', '22369.png', '21598.png', '22849.png', '20614.png', '02387.png', '17674.png', '23872.png', '23018.png', '24679.png', '12528.png', '09601.png', '24654.png', '20467.png', '03555.png', '19444.png', '12455.png', '18299.png', '15111.png', '22937.png', '19411.png', '01217.png', '08808.png', '13073.png', '03034.png', '00451.png', '00375.png', '04230.png', '16179.png', '22423.png', '24147.png', '17319.png', '13857.png', '02110.png', '01915.png', '02472.png', '04020.png', '22356.png', '11137.png', '23665.png', '02956.png', '17884.png', '22443.png', '10063.png', '14463.png', '06754.png', '02288.png', '23020.png', '21056.png', '15198.png', '12080.png', '21954.png', '11624.png', '08537.png', '09561.png', '18555.png', '06434.png', '11879.png', '06759.png', '11484.png', '23411.png', '16096.png', '15225.png', '21709.png', '22173.png', '13223.png', '01084.png', '19710.png', '12584.png', '06340.png', '09998.png', '10136.png', '04601.png', '21121.png', '16300.png', '15336.png', '06453.png', '21897.png', '18727.png', '20375.png', '00195.png', '19236.png', '24852.png', '23170.png', '24077.png', '13868.png', '16510.png', '00220.png', '22447.png', '12548.png', '21744.png', '22065.png', '06658.png', '20273.png', '07691.png', '01359.png', '12836.png', '07447.png', '00417.png', '17961.png', '15178.png', '00898.png', '02340.png', '04061.png', '03129.png', '13462.png', '04559.png', '16867.png', '06209.png', '16643.png', '12089.png', '06538.png', '02599.png', '13594.png', '07218.png', '18794.png', '15005.png', '19739.png', '24012.png', '13645.png', '01437.png', '01279.png', '04658.png', '08754.png', '00091.png', '20358.png', '06344.png', '01691.png', '16887.png', '23245.png', '17639.png', '05628.png', '17778.png', '21114.png', '14935.png', '04541.png', '11820.png', '04738.png', '15137.png', '07335.png', '11981.png', '19182.png', '13257.png', '02013.png', '16221.png', '01020.png', '11594.png', '18394.png', '09262.png', '04623.png', '04984.png', '22663.png', '09554.png', '13651.png', '23142.png', '14762.png', '07099.png', '14596.png', '09422.png', '15443.png', '20363.png', '23185.png', '23936.png', '07813.png', '22332.png', '14850.png', '20519.png', '00155.png', '06552.png', '22911.png', '10066.png', '09633.png', '07984.png', '11177.png', '12691.png', '00067.png', '15321.png', '04160.png', '17901.png', '24910.png', '17438.png', '00843.png', '11307.png', '09859.png', '02949.png', '10569.png', '09547.png', '05479.png', '20561.png', '13334.png', '16347.png', '09743.png', '01782.png', '16698.png', '03517.png', '22291.png', '16675.png', '20434.png', '12541.png', '22873.png', '23466.png', '02435.png', '19609.png', '14797.png', '04873.png', '16500.png', '03882.png', '18489.png', '05015.png', '21354.png', '15623.png', '23358.png', '01858.png', '13638.png', '04443.png', '11370.png', '14891.png', '13970.png', '13566.png', '01269.png', '24216.png', '06090.png', '18768.png', '05173.png', '01928.png', '13445.png', '10687.png', '22274.png', '02683.png', '18314.png', '00310.png', '19348.png', '19637.png', '21585.png', '16188.png', '06730.png', '09207.png', '22821.png', '01048.png', '16280.png', '09143.png', '10028.png', '24848.png', '08708.png', '14919.png', '01965.png', '04090.png', '05338.png', '24300.png', '09254.png', '07430.png', '09192.png', '05572.png', '19017.png', '09027.png', '04523.png', '10272.png', '11963.png', '08828.png', '14634.png', '13301.png', '03093.png', '22073.png', '08177.png', '24626.png', '14686.png', '01630.png', '00491.png', '01219.png', '21617.png', '02083.png', '19532.png', '11316.png', '15500.png', '21346.png', '08026.png', '21019.png', '13786.png', '19708.png', '17360.png', '17262.png', '10823.png', '18997.png', '03468.png', '13899.png', '16533.png', '09148.png', '14259.png', '12642.png', '11857.png', '11405.png', '22949.png', '19937.png', '20218.png', '22969.png', '04558.png', '00817.png', '08471.png', '13547.png', '19721.png', '16104.png', '02115.png', '15529.png', '08216.png', '02317.png', '07130.png', '18090.png', '23314.png', '01082.png', '17816.png', '03733.png', '08917.png', '08460.png', '15223.png', '16083.png', '04117.png', '10069.png', '23898.png', '03668.png', '10328.png', '00628.png', '07627.png', '15359.png', '06565.png', '15330.png', '11758.png', '05140.png', '08836.png', '08466.png', '09266.png', '15601.png', '13729.png', '21215.png', '05462.png', '02947.png', '24119.png', '20055.png', '08399.png', '01215.png', '19690.png', '21162.png', '13921.png', '12687.png', '24034.png', '00398.png', '23901.png', '08853.png', '09270.png', '13958.png', '03145.png', '15258.png', '19926.png', '11286.png', '07672.png', '11630.png', '16745.png', '02407.png', '05533.png', '07974.png', '15814.png', '12560.png', '22915.png', '17046.png', '20067.png', '00466.png', '14860.png', '03354.png', '24204.png', '12728.png', '14518.png', '17708.png', '19003.png', '03563.png', '08381.png', '21040.png', '06891.png', '23565.png', '20927.png', '13146.png', '16384.png', '15509.png', '05926.png', '06616.png', '01171.png', '18048.png', '24099.png', '09703.png', '03268.png', '22424.png', '03656.png', '05676.png', '05671.png', '04019.png', '20912.png', '05381.png', '20928.png', '21373.png', '15508.png', '00964.png', '22560.png', '02838.png', '10622.png', '11456.png', '17194.png', '15572.png', '19025.png', '15816.png', '23960.png', '18638.png', '05735.png', '24219.png', '01348.png', '12618.png', '00550.png', '12864.png', '13655.png', '17403.png', '08607.png', '16183.png', '20540.png', '04344.png', '02968.png', '11684.png', '14438.png', '00579.png', '10797.png', '18064.png', '21217.png', '04663.png', '18329.png', '02617.png', '12073.png', '14230.png', '07803.png', '00034.png', '08275.png', '05218.png', '16191.png', '20755.png', '06273.png', '14535.png', '16999.png', '14752.png', '04489.png', '14874.png', '15687.png', '22501.png', '02210.png', '08803.png', '17860.png', '13159.png', '07118.png', '16760.png', '17973.png', '19093.png', '09493.png', '07769.png', '24185.png', '09529.png', '19732.png', '16245.png', '06243.png', '22735.png', '12331.png', '22760.png', '11201.png', '05139.png', '15047.png', '07431.png', '03762.png', '11407.png', '17855.png', '21553.png', '00926.png', '16255.png', '23014.png', '22013.png', '20840.png', '12710.png', '18957.png', '11065.png', '11039.png', '18860.png', '08820.png', '21698.png', '15377.png', '04603.png', '02309.png', '02748.png', '21927.png', '24769.png', '24463.png', '06500.png', '00779.png', '20722.png', '17414.png', '20289.png', '23654.png', '07330.png', '05785.png', '15929.png', '05092.png', '05903.png', '18686.png', '11675.png', '18234.png', '01516.png', '23606.png', '11937.png', '00486.png', '05980.png', '08285.png', '08829.png', '23952.png', '05266.png', '00858.png', '08692.png', '22494.png', '10434.png', '04700.png', '04302.png', '12204.png', '02549.png', '10597.png', '18771.png', '18019.png', '14221.png', '04939.png', '22537.png', '13662.png', '01310.png', '19553.png', '12573.png', '21978.png', '19636.png', '07073.png', '24576.png', '05231.png', '09735.png', '24742.png', '15664.png', '10933.png', '09508.png', '13552.png', '02993.png', '17556.png', '18746.png', '17219.png', '06679.png', '11255.png', '05504.png', '03832.png', '23436.png', '13864.png', '17234.png', '23961.png', '21468.png', '16577.png', '18172.png', '24321.png', '14769.png', '01196.png', '03928.png', '02282.png', '18494.png', '13201.png', '11805.png', '09359.png', '09919.png', '00661.png', '05335.png', '12504.png', '08972.png', '11423.png', '04506.png', '11210.png', '06049.png', '01375.png', '08238.png', '11887.png', '09955.png', '12723.png', '22683.png', '06027.png', '24302.png', '03246.png', '07630.png', '21256.png', '19802.png', '00805.png', '00751.png', '20677.png', '09645.png', '11409.png', '10555.png', '00791.png', '22360.png', '09651.png', '00101.png', '22971.png', '12648.png', '23978.png', '08986.png', '09413.png', '09911.png', '02989.png', '19002.png', '15127.png', '09076.png', '23246.png', '11861.png', '21556.png', '02077.png', '03265.png', '09124.png', '13324.png', '14537.png', '12018.png', '05756.png', '06660.png', '09158.png', '01281.png', '20686.png', '08880.png', '03531.png', '13172.png', '20010.png', '13344.png', '02253.png', '15994.png', '01138.png', '02817.png', '03457.png', '20475.png', '12191.png', '23513.png', '12309.png', '24627.png', '04254.png', '20413.png', '10416.png', '19081.png', '06216.png', '21219.png', '24303.png', '01428.png', '08907.png', '22125.png', '04913.png', '01493.png', '03433.png', '19779.png', '23552.png', '16390.png', '04332.png', '01318.png', '04307.png', '01720.png', '16418.png', '02883.png', '01080.png', '18378.png', '12601.png', '14650.png', '03187.png', '20278.png', '18049.png', '08486.png', '21449.png', '17964.png', '20061.png', '09151.png', '10866.png', '05016.png', '06153.png', '05131.png', '07801.png', '17527.png', '11449.png', '19056.png', '09195.png', '06917.png', '23120.png', '00518.png', '22702.png', '14244.png', '14871.png', '17520.png', '21377.png', '19146.png', '20304.png', '15326.png', '12729.png', '05082.png', '13798.png', '19514.png', '07229.png', '02121.png', '08547.png', '14405.png', '14857.png', '13778.png', '18207.png', '03192.png', '17759.png', '10643.png', '24082.png', '09081.png', '03050.png', '24705.png', '01376.png', '18003.png', '04841.png', '14585.png', '00855.png', '02137.png', '14116.png', '00181.png', '04525.png', '17828.png', '08586.png', '06452.png', '17416.png', '08411.png', '18159.png', '19720.png', '23615.png', '09768.png', '13379.png', '15050.png', '14839.png', '11892.png', '14030.png', '05573.png', '13136.png', '10482.png', '07576.png', '01203.png', '12320.png', '14290.png', '13593.png', '16681.png', '24183.png', '12777.png', '09123.png', '24662.png', '10794.png', '24590.png', '18531.png', '07160.png', '18474.png', '23530.png', '17616.png', '21890.png', '09488.png', '22462.png', '02351.png', '00576.png', '12833.png', '24149.png', '07612.png', '03305.png', '05743.png', '17518.png', '04621.png', '14192.png', '11459.png', '06789.png', '24546.png', '11710.png', '14779.png', '19133.png', '20760.png', '18250.png', '17200.png', '07233.png', '15333.png', '24486.png', '19303.png', '01480.png', '12104.png', '14854.png', '14216.png', '12876.png', '16199.png', '13546.png', '12965.png', '05425.png', '01042.png', '17264.png', '19205.png', '21998.png', '00256.png', '06087.png', '19857.png', '23906.png', '12178.png', '20972.png', '24202.png', '03348.png', '08060.png', '19252.png', '21052.png', '09026.png', '07873.png', '07914.png', '05994.png', '01128.png', '09393.png', '01319.png', '03124.png', '05889.png', '01235.png', '19461.png', '23469.png', '16667.png', '13471.png', '11294.png', '22594.png', '10906.png', '11793.png', '19023.png', '06608.png', '03627.png', '20844.png', '03446.png', '14689.png', '09921.png', '24456.png', '16052.png', '01265.png', '06402.png', '22460.png', '19048.png', '17735.png', '10630.png', '13792.png', '23812.png', '05604.png', '16321.png', '16448.png', '06275.png', '22257.png', '15076.png', '07855.png', '02256.png', '22311.png', '00807.png', '10143.png', '06430.png', '08206.png', '16141.png', '04503.png', '24621.png', '00407.png', '06305.png', '04466.png', '02026.png', '07401.png', '03956.png', '06459.png', '06044.png', '01587.png', '10855.png', '21190.png', '17670.png', '01611.png', '12568.png', '09647.png', '24139.png', '24157.png', '03436.png', '23774.png', '12760.png', '11599.png', '13609.png', '02918.png', '00790.png', '05839.png', '22383.png', '10440.png', '12116.png', '12711.png', '04410.png', '24598.png', '14079.png', '00965.png', '14371.png', '07826.png', '07668.png', '14977.png', '08213.png', '15545.png', '20770.png', '01386.png', '17564.png', '15288.png', '16385.png', '07139.png', '07204.png', '03140.png', '16328.png', '01182.png', '14500.png', '01865.png', '03590.png', '06722.png', '21451.png', '08232.png', '04372.png', '15641.png', '07784.png', '22622.png', '01729.png', '03542.png', '15735.png', '13027.png', '03891.png', '08712.png', '20105.png', '14360.png', '19707.png', '11124.png', '15730.png', '13978.png', '21881.png', '03731.png', '17199.png', '24944.png', '15446.png', '02010.png', '16899.png', '10936.png', '18788.png', '01692.png', '24961.png', '04713.png', '05590.png', '12901.png', '16765.png', '04705.png', '12413.png', '14454.png', '19786.png', '06296.png', '17405.png', '06411.png', '09156.png', '11101.png', '24805.png', '07910.png', '14384.png', '04402.png', '08420.png', '19442.png', '00531.png', '19148.png', '00264.png', '07210.png', '01029.png', '11392.png', '01966.png', '11411.png', '13681.png', '09162.png', '21781.png', '01736.png', '05037.png', '04842.png', '23260.png', '01221.png', '11840.png', '22160.png', '15689.png', '20107.png', '12791.png', '05499.png', '16031.png', '04587.png', '05804.png', '00853.png', '01972.png', '21247.png', '06610.png', '06520.png', '02442.png', '07579.png', '12757.png', '12800.png', '15184.png', '00477.png', '07828.png', '05163.png', '18568.png', '02225.png', '02986.png', '04031.png', '14410.png', '15841.png', '10108.png', '18824.png', '19985.png', '01768.png', '04769.png', '05865.png', '09560.png', '04620.png', '02469.png', '20479.png', '20994.png', '05938.png', '08582.png', '06721.png', '21694.png', '16691.png', '12826.png', '20556.png', '06984.png', '15553.png', '22410.png', '05245.png', '12470.png', '15254.png', '11084.png', '07122.png', '10957.png', '20904.png', '19733.png', '08645.png', '01365.png', '13524.png', '08769.png', '01188.png', '08186.png', '06260.png', '07468.png', '14163.png', '09314.png', '02550.png', '02736.png', '20004.png', '24189.png', '15188.png', '21055.png', '14130.png', '05759.png', '23238.png', '09279.png', '15313.png', '19124.png', '17259.png', '22106.png', '17806.png', '17717.png', '18774.png', '20632.png', '16622.png', '13325.png', '23145.png', '23241.png', '02887.png', '03096.png', '12835.png', '19046.png', '09323.png', '06448.png', '04695.png', '06312.png', '17774.png', '20023.png', '06598.png', '11848.png', '09611.png', '07779.png', '02480.png', '17549.png', '12378.png', '15963.png', '05516.png', '24160.png', '11042.png', '21507.png', '08518.png', '03356.png', '18586.png', '03614.png', '15646.png', '14536.png', '22418.png', '21592.png', '16692.png', '08954.png', '18342.png', '24221.png', '17904.png', '23157.png', '20331.png', '15343.png', '17646.png', '23660.png', '08897.png', '00732.png', '21081.png', '00367.png', '19067.png', '01424.png', '20796.png', '03892.png', '21319.png', '23852.png', '04138.png', '01848.png', '13813.png', '02094.png', '07277.png', '09616.png', '02316.png', '01501.png', '24831.png', '12819.png', '10340.png', '22316.png', '02456.png', '04142.png', '01211.png', '03456.png', '01646.png', '02314.png', '22252.png', '18139.png', '17449.png', '19095.png', '06904.png', '12053.png', '08251.png', '24573.png', '20145.png', '23757.png', '07341.png', '02664.png', '19635.png', '06214.png', '15966.png', '12158.png', '06993.png', '15034.png', '16136.png', '09889.png', '01498.png', '07545.png', '13240.png', '05706.png', '23126.png', '03291.png', '05357.png', '19592.png', '00244.png', '17844.png', '24035.png', '04013.png', '15491.png', '00474.png', '20222.png', '22421.png', '01473.png', '02655.png', '15861.png', '22527.png', '02905.png', '03890.png', '07333.png', '05679.png', '13072.png', '05905.png', '14412.png', '02520.png', '18737.png', '15919.png', '09457.png', '23156.png', '20922.png', '03166.png', '22896.png', '11907.png', '15295.png', '10679.png', '18164.png', '18652.png', '11247.png', '08289.png', '24048.png', '16881.png', '05934.png', '23593.png', '08609.png', '12370.png', '05531.png', '04097.png', '17742.png', '20339.png', '18253.png', '01973.png', '22154.png', '02890.png', '06640.png', '15819.png', '17170.png', '17872.png', '23890.png']
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
        
        if i_iter == 410411:
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
