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
    a = ['18280.png', '24527.png', '08630.png', '21652.png', '11589.png', '02938.png', '04182.png', '22284.png', '11076.png', '15096.png', '07062.png', '12936.png', '09675.png', '16598.png', '02851.png', '18288.png', '12581.png', '11796.png', '16932.png', '21105.png', '05672.png', '13491.png', '24021.png', '11307.png', '14825.png', '06346.png', '10332.png', '05024.png', '08694.png', '12500.png', '23043.png', '11406.png', '19648.png', '00247.png', '24409.png', '18977.png', '21232.png', '20026.png', '22684.png', '11488.png', '24897.png', '05044.png', '06238.png', '08495.png', '08011.png', '24006.png', '02186.png', '10625.png', '03365.png', '04272.png', '02103.png', '16469.png', '19953.png', '12541.png', '14162.png', '02642.png', '18834.png', '05975.png', '08403.png', '12555.png', '12353.png', '02084.png', '04250.png', '24396.png', '21574.png', '10345.png', '08371.png', '09167.png', '21522.png', '21176.png', '19117.png', '08610.png', '03513.png', '00683.png', '14595.png', '21772.png', '21930.png', '05927.png', '12052.png', '12548.png', '07068.png', '08397.png', '08831.png', '08829.png', '08164.png', '17005.png', '01935.png', '12103.png', '20362.png', '24587.png', '03287.png', '10122.png', '04604.png', '00174.png', '20900.png', '18912.png', '04880.png', '05129.png', '20847.png', '14650.png', '24564.png', '07024.png', '03166.png', '09841.png', '24229.png', '17091.png', '19823.png', '21395.png', '04941.png', '11245.png', '03548.png', '17157.png', '06387.png', '05998.png', '09364.png', '10876.png', '16630.png', '21846.png', '11759.png', '14905.png', '04489.png', '09369.png', '15344.png', '00489.png', '13425.png', '08834.png', '01753.png', '12525.png', '23922.png', '11604.png', '20682.png', '09266.png', '07084.png', '06442.png', '16951.png', '10061.png', '05256.png', '08781.png', '21792.png', '16417.png', '10751.png', '00575.png', '15903.png', '23509.png', '09131.png', '20521.png', '16214.png', '21378.png', '03894.png', '07772.png', '07336.png', '11351.png', '13295.png', '14325.png', '19944.png', '07501.png', '12661.png', '08441.png', '06425.png', '20851.png', '00137.png', '12287.png', '07927.png', '08159.png', '06017.png', '03611.png', '07624.png', '04734.png', '13714.png', '13329.png', '18817.png', '22311.png', '23037.png', '17619.png', '05772.png', '05512.png', '10453.png', '12149.png', '11187.png', '24390.png', '10528.png', '01755.png', '10609.png', '04296.png', '10006.png', '01976.png', '15921.png', '03963.png', '18227.png', '05141.png', '03080.png', '12207.png', '09516.png', '06547.png', '13534.png', '11030.png', '10215.png', '22716.png', '13999.png', '22216.png', '00408.png', '24455.png', '24198.png', '11494.png', '23783.png', '09898.png', '03844.png', '18972.png', '23832.png', '08646.png', '00381.png', '14100.png', '03748.png', '04199.png', '16008.png', '23091.png', '00534.png', '01527.png', '00200.png', '08563.png', '02063.png', '00592.png', '23201.png', '04597.png', '12365.png', '07114.png', '16901.png', '11281.png', '00597.png', '01850.png', '10978.png', '21428.png', '19510.png', '14123.png', '18682.png', '05734.png', '07095.png', '19613.png', '12249.png', '00481.png', '22614.png', '16110.png', '05462.png', '21400.png', '18978.png', '09608.png', '01947.png', '00653.png', '20173.png', '12474.png', '01209.png', '19543.png', '24762.png', '24057.png', '24965.png', '15806.png', '00744.png', '18903.png', '20664.png', '09906.png', '24799.png', '18965.png', '22111.png', '05307.png', '19705.png', '14448.png', '10525.png', '15139.png', '11784.png', '22631.png', '16950.png', '16104.png', '06802.png', '11547.png', '04162.png', '00951.png', '11937.png', '11191.png', '22402.png', '08901.png', '14059.png', '09450.png', '15938.png', '14847.png', '07754.png', '09782.png', '07140.png', '09285.png', '08045.png', '00450.png', '09815.png', '21035.png', '08327.png', '16802.png', '23760.png', '11987.png', '20507.png', '12651.png', '00738.png', '02290.png', '05615.png', '12462.png', '09033.png', '06886.png', '19045.png', '10243.png', '03315.png', '14152.png', '11633.png', '17943.png', '04501.png', '17392.png', '16496.png', '16386.png', '04746.png', '04293.png', '12225.png', '15857.png', '00496.png', '09982.png', '02095.png', '14967.png', '14613.png', '12211.png', '24780.png', '04009.png', '18740.png', '08672.png', '03777.png', '07139.png', '12251.png', '09232.png', '17821.png', '22020.png', '14258.png', '23231.png', '05719.png', '01403.png', '24059.png', '11205.png', '14501.png', '13746.png', '09463.png', '19189.png', '21886.png', '06700.png', '14668.png', '01127.png', '17603.png', '17334.png', '24615.png', '14879.png', '23896.png', '02029.png', '06860.png', '05327.png', '08600.png', '09377.png', '15591.png', '22971.png', '18773.png', '24471.png', '13113.png', '24301.png', '10792.png', '06420.png', '13657.png', '03986.png', '20156.png', '08634.png', '15154.png', '23787.png', '22071.png', '19512.png', '17043.png', '21545.png', '00274.png', '08475.png', '11868.png', '02286.png', '01801.png', '19508.png', '14524.png', '01959.png', '23436.png', '12728.png', '21847.png', '15770.png', '18113.png', '24674.png', '14072.png', '07623.png', '13437.png', '21230.png', '06296.png', '13287.png', '05908.png', '15579.png', '00171.png', '01614.png', '14655.png', '20585.png', '23269.png', '22405.png', '24741.png', '09705.png', '09739.png', '01942.png', '19017.png', '23519.png', '10837.png', '05622.png', '01512.png', '20118.png', '00299.png', '09332.png', '23407.png', '18897.png', '08453.png', '23126.png', '08536.png', '09011.png', '15787.png', '12705.png', '12940.png', '02695.png', '09659.png', '10712.png', '11689.png', '14944.png', '21879.png', '08648.png', '12598.png', '23145.png', '00720.png', '24936.png', '07788.png', '02108.png', '24191.png', '02136.png', '13269.png', '09217.png', '17965.png', '14111.png', '21700.png', '19844.png', '15967.png', '01999.png', '02936.png', '06896.png', '01139.png', '08660.png', '03961.png', '24687.png', '02243.png', '01537.png', '01277.png', '24819.png', '23074.png', '03404.png', '18343.png', '07528.png', '22653.png', '10222.png', '15190.png', '10443.png', '14304.png', '21465.png', '07208.png', '23956.png', '09670.png', '15920.png', '16801.png', '05981.png', '18129.png', '03948.png', '11485.png', '02592.png', '19274.png', '06978.png', '16502.png', '11251.png', '05907.png', '19819.png', '17453.png', '23155.png', '10700.png', '22980.png', '18764.png', '03874.png', '23817.png', '23183.png', '01905.png', '11765.png', '07765.png', '20792.png', '07207.png', '05319.png', '20034.png', '09988.png', '14312.png', '15539.png', '05323.png', '09214.png', '07817.png', '07059.png', '14659.png', '18128.png', '11922.png', '22477.png', '16142.png', '03546.png', '18284.png', '16626.png', '15864.png', '19431.png', '15707.png', '02092.png', '11993.png', '15759.png', '12586.png', '07332.png', '16720.png', '17220.png', '23608.png', '04999.png', '05117.png', '23931.png', '05540.png', '11408.png', '08336.png', '04932.png', '24536.png', '11303.png', '13939.png', '14492.png', '00522.png', '12830.png', '08227.png', '09496.png', '19556.png', '13656.png', '08844.png', '14215.png', '16631.png', '09270.png', '11124.png', '06623.png', '24030.png', '02778.png', '11598.png', '01903.png', '18956.png', '16910.png', '01544.png', '00056.png', '11117.png', '08240.png', '17544.png', '13463.png', '20638.png', '11182.png', '13983.png', '11623.png', '00552.png', '08207.png', '19735.png', '03996.png', '02254.png', '14727.png', '22642.png', '22408.png', '13107.png', '06814.png', '08151.png', '21952.png', '08514.png', '10361.png', '13899.png', '21113.png', '03765.png', '24399.png', '13261.png', '03629.png', '12042.png', '06082.png', '24329.png', '12652.png', '00153.png', '02418.png', '10181.png', '14320.png', '22238.png', '15305.png', '11616.png', '01656.png', '11902.png', '13702.png', '00486.png', '17143.png', '21369.png', '02821.png', '15133.png', '18312.png', '15720.png', '16049.png', '09548.png', '18866.png', '19011.png', '02473.png', '04390.png', '24322.png', '06485.png', '16187.png', '04176.png', '22138.png', '08191.png', '01190.png', '24317.png', '01081.png', '19551.png', '01525.png', '09324.png', '16700.png', '17266.png', '05148.png', '23619.png', '22006.png', '07471.png', '18257.png', '01335.png', '07480.png', '23238.png', '10055.png', '07400.png', '22698.png', '01063.png', '01557.png', '20268.png', '11833.png', '18355.png', '14004.png', '19338.png', '08978.png', '06276.png', '07565.png', '13370.png', '04995.png', '22489.png', '06803.png', '03399.png', '02167.png', '12010.png', '14548.png', '04539.png', '05952.png', '03757.png', '02733.png', '09910.png', '05477.png', '01009.png', '16636.png', '13909.png', '18208.png', '22200.png', '15757.png', '00644.png', '24586.png', '14286.png', '13975.png', '03650.png', '18463.png', '06216.png', '22158.png', '18671.png', '23718.png', '01817.png', '04953.png', '15576.png', '20286.png', '05023.png', '04267.png', '11770.png', '24932.png', '00049.png', '14882.png', '18296.png', '08152.png', '01467.png', '14019.png', '20545.png', '17436.png', '19421.png', '17305.png', '01639.png', '02274.png', '16655.png', '23309.png', '23965.png', '14156.png', '05449.png', '08048.png', '02874.png', '04865.png', '01635.png', '08543.png', '19361.png', '22328.png', '00120.png', '17598.png', '17461.png', '24212.png', '12405.png', '14622.png', '07083.png', '02755.png', '03526.png', '00833.png', '06621.png', '17747.png', '15842.png', '18741.png', '10324.png', '19865.png', '08987.png', '21592.png', '08895.png', '01890.png', '13495.png', '21034.png', '01482.png', '20513.png', '19193.png', '10801.png', '23481.png', '03707.png', '14647.png', '07329.png', '21272.png', '04622.png', '23806.png', '11435.png', '22085.png', '00967.png', '19477.png', '09350.png', '07879.png', '24662.png', '00536.png', '07818.png', '14830.png', '21341.png', '23741.png', '12836.png', '15983.png', '22694.png', '23096.png', '16102.png', '12063.png', '07699.png', '04872.png', '06697.png', '21808.png', '09486.png', '08401.png', '16472.png', '06634.png', '12257.png', '03585.png', '00962.png', '17984.png', '04185.png', '24205.png', '14963.png', '22415.png', '16816.png', '13065.png', '13990.png', '21041.png', '23168.png', '23253.png', '23008.png', '01503.png', '13279.png', '00805.png', '11814.png', '04072.png', '15301.png', '18628.png', '06565.png', '17355.png', '08267.png', '14579.png', '08695.png', '02690.png', '21546.png', '01317.png', '13731.png', '14003.png', '12333.png', '18825.png', '22664.png', '15520.png', '23605.png', '13457.png', '13750.png', '14482.png', '05079.png', '12949.png', '07799.png', '05823.png', '00055.png', '12006.png', '09472.png', '21748.png', '05549.png', '07170.png', '04810.png', '03764.png', '05260.png', '02883.png', '09588.png', '16623.png', '20410.png', '17869.png', '11010.png', '11609.png', '23267.png', '22592.png', '01814.png', '16633.png', '08932.png', '14272.png', '06383.png', '18199.png', '00280.png', '21416.png', '03809.png', '19777.png', '07651.png', '02966.png', '18778.png', '13265.png', '22702.png', '05294.png', '04582.png', '20842.png', '10438.png', '09630.png', '09719.png', '05976.png', '24387.png', '16140.png', '07452.png', '10814.png', '20793.png', '15529.png', '10810.png', '11847.png', '21120.png', '18315.png', '15328.png', '01746.png', '05100.png', '07716.png', '05483.png', '21146.png', '16956.png', '17729.png', '13348.png', '05794.png', '08203.png', '17874.png', '20948.png', '18913.png', '15393.png', '15059.png', '07670.png', '15435.png', '00371.png', '02556.png', '07488.png', '06443.png', '19785.png', '08088.png', '07833.png', '01629.png', '22976.png', '09536.png', '01848.png', '24714.png', '10592.png', '13562.png', '23498.png', '24728.png', '12641.png', '16844.png', '09317.png', '12076.png', '10656.png', '22106.png', '05583.png', '23496.png', '11884.png', '07999.png', '22373.png', '05513.png', '01447.png', '04767.png', '00380.png', '03245.png', '22030.png', '01715.png', '19244.png', '23166.png', '16437.png', '21612.png', '21797.png', '19948.png', '00209.png', '00151.png', '03512.png', '09886.png', '03830.png', '05115.png', '11643.png', '16447.png', '03708.png', '23784.png', '22908.png', '14365.png', '14001.png', '08774.png', '14770.png', '15105.png', '17602.png', '01449.png', '06757.png', '18731.png', '02616.png', '23197.png', '09761.png', '04002.png', '03564.png', '16886.png', '02676.png', '01362.png', '13280.png', '05514.png', '01749.png', '04765.png', '00749.png', '08493.png', '00196.png', '18758.png', '16407.png', '07742.png', '01129.png', '24112.png', '12855.png', '03257.png', '21763.png', '05756.png', '14466.png', '12825.png', '08955.png', '24941.png', '21454.png', '17954.png', '19811.png', '11034.png', '00246.png', '05508.png', '13384.png', '23454.png', '11164.png', '22557.png', '12016.png', '03984.png', '02830.png', '05253.png', '01034.png', '22960.png', '21760.png', '23990.png', '23093.png', '19897.png', '16824.png', '00451.png', '06057.png', '08928.png', '11021.png', '08794.png', '02466.png', '02681.png', '03858.png', '05951.png', '05992.png', '19636.png', '13843.png', '06641.png', '07823.png', '05701.png', '15001.png', '08218.png', '18002.png', '19174.png', '08292.png', '02712.png', '19146.png', '04135.png', '01743.png', '00189.png', '14489.png', '03964.png', '24081.png', '19929.png', '04827.png', '17367.png', '14644.png', '19545.png', '24758.png', '09725.png', '21322.png', '17934.png', '04235.png', '22370.png', '02056.png', '07665.png', '10933.png', '10403.png', '20075.png', '12273.png', '15719.png', '10512.png', '00696.png', '15206.png', '13170.png', '08919.png', '18612.png', '05632.png', '02549.png', '02279.png', '05122.png', '09114.png', '05539.png', '23745.png', '24166.png', '09402.png', '16657.png', '06456.png', '09227.png', '07216.png', '20873.png', '16307.png', '08396.png', '00869.png', '00436.png', '02181.png', '23348.png', '00391.png', '08210.png', '22893.png', '17502.png', '09564.png', '22469.png', '07341.png', '22488.png', '04279.png', '17154.png', '06840.png', '15758.png', '24853.png', '08302.png', '18125.png', '20766.png', '19061.png', '22985.png', '10113.png', '05301.png', '09019.png', '17267.png', '24620.png', '08832.png', '14688.png', '08061.png', '18828.png', '16128.png', '01244.png', '21422.png', '08574.png', '10699.png', '09751.png', '03084.png', '20977.png', '17985.png', '09397.png', '07465.png', '20644.png', '13922.png', '18793.png', '18008.png', '01764.png', '21217.png', '08720.png', '17074.png', '01828.png', '18230.png', '17153.png', '16278.png', '17736.png', '12859.png', '02059.png', '01862.png', '18137.png', '02955.png', '21340.png', '12590.png', '02412.png', '02947.png', '03263.png', '23814.png', '09924.png', '07382.png', '05406.png', '00252.png', '06196.png', '11329.png', '06155.png', '10495.png', '00823.png', '23460.png', '19150.png', '15446.png', '01030.png', '09464.png', '09429.png', '10711.png', '08144.png', '06006.png', '05190.png', '03895.png', '04291.png', '07884.png', '19815.png', '11252.png', '01645.png', '08968.png', '01312.png', '04887.png', '11444.png', '03584.png', '15120.png', '22637.png', '13968.png', '00603.png', '04266.png', '22307.png', '12995.png', '13784.png', '12746.png', '12397.png', '10308.png', '03266.png', '19541.png', '05665.png', '15482.png', '21843.png', '10991.png', '06409.png', '14053.png', '17265.png', '21560.png', '24022.png', '15777.png', '05631.png', '05931.png', '09946.png', '22560.png', '10568.png', '11966.png', '23809.png', '01772.png', '07394.png', '00874.png', '13729.png', '08174.png', '06098.png', '04858.png', '12621.png', '01509.png', '16980.png', '17483.png', '05436.png', '17811.png', '16132.png', '20298.png', '03040.png', '00398.png', '16450.png', '08606.png', '23822.png', '09762.png', '16005.png', '20076.png', '15566.png', '24361.png', '12946.png', '18052.png', '02730.png', '01458.png', '04519.png', '21573.png', '01360.png', '24371.png', '17004.png', '22599.png', '03309.png', '00507.png', '21021.png', '08410.png', '11713.png', '10481.png', '12338.png', '14133.png', '08805.png', '22935.png', '16525.png', '06676.png', '07899.png', '07925.png', '13246.png', '08960.png', '11322.png', '24308.png', '12672.png', '14947.png', '18718.png', '08256.png', '02323.png', '15358.png', '18629.png', '24618.png', '07045.png', '01963.png', '21767.png', '02058.png', '01206.png', '07947.png', '04068.png', '19609.png', '03739.png', '00966.png', '09655.png', '19206.png', '02275.png', '19756.png', '13235.png', '13770.png', '10758.png', '07076.png', '24214.png', '11799.png', '20043.png', '10648.png', '09769.png', '15261.png', '14469.png', '24804.png', '14563.png', '04714.png', '18616.png', '07945.png', '13419.png', '07254.png', '03540.png', '15851.png', '22750.png', '03167.png', '20248.png', '09357.png', '04136.png', '23536.png', '21404.png', '18657.png', '16559.png', '19266.png', '16037.png', '03244.png', '17028.png', '20340.png', '24178.png', '19183.png', '11236.png', '06215.png', '12734.png', '19534.png', '05110.png', '19471.png', '17080.png', '00994.png', '03071.png', '19640.png', '18864.png', '13450.png', '24221.png', '08370.png', '07726.png', '01763.png', '08133.png', '07153.png', '13158.png', '17520.png', '22421.png', '23171.png', '19719.png', '09928.png', '00263.png', '07316.png', '01094.png', '23459.png', '00613.png', '24281.png', '09064.png', '20736.png', '10489.png', '06692.png', '18369.png', '21015.png', '10616.png', '00414.png', '03449.png', '12193.png', '15535.png', '22507.png', '23295.png', '24295.png', '23898.png', '09209.png', '08125.png', '23190.png', '16582.png', '24355.png', '00910.png', '24592.png', '03960.png', '22272.png', '09602.png', '12616.png', '23341.png', '15022.png', '03269.png', '16653.png', '12427.png', '11033.png', '15477.png', '08941.png', '13333.png', '22156.png', '18306.png', '00388.png', '15003.png', '08688.png', '13953.png', '17401.png', '12626.png', '10858.png', '08209.png', '17884.png', '17596.png', '09345.png', '02885.png', '16621.png', '16549.png', '15588.png', '11266.png', '12742.png', '20727.png', '19057.png', '05741.png', '14438.png', '15433.png', '20939.png', '18738.png', '07511.png', '23782.png', '10683.png', '06614.png', '14442.png', '16765.png', '18940.png', '16784.png', '06918.png', '00509.png', '06211.png', '00933.png', '17852.png', '04573.png', '13889.png', '23308.png', '21837.png', '11791.png', '03545.png', '18309.png', '22749.png', '16964.png', '05489.png', '20537.png', '24702.png', '10529.png', '24683.png', '09959.png', '22221.png', '06620.png', '17096.png', '14157.png', '05350.png', '15172.png', '04340.png', '02588.png', '07464.png', '10277.png', '08653.png', '16625.png', '01921.png', '07681.png', '01618.png', '13407.png', '20439.png', '03664.png', '16849.png', '05221.png', '19072.png', '09363.png', '21123.png', '22048.png', '00581.png', '08899.png', '09692.png', '10862.png', '03297.png', '15443.png', '05313.png', '03123.png', '16499.png', '04169.png', '09716.png', '16510.png', '14142.png', '23104.png', '08586.png', '05523.png', '09746.png', '21480.png', '04659.png', '10016.png', '04332.png', '15392.png', '09717.png', '20743.png', '13481.png', '08001.png', '13094.png', '18018.png', '19961.png', '24834.png', '20981.png', '08525.png', '06099.png', '08175.png', '22953.png', '15886.png', '05331.png', '21262.png', '01796.png', '24663.png', '02304.png', '11842.png', '17578.png', '12431.png', '12624.png', '23016.png', '08787.png', '13966.png', '12945.png', '17086.png', '17504.png', '08771.png', '13858.png', '08485.png', '04749.png', '00868.png', '10585.png', '22779.png', '16006.png', '09318.png', '22936.png', '03300.png', '20922.png', '05499.png', '23835.png', '21722.png', '16332.png', '13155.png', '12670.png', '05541.png', '04523.png', '10618.png', '15260.png', '05041.png', '04556.png', '01137.png', '24223.png', '04892.png', '08459.png', '15455.png', '09412.png', '02341.png', '20971.png', '08231.png', '01892.png', '07578.png', '05715.png', '18037.png', '11008.png', '08547.png', '10743.png', '08984.png', '13402.png', '16639.png', '01742.png', '15531.png', '20520.png', '15607.png', '01235.png', '22142.png', '24705.png', '07390.png', '00273.png', '00539.png', '19180.png', '15542.png', '06760.png', '14927.png', '16449.png', '16974.png', '12229.png', '13916.png', '20400.png', '20105.png', '21151.png', '02021.png', '16247.png', '23441.png', '08566.png', '07406.png', '07575.png', '04600.png', '04917.png', '10914.png', '24036.png', '14080.png', '02657.png', '09267.png', '04276.png', '22951.png', '04849.png', '06003.png', '02724.png', '10160.png', '01483.png', '18267.png', '22346.png', '08412.png', '24231.png', '18219.png', '07495.png', '16961.png', '15626.png', '23400.png', '18931.png', '15965.png', '22692.png', '04957.png', '17243.png', '00061.png', '04536.png', '09616.png', '24473.png', '00141.png', '07687.png', '07248.png', '06932.png', '10627.png', '18975.png', '02004.png', '00981.png', '22041.png', '01050.png', '06075.png', '05747.png', '02956.png', '20455.png', '22161.png', '09510.png', '23360.png', '23645.png', '13003.png', '23796.png', '16996.png', '03094.png', '09282.png', '22532.png', '09242.png', '19839.png', '23650.png', '13018.png', '24045.png', '13525.png', '19766.png', '04454.png', '06114.png', '14025.png', '14296.png', '14193.png', '02259.png', '09749.png', '06643.png', '11798.png', '08100.png', '03766.png', '02713.png', '15436.png', '21923.png', '16867.png', '15377.png', '18609.png', '14198.png', '15256.png', '19516.png', '18489.png', '05570.png', '17634.png', '00038.png', '05067.png', '01621.png', '07485.png', '12422.png', '19029.png', '21061.png', '21111.png', '04711.png', '11332.png', '05913.png', '05727.png', '03775.png', '24775.png', '00048.png', '22302.png', '23284.png', '18283.png', '15295.png', '08790.png', '21350.png', '13436.png', '21000.png', '23443.png', '21144.png', '05628.png', '13210.png', '02978.png', '21927.png', '19588.png', '00582.png', '20541.png', '08980.png', '11807.png', '05009.png', '07588.png', '20406.png', '13902.png', '21124.png', '13717.png', '23611.png', '18605.png', '06342.png', '06855.png', '12243.png', '00560.png', '06356.png', '05603.png', '04861.png', '02895.png', '13524.png', '24016.png', '10088.png', '04970.png', '17639.png', '18485.png', '24335.png', '04380.png', '05208.png', '07446.png', '19816.png', '16799.png', '03035.png', '18022.png', '09813.png', '19482.png', '11213.png', '06823.png', '14669.png', '19977.png', '06156.png', '08136.png', '22359.png', '14730.png', '04928.png', '10635.png', '20221.png', '20817.png', '20906.png', '05020.png', '18173.png', '03673.png', '12788.png', '15945.png', '22848.png', '17407.png', '15567.png', '04468.png', '06472.png', '08068.png', '02432.png', '14249.png', '03285.png', '18218.png', '24812.png', '09744.png', '09165.png', '05031.png', '18319.png', '17866.png', '12612.png', '22427.png', '22828.png', '22122.png', '13918.png', '19457.png', '06452.png', '20343.png', '03578.png', '08022.png', '11148.png', '17982.png', '08035.png', '03465.png', '24321.png', '09819.png', '15124.png', '09422.png', '19156.png', '03931.png', '02385.png', '00064.png', '20532.png', '15367.png', '20658.png', '10119.png', '04107.png', '19276.png', '03646.png', '00306.png', '08926.png', '03407.png', '16632.png', '18752.png', '04656.png', '11280.png', '24172.png', '18664.png', '07551.png', '21032.png', '01982.png', '02156.png', '01766.png', '09900.png', '15731.png', '16685.png', '02667.png', '05868.png', '07357.png', '21053.png', '10981.png', '03265.png', '01017.png', '11885.png', '22445.png', '13506.png', '24737.png', '00925.png', '08584.png', '20388.png', '22621.png', '03132.png', '00206.png', '07013.png', '19838.png', '18963.png', '16682.png', '03491.png', '07630.png', '04974.png', '10701.png', '06214.png', '00851.png', '11294.png', '10750.png', '15653.png', '11143.png', '08222.png', '07122.png', '06654.png', '01195.png', '04456.png', '23750.png', '13620.png', '22903.png', '03791.png', '10760.png', '18884.png', '23856.png', '09535.png', '11531.png', '04971.png', '11102.png', '14404.png', '01408.png', '02100.png', '15334.png', '13793.png', '02246.png', '22442.png', '00604.png', '23010.png', '02502.png', '24275.png', '06842.png', '04793.png', '14395.png', '12402.png', '04081.png', '12215.png', '24465.png', '19395.png', '11625.png', '19573.png', '14578.png', '20912.png', '14063.png', '19472.png', '13125.png', '16938.png', '19325.png', '22755.png', '02404.png', '21310.png', '06956.png', '10620.png', '23473.png', '09811.png', '15680.png', '18012.png', '06039.png', '20486.png', '09086.png', '14534.png', '11699.png', '24069.png', '03389.png', '24378.png', '22237.png', '24543.png', '22710.png', '21578.png', '04955.png', '19624.png', '04217.png', '19169.png', '03624.png', '02871.png', '00262.png', '22004.png', '00782.png', '05650.png', '21248.png', '17853.png', '21392.png', '00884.png', '21186.png', '10727.png', '17378.png', '08027.png', '12135.png', '22059.png', '16127.png', '21977.png', '14585.png', '04435.png', '23641.png', '14414.png', '12007.png', '04517.png', '10849.png', '11001.png', '12505.png', '08801.png', '02761.png', '10820.png', '07149.png', '11916.png', '08075.png', '04200.png', '01683.png', '09570.png', '15918.png', '15936.png', '00817.png', '21868.png', '20072.png', '02715.png', '11146.png', '23914.png', '14319.png', '16714.png', '17379.png', '04639.png', '22355.png', '20946.png', '05485.png', '12274.png', '24919.png', '18362.png', '17218.png', '24743.png', '13615.png', '02797.png', '11890.png', '08129.png', '19568.png', '08604.png', '21263.png', '15425.png', '22065.png', '05191.png', '20677.png', '09809.png', '02744.png', '02232.png', '11400.png', '24253.png', '12262.png', '23764.png', '23455.png', '07592.png', '18031.png', '15863.png', '04005.png', '06922.png', '04495.png', '22927.png', '06721.png', '09656.png', '03147.png', '20138.png', '07462.png', '16376.png', '02958.png', '14906.png', '22319.png', '02144.png', '18405.png', '12279.png', '01039.png', '07505.png', '05084.png', '08000.png', '12334.png', '24843.png', '15669.png', '12592.png', '04915.png', '14779.png', '17582.png', '14634.png', '15179.png', '00358.png', '08057.png', '00462.png', '10282.png', '03803.png', '01691.png', '00518.png', '19069.png', '01489.png', '18813.png', '03870.png', '17576.png', '16468.png', '04978.png', '17989.png', '12292.png', '06959.png', '10596.png', '22498.png', '24143.png', '06939.png', '04916.png', '20141.png', '15211.png', '10765.png', '22497.png', '03684.png', '13831.png', '22522.png', '02964.png', '08300.png', '22296.png', '10407.png', '20763.png', '20514.png', '23085.png', '03347.png', '04172.png', '24794.png', '22555.png', '05749.png', '08783.png', '16600.png', '17923.png', '15191.png', '06509.png', '02904.png', '03011.png', '02179.png', '08468.png', '08822.png', '12259.png', '00084.png', '07369.png', '11458.png', '14308.png', '04317.png', '15824.png', '09254.png', '09190.png', '01767.png', '17036.png', '00825.png', '05392.png', '13149.png', '14665.png', '15975.png', '10326.png', '01879.png', '03489.png', '10246.png', '04385.png', '01567.png', '11483.png', '01602.png', '07113.png', '01303.png', '14032.png', '14209.png', '10214.png', '23063.png', '06884.png', '23217.png', '06405.png', '05705.png', '01696.png', '01345.png', '08559.png', '23448.png', '09497.png', '10850.png', '24557.png', '01835.png', '16524.png', '24456.png', '22865.png', '05685.png', '03906.png', '11450.png', '11582.png', '16096.png', '06532.png', '01933.png', '21971.png', '07745.png', '14705.png', '19958.png', '23705.png', '16819.png', '00142.png', '13140.png', '21178.png', '16576.png', '21235.png', '14503.png', '08036.png', '22334.png', '04431.png', '23520.png', '13365.png', '19801.png', '05114.png', '02701.png', '00668.png', '03580.png', '16811.png', '04018.png', '16827.png', '20631.png', '11735.png', '12523.png', '21544.png', '15253.png', '13211.png', '06809.png', '21192.png', '00132.png', '19987.png', '08239.png', '14486.png', '14902.png', '09259.png', '05036.png', '00554.png', '16766.png', '13104.png', '01208.png', '13748.png', '00440.png', '24358.png', '01579.png', '05057.png', '01846.png', '01280.png', '08590.png', '20660.png', '23462.png', '24393.png', '20947.png', '12380.png', '09893.png', '20031.png', '20160.png', '08225.png', '00930.png', '09108.png', '09202.png', '20481.png', '10242.png', '16790.png', '04000.png', '01895.png', '14186.png', '20007.png', '05845.png', '09579.png', '02027.png', '07610.png', '16393.png', '08154.png', '15974.png', '04423.png', '02081.png', '00465.png', '21325.png', '09822.png', '00355.png', '01424.png', '09393.png', '05233.png', '01358.png', '19996.png', '13188.png', '15686.png', '01118.png', '00095.png', '19939.png', '09997.png', '12674.png', '19433.png', '11306.png', '05005.png', '07143.png', '14402.png', '17396.png', '15214.png', '10847.png', '06941.png', '04816.png', '18584.png', '23280.png', '11479.png', '01670.png', '19661.png', '14382.png', '23034.png', '08954.png', '03467.png', '20632.png', '04856.png', '10041.png', '01250.png', '06302.png', '24319.png', '24569.png', '06590.png', '10369.png', '13321.png', '03887.png', '13933.png', '20633.png', '20937.png', '18253.png', '05398.png', '22285.png', '10526.png', '13761.png', '07130.png', '08605.png', '23262.png', '13455.png', '11720.png', '13885.png', '19216.png', '10379.png', '04349.png', '22633.png', '09192.png', '07639.png', '14910.png', '04444.png', '15289.png', '24259.png', '19449.png', '21687.png', '14437.png', '01105.png', '04874.png', '06745.png', '22836.png', '01549.png', '05752.png', '18953.png', '18791.png', '09070.png', '24233.png', '06036.png', '24067.png', '10342.png', '12026.png', '11972.png', '17351.png', '21122.png', '19272.png', '16750.png', '15990.png', '06126.png', '21379.png', '10963.png', '08072.png', '08066.png', '18045.png', '15810.png', '16322.png', '00457.png', '12969.png', '06832.png', '11866.png', '11194.png', '14691.png', '06384.png', '05588.png', '05287.png', '09174.png', '11374.png', '16501.png', '03001.png', '05883.png', '06379.png', '14920.png', '13931.png', '01263.png', '02631.png', '00938.png', '03591.png', '19493.png', '05584.png', '11083.png', '09308.png', '14738.png', '20512.png', '14849.png', '06182.png', '08785.png', '04383.png', '10186.png', '01253.png', '03059.png', '14875.png', '14860.png', '16490.png', '12569.png', '19595.png', '21906.png', '04314.png', '17537.png', '12086.png', '18578.png', '19102.png', '23098.png', '21658.png', '01135.png', '08291.png', '12315.png', '23256.png', '12620.png', '04133.png', '09596.png', '06929.png', '02061.png', '03658.png', '20168.png', '20523.png', '12175.png', '20788.png', '13876.png', '18303.png', '15387.png', '21552.png', '05103.png', '08197.png', '01590.png', '05819.png', '15132.png', '14322.png', '12084.png', '05965.png', '04664.png', '02941.png', '02887.png', '24869.png', '19623.png', '10769.png', '00399.png', '08561.png', '21973.png', '17974.png', '05776.png', '11726.png', '21203.png', '21758.png', '00863.png', '16577.png', '19714.png', '23397.png', '10831.png', '01193.png', '04118.png', '12150.png', '01383.png', '22602.png', '09646.png', '06466.png', '04801.png', '00846.png', '06016.png', '05234.png', '06093.png', '16177.png', '21166.png', '00488.png', '23196.png', '14516.png', '14886.png', '23627.png', '16523.png', '21188.png', '04503.png', '15628.png', '19687.png', '23288.png', '12843.png', '14788.png', '04008.png', '17101.png', '03117.png', '09384.png', '20180.png', '07563.png', '07564.png', '10248.png', '04616.png', '05812.png', '10838.png', '22165.png', '16796.png', '16479.png', '16594.png', '00019.png', '05937.png', '22056.png', '07747.png', '20893.png', '16060.png', '14329.png', '14484.png', '24862.png', '03486.png', '00235.png', '12747.png', '00548.png', '24429.png', '04735.png', '23058.png', '24135.png', '02356.png', '10611.png', '02480.png', '03272.png', '16492.png', '14159.png', '21890.png', '00139.png', '01046.png', '19575.png', '11880.png', '10970.png', '11282.png', '11298.png', '18535.png', '07534.png', '21713.png', '07793.png', '21070.png', '21302.png', '13736.png', '17988.png', '03822.png', '17525.png', '18480.png', '05695.png', '22660.png', '16722.png', '24935.png', '10297.png', '18508.png', '23300.png', '13802.png', '06273.png', '13398.png', '15234.png', '14862.png', '23366.png', '07895.png', '12629.png', '19732.png', '21210.png', '18088.png', '01873.png', '24384.png', '13623.png', '14726.png', '24300.png', '20254.png', '23495.png', '21679.png', '03052.png', '08792.png', '11480.png', '08200.png', '16182.png', '24917.png', '23006.png', '00379.png', '09870.png', '22691.png', '12289.png', '12141.png', '08021.png', '11183.png', '04404.png', '24823.png', '01170.png', '23429.png', '24004.png', '07729.png', '03768.png', '05760.png', '13812.png', '04059.png', '20901.png', '23020.png', '12493.png', '23110.png', '02018.png', '16914.png', '06369.png', '24243.png', '20319.png', '19574.png', '24497.png', '19890.png', '00944.png', '06573.png', '05181.png', '08972.png', '16196.png', '22690.png', '09707.png', '20425.png', '13544.png', '22454.png', '24121.png', '06426.png', '05450.png', '04445.png', '14811.png', '04157.png', '15415.png', '01833.png', '09446.png', '18422.png', '05411.png', '05912.png', '17352.png', '03292.png', '07229.png', '01811.png', '08760.png', '05922.png', '23556.png', '08873.png', '05856.png', '04303.png', '23942.png', '17406.png', '10915.png', '00313.png', '10372.png', '21984.png', '18580.png', '21579.png', '01117.png', '14986.png', '12240.png', '14547.png', '05997.png', '21071.png', '07968.png', '10358.png', '07286.png', '14670.png', '22540.png', '03410.png', '14297.png', '07902.png', '00947.png', '19464.png', '06986.png', '16676.png', '23144.png', '20530.png', '04190.png', '17519.png', '17047.png', '05799.png', '22946.png', '24196.png', '08074.png', '02935.png', '16841.png', '07188.png', '11953.png', '08775.png', '13074.png', '22101.png', '02766.png', '20814.png', '18222.png', '24523.png', '00054.png', '10138.png', '24365.png', '17168.png', '13912.png', '00735.png', '15639.png', '09629.png', '19492.png', '22139.png', '05040.png', '01343.png', '11850.png', '22080.png', '02943.png', '08063.png', '00288.png', '10157.png', '14350.png', '19181.png', '09697.png', '00395.png', '05060.png', '14949.png', '09921.png', '12126.png', '20322.png', '05076.png', '11417.png', '03382.png', '19327.png', '14390.png', '12668.png', '24660.png', '17717.png', '13311.png', '15168.png', '24661.png', '14956.png', '15495.png', '02940.png', '02288.png', '01872.png', '12342.png', '21198.png', '19651.png', '23178.png', '17071.png', '06663.png', '15161.png', '22778.png', '00770.png', '16212.png', '09837.png', '09674.png', '15559.png', '18815.png', '07822.png', '06544.png', '12554.png', '14436.png', '05033.png', '08573.png', '23229.png', '01407.png', '13580.png', '17206.png', '10937.png', '04400.png', '02930.png', '13936.png', '17368.png', '22253.png', '08234.png', '18739.png', '19507.png', '12413.png', '15419.png', '24878.png', '18028.png', '20450.png', '08619.png', '12739.png', '07030.png', '14121.png', '23780.png', '17861.png', '15147.png', '04770.png', '07689.png', '08182.png', '16999.png', '15970.png', '05900.png', '08943.png', '01844.png', '13275.png', '02689.png', '23444.png', '13035.png', '09410.png', '17836.png', '11656.png', '14897.png', '22170.png', '10129.png', '04273.png', '17108.png', '19260.png', '23587.png', '19369.png', '13955.png', '14834.png', '07905.png', '13102.png', '04683.png', '06221.png', '06955.png', '24777.png', '04023.png', '07082.png', '20014.png', '04210.png', '15457.png', '24538.png', '04396.png', '23466.png', '12496.png', '17298.png', '00946.png', '04654.png', '00296.png', '07751.png', '15466.png', '22268.png', '11931.png', '12401.png', '11695.png', '23711.png', '12070.png', '14708.png', '23322.png', '15233.png', '06653.png', '23468.png', '08384.png', '16835.png', '08620.png', '01589.png', '10474.png', '04011.png', '22544.png', '08420.png', '07992.png', '15969.png', '13424.png', '10162.png', '21278.png', '23819.png', '08652.png', '04027.png', '20859.png', '23121.png', '08691.png', '14166.png', '02970.png', '12540.png', '04817.png', '14844.png', '24844.png', '05971.png', '16463.png', '15008.png', '14291.png', '24190.png', '05469.png', '08433.png', '12322.png', '09068.png', '19467.png', '12034.png', '21030.png', '12994.png', '21605.png', '04666.png', '16970.png', '11949.png', '06554.png', '00433.png', '11523.png', '20469.png', '13304.png', '03855.png', '15011.png', '10630.png', '09279.png', '05592.png', '21997.png', '09594.png', '08896.png', '23087.png', '06779.png', '14368.png', '07182.png', '20407.png', '10503.png', '10347.png', '19419.png', '17690.png', '14301.png', '17933.png', '12628.png', '03508.png', '24937.png', '07883.png', '10368.png', '13033.png', '20929.png', '08004.png', '06891.png', '07712.png', '10177.png', '18225.png', '09721.png', '04111.png', '12375.png', '19203.png', '09180.png', '12789.png', '18260.png', '12216.png', '11348.png', '03800.png', '08782.png', '10504.png', '04574.png', '01682.png', '15205.png', '05374.png', '20333.png', '09043.png', '22298.png', '10702.png', '00950.png', '09838.png', '23816.png', '07896.png', '08580.png', '09730.png', '18295.png', '03444.png', '23442.png', '02551.png', '18877.png', '03862.png', '17018.png', '04275.png', '15841.png', '06100.png', '16076.png', '11122.png', '14191.png', '01613.png', '07311.png', '03522.png', '10196.png', '22168.png', '16862.png', '09992.png', '16797.png', '15632.png', '23607.png', '23678.png', '09702.png', '09207.png', '23182.png', '16875.png', '21003.png', '06390.png', '00871.png', '23642.png', '10542.png', '14055.png', '24051.png', '15907.png', '22447.png', '03924.png', '22634.png', '13819.png', '17633.png', '07944.png', '03335.png', '22025.png', '17611.png', '10549.png', '05194.png', '09134.png', '21356.png', '14027.png', '05133.png', '09297.png', '17643.png', '12758.png', '12466.png', '04471.png', '13668.png', '24910.png', '04410.png', '11717.png', '19032.png', '05590.png', '18845.png', '12735.png', '05932.png', '06772.png', '03840.png', '19261.png', '08440.png', '22225.png', '08049.png', '05638.png', '17139.png', '03839.png', '02086.png', '13044.png', '24489.png', '05963.png', '09876.png', '10206.png', '15698.png', '21437.png', '03215.png', '18797.png', '02989.png', '00929.png', '22468.png', '00474.png', '11548.png', '13255.png', '14510.png', '23658.png', '13747.png', '14073.png', '12571.png', '16477.png', '16815.png', '01439.png', '20742.png', '12593.png', '14836.png', '21501.png', '07830.png', '10331.png', '14610.png', '11154.png', '20087.png', '05120.png', '14273.png', '16316.png', '10839.png', '17167.png', '11481.png', '19922.png', '04350.png', '08579.png', '08664.png', '24192.png', '08548.png', '04419.png', '11190.png', '09322.png', '22699.png', '16399.png', '19005.png', '18330.png', '11992.png', '00341.png', '13723.png', '17308.png', '21159.png', '21594.png', '17608.png', '20801.png', '20451.png', '18340.png', '10583.png', '14237.png', '01438.png', '03712.png', '19781.png', '13907.png', '22990.png', '10819.png', '08463.png', '21728.png', '03028.png', '14171.png', '17539.png', '18258.png', '00422.png', '01984.png', '06504.png', '06885.png', '05216.png', '05055.png', '17231.png', '22717.png', '24720.png', '05988.png', '12167.png', '00941.png', '22093.png', '22393.png', '16895.png', '00718.png', '21648.png', '20039.png', '16357.png', '10124.png', '17679.png', '20337.png', '02451.png', '20280.png', '05424.png', '16917.png', '14888.png', '07094.png', '20262.png', '11109.png', '23524.png', '10641.png', '20423.png', '04370.png', '10657.png', '06674.png', '08687.png', '12597.png', '13460.png', '12725.png', '14859.png', '02467.png', '12004.png', '11581.png', '17057.png', '18719.png', '19864.png', '03227.png', '02829.png', '21757.png', '01532.png', '14234.png', '07087.png', '08993.png', '19709.png', '06004.png', '24833.png', '16969.png', '02047.png', '10210.png', '03418.png', '02721.png', '09286.png', '01542.png', '06810.png', '24138.png', '14542.png', '06144.png', '08105.png', '16066.png', '09475.png', '24164.png', '18679.png', '03659.png', '00122.png', '03026.png', '05325.png', '06753.png', '14912.png', '10440.png', '09103.png', '10646.png', '17476.png', '16704.png', '17734.png', '10114.png', '22987.png', '16179.png', '16138.png', '14519.png', '11991.png', '00755.png', '05151.png', '20295.png', '05416.png', '17326.png', '14656.png', '00897.png', '04931.png', '13864.png', '03165.png', '22663.png', '07566.png', '05328.png', '06825.png', '07882.png', '02400.png', '16742.png', '20909.png', '16216.png', '21632.png', '08101.png', '11545.png', '01913.png', '18142.png', '01354.png', '09385.png', '24597.png', '04958.png', '04972.png', '04607.png', '07482.png', '08220.png', '10969.png', '19453.png', '01442.png', '18774.png', '02461.png', '18891.png', '05475.png', '16850.png', '15403.png', '09951.png', '06608.png', '11838.png', '15048.png', '16344.png', '22895.png', '10258.png', '05844.png', '00168.png', '03841.png', '00359.png', '08430.png', '16550.png', '15231.png', '06169.png', '20753.png', '03256.png', '18367.png', '16546.png', '12847.png', '15000.png', '24068.png', '21833.png', '22463.png', '01551.png', '22205.png', '20290.png', '21918.png', '18927.png', '23094.png', '19498.png', '24342.png', '23569.png', '00517.png', '09525.png', '09042.png', '18809.png', '12864.png', '09779.png', '16556.png', '03212.png', '03477.png', '17040.png', '09151.png', '14587.png', '07965.png', '17492.png', '07258.png', '07039.png', '06515.png', '01554.png', '24165.png', '12762.png', '22448.png', '04603.png', '15073.png', '23124.png', '12244.png', '14358.png', '11605.png', '20152.png', '22556.png', '11497.png', '05021.png', '19835.png', '23765.png', '23025.png', '16310.png', '12208.png', '22566.png', '19896.png', '22159.png', '06755.png', '14933.png', '01356.png', '15412.png', '14786.png', '13297.png', '01893.png', '13136.png', '09549.png', '21493.png', '10510.png', '04727.png', '09144.png', '07940.png', '24228.png', '02350.png', '14881.png', '17387.png', '22096.png', '20259.png', '03222.png', '22525.png', '08830.png', '10575.png', '00519.png', '18670.png', '08404.png', '08285.png', '09676.png', '04327.png', '04417.png', '13114.png', '06622.png', '04870.png', '06574.png', '11588.png', '14396.png', '03213.png', '14714.png', '01052.png', '12646.png', '06040.png', '03380.png', '19990.png', '05750.png', '02814.png', '24232.png', '19503.png', '06636.png', '18553.png', '15752.png', '20208.png', '22945.png', '03918.png', '18933.png', '06418.png', '10570.png', '23665.png', '15071.png', '19296.png', '02271.png', '01646.png', '11462.png', '18805.png', '05370.png', '04693.png', '05487.png', '16679.png', '02763.png', '17382.png', '11243.png', '01962.png', '08915.png', '04893.png', '08290.png', '07415.png', '22300.png', '11240.png', '17966.png', '08447.png', '16569.png', '10807.png', '10008.png', '18695.png', '15715.png', '14384.png', '10327.png', '07862.png', '10500.png', '17645.png', '04564.png', '10891.png', '20068.png', '02326.png', '00730.png', '03752.png', '06687.png', '16296.png', '20121.png', '20355.png', '01271.png', '15767.png', '10610.png', '10249.png', '08351.png', '07982.png', '15363.png', '01265.png', '01291.png', '17427.png', '05375.png', '00212.png', '18641.png', '15706.png', '18150.png', '15634.png', '18642.png', '19527.png', '22154.png', '13605.png', '04993.png', '09993.png', '14945.png', '14732.png', '17947.png', '03105.png', '08298.png', '05282.png', '10292.png', '20440.png', '07631.png', '21294.png', '05480.png', '20647.png', '07946.png', '02138.png', '24362.png', '10540.png', '00116.png', '19250.png', '05295.png', '01575.png', '06833.png', '01163.png', '12773.png', '17949.png', '23237.png', '16820.png', '04058.png', '00502.png', '03971.png', '21794.png', '01842.png', '23050.png', '02442.png', '03000.png', '21452.png', '03478.png', '21292.png', '22907.png', '24886.png', '20006.png', '15326.png', '09438.png', '17751.png', '17171.png', '14149.png', '06305.png', '15280.png', '17234.png', '14196.png', '21072.png', '23853.png', '20640.png', '17839.png', '05182.png', '17335.png', '22777.png', '03604.png', '19359.png', '21881.png', '01332.png', '13813.png', '05073.png', '02988.png', '05290.png', '00976.png', '11309.png', '21504.png', '01249.png', '13045.png', '03823.png', '11274.png', '15968.png', '23748.png', '23474.png', '00685.png', '21234.png', '13796.png', '23428.png', '05078.png', '10325.png', '00369.png', '01400.png', '23337.png', '06124.png', '07561.png', '16168.png', '11915.png', '22289.png', '08261.png', '02199.png', '23924.png', '00859.png', '12703.png', '06870.png', '09638.png', '12341.png', '21209.png', '12484.png', '15448.png', '24651.png', '10387.png', '16488.png', '19829.png', '20306.png', '12681.png', '14870.png', '18285.png', '19750.png', '17027.png', '07811.png', '24814.png', '01019.png', '16312.png', '08757.png', '16157.png', '13462.png', '10645.png', '21339.png', '02114.png', '24208.png', '22819.png', '03710.png', '17766.png', '05744.png', '00129.png', '00712.png', '21960.png', '15597.png', '03013.png', '11399.png', '22125.png', '17452.png', '01212.png', '05205.png', '15521.png', '07658.png', '18966.png', '17339.png', '11684.png', '09454.png', '08284.png', '00746.png', '15880.png', '05544.png', '17632.png', '15687.png', '16144.png', '13051.png', '19478.png', '00945.png', '02969.png', '12340.png', '19725.png', '06523.png', '10531.png', '19880.png', '15391.png', '09509.png', '24735.png', '24315.png', '00329.png', '08945.png', '21811.png', '24230.png', '18083.png', '13202.png', '09146.png', '10976.png', '00978.png', '12179.png', '11636.png', '04026.png', '16263.png', '20495.png', '11632.png', '09816.png', '14239.png', '18415.png', '18085.png', '06569.png', '04797.png', '17144.png', '15957.png', '19293.png', '00138.png', '14224.png', '20542.png', '01120.png', '07375.png', '03223.png', '04492.png', '13712.png', '07418.png', '10056.png', '11909.png', '16809.png', '21284.png', '24629.png', '02807.png', '06164.png', '10855.png', '02595.png', '18458.png', '02034.png', '15072.png', '14545.png', '16258.png', '23549.png', '04551.png', '00469.png', '00304.png', '15813.png', '07698.png', '15939.png', '23029.png', '13568.png', '08766.png', '12469.png', '00443.png', '02485.png', '19616.png', '03152.png', '03032.png', '17381.png', '13240.png', '11046.png', '18627.png', '11037.png', '06064.png', '18583.png', '03801.png', '02168.png', '16184.png', '07536.png', '23777.png', '19817.png', '00082.png', '16456.png', '17186.png', '08809.png', '07682.png', '16275.png', '22786.png', '07593.png', '12714.png', '01768.png', '23265.png', '04240.png', '12657.png', '24761.png', '07714.png', '19443.png', '14857.png', '24650.png', '23604.png', '21128.png', '01245.png', '11862.png', '00423.png', '19814.png', '21873.png', '10030.png', '16543.png', '03392.png', '01562.png', '05656.png', '07941.png', '22038.png', '08166.png', '21121.png', '24934.png', '14497.png', '07414.png', '02462.png', '24189.png', '11134.png', '23351.png', '17559.png', '06494.png', '08118.png', '15040.png', '13633.png', '11448.png', '24448.png', '03301.png', '07409.png', '02740.png', '07757.png', '14678.png', '24340.png', '12966.png', '13101.png', '15895.png', '13443.png', '22036.png', '02952.png', '10709.png', '15359.png', '17418.png', '11956.png', '02815.png', '00987.png', '20307.png', '07850.png', '03171.png', '10095.png', '12111.png', '14241.png', '00681.png', '22622.png', '14989.png', '08137.png', '07177.png', '15306.png', '03897.png', '20106.png', '12281.png', '01689.png', '16178.png', '08837.png', '08253.png', '05901.png', '19533.png', '17204.png', '21127.png', '05317.png', '14502.png', '10250.png', '10717.png', '01398.png', '04394.png', '23059.png', '06836.png', '00888.png', '09836.png', '19043.png', '15711.png', '09775.png', '13230.png', '02391.png', '05366.png', '19673.png', '18706.png', '15251.png', '08882.png', '06121.png', '15056.png', '04653.png', '15137.png', '15054.png', '04425.png', '21718.png', '18900.png', '01054.png', '15916.png', '03951.png', '16064.png', '02529.png', '17659.png', '06230.png', '04994.png', '11499.png', '16414.png', '20458.png', '23336.png', '09420.png', '22478.png', '05412.png', '13416.png', '15195.png', '10731.png', '11516.png', '18932.png', '22687.png', '17215.png', '13790.png', '03450.png', '22192.png', '11090.png', '01037.png', '16687.png', '23387.png', '03445.png', '11319.png', '10674.png', '04581.png', '18217.png', '06127.png', '09213.png', '23055.png', '16663.png', '18292.png', '21633.png', '12332.png', '14285.png', '02858.png', '00986.png', '12576.png', '05569.png', '15774.png', '08697.png', '02306.png', '14937.png', '04591.png', '11762.png', '24470.png', '10097.png', '18510.png', '10637.png', '00725.png', '08918.png', '06347.png', '06295.png', '00316.png', '07219.png', '03620.png', '10643.png', '20838.png', '18200.png', '08910.png', '12545.png', '23125.png', '20030.png', '17782.png', '04703.png', '07618.png', '13093.png', '16035.png', '06408.png', '06279.png', '13555.png', '11445.png', '12079.png', '10262.png', '17371.png', '09693.png', '19710.png', '05088.png', '02629.png', '07453.png', '17772.png', '03994.png', '06382.png', '08626.png', '14787.png', '19571.png', '05942.png', '24561.png', '05051.png', '09002.png', '19144.png', '10413.png', '05059.png', '13987.png', '02873.png', '00520.png', '20376.png', '01548.png', '21225.png', '14932.png', '21445.png', '10385.png', '22627.png', '18170.png', '11739.png', '24074.png', '04511.png', '13952.png', '11216.png', '02405.png', '08503.png', '08187.png', '22673.png', '11900.png', '22656.png', '05582.png', '20952.png', '18138.png', '22411.png', '00583.png', '04287.png', '02673.png', '19314.png', '18229.png', '01186.png', '24692.png', '19587.png', '15031.png', '01576.png', '15874.png', '07885.png', '10217.png', '00954.png', '07917.png', '16116.png', '00258.png', '07887.png', '07930.png', '15737.png', '17357.png', '05625.png', '06813.png', '16786.png', '06361.png', '16644.png', '08920.png', '10460.png', '21048.png', '24338.png', '00508.png', '09122.png', '02594.png', '16930.png', '06218.png', '17560.png', '01285.png', '19317.png', '00208.png', '00832.png', '10354.png', '07962.png', '01830.png', '23517.png', '13403.png', '00688.png', '23557.png', '15681.png', '11854.png', '01382.png', '13921.png', '07335.png', '03549.png', '00988.png', '13203.png', '13728.png', '21845.png', '24254.png', '07455.png', '15445.png', '09432.png', '13300.png', '19979.png', '16843.png', '14280.png', '23926.png', '08622.png', '00998.png', '08334.png', '10946.png', '23130.png', '01958.png', '00311.png', '17290.png', '00605.png', '10885.png', '13108.png', '15200.png', '11291.png', '05261.png', '18943.png', '02068.png', '11442.png', '06415.png', '18595.png', '21626.png', '22199.png', '18783.png', '23084.png', '20877.png', '12438.png', '03929.png', '08097.png', '22734.png', '14763.png', '24424.png', '11272.png', '01179.png', '05486.png', '04204.png', '17247.png', '08770.png', '18586.png', '06020.png', '08681.png', '05759.png', '08380.png', '09512.png', '13700.png', '11517.png', '17895.png', '19930.png', '04071.png', '18298.png', '19368.png', '20560.png', '24238.png', '10572.png', '21126.png', '02992.png', '12993.png', '06958.png', '09200.png', '05064.png', '14381.png', '13142.png', '14174.png', '13231.png', '12237.png', '22906.png', '06903.png', '17399.png', '17849.png', '23292.png', '06366.png', '12684.png', '04032.png', '06761.png', '21435.png', '18717.png', '17429.png', '21622.png', '01128.png', '03251.png', '10167.png', '09976.png', '02344.png', '10134.png', '09145.png', '10696.png', '22682.png', '11487.png', '21907.png', '22208.png', '10730.png', '18486.png', '06286.png', '03566.png', '02318.png', '19871.png', '07572.png', '05935.png', '22306.png', '02971.png', '07010.png', '10192.png', '00541.png', '18871.png', '21315.png', '03020.png', '16677.png', '10343.png', '15010.png', '01662.png', '04091.png', '11095.png', '20543.png', '02334.png', '08587.png', '05959.png', '03182.png', '13681.png', '21540.png', '05723.png', '07813.png', '02116.png', '24007.png', '09130.png', '10067.png', '24740.png', '11232.png', '08176.png', '12168.png', '20283.png', '16321.png', '00007.png', '13772.png', '10561.png', '24210.png', '07953.png', '06268.png', '16331.png', '22190.png', '06096.png', '09832.png', '24624.png', '01007.png', '19717.png', '21051.png', '23722.png', '08507.png', '14360.png', '21270.png', '00983.png', '21787.png', '10490.png', '00876.png', '10896.png', '13380.png', '08413.png', '22383.png', '15712.png', '11587.png', '10023.png', '24140.png', '08949.png', '04030.png', '04783.png', '15922.png', '07035.png', '17273.png', '01511.png', '05969.png', '19042.png', '03529.png', '18604.png', '23127.png', '04430.png', '06200.png', '11404.png', '22932.png', '01493.png', '17232.png', '01257.png', '08016.png', '05409.png', '09170.png', '07923.png', '04405.png', '00893.png', '21361.png', '17787.png', '10905.png', '11888.png', '01331.png', '16868.png', '11628.png', '16798.png', '15002.png', '12081.png', '16294.png', '09466.png', '06107.png', '07926.png', '07034.png', '24724.png', '00731.png', '02802.png', '18694.png', '17897.png', '15163.png', '18239.png', '11018.png', '05142.png', '11813.png', '20836.png', '02444.png', '05027.png', '02998.png', '21508.png', '07397.png', '23714.png', '21791.png', '10584.png', '02900.png', '07853.png', '02360.png', '02982.png', '18122.png', '24147.png', '21354.png', '22434.png', '06549.png', '09320.png', '11540.png', '16172.png', '23273.png', '20681.png', '09607.png', '19536.png', '18181.png', '24313.png', '11634.png', '18034.png', '02247.png', '07866.png', '02720.png', '19389.png', '07849.png', '08662.png', '01667.png', '12929.png', '14943.png', '07696.png', '06451.png', '01083.png', '08123.png', '04475.png', '10350.png', '12663.png', '01315.png', '19114.png', '18441.png', '12803.png', '09371.png', '19400.png', '06646.png', '06516.png', '15629.png', '11872.png', '21823.png', '20194.png', '06395.png', '03295.png', '24929.png', '06217.png', '15335.png', '18195.png', '09833.png', '16923.png', '04642.png', '21824.png', '09053.png', '02587.png', '03651.png', '13529.png', '04942.png', '11467.png', '24858.png', '21409.png', '03007.png', '11199.png', '13670.png', '00301.png', '12997.png', '13404.png', '15676.png', '06329.png', '02284.png', '08165.png', '11835.png', '20657.png', '02811.png', '22098.png', '09243.png', '13777.png', '20303.png', '08517.png', '07027.png', '17485.png', '02528.png', '23830.png', '21535.png', '16236.png', '14009.png', '22281.png', '13245.png', '23487.png', '22280.png', '09388.png', '00530.png', '01684.png', '11968.png', '19590.png', '06201.png', '24369.png', '18185.png', '02036.png', '11941.png', '08184.png', '23331.png', '23545.png', '19425.png', '09830.png', '06524.png', '06579.png', '16568.png', '09690.png', '18245.png', '08015.png', '20358.png', '15673.png', '23151.png', '16309.png', '13978.png', '01953.png', '08376.png', '23825.png', '11986.png', '04104.png', '17050.png', '17203.png', '10977.png', '20643.png', '15213.png', '15723.png', '15389.png', '05164.png', '06137.png', '03722.png', '04440.png', '00691.png', '22888.png', '02597.png', '14281.png', '12314.png', '08593.png', '09814.png', '02505.png', '08274.png', '15980.png', '06255.png', '12492.png', '16319.png', '00158.png', '15868.png', '07694.png', '14041.png', '01035.png', '11263.png', '21027.png', '19323.png', '20151.png', '10941.png', '09787.png', '03579.png', '16627.png', '05017.png', '18693.png', '10659.png', '21614.png', '16545.png', '14477.png', '17332.png', '14661.png', '10682.png', '01566.png', '01724.png', '22189.png', '07640.png', '03461.png', '15109.png', '11078.png', '18881.png', '14369.png', '06306.png', '00439.png', '01803.png', '22259.png', '20231.png', '18558.png', '09544.png', '21603.png', '14185.png', '18533.png', '06717.png', '05923.png', '05443.png', '17385.png', '02498.png', '11591.png', '22796.png', '15570.png', '08845.png', '04598.png', '06785.png', '16272.png', '02546.png', '12876.png', '05888.png', '13508.png', '19882.png', '00267.png', '09551.png', '18600.png', '21584.png', '24090.png', '06228.png', '15502.png', '21702.png', '20856.png', '21423.png', '10220.png', '10264.png', '08553.png', '15441.png', '23562.png', '23263.png', '10021.png', '18256.png', '21396.png', '08690.png', '17875.png', '16108.png', '02435.png', '19635.png', '15500.png', '14478.png', '09680.png', '03573.png', '15497.png', '23053.png', '15979.png', '18317.png', '07118.png', '07292.png', '04900.png', '22539.png', '21222.png', '10784.png', '20158.png', '17196.png', '16893.png', '19762.png', '11512.png', '11893.png', '18649.png', '13822.png', '02650.png', '05419.png', '08814.png', '09774.png', '13009.png', '09055.png', '05533.png', '23602.png', '03146.png', '15364.png', '24193.png', '05277.png', '02784.png', '17277.png', '01418.png', '22835.png', '09728.png', '20064.png', '14021.png', '02276.png', '09333.png', '01410.png', '06202.png', '13438.png', '10514.png', '09672.png', '00768.png', '02545.png', '15259.png', '00745.png', '18201.png', '14016.png', '17079.png', '21946.png', '16290.png', '07120.png', '08673.png', '02931.png', '12013.png', '20445.png', '04470.png', '17586.png', '17289.png', '11938.png', '18455.png', '19913.png', '04875.png', '00033.png', '21193.png', '23500.png', '06787.png', '17673.png', '05225.png', '04137.png', '07909.png', '16564.png', '00615.png', '09291.png', '07807.png', '01349.png', '15727.png', '24309.png', '19480.png', '10910.png', '16208.png', '23147.png', '08482.png', '21312.png', '18684.png', '16602.png', '21280.png', '16985.png', '16664.png', '13006.png', '11924.png', '02349.png', '09854.png', '19537.png', '07545.png', '20220.png', '21769.png', '16656.png', '11707.png', '17726.png', '12790.png', '07761.png', '17771.png', '23277.png', '01829.png', '20998.png', '10833.png', '00445.png', '19529.png', '10475.png', '18389.png', '17776.png', '04621.png', '21638.png', '05728.png', '09360.png', '16430.png', '11999.png', '15735.png', '12089.png', '21431.png', '06570.png', '07270.png', '06489.png', '03232.png', '16738.png', '18148.png', '10476.png', '15503.png', '08698.png', '10432.png', '15593.png', '10746.png', '22552.png', '05184.png', '00453.png', '19629.png', '20510.png', '17375.png', '10314.png', '10467.png', '00676.png', '07261.png', '18746.png', '18412.png', '03510.png', '20608.png', '08819.png', '11363.png', '15405.png', '05699.png', '23378.png', '02083.png', '21782.png', '05376.png', '21555.png', '18383.png', '20472.png', '12348.png', '22349.png', '02824.png', '02648.png', '15875.png', '17443.png', '23169.png', '15544.png', '22258.png', '03736.png', '14135.png', '24859.png', '15090.png', '08084.png', '17785.png', '15827.png', '20183.png', '11198.png', '05238.png', '04113.png', '21443.png', '21668.png', '05766.png', '14181.png', '23916.png', '08795.png', '14566.png', '08411.png', '09040.png', '15505.png', '17478.png', '07181.png', '03937.png', '15747.png', '06177.png', '13359.png', '01535.png', '04274.png', '20313.png', '11366.png', '10559.png', '05566.png', '18735.png', '06550.png', '02241.png', '21346.png', '07230.png', '24274.png', '19083.png', '03677.png', '03683.png', '24271.png', '07783.png', '22118.png', '08786.png', '12982.png', '12309.png', '10071.png', '03067.png', '17193.png', '03949.png', '04866.png', '02013.png', '24605.png', '22135.png', '18427.png', '15501.png', '01008.png', '02493.png', '02612.png', '05455.png', '24132.png', '12840.png', '04121.png', '18313.png', '22708.png', '18314.png', '22584.png', '13943.png', '13083.png', '08705.png', '13458.png', '11940.png', '00907.png', '24219.png', '19526.png', '02157.png', '18093.png', '19037.png', '14107.png', '12679.png', '21550.png', '06067.png', '11200.png', '17572.png', '00862.png', '22992.png', '23491.png', '12968.png', '11114.png', '16026.png', '03318.png', '02351.png', '12781.png', '23964.png', '04813.png', '06659.png', '09089.png', '15610.png', '01422.png', '10703.png', '23134.png', '17238.png', '02562.png', '00013.png', '21109.png', '08293.png', '17658.png', '04661.png', '15193.png', '18248.png', '15112.png', '09052.png', '16325.png', '22751.png', '18915.png', '00303.png', '11537.png', '13351.png', '10227.png', '16937.png', '14462.png', '06265.png', '03403.png', '06837.png', '20556.png', '12732.png', '06078.png', '06709.png', '16975.png', '08699.png', '18147.png', '00467.png', '24790.png', '14804.png', '19003.png', '16606.png', '23770.png', '09915.png', '22011.png', '01448.png', '24940.png', '22201.png', '18720.png', '01000.png', '02559.png', '22606.png', '00085.png', '02843.png', '02371.png', '07105.png', '06744.png', '11878.png', '05895.png', '23078.png', '21696.png', '09634.png', '08924.png', '10742.png', '04788.png', '24817.png', '18855.png', '10811.png', '21643.png', '18399.png', '13238.png', '01692.png', '04062.png', '11139.png', '06002.png', '22162.png', '01290.png', '14833.png', '04366.png', '15848.png', '01497.png', '24082.png', '01301.png', '02007.png', '04513.png', '06839.png', '02038.png', '04629.png', '14515.png', '18749.png', '01824.png', '04570.png', '08727.png', '11188.png', '04171.png', '03556.png', '04548.png', '01669.png', '05693.png', '06460.png', '04521.png', '10104.png', '06467.png', '15674.png', '24197.png', '01638.png', '04312.png', '12605.png', '12952.png', '20868.png', '12021.png', '15534.png', '22670.png', '23470.png', '18655.png', '09292.png', '23975.png', '14729.png', '21066.png', '10340.png', '00207.png', '00018.png', '13338.png', '06796.png', '15411.png', '14951.png', '02897.png', '10815.png', '15079.png', '21138.png', '04863.png', '17311.png', '03497.png', '10291.png', '17929.png', '16287.png', '09524.png', '21173.png', '06999.png', '24779.png', '02490.png', '17146.png', '23891.png', '17995.png', '22077.png', '08971.png', '22303.png', '21382.png', '16243.png', '12351.png', '02704.png', '23176.png', '01425.png', '14624.png', '00412.png', '16022.png', '09557.png', '14251.png', '05621.png', '22341.png', '16246.png', '24697.png', '05950.png', '08304.png', '10544.png', '09603.png', '18662.png', '20587.png', '09592.png', '07067.png', '18814.png', '00646.png', '11856.png', '07441.png', '07908.png', '00982.png', '18464.png', '15246.png', '00772.png', '17202.png', '04710.png', '02329.png', '10116.png', '16245.png', '09457.png', '02458.png', '19440.png', '20925.png', '21473.png', '00366.png', '04862.png', '09729.png', '20044.png', '19187.png', '03543.png', '20706.png', '07070.png', '12015.png', '01202.png', '23447.png', '02338.png', '16864.png', '14105.png', '15732.png', '13960.png', '21624.png', '19108.png', '20112.png', '12112.png', '00649.png', '11378.png', '15496.png', '07299.png', '08139.png', '18123.png', '17420.png', '09981.png', '19546.png', '17217.png', '05415.png', '12986.png', '03259.png', '12267.png', '20270.png', '03385.png', '10069.png', '11960.png', '04787.png', '17415.png', '15268.png', '07318.png', '01130.png', '03519.png', '03162.png', '04426.png', '11606.png', '03603.png', '03613.png', '20609.png', '06447.png', '14141.png', '13848.png', '15088.png', '12306.png', '10045.png', '01293.png', '14717.png', '19251.png', '06577.png', '17069.png', '13137.png', '12373.png', '07080.png', '00686.png', '05304.png', '23730.png', '15053.png', '06762.png', '21306.png', '21521.png', '01702.png', '13865.png', '00769.png', '13666.png', '12301.png', '08115.png', '05077.png', '23355.png', '07156.png', '15709.png', '02668.png', '05774.png', '00557.png', '05627.png', '03561.png', '08382.png', '21972.png', '16285.png', '06151.png', '04852.png', '13787.png', '11555.png', '02688.png', '05249.png', '14398.png', '04845.png', '19232.png', '06562.png', '18507.png', '14710.png', '19255.png', '08643.png', '02547.png', '18509.png', '13089.png', '02223.png', '05836.png', '07840.png', '11388.png', '04325.png', '13471.png', '05560.png', '19544.png', '07260.png', '02878.png', '02844.png', '13292.png', '00210.png', '18672.png', '05326.png', '09298.png', '10857.png', '06752.png', '02656.png', '00374.png', '04712.png', '04215.png', '06219.png', '20131.png', '14837.png', '18158.png', '12272.png', '15984.png', '19414.png', '05641.png', '21223.png', '19738.png', '13426.png', '04439.png', '19789.png', '22765.png', '12572.png', '24816.png', '24765.png', '07663.png', '23343.png', '09828.png', '00957.png', '20223.png', '20282.png', '16604.png', '03990.png', '18920.png', '23475.png', '16318.png', '09023.png', '16732.png', '02503.png', '11958.png', '14277.png', '13213.png', '08854.png', '07252.png', '09271.png', '07393.png', '00761.png', '20750.png', '18410.png', '23723.png', '21849.png', '05347.png', '16011.png', '06436.png', '04036.png', '09302.png', '03357.png', '02002.png', '23534.png', '13374.png', '10190.png', '03751.png', '07065.png', '02394.png', '18840.png', '15550.png', '00877.png', '22021.png', '03942.png', '17720.png', '24850.png', '00729.png', '13548.png', '15746.png', '17773.png', '12488.png', '04061.png', '10546.png', '05417.png', '20236.png', '17810.png', '09513.png', '03070.png', '01399.png', '17872.png', '13758.png', '11594.png', '07834.png', '15766.png', '00559.png', '02312.png', '15648.png', '07202.png', '23386.png', '00632.png', '01219.png', '11848.png', '10223.png', '04401.png', '10323.png', '05505.png', '13517.png', '23699.png', '14959.png', '14353.png', '03698.png', '06023.png', '16508.png', '00050.png', '23598.png', '08248.png', '01350.png', '00703.png', '04455.png', '17172.png', '03366.png', '21747.png', '01295.png', '06690.png', '22385.png', '04165.png', '22235.png', '12871.png', '04490.png', '11635.png', '08676.png', '05852.png', '01909.png', '06005.png', '23173.png', '00716.png', '11121.png', '09599.png', '16888.png', '21719.png', '15473.png', '17757.png', '07825.png', '10111.png', '06373.png', '23422.png', '10011.png', '13687.png', '18402.png', '00136.png', '05691.png', '22336.png', '12692.png', '13830.png', '11259.png', '11747.png', '07365.png', '18027.png', '22563.png', '05545.png', '06656.png', '19200.png', '11994.png', '07842.png', '13948.png', '22789.png', '04094.png', '24526.png', '01006.png', '10039.png', '01495.png', '04125.png', '01471.png', '21421.png', '16145.png', '08126.png', '19699.png', '08122.png', '05011.png', '09533.png', '04331.png', '04834.png', '21163.png', '22897.png', '03361.png', '15243.png', '06940.png', '01875.png', '21654.png', '06895.png', '09726.png', '15738.png', '00497.png', '15047.png', '15198.png', '08455.png', '15274.png', '12232.png', '16150.png', '00012.png', '15029.png', '24014.png', '15021.png', '02662.png', '11369.png', '03666.png', '21408.png', '08627.png', '02337.png', '07614.png', '09083.png', '09456.png', '03968.png', '07837.png', '15996.png', '17063.png', '20924.png', '03414.png', '21237.png', '13901.png', '15329.png', '07596.png', '15237.png', '01608.png', '22952.png', '15549.png', '09379.png', '03813.png', '10371.png', '21717.png', '05905.png', '11846.png', '08827.png', '08897.png', '00804.png', '01885.png', '01826.png', '04371.png', '10812.png', '20281.png', '15308.png', '02805.png', '00665.png', '11934.png', '00523.png', '24099.png', '13583.png', '22429.png', '17915.png', '04054.png', '04602.png', '10736.png', '13328.png', '20870.png', '23894.png', '19396.png', '23821.png', '11647.png', '23805.png', '09120.png', '13514.png', '22130.png', '09194.png', '01492.png', '17589.png', '20025.png', '08937.png']
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
        
        if i_iter == 346470:
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
