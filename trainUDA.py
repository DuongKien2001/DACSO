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
    a = 68	['07077.png', '06370.png', '03163.png', '19148.png', '14520.png', '10705.png', '00419.png', '18528.png', '09804.png', '10944.png', '02704.png', '24601.png', '07114.png', '08883.png', '02371.png', '06069.png', '18718.png', '04660.png', '24270.png', '04210.png', '03939.png', '11650.png', '12793.png', '02580.png', '07795.png', '07805.png', '16362.png', '10125.png', '05355.png', '20006.png', '03094.png', '23256.png', '18460.png', '04181.png', '16930.png', '13186.png', '23937.png', '02741.png', '12823.png', '21734.png', '15291.png', '19141.png', '00617.png', '20098.png', '23032.png', '06613.png', '05644.png', '20595.png', '12477.png', '24205.png', '11491.png', '12107.png', '06404.png', '19113.png', '22491.png', '24088.png', '22087.png', '23919.png', '22963.png', '19720.png', '00321.png', '08265.png', '13278.png', '23993.png', '08209.png', '23783.png', '16607.png', '05720.png', '19242.png', '04874.png', '13206.png', '19350.png', '13381.png', '07104.png', '14891.png', '01671.png', '07656.png', '08508.png', '00880.png', '19670.png', '14083.png', '11392.png', '16059.png', '01455.png', '18185.png', '16117.png', '17593.png', '03301.png', '09611.png', '14945.png', '04948.png', '09116.png', '05420.png', '20992.png', '22270.png', '02463.png', '19142.png', '02745.png', '07818.png', '22946.png', '00960.png', '11471.png', '16981.png', '02209.png', '03011.png', '00079.png', '19892.png', '03640.png', '22128.png', '20344.png', '04111.png', '22528.png', '13562.png', '05912.png', '03988.png', '17717.png', '09734.png', '12848.png', '00224.png', '24544.png', '03830.png', '09385.png', '07787.png', '00394.png', '15603.png', '10896.png', '13986.png', '21421.png', '23754.png', '03348.png', '22254.png', '24142.png', '20639.png', '24343.png', '09806.png', '13664.png', '07174.png', '13582.png', '17423.png', '24132.png', '08231.png', '01805.png', '11696.png', '15742.png', '03856.png', '14942.png', '19206.png', '18558.png', '12102.png', '02022.png', '10451.png', '05189.png', '05121.png', '06187.png', '06990.png', '17907.png', '19324.png', '09320.png', '09090.png', '21759.png', '03139.png', '08951.png', '17883.png', '19230.png', '01288.png', '08744.png', '23788.png', '17322.png', '14897.png', '09024.png', '06948.png', '01627.png', '08332.png', '08278.png', '08736.png', '10225.png', '20352.png', '09672.png', '08304.png', '21359.png', '02478.png', '04213.png', '11348.png', '24490.png', '16836.png', '18478.png', '01528.png', '00402.png', '13637.png', '12037.png', '17548.png', '14539.png', '02774.png', '11086.png', '14615.png', '17813.png', '05124.png', '02310.png', '06182.png', '06945.png', '10715.png', '02914.png', '07697.png', '14802.png', '02602.png', '10655.png', '04385.png', '01286.png', '10764.png', '23660.png', '07255.png', '21169.png', '09847.png', '13684.png', '09921.png', '11984.png', '00502.png', '04563.png', '09321.png', '14483.png', '07300.png', '03357.png', '04251.png', '16395.png', '19317.png', '16797.png', '17171.png', '08900.png', '24127.png', '09645.png', '01026.png', '24799.png', '06220.png', '21658.png', '19186.png', '02725.png', '07113.png', '16332.png', '15041.png', '22310.png', '24748.png', '12971.png', '22370.png', '12962.png', '24527.png', '20485.png', '12056.png', '15631.png', '02119.png', '17510.png', '18692.png', '18540.png', '23883.png', '04236.png', '10357.png', '19824.png', '19516.png', '10857.png', '18451.png', '05950.png', '16036.png', '05624.png', '19392.png', '24508.png', '15174.png', '08296.png', '05439.png', '12738.png', '11111.png', '11760.png', '06254.png', '12259.png', '23727.png', '19711.png', '05003.png', '08121.png', '10113.png', '22547.png', '09442.png', '09272.png', '18601.png', '06354.png', '01351.png', '07625.png', '11224.png', '09537.png', '01082.png', '02735.png', '19889.png', '04556.png', '09738.png', '15201.png', '24381.png', '00286.png', '19508.png', '17737.png', '07794.png', '23098.png', '03642.png', '18074.png', '01586.png', '09936.png', '07833.png', '08980.png', '10442.png', '03639.png', '09164.png', '09451.png', '22688.png', '10634.png', '09162.png', '07746.png', '14637.png', '19752.png', '03905.png', '18757.png', '20187.png', '13696.png', '16231.png', '13526.png', '05369.png', '12419.png', '13654.png', '09310.png', '01551.png', '13404.png', '12435.png', '11050.png', '05437.png', '14537.png', '24531.png', '24393.png', '09262.png', '05638.png', '13367.png', '04429.png', '14056.png', '05131.png', '07610.png', '19066.png', '01198.png', '04736.png', '19622.png', '02373.png', '14812.png', '07046.png', '03530.png', '04081.png', '20434.png', '10465.png', '06378.png', '05231.png', '10146.png', '05303.png', '24034.png', '18865.png', '01479.png', '09906.png', '01510.png', '11095.png', '16504.png', '06692.png', '03387.png', '01354.png', '00767.png', '02525.png', '17466.png', '02750.png', '21698.png', '21097.png', '08358.png', '00251.png', '23676.png', '09917.png', '18982.png', '24157.png', '16175.png', '00851.png', '17832.png', '05342.png', '03989.png', '15617.png', '21969.png', '13777.png', '02047.png', '02061.png', '10688.png', '02755.png', '14727.png', '19183.png', '04793.png', '09845.png', '19838.png', '09118.png', '08460.png', '04589.png', '09649.png', '10120.png', '02513.png', '18589.png', '11909.png', '16227.png', '12879.png', '04378.png', '07980.png', '12807.png', '12565.png', '21892.png', '20156.png', '12658.png', '10976.png', '14991.png', '02917.png', '05139.png', '11724.png', '22002.png', '01230.png', '11038.png', '21346.png', '02787.png', '17252.png', '13622.png', '03481.png', '01035.png', '05603.png', '22474.png', '03063.png', '24714.png', '14155.png', '08217.png', '22479.png', '03418.png', '06621.png', '18123.png', '01824.png', '01347.png', '21083.png', '17745.png', '10609.png', '23890.png', '15849.png', '18997.png', '12386.png', '09416.png', '11453.png', '19086.png', '06441.png', '23143.png', '01935.png', '11517.png', '08340.png', '06347.png', '01413.png', '22561.png', '05066.png', '11268.png', '00002.png', '01635.png', '22883.png', '13759.png', '22947.png', '06611.png', '18800.png', '05981.png', '04012.png', '23262.png', '21289.png', '04982.png', '13265.png', '23211.png', '07642.png', '02768.png', '13428.png', '20734.png', '14495.png', '08120.png', '09330.png', '15676.png', '19564.png', '09698.png', '05127.png', '00074.png', '03482.png', '22369.png', '15251.png', '08680.png', '06988.png', '07564.png', '21492.png', '12063.png', '18406.png', '13844.png', '22654.png', '08188.png', '00548.png', '10867.png', '17869.png', '09123.png', '17847.png', '11013.png', '09324.png', '04286.png', '18766.png', '00482.png', '21200.png', '07475.png', '03934.png', '06137.png', '09158.png', '12532.png', '10823.png', '00320.png', '14604.png', '19053.png', '19764.png', '23892.png', '11466.png', '00456.png', '02490.png', '04889.png', '07178.png', '02289.png', '23113.png', '21895.png', '07003.png', '24010.png', '02217.png', '24213.png', '18720.png', '22094.png', '15728.png', '17503.png', '19464.png', '21055.png', '09105.png', '03131.png', '08461.png', '18602.png', '21972.png', '00082.png', '16371.png', '23122.png', '08517.png', '15186.png', '17867.png', '06804.png', '07340.png', '10472.png', '22703.png', '10816.png', '20268.png', '02252.png', '13609.png', '17981.png', '24757.png', '01280.png', '08734.png', '17534.png', '07240.png', '00786.png', '02477.png', '10243.png', '14026.png', '20396.png', '05228.png', '12028.png', '22721.png', '04584.png', '11140.png', '14975.png', '01171.png', '07505.png', '07577.png', '15750.png', '17901.png', '18686.png', '04005.png', '19275.png', '09525.png', '24732.png', '12526.png', '11317.png', '02160.png', '13646.png', '14533.png', '04577.png', '10801.png', '19881.png', '05517.png', '15082.png', '07496.png', '23161.png', '06384.png', '20085.png', '03220.png', '03923.png', '16718.png', '02933.png', '01828.png', '01895.png', '05768.png', '14935.png', '19212.png', '11769.png', '07224.png', '11405.png', '13453.png', '24276.png', '07926.png', '15566.png', '11487.png', '19343.png', '20998.png', '01298.png', '04818.png', '02889.png', '18652.png', '01716.png', '00042.png', '17138.png', '20951.png', '24860.png', '19129.png', '24577.png', '20084.png', '20112.png', '19689.png', '21722.png', '24523.png', '00887.png', '01827.png', '23376.png', '11831.png', '01641.png', '16248.png', '09271.png', '13851.png', '09191.png', '23470.png', '07001.png', '20680.png', '03285.png', '17093.png', '08252.png', '22615.png', '08601.png', '03893.png', '03776.png', '20393.png', '00305.png', '10872.png', '18896.png', '12390.png', '24123.png', '08719.png', '03087.png', '11077.png', '05375.png', '00124.png', '22867.png', '24524.png', '19337.png', '16200.png', '12460.png', '24414.png', '13675.png', '19790.png', '17431.png', '06604.png', '17554.png', '00778.png', '24359.png', '23587.png', '12368.png', '21172.png', '23946.png', '05689.png', '16296.png', '22469.png', '06209.png', '23444.png', '16553.png', '19522.png', '18393.png', '09930.png', '07952.png', '20675.png', '07512.png', '21324.png', '18140.png', '17234.png', '23997.png', '23121.png', '02691.png', '10616.png', '21033.png', '07426.png', '14606.png', '03978.png', '15551.png', '02608.png', '18418.png', '18599.png', '17539.png', '04791.png', '02190.png', '19208.png', '24434.png', '12938.png', '07027.png', '23792.png', '20961.png', '16104.png', '02563.png', '17026.png', '23643.png', '01234.png', '16206.png', '19171.png', '09618.png', '23442.png', '10447.png', '01750.png', '05529.png', '17152.png', '15360.png', '03426.png', '04230.png', '23800.png', '21842.png', '24694.png', '07414.png', '12146.png', '24675.png', '14614.png', '13610.png', '02065.png', '19225.png', '23326.png', '21715.png', '00582.png', '11537.png', '21049.png', '10149.png', '02082.png', '24798.png', '06344.png', '24942.png', '24573.png', '24201.png', '20698.png', '02546.png', '01769.png', '03568.png', '00587.png', '05267.png', '07647.png', '18269.png', '13457.png', '16766.png', '23015.png', '12730.png', '02929.png', '23102.png', '08118.png', '21730.png', '11562.png', '19710.png', '05318.png', '16587.png', '20895.png', '24879.png', '08081.png', '16161.png', '21567.png', '11257.png', '17214.png', '01907.png', '17300.png', '10230.png', '20449.png', '12262.png', '11041.png', '02509.png', '23188.png', '13625.png', '20705.png', '16969.png', '13169.png', '04824.png', '00114.png', '18379.png', '09155.png', '09168.png', '22314.png', '11704.png', '14786.png', '10038.png', '11409.png', '21589.png', '09692.png', '06047.png', '03273.png', '02746.png', '21365.png', '23950.png', '23681.png', '17718.png', '16650.png', '19646.png', '11945.png', '11731.png', '23626.png', '23906.png', '06769.png', '03057.png', '06459.png', '22182.png', '19197.png', '03168.png', '10606.png', '19319.png', '04688.png', '23216.png', '08283.png', '22100.png', '11341.png', '07707.png', '10436.png', '01401.png', '12202.png', '10027.png', '06159.png', '00881.png', '10815.png', '20060.png', '19758.png', '09364.png', '07827.png', '00955.png', '06842.png', '07918.png', '22691.png', '02332.png', '24710.png', '09454.png', '03281.png', '00314.png', '23361.png', '01368.png', '09629.png', '17383.png', '23080.png', '01279.png', '15713.png', '15224.png', '17183.png', '04030.png', '00909.png', '12480.png', '18611.png', '24432.png', '02556.png', '12364.png', '20846.png', '17844.png', '12461.png', '11606.png', '14768.png', '07942.png', '21932.png', '04830.png', '00581.png', '11125.png', '00755.png', '08585.png', '19360.png', '03832.png', '19193.png', '23106.png', '05726.png', '18288.png', '02364.png', '11339.png', '07223.png', '24950.png', '07555.png', '18155.png', '04973.png', '20684.png', '11498.png', '22344.png', '11333.png', '24185.png', '11372.png', '08124.png', '13565.png', '04859.png', '15282.png', '18328.png', '09656.png', '24374.png', '10821.png', '04550.png', '04626.png', '09597.png', '17584.png', '10469.png', '00876.png', '20968.png', '02527.png', '17924.png', '16150.png', '06829.png', '08258.png', '23355.png', '21515.png', '12577.png', '14022.png', '24143.png', '03452.png', '02507.png', '14765.png', '01676.png', '11826.png', '00666.png', '07633.png', '06703.png', '19667.png', '09449.png', '04558.png', '03526.png', '10267.png', '07983.png', '12652.png', '06582.png', '06132.png', '10755.png', '16068.png', '00831.png', '22221.png', '01793.png', '02044.png', '07495.png', '03432.png', '19493.png', '06836.png', '01818.png', '07984.png', '12908.png', '13048.png', '06191.png', '07640.png', '11283.png', '07323.png', '21386.png', '21573.png', '19778.png', '14236.png', '23705.png', '03088.png', '20807.png', '21145.png', '07685.png', '10935.png', '01978.png', '15020.png', '00886.png', '04567.png', '05905.png', '09073.png', '04281.png', '07341.png', '03358.png', '13301.png', '14282.png', '07456.png', '10864.png', '03363.png', '00804.png', '18024.png', '19205.png', '01302.png', '12044.png', '17129.png', '02732.png', '10404.png', '09457.png', '20323.png', '16250.png', '20035.png', '08669.png', '01421.png', '08897.png', '20259.png', '02545.png', '18136.png', '06310.png', '14934.png', '08218.png', '08065.png', '07874.png', '17657.png', '02957.png', '02799.png', '05840.png', '03789.png', '02976.png', '09887.png', '08233.png', '15454.png', '22709.png', '24015.png', '21962.png', '16038.png', '00410.png', '01156.png', '15558.png', '13883.png', '09650.png', '22465.png', '16753.png', '21544.png', '07338.png', '12248.png', '06992.png', '16421.png', '01459.png', '03751.png', '14530.png', '19780.png', '23876.png', '23042.png', '01648.png', '06214.png', '20514.png', '12508.png', '21442.png', '10736.png', '18110.png', '00871.png', '19076.png', '15376.png', '15412.png', '04278.png', '24834.png', '21841.png', '15057.png', '02994.png', '21936.png', '19481.png', '22823.png', '06578.png', '00703.png', '20553.png', '09345.png', '20728.png', '18576.png', '17266.png', '02517.png', '14719.png', '00092.png', '24097.png', '11431.png', '21995.png', '06958.png', '06772.png', '14366.png', '22623.png', '13556.png', '06512.png', '22879.png', '05825.png', '05394.png', '18321.png', '20886.png', '13422.png', '10131.png', '18470.png', '00331.png', '03529.png', '23994.png', '19789.png', '01543.png', '21605.png', '04468.png', '18846.png', '23274.png', '14347.png', '11159.png', '14067.png', '16143.png', '03573.png', '05848.png', '20772.png', '10305.png', '06732.png', '18706.png', '24611.png', '23329.png', '02327.png', '08507.png', '06307.png', '22191.png', '01191.png', '21170.png', '16880.png', '07905.png', '07259.png', '01036.png', '18489.png', '24780.png', '15859.png', '13122.png', '01751.png', '03690.png', '20543.png', '10369.png', '16641.png', '05363.png', '05662.png', '15767.png', '03477.png', '05473.png', '10949.png', '07975.png', '09109.png', '06437.png', '19296.png', '23974.png', '02366.png', '20976.png', '04103.png', '08685.png', '01445.png', '24112.png', '15999.png', '03492.png', '03207.png', '12035.png', '24072.png', '12023.png', '07484.png', '24592.png', '13095.png', '06383.png', '03507.png', '11857.png', '13729.png', '13638.png', '20495.png', '12924.png', '00821.png', '20274.png', '16848.png', '13732.png', '01498.png', '09223.png', '04516.png', '15771.png', '05806.png', '15953.png', '15195.png', '21495.png', '08356.png', '19085.png', '12808.png', '10000.png', '16458.png', '00641.png', '14362.png', '08014.png', '11568.png', '01603.png', '08013.png', '11390.png', '08996.png', '07608.png', '08143.png', '02219.png', '22842.png', '22902.png', '18246.png', '10786.png', '03809.png', '23499.png', '20670.png', '15602.png', '24316.png', '21631.png', '03041.png', '22005.png', '22034.png', '24120.png', '19929.png', '09154.png', '13177.png', '13110.png', '11439.png', '17802.png', '09081.png', '04431.png', '00230.png', '00013.png', '18344.png', '20214.png', '22451.png', '20116.png', '16628.png', '21867.png', '04732.png', '14258.png', '20277.png', '22704.png', '23912.png', '10437.png', '02297.png', '06628.png', '17676.png', '18542.png', '17443.png', '21934.png', '13109.png', '07709.png', '16791.png', '09285.png', '17900.png', '08917.png', '05105.png', '16136.png', '15315.png', '20132.png', '14367.png', '06515.png', '14668.png', '24642.png', '12754.png', '20282.png', '21664.png', '03825.png', '02336.png', '17866.png', '04145.png', '10880.png', '04832.png', '24181.png', '18149.png', '02077.png', '10760.png', '05452.png', '20257.png', '24886.png', '13442.png', '11703.png', '00777.png', '18926.png', '10659.png', '24906.png', '08852.png', '18850.png', '11196.png', '11546.png', '12545.png', '07915.png', '11112.png', '09291.png', '23874.png', '02414.png', '03592.png', '21235.png', '00284.png', '00888.png', '13004.png', '10672.png', '04959.png', '21906.png', '06774.png', '10555.png', '14566.png', '12977.png', '10280.png', '15372.png', '08518.png', '05308.png', '00008.png', '02239.png', '24766.png', '02845.png', '13155.png', '19431.png', '08391.png', '22472.png', '08915.png', '23786.png', '10080.png', '14234.png', '24443.png', '12440.png', '21586.png', '17435.png', '01444.png', '23146.png', '12416.png', '08921.png', '23768.png', '20350.png', '10260.png', '00522.png', '22393.png', '18791.png', '03384.png', '12444.png', '12200.png', '19738.png', '21047.png', '15326.png', '12304.png', '16770.png', '08727.png', '22343.png', '14028.png', '24436.png', '03875.png', '07425.png', '21550.png', '06482.png', '11109.png', '07278.png', '13959.png', '11866.png', '15214.png', '12600.png', '21018.png', '01357.png', '15101.png', '01300.png', '08715.png', '19759.png', '16334.png', '00261.png', '08378.png', '08926.png', '19353.png', '19003.png', '22529.png', '18844.png', '09924.png', '09343.png', '03050.png', '00875.png', '00721.png', '10745.png', '05930.png', '02574.png', '02277.png', '08956.png', '20635.png', '23092.png', '12562.png', '19609.png', '07025.png', '14354.png', '11073.png', '24832.png', '08870.png', '02429.png', '17302.png', '15935.png', '17913.png', '06205.png', '23381.png', '21711.png', '09961.png', '07334.png', '23159.png', '23763.png', '20932.png', '20902.png', '23998.png', '05636.png', '04054.png', '20795.png', '10831.png', '09721.png', '01105.png', '21604.png', '19673.png', '21978.png', '16242.png', '06938.png', '10788.png', '23325.png', '08023.png', '01346.png', '02372.png', '01539.png', '16656.png', '19503.png', '13097.png', '21298.png', '02652.png', '05194.png', '15299.png', '18357.png', '08198.png', '19491.png', '24321.png', '02473.png', '10997.png', '02885.png', '00513.png', '23518.png', '02486.png', '22495.png', '21517.png', '04014.png', '21597.png', '13611.png', '20949.png', '00486.png', '23022.png', '12085.png', '23447.png', '01179.png', '23377.png', '18075.png', '17345.png', '01877.png', '14927.png', '16807.png', '13889.png', '19927.png', '13052.png', '23553.png', '19196.png', '02811.png', '04333.png', '21032.png', '22824.png', '09462.png', '19785.png', '10925.png', '15983.png', '11430.png', '22613.png', '21343.png', '08338.png', '14221.png', '08888.png', '04568.png', '20266.png', '03737.png', '09958.png', '21195.png', '05971.png', '10734.png', '09648.png', '17942.png', '16144.png', '16028.png', '02262.png', '19060.png', '04386.png', '01332.png', '09898.png', '18234.png', '22668.png', '12548.png', '15331.png', '11179.png', '02647.png', '17607.png', '15459.png', '13020.png', '06607.png', '02694.png', '17116.png', '04349.png', '15970.png', '01792.png', '23482.png', '15447.png', '09062.png', '15420.png', '01141.png', '19475.png', '24581.png', '13413.png', '18795.png', '22105.png', '20375.png', '08648.png', '16512.png', '08513.png', '07732.png', '20330.png', '15620.png', '18685.png', '01969.png', '01544.png', '21213.png', '11616.png', '08919.png', '05100.png', '03817.png', '16005.png', '10675.png', '09176.png', '01730.png', '07186.png', '09882.png', '13474.png', '20985.png', '02857.png', '02048.png', '16655.png', '24858.png', '14602.png', '00355.png', '23339.png', '12544.png', '10502.png', '24555.png', '11002.png', '24880.png', '00437.png', '10557.png', '03370.png', '00047.png', '05324.png', '11182.png', '12811.png', '06262.png', '01858.png', '03115.png', '12447.png', '19916.png', '01448.png', '06235.png', '02573.png', '13173.png', '18151.png', '01976.png', '13348.png', '12574.png', '05763.png', '05970.png', '04483.png', '08807.png', '06295.png', '23076.png', '14850.png', '23765.png', '01189.png', '23299.png', '17166.png', '11241.png', '10813.png', '23757.png', '10115.png', '17781.png', '14245.png', '09483.png', '24328.png', '22120.png', '00237.png', '14681.png', '04402.png', '10784.png', '13619.png', '15803.png', '21829.png', '00409.png', '24784.png', '23309.png', '08523.png', '09983.png', '09776.png', '22797.png', '01002.png', '08005.png', '21812.png', '20891.png', '17347.png', '01936.png', '01372.png', '17254.png', '04855.png', '06776.png', '02791.png', '04264.png', '03302.png', '07617.png', '06359.png', '12089.png', '20311.png', '23383.png', '14125.png', '19671.png', '04817.png', '05016.png', '23659.png', '04528.png', '08675.png', '01617.png', '05078.png', '04098.png', '02069.png', '03197.png', '22018.png', '00537.png', '13951.png', '04529.png', '15871.png', '14598.png', '03294.png', '24753.png', '09793.png', '19472.png', '22833.png', '21843.png', '23831.png', '24759.png', '04542.png', '22190.png', '17139.png', '17437.png', '12556.png', '07619.png', '09077.png', '21132.png', '08963.png', '04040.png', '18242.png', '07961.png', '22999.png', '01662.png', '13448.png', '06911.png', '16065.png', '11688.png', '20940.png', '19160.png', '05583.png', '06557.png', '07435.png', '00610.png', '11069.png', '22372.png', '19093.png', '10314.png', '19533.png', '23964.png', '02743.png', '16194.png', '12158.png', '15294.png', '19990.png', '10988.png', '00707.png', '11638.png', '16646.png', '03522.png', '08803.png', '16034.png', '12918.png', '11401.png', '18675.png', '19703.png', '01314.png', '22290.png', '12815.png', '00729.png', '03292.png', '01167.png', '24744.png', '18584.png', '04507.png', '13489.png', '02057.png', '20095.png', '03179.png', '11127.png', '18427.png', '20935.png', '20024.png', '11504.png', '16050.png', '19831.png', '07749.png', '02094.png', '04080.png', '01423.png', '24244.png', '07468.png', '14115.png', '04274.png', '00413.png', '10372.png', '03245.png', '05354.png', '05444.png', '17353.png', '20507.png', '10432.png', '11253.png', '03069.png', '01901.png', '21012.png', '06867.png', '03559.png', '14008.png', '18769.png', '12463.png', '08707.png', '02687.png', '18723.png', '08466.png', '19130.png', '08489.png', '12417.png', '06977.png', '24058.png', '05841.png', '01555.png', '19025.png', '17601.png', '12332.png', '11905.png', '15278.png', '08242.png', '21232.png', '21848.png', '23348.png', '06002.png', '14459.png', '04687.png', '13795.png', '17792.png', '24699.png', '10615.png', '09593.png', '24653.png', '05908.png', '16199.png', '22606.png', '14190.png', '14073.png', '17496.png', '16333.png', '02158.png', '18521.png', '21866.png', '21719.png', '13461.png', '05695.png', '04875.png', '21494.png', '03653.png', '24715.png', '23507.png', '01233.png', '17912.png', '08214.png', '23058.png', '02114.png', '07658.png', '03792.png', '17323.png', '08337.png', '02066.png', '03835.png', '08501.png', '05206.png', '11102.png', '20074.png', '22665.png', '17945.png', '16487.png', '14636.png', '10342.png', '01133.png', '05332.png', '18405.png', '18117.png', '11659.png', '08236.png', '12484.png', '23679.png', '02531.png', '16037.png', '00957.png', '12536.png', '11329.png', '24326.png', '14617.png', '20149.png', '20036.png', '08025.png', '00232.png', '05544.png', '18805.png', '15093.png', '11808.png', '10563.png', '10162.png', '05246.png', '10418.png', '18955.png', '03652.png', '03572.png', '01261.png', '20290.png', '03068.png', '18771.png', '24352.png', '18095.png', '03279.png', '09538.png', '01093.png', '04912.png', '03955.png', '21168.png', '12950.png', '13984.png', '09298.png', '23192.png', '20617.png', '21209.png', '05027.png', '16025.png', '13019.png', '15674.png', '12034.png', '24867.png', '21577.png', '21852.png', '02872.png', '23632.png', '07681.png', '09386.png', '19719.png', '01837.png', '13857.png', '06204.png', '18051.png', '15626.png', '10331.png', '16360.png', '07724.png', '18157.png', '22213.png', '22564.png', '08617.png', '02714.png', '19875.png', '23665.png', '02488.png', '13409.png', '00688.png', '03056.png', '19058.png', '19542.png', '15402.png', '24260.png', '14757.png', '22794.png', '14739.png', '18648.png', '05627.png', '06154.png', '23485.png', '10798.png', '18900.png', '11582.png', '01063.png', '07508.png', '04620.png', '01850.png', '08276.png', '00797.png', '10545.png', '20161.png', '22854.png', '13465.png', '12420.png', '17371.png', '23728.png', '11732.png', '03250.png', '02418.png', '06203.png', '03570.png', '20444.png', '22553.png', '10714.png', '15629.png', '07315.png', '21496.png', '15973.png', '22646.png', '18010.png', '09943.png', '23712.png', '13051.png', '20134.png', '03374.png', '23696.png', '09787.png', '17731.png', '04609.png', '03218.png', '22361.png', '17060.png', '09619.png', '02944.png', '21336.png', '18016.png', '04221.png', '07410.png', '11819.png', '05732.png', '02422.png', '02895.png', '02860.png', '02120.png', '01222.png', '21173.png', '07679.png', '20969.png', '00559.png', '22892.png', '10384.png', '01322.png', '13046.png', '24509.png', '00645.png', '00379.png', '15473.png', '21085.png', '20319.png', '01763.png', '07260.png', '04004.png', '22401.png', '04253.png', '20722.png', '21270.png', '08728.png', '16195.png', '13419.png', '15709.png', '20301.png', '18189.png', '07634.png', '12844.png', '01290.png', '16972.png', '22609.png', '17914.png', '01934.png', '05215.png', '08026.png', '15929.png', '06717.png', '00792.png', '18402.png', '15449.png', '18070.png', '07008.png', '18770.png', '19978.png', '00872.png', '07280.png', '22719.png', '19612.png', '02453.png', '10877.png', '09450.png', '23772.png', '06127.png', '14556.png', '16293.png', '02125.png', '12612.png', '11158.png', '04344.png', '02078.png', '01497.png', '09353.png', '09786.png', '16047.png', '01623.png', '08237.png', '04865.png', '05563.png', '19614.png', '11904.png', '24847.png', '15815.png', '19333.png', '17622.png', '13680.png', '22620.png', '00353.png', '05419.png', '22255.png', '00524.png', '12821.png', '16156.png', '18579.png', '18314.png', '18928.png', '15218.png', '24587.png', '11273.png', '19909.png', '17280.png', '18681.png', '21303.png', '20500.png', '10546.png', '18031.png', '21277.png', '19359.png', '19581.png', '22092.png', '05775.png', '15271.png', '02802.png', '17257.png', '08612.png', '13549.png', '02505.png', '05884.png', '19126.png', '10375.png', '20774.png', '21489.png', '22453.png', '05781.png', '07075.png', '14426.png', '14035.png', '05262.png', '16252.png', '05024.png', '17735.png', '04412.png', '23320.png', '05641.png', '00063.png', '04719.png', '21537.png', '16774.png', '12296.png', '03535.png', '20213.png', '00710.png', '04302.png', '04777.png', '21741.png', '04714.png', '18586.png', '15182.png', '05900.png', '15009.png', '13128.png', '06460.png', '10109.png', '17096.png', '08226.png', '07676.png', '13584.png', '19644.png', '12714.png', '11418.png', '22987.png', '07126.png', '23564.png', '04359.png', '04296.png', '24336.png', '10044.png', '03450.png', '02792.png', '19320.png', '11804.png', '24686.png', '01228.png', '16174.png', '05459.png', '03214.png', '01876.png', '03089.png', '04343.png', '07802.png', '08028.png', '06323.png', '15689.png', '04926.png', '01109.png', '02821.png', '23101.png', '06811.png', '17960.png', '15424.png', '15112.png', '00146.png', '23331.png', '11687.png', '10448.png', '17497.png', '19455.png', '19651.png', '09120.png', '22582.png', '17816.png', '18302.png', '00814.png', '17522.png', '06085.png', '07271.png', '07205.png', '23064.png', '12945.png', '07917.png', '09338.png', '13942.png', '05020.png', '23929.png', '18073.png', '10812.png', '23378.png', '04980.png', '00243.png', '03040.png', '24841.png', '23567.png', '13148.png', '22202.png', '16792.png', '08943.png', '17533.png', '08923.png', '01328.png', '11829.png', '24789.png', '05364.png', '09666.png', '07274.png', '19938.png', '16000.png', '19119.png', '10271.png', '02038.png', '18047.png', '22812.png', '04150.png', '04766.png', '07717.png', '10678.png', '05958.png', '18329.png', '08366.png', '03790.png', '13187.png', '17225.png', '09658.png', '14751.png', '02377.png', '04336.png', '03906.png', '09122.png', '20295.png', '00595.png', '22660.png', '20541.png', '09172.png', '02302.png', '03472.png', '08868.png', '15301.png', '10900.png', '07290.png', '11366.png', '14720.png', '16079.png', '00783.png', '18387.png', '21162.png', '20401.png', '14647.png', '10853.png', '23304.png', '05162.png', '19629.png', '14110.png', '20252.png', '03155.png', '15893.png', '17264.png', '08123.png', '19163.png', '12319.png', '19124.png', '14576.png', '23509.png', '22643.png', '03858.png', '17482.png', '12641.png', '10777.png', '14187.png', '02715.png', '21520.png', '02032.png', '04178.png', '22436.png', '03775.png', '08040.png', '08574.png', '00356.png', '14949.png', '04188.png', '03614.png', '06399.png', '06841.png', '13270.png', '05220.png', '20083.png', '18927.png', '09878.png', '12349.png', '10625.png', '23816.png', '14794.png', '16447.png', '13723.png', '01847.png', '06091.png', '07939.png', '12091.png', '07906.png', '04217.png', '16378.png', '05733.png', '01031.png', '06955.png', '08509.png', '12404.png', '00593.png', '21818.png', '02489.png', '20798.png', '08878.png', '13401.png', '04984.png', '18015.png', '09716.png', '10110.png', '20868.png', '15561.png', '05244.png', '06623.png', '10032.png', '13217.png', '20697.png', '09941.png', '13818.png', '13119.png', '16391.png', '12705.png', '10365.png', '01842.png', '12632.png', '12205.png', '11263.png', '14429.png', '17334.png', '13390.png', '24821.png', '01652.png', '11639.png', '15664.png', '18359.png', '13881.png', '01925.png', '19413.png', '07066.png', '04417.png', '10710.png', '10696.png', '24282.png', '14391.png', '17185.png', '03938.png', '05126.png', '19288.png', '19658.png', '11952.png', '07488.png', '15100.png', '23006.png', '12370.png', '22940.png', '18556.png', '03883.png', '15396.png', '06812.png', '13285.png', '09069.png', '08597.png', '03028.png', '21816.png', '12864.png', '19526.png', '11381.png', '21380.png', '21142.png', '00101.png', '12636.png', '03468.png', '15532.png', '19367.png', '22903.png', '03414.png', '18183.png', '06845.png', '07718.png', '05065.png', '20153.png', '14332.png', '10169.png', '22349.png', '20172.png', '04340.png', '19013.png', '05705.png', '14948.png', '01632.png', '12265.png', '08216.png', '14029.png', '14805.png', '10396.png', '17265.png', '07602.png', '20518.png', '08098.png', '09502.png', '15506.png', '23465.png', '12940.png', '06676.png', '14473.png', '04671.png', '10082.png', '10518.png', '01710.png', '17075.png', '04513.png', '17053.png', '19802.png', '01774.png', '20285.png', '12459.png', '21282.png', '03082.png', '03346.png', '01556.png', '09307.png', '08019.png', '06649.png', '04907.png', '04013.png', '07213.png', '09561.png', '04622.png', '21174.png', '08173.png', '04348.png', '21802.png', '08148.png', '11741.png', '08368.png', '00640.png', '04705.png', '10568.png', '03688.png', '23972.png', '06764.png', '09455.png', '00683.png', '05214.png', '07235.png', '18820.png', '04331.png', '20407.png', '08046.png', '05736.png', '18653.png', '03362.png', '12780.png', '18901.png', '03557.png', '22184.png', '01028.png', '20445.png', '01532.png', '15901.png', '08126.png', '10276.png', '14097.png', '01174.png', '13952.png', '23368.png', '13487.png', '23493.png', '05737.png', '04852.png', '00529.png', '01816.png', '23231.png', '17617.png', '15262.png', '03282.png', '03377.png', '10224.png', '12007.png', '21205.png', '18825.png', '22549.png', '02812.png', '05909.png', '02396.png', '14150.png', '16018.png', '21056.png', '15754.png', '14971.png', '07055.png', '20725.png', '10950.png', '11477.png', '21046.png', '17643.png', '24957.png', '09585.png', '19910.png', '20633.png', '04156.png', '14436.png', '22530.png', '05927.png', '17559.png', '15193.png', '16835.png', '04368.png', '06355.png', '12502.png', '07383.png', '06351.png', '17453.png', '01781.png', '07020.png', '14531.png', '11547.png', '10630.png', '22964.png', '00068.png', '04367.png', '18118.png', '20388.png', '05279.png', '18079.png', '01711.png', '14820.png', '02348.png', '03674.png', '21215.png', '05907.png', '08555.png', '03725.png', '03973.png', '24781.png', '11515.png', '09433.png', '06495.png', '23959.png', '20263.png', '05300.png', '14844.png', '08389.png', '09968.png', '07002.png', '03769.png', '02263.png', '22661.png', '19203.png', '24863.png', '04240.png', '23810.png', '16746.png', '14292.png', '03813.png', '23738.png', '00225.png', '20040.png', '04928.png', '12016.png', '14745.png', '08387.png', '00653.png', '10790.png', '06446.png', '13006.png', '03101.png', '09052.png', '07592.png', '19933.png', '06308.png', '06079.png', '15279.png', '06959.png', '05030.png', '17339.png', '18841.png', '10496.png', '10776.png', '14690.png', '22286.png', '10201.png', '02342.png', '01315.png', '07094.png', '08106.png', '06149.png', '13488.png', '08905.png', '20836.png', '23248.png', '10621.png', '15080.png', '03744.png', '22575.png', '09468.png', '01231.png', '21930.png', '03987.png', '03685.png', '18809.png', '11411.png', '03392.png', '01081.png', '24746.png', '20164.png', '01794.png', '10084.png', '09203.png', '22408.png', '13028.png', '20335.png', '12590.png', '10648.png', '08370.png', '22590.png', '20349.png', '24812.png', '07594.png', '17111.png', '03846.png', '20415.png', '09553.png', '00598.png', '06087.png', '15387.png', '10690.png', '22085.png', '00036.png', '07111.png', '03752.png', '24350.png', '24209.png', '05524.png', '02577.png', '15801.png', '14413.png', '14174.png', '16621.png', '15344.png', '16743.png', '21885.png', '13456.png', '15649.png', '19701.png', '06260.png', '14838.png', '05313.png', '13495.png', '08498.png', '06891.png', '05832.png', '08108.png', '14781.png', '02145.png', '00756.png', '11564.png', '02506.png', '12602.png', '08442.png', '16738.png', '17148.png', '07872.png', '07844.png', '24762.png', '14834.png', '15706.png', '15026.png', '21453.png', '11620.png', '08243.png', '10649.png', '18698.png', '03117.png', '00181.png', '22088.png', '22060.png', '07134.png', '13473.png', '24850.png', '14701.png', '19812.png', '24500.png', '10689.png', '04911.png', '03860.png', '05820.png', '23611.png', '17244.png', '07792.png', '00903.png', '20089.png', '20788.png', '03092.png', '19187.png', '22244.png', '15158.png', '00998.png', '02720.png', '07345.png', '04895.png', '06274.png', '14076.png', '24791.png', '12797.png', '07108.png', '18223.png', '21526.png', '05373.png', '24627.png', '22653.png', '19884.png', '18348.png', '23532.png', '04416.png', '16370.png', '13226.png', '19905.png', '04101.png', '18206.png', '08911.png', '17677.png', '07366.png', '20757.png', '23312.png', '06971.png', '02613.png', '04570.png', '13999.png', '08863.png', '12391.png', '21265.png', '22305.png', '23287.png', '18126.png', '18832.png', '17672.png', '16390.png', '04998.png', '11839.png', '16383.png', '12781.png', '14054.png', '04739.png', '05632.png', '04401.png', '08022.png', '09895.png', '11223.png', '04747.png', '08439.png', '06863.png', '21054.png', '23759.png', '24613.png', '13887.png', '23408.png', '24133.png', '18092.png', '11548.png', '07826.png', '00093.png', '13992.png', '05129.png', '00607.png', '23416.png', '06062.png', '15668.png', '21038.png', '05299.png', '07754.png', '12388.png', '14828.png', '09761.png', '22329.png', '08138.png', '04017.png', '07969.png', '00563.png', '23347.png', '18670.png', '14605.png', '12156.png', '06000.png', '19287.png', '00775.png', '07638.png', '03515.png', '14908.png', '05575.png', '06854.png', '11643.png', '05475.png', '07553.png', '10989.png', '07662.png', '02606.png', '20165.png', '21454.png', '04517.png', '11781.png', '14499.png', '01248.png', '10482.png', '11863.png', '10975.png', '13912.png', '06597.png', '03230.png', '19043.png', '08415.png', '09676.png', '10808.png', '19458.png', '15043.png', '03015.png', '06558.png', '05986.png', '10494.png', '08991.png', '09490.png', '07347.png', '21525.png', '18631.png', '22320.png', '15806.png', '17270.png', '15492.png', '24103.png', '18476.png', '13943.png', '03299.png', '18612.png', '06754.png', '20676.png', '24442.png', '23819.png', '12263.png', '13640.png', '15401.png', '04619.png', '21832.png', '01937.png', '15108.png', '02171.png', '15221.png', '21718.png', '22843.png', '18462.png', '05650.png', '16619.png', '14196.png', '10996.png', '01209.png', '03198.png', '23609.png', '13104.png', '20222.png', '05804.png', '09926.png', '06934.png', '00724.png', '22674.png', '22216.png', '04977.png', '00223.png', '16674.png', '13693.png', '05545.png', '03826.png', '23354.png', '16957.png', '23563.png', '08206.png', '15959.png', '22768.png', '00226.png', '19468.png', '16721.png', '24898.png', '15442.png', '04938.png', '23373.png', '24297.png', '01939.png', '00365.png', '21160.png', '11591.png', '12901.png', '02654.png', '01102.png', '13163.png', '21530.png', '07256.png', '16984.png', '17143.png', '00807.png', '11281.png', '02788.png', '12712.png', '04555.png', '15829.png', '05152.png', '09108.png', '24197.png', '15778.png', '01887.png', '11809.png', '05387.png', '21435.png', '22463.png', '20524.png', '05868.png', '19860.png', '00248.png', '05073.png', '10617.png', '18712.png', '06431.png', '02374.png', '08619.png', '12462.png', '16214.png', '10684.png', '19992.png', '18255.png', '22051.png', '03602.png', '21588.png', '11805.png', '17560.png', '01773.png', '20936.png', '14487.png', '11729.png', '24035.png', '20867.png', '16701.png', '05674.png', '24633.png', '07583.png', '12445.png', '10730.png', '23624.png', '03187.png', '02220.png', '13126.png', '15418.png', '09231.png', '04078.png', '12049.png', '16639.png', '07432.png', '16977.png', '01176.png', '01924.png', '12732.png', '14401.png', '22130.png', '20008.png', '23650.png', '23646.png', '10479.png', '08447.png', '05843.png', '20921.png', '01203.png', '06004.png', '21077.png', '00431.png', '10236.png', '19696.png', '23278.png', '11118.png', '17513.png', '21656.png', '11192.png', '07363.png', '11377.png', '17268.png', '20910.png', '02143.png', '15064.png', '10742.png', '17351.png', '15439.png', '05113.png', '22374.png', '09824.png', '13982.png', '22921.png', '04548.png', '14361.png', '04522.png', '06798.png', '08134.png', '04204.png', '08241.png', '09723.png', '01269.png', '22517.png', '02379.png', '05887.png', '23723.png', '03383.png', '24238.png', '08379.png', '12080.png', '23110.png', '13218.png', '16838.png', '05809.png', '07125.png', '18845.png', '13583.png', '20599.png', '16236.png', '07873.png', '04074.png', '01737.png', '01074.png', '01010.png', '15008.png', '10078.png', '24938.png', '14396.png', '21926.png', '11654.png', '04709.png', '04527.png', '13081.png', '18072.png', '09480.png', '05484.png', '01056.png', '14744.png', '21909.png', '06506.png', '09156.png', '00058.png', '02443.png', '14299.png', '21253.png', '22642.png', '07487.png', '05499.png', '21154.png', '12169.png', '14686.png', '15923.png', '21884.png', '15624.png', '24548.png', '15842.png', '16158.png', '09742.png', '09074.png', '00507.png', '17062.png', '18268.png', '01968.png', '12675.png', '06070.png', '22180.png', '04053.png', '04715.png', '03600.png', '18404.png', '18947.png', '19523.png', '18202.png', '14626.png', '21580.png', '03090.png', '10184.png', '19474.png', '06183.png', '01018.png', '16524.png', '24233.png', '15222.png', '03721.png', '17426.png', '18349.png', '01060.png', '01194.png', '19620.png', '20314.png', '21939.png', '10480.png', '09976.png', '20990.png', '20081.png', '00997.png', '12207.png', '02487.png', '13771.png', '09718.png', '14327.png', '14807.png', '10508.png', '17991.png', '01142.png', '12847.png', '12402.png', '21791.png', '05703.png', '05458.png', '17465.png', '23172.png', '13464.png', '13838.png', '14166.png', '23589.png', '03029.png', '19597.png', '07666.png', '08528.png', '04632.png', '06608.png', '06668.png', '20046.png', '22326.png', '15647.png', '19945.png', '06244.png', '11599.png', '19470.png', '04942.png', '19845.png', '05137.png', '20115.png', '10346.png', '05784.png', '02826.png', '02611.png', '15162.png', '16578.png', '19394.png', '08055.png', '03591.png', '14813.png', '05639.png', '12871.png', '23457.png', '16877.png', '16722.png', '18490.png', '07227.png', '05047.png', '07551.png', '14585.png', '08848.png', '12623.png', '06270.png', '23982.png', '19291.png', '24729.png', '18680.png', '03255.png', '17846.png', '04440.png', '24347.png', '19688.png', '20550.png', '07217.png', '15121.png', '24275.png', '18824.png', '24646.png', '01101.png', '13190.png', '00011.png', '10081.png', '19174.png', '06931.png', '06052.png', '15692.png', '00733.png', '18094.png', '12363.png', '12955.png', '18102.png', '12563.png', '02759.png', '16432.png', '00060.png', '22799.png', '04025.png', '10553.png', '16024.png', '12773.png', '14333.png', '01422.png', '08355.png', '19369.png', '11600.png', '07876.png', '04380.png', '11802.png', '12225.png', '19975.png', '00210.png', '07120.png', '11291.png', '08168.png', '04547.png', '12592.png', '23655.png', '17475.png', '18593.png', '10984.png', '18053.png', '17588.png', '21901.png', '09952.png', '11037.png', '08154.png', '10665.png', '05186.png', '17733.png', '07346.png', '15984.png', '16975.png', '14990.png', '04022.png', '18277.png', '17965.png', '05977.png', '11357.png', '23968.png', '10330.png', '05670.png', '10632.png', '02620.png', '06238.png', '17141.png', '08795.png', '23365.png', '14104.png', '20668.png', '16852.png', '22272.png', '20356.png', '24198.png', '10932.png', '01557.png', '24104.png', '21617.png', '11747.png', '11881.png', '14369.png', '10141.png', '18107.png', '01313.png', '15731.png', '14133.png', '12064.png', '02233.png', '02952.png', '09089.png', '06973.png', '10840.png', '04761.png', '11314.png', '08398.png', '19694.png', '11989.png', '15537.png', '22982.png', '12840.png', '15601.png', '24875.png', '07661.png', '06375.png', '06832.png', '05371.png', '09745.png', '19864.png', '01582.png', '21413.png', '17946.png', '10049.png', '02605.png', '18258.png', '19213.png', '16394.png', '18893.png', '16489.png', '18167.png', '11891.png', '24399.png', '03566.png', '01891.png', '13561.png', '16166.png', '10851.png', '23259.png', '00179.png', '24826.png', '19068.png', '16452.png', '09850.png', '18005.png', '20217.png', '18368.png', '12598.png', '15381.png', '20147.png', '14919.png', '03382.png', '13726.png', '16706.png', '03821.png', '17310.png', '23267.png', '24248.png', '16842.png', '17540.png', '10686.png', '03356.png', '00404.png', '13394.png', '12584.png', '09899.png', '04596.png', '20104.png', '08420.png', '09467.png', '06682.png', '10138.png', '21230.png', '01692.png', '08463.png', '09029.png', '13711.png', '09966.png', '24528.png', '18969.png', '02838.png', '08746.png', '00933.png', '01013.png', '12756.png', '05711.png', '18669.png', '14587.png', '23252.png', '11505.png', '03210.png', '01705.png', '05787.png', '03195.png', '15184.png', '08823.png', '10854.png', '02740.png', '16817.png', '04664.png', '01634.png', '12518.png', '11104.png', '17968.png', '14404.png', '09608.png', '13843.png', '08869.png', '13260.png', '09612.png', '04108.png', '04908.png', '10713.png', '16982.png', '18299.png', '08667.png', '13099.png', '03953.png', '21084.png', '20338.png', '09851.png', '07675.png', '17486.png', '20547.png', '06985.png', '14293.png', '09582.png', '24415.png', '02868.png', '14973.png', '02365.png', '16898.png', '23725.png', '12474.png', '12397.png', '24179.png', '11838.png', '11099.png', '08339.png', '16164.png', '11330.png', '12961.png', '06691.png', '12058.png', '03241.png', '08088.png', '00946.png', '22607.png', '24851.png', '20660.png', '18353.png', '02867.png', '13833.png', '17275.png', '12537.png', '19099.png', '22791.png', '15933.png', '18249.png', '22388.png', '09967.png', '11194.png', '10250.png', '07689.png', '17263.png', '00247.png', '04389.png', '18960.png', '19011.png', '02432.png', '13918.png', '23943.png', '01428.png', '10610.png', '24273.png', '00312.png', '15997.png', '13540.png', '00514.png', '23884.png', '22407.png', '22979.png', '02709.png', '01412.png', '00850.png', '03583.png', '07521.png', '18585.png', '21755.png', '15993.png', '05350.png', '07562.png', '01834.png', '16907.png', '11756.png', '17845.png', '03831.png', '23467.png', '17495.png', '19619.png', '16272.png', '11296.png', '13876.png', '01373.png', '14797.png', '20189.png', '11948.png', '17441.png', '01527.png', '02579.png', '10611.png', '00010.png', '09360.png', '16951.png', '14344.png', '08859.png', '16729.png', '00958.png', '11312.png', '20802.png', '21611.png', '17054.png', '02193.png', '17890.png', '06782.png', '02474.png', '15547.png', '21646.png', '16839.png', '23134.png', '09740.png', '04085.png', '13482.png', '20414.png', '20946.png', '22385.png', '21481.png', '02426.png', '06675.png', '10349.png', '14408.png', '20943.png', '23776.png', '12501.png', '12414.png', '07667.png', '06009.png', '17719.png', '09412.png', '14055.png', '20870.png', '22268.png', '22578.png', '08012.png', '15823.png', '09821.png', '21756.png', '03440.png', '20205.png', '11252.png', '23152.png', '11251.png', '04857.png', '09186.png', '11561.png', '05199.png', '21600.png', '07081.png', '19389.png', '15961.png', '21401.png', '03062.png', '11767.png', '21651.png', '08969.png', '05607.png', '00493.png', '21801.png', '03129.png', '12867.png', '21763.png', '06704.png', '19227.png', '04616.png', '13434.png', '18569.png', '23732.png', '21838.png', '15725.png', '17808.png', '22098.png', '16651.png', '06787.png', '07117.png', '06834.png', '04586.png', '16067.png', '19870.png', '15517.png', '17970.png', '05263.png', '04011.png', '19546.png', '16501.png', '10403.png', '20364.png', '09747.png', '14782.png', '01425.png', '18924.png', '03419.png', '06522.png', '24128.png', '10069.png', '16750.png', '03668.png', '04786.png', '13411.png', '04679.png', '17897.png', '10273.png', '19420.png', '21237.png', '19435.png', '16317.png', '20150.png', '05405.png', '14039.png', '16931.png', '12052.png', '11690.png', '03201.png', '08611.png', '15126.png', '15398.png', '09384.png', '06400.png', '08590.png', '15677.png', '07591.png', '04279.png', '20001.png', '12036.png', '16460.png', '04787.png', '06508.png', '09424.png', '23496.png', '04079.png', '12777.png', '13085.png', '05663.png', '06372.png', '15161.png', '01926.png', '24497.png', '17014.png', '21031.png', '15819.png', '11887.png', '11985.png', '21143.png', '20857.png', '17434.png', '05635.png', '02997.png', '17317.png', '11962.png', '21297.png', '01613.png', '23459.png', '23834.png', '04627.png', '18331.png', '11450.png', '12342.png', '00453.png', '16091.png', '12098.png', '19064.png', '16042.png', '10170.png', '09532.png', '13322.png', '19616.png', '20781.png', '16912.png', '09717.png', '00671.png', '05175.png', '21417.png', '23388.png', '09349.png', '24606.png', '05772.png', '06120.png', '11603.png', '12143.png', '00116.png', '15237.png', '24558.png', '08936.png', '05454.png', '05931.png', '17107.png', '19262.png', '06960.png', '24299.png', '01522.png', '17532.png', '14308.png', '20307.png', '17749.png', '10424.png', '07121.png', '23764.png', '18676.png', '19825.png', '23318.png', '16961.png', '17373.png', '24025.png', '21234.png', '13043.png', '01698.png', '16351.png', '10680.png', '10183.png', '02280.png', '16386.png', '13767.png', '14300.png', '09555.png', '01024.png', '05421.png', '00583.png', '24877.png', '11135.png', '00862.png', '13782.png', '06423.png', '23928.png', '03892.png', '24649.png', '06665.png', '23009.png', '17797.png', '04602.png', '23340.png', '19662.png', '19567.png', '03373.png', '08394.png', '16140.png', '22785.png', '24081.png', '23062.png', '16747.png', '10259.png', '21005.png', '20820.png', '05959.png', '08687.png', '16234.png', '22967.png', '10380.png', '16565.png', '18303.png', '05413.png', '13416.png', '05790.png', '19209.png', '20312.png', '23886.png', '23166.png', '08849.png', '11872.png', '12051.png', '18443.png', '11722.png', '22295.png', '22056.png', '17226.png', '22698.png', '14001.png', '12487.png', '15542.png', '22543.png', '12488.png', '17542.png', '01438.png', '19868.png', '23401.png', '14373.png', '20206.png', '04228.png', '04192.png', '04546.png', '01030.png', '08290.png', '24037.png', '13568.png', '23475.png', '14432.png', '22727.png', '19732.png', '01872.png', '23793.png', '22302.png', '12868.png', '15916.png', '18613.png', '13618.png', '07147.png', '12913.png', '16306.png', '03677.png', '10862.png', '13910.png', '13090.png', '03261.png', '10752.png', '08550.png', '11385.png', '24709.png', '24002.png', '11626.png', '12876.png', '21267.png', '06551.png', '22354.png', '21522.png', '16594.png', '12646.png', '12952.png', '17471.png', '01180.png', '09289.png', '01415.png', '17035.png', '22077.png', '16762.png', '17099.png', '08798.png', '13826.png', '17105.png', '00970.png', '21269.png', '01767.png', '06053.png', '14334.png', '01677.png', '06268.png', '10458.png', '03225.png', '17740.png', '19606.png', '08689.png', '05097.png', '19773.png', '12305.png', '18774.png', '07396.png', '23837.png', '14153.png', '04545.png', '11203.png', '01160.png', '14916.png', '12509.png', '17267.png', '20190.png', '08128.png', '23440.png', '16856.png', '12130.png', '24435.png', '11407.png', '11363.png', '21784.png', '15089.png', '24961.png', '04827.png', '18733.png', '04169.png', '16494.png', '09233.png', '02629.png', '23068.png', '09929.png', '10427.png', '16516.png', '12054.png', '09946.png', '07437.png', '19664.png', '08979.png', '22758.png', '04745.png', '02752.png', '20786.png', '02146.png', '20439.png', '11679.png', '11960.png', '20848.png', '21151.png', '08421.png', '17149.png', '04376.png', '23225.png', '19405.png', '12046.png', '22183.png', '07203.png', '24130.png', '08577.png', '21250.png', '18217.png', '23579.png', '08503.png', '17329.png', '07778.png', '18550.png', '18129.png', '21226.png', '17938.png', '20596.png', '22539.png', '22863.png', '21667.png', '17127.png', '20869.png', '00430.png', '19318.png', '07476.png', '12365.png', '13722.png', '19326.png', '08417.png', '02226.png', '23804.png', '19595.png', '09699.png', '04197.png', '21382.png', '17241.png', '10759.png', '21723.png', '19721.png', '13337.png', '23610.png', '05921.png', '19201.png', '20117.png', '20326.png', '06887.png', '16363.png', '09939.png', '00328.png', '00550.png', '17679.png', '16040.png', '15776.png', '12191.png', '15127.png', '23683.png', '23839.png', '10928.png', '19857.png', '15118.png', '05115.png', '14939.png', '17092.png', '08392.png', '00637.png', '02901.png', '23537.png', '05699.png', '05010.png', '23397.png', '22294.png', '08683.png', '11880.png', '23921.png', '22829.png', '16915.png', '12689.png', '06143.png', '09830.png', '09342.png', '06825.png', '23349.png', '07871.png', '21541.png', '12782.png', '05049.png', '20420.png', '11133.png', '24752.png', '14124.png', '08845.png', '22624.png', '05801.png', '24440.png', '22942.png', '11023.png', '10869.png', '19505.png', '11091.png', '03084.png', '20763.png', '23409.png', '18495.png', '03119.png', '09086.png', '01096.png', '13573.png', '04352.png', '19685.png', '06523.png', '23663.png', '16267.png', '00907.png', '15090.png', '05584.png', '07688.png', '13327.png', '06700.png', '10822.png', '20446.png', '18280.png', '23227.png', '17582.png', '11972.png', '10504.png', '13418.png', '15272.png', '18738.png', '03551.png', '14070.png', '14182.png', '03415.png', '20893.png', '17516.png', '11178.png', '11524.png', '21285.png', '07030.png', '24190.png', '00200.png', '14171.png', '12493.png', '12851.png', '02968.png', '22670.png', '11527.png', '10005.png', '10583.png', '17843.png', '03760.png', '15869.png', '06589.png', '23107.png', '05974.png', '12902.png', '16990.png', '18310.png', '08733.png', '12721.png', '20754.png', '21648.png', '08017.png', '19528.png', '03993.png', '13384.png', '23925.png', '04684.png', '18456.png', '19866.png', '23094.png', '04056.png', '00014.png', '15958.png', '08169.png', '12803.png', '01224.png', '13498.png', '15641.png', '10047.png', '14445.png', '00911.png', '01369.png', '17514.png', '15519.png', '02424.png', '01637.png', '11577.png', '14451.png', '07573.png', '21178.png', '04462.png', '15810.png', '07606.png', '06011.png', '08343.png', '02479.png', '12709.png', '06093.png', '24261.png', '17255.png', '19991.png', '12710.png', '00246.png', '08867.png', '20528.png', '08600.png', '00313.png', '17790.png', '05796.png', '07130.png', '15106.png', '06477.png', '08708.png', '02610.png', '19423.png', '18856.png', '04227.png', '24931.png', '03889.png', '04599.png', '13669.png', '19415.png', '03232.png', '08558.png', '09910.png', '13681.png', '04199.png', '24026.png', '22384.png', '11249.png', '09794.png', '21908.png', '14586.png', '12613.png', '05748.png', '16615.png', '01956.png', '22927.png', '08316.png', '02582.png', '23940.png', '19390.png', '15915.png', '14831.png', '07758.png', '10599.png', '21096.png', '19062.png', '13236.png', '13070.png', '07199.png', '14629.png', '09448.png', '19843.png', '05395.png', '07699.png', '09907.png', '20208.png', '14034.png', '16414.png', '05028.png', '01252.png', '15273.png', '18488.png', '14864.png', '11811.png', '02657.png', '07382.png', '23090.png', '10196.png', '04935.png', '07344.png', '07618.png', '12898.png', '07365.png', '08987.png', '13111.png', '07411.png', '06212.png', '15049.png', '23295.png', '22825.png', '23885.png', '02030.png', '07249.png', '08286.png', '11587.png', '23154.png', '11519.png', '12344.png', '12720.png', '21999.png', '04094.png', '07063.png', '12744.png', '19179.png', '23194.png', '14892.png', '05512.png', '20768.png', '11277.png', '19348.png', '00651.png', '01996.png', '00857.png', '01124.png', '24378.png', '21086.png', '04821.png', '08965.png', '16579.png', '02102.png', '03133.png', '17573.png', '13354.png', '07440.png', '18060.png', '10354.png', '16076.png', '12580.png', '23560.png', '22672.png', '12075.png', '24152.png', '02639.png', '22630.png', '24327.png', '06803.png', '17091.png', '15971.png', '03262.png', '24411.png', '05480.png', '03438.png', '23002.png', '16630.png', '01987.png', '07449.png', '17535.png', '00757.png', '02161.png', '09194.png', '13075.png', '22858.png', '16080.png', '01466.png', '07878.png', '15451.png', '19718.png', '04637.png', '24944.png', '05114.png', '21793.png', '09999.png', '13530.png', '13842.png', '06984.png', '17758.png', '03151.png', '03058.png', '13197.png', '11672.png', '11164.png', '15261.png', '04044.png', '18730.png', '12524.png', '01182.png', '21150.png', '07116.png', '08246.png', '05445.png', '04235.png', '15926.png', '02501.png', '12111.png', '18563.png', '00680.png', '02793.png', '19944.png', '09482.png', '19444.png', '16021.png', '19515.png', '21358.png', '21450.png', '12399.png', '01583.png', '13672.png', '07124.png', '14319.png', '20236.png', '16859.png', '07244.png', '17425.png', '01785.png', '06824.png', '07504.png', '07924.png', '21771.png', '01525.png', '02887.png', '12983.png', '10608.png', '11478.png', '03421.png', '04466.png', '23028.png', '18760.png', '20924.png', '11661.png', '17591.png', '19676.png', '16377.png', '16287.png', '17842.png', '02484.png', '06390.png', '15539.png', '21620.png', '19094.png', '11435.png', '08741.png', '19424.png', '05222.png', '17755.png', '19877.png', '08855.png', '20654.png', '12124.png', '18363.png', '11651.png', '19524.png', '15894.png', '15938.png', '00264.png', '20075.png', '14134.png', '19388.png', '17382.png', '00303.png', '07892.png', '18762.png', '04822.png', '01870.png', '16411.png', '18386.png', '04704.png', '11584.png', '22484.png', '17930.png', '20053.png', '11637.png', '11851.png', '16461.png', '17858.png', '07061.png', '07397.png', '24218.png', '24868.png', '02841.png', '02839.png', '02909.png', '24946.png', '23535.png', '16866.png', '06653.png', '18621.png', '02026.png', '24667.png', '05795.png', '09040.png', '09460.png', '05023.png', '10952.png', '14284.png', '12434.png', '13908.png', '16785.png', '12890.png', '01790.png', '15338.png', '16499.png', '23240.png', '08639.png', '13123.png', '05448.png', '00207.png', '17027.png', '06096.png', '05135.png', '21678.png', '06837.png', '14061.png', '04503.png', '03372.png', '01275.png', '18059.png', '24161.png', '03311.png', '14126.png', '21071.png', '02476.png', '19584.png', '22971.png', '07590.png', '19963.png', '17506.png', '07149.png', '09494.png', '16546.png', '24947.png', '03007.png', '11335.png', '03902.png', '12018.png', '18958.png', '00315.png', '22777.png', '11996.png', '16105.png', '18534.png', '03359.png', '04272.png', '10775.png', '15951.png', '15334.png', '08538.png', '16162.png', '24266.png', '03897.png', '15501.png', '24162.png', '24006.png', '24846.png', '11827.png', '20934.png', '18356.png', '01944.png', '11772.png', '09137.png', '21630.png', '11358.png', '18498.png', '06153.png', '10228.png', '09405.png', '05702.png', '12528.png', '21612.png', '19565.png', '19917.png', '11173.png', '23827.png', '22228.png', '05791.png', '17855.png', '06930.png', '20749.png', '18222.png', '16467.png', '16237.png', '05467.png', '21834.png', '20615.png', '18864.png', '16714.png', '06937.png', '06888.png', '15489.png', '21449.png', '13054.png', '24882.png', '18811.png', '00626.png', '05288.png', '16816.png', '07478.png', '23469.png', '16698.png', '18883.png', '19922.png', '24259.png', '21485.png', '07377.png', '08477.png', '00611.png', '10484.png', '14032.png', '09897.png', '14099.png', '14249.png', '16862.png', '08533.png', '19015.png', '05494.png', '09637.png', '23770.png', '18651.png', '21644.png', '04629.png', '21229.png', '01123.png', '09072.png', '23029.png', '17319.png', '07950.png', '23181.png', '13288.png', '06352.png', '05640.png', '13063.png', '02386.png', '20467.png', '09418.png', '10317.png', '20801.png', '17417.png', '13588.png', '20159.png', '10213.png', '23636.png', '04133.png', '21654.png', '17356.png', '17853.png', '07137.png', '06729.png', '02676.png', '00934.png', '13914.png', '09531.png', '17387.png', '12273.png', '10535.png', '23947.png', '05001.png', '21278.png', '21266.png', '03203.png', '19127.png', '08706.png', '21561.png', '08884.png', '06778.png', '09589.png', '09572.png', '06671.png', '01667.png', '09401.png', '13762.png', '14974.png', '13935.png', '24788.png', '23168.png', '06281.png', '04372.png', '04746.png', '13802.png', '15657.png', '12242.png', '08223.png', '11743.png', '05138.png', '01889.png', '04393.png', '11142.png', '24265.png', '03547.png', '05675.png', '20188.png', '07575.png', '09883.png', '24719.png', '17081.png', '23984.png', '10794.png', '11850.png', '11691.png', '08610.png', '06947.png', '03554.png', '08068.png', '06119.png', '23016.png', '12280.png', '19788.png', '17934.png', '10381.png', '23604.png', '18056.png', '22147.png', '07803.png', '00753.png', '15156.png', '23013.png', '07493.png', '22022.png', '06231.png', '04254.png', '07911.png', '04151.png', '15036.png', '09633.png', '10015.png', '23941.png', '02853.png', '17424.png', '18562.png', '19200.png', '10994.png', '16373.png', '01909.png', '11861.png', '24958.png', '00376.png', '21569.png', '19525.png', '19190.png', '16185.png', '10238.png', '13504.png', '12361.png', '01959.png', '12073.png', '20673.png', '00812.png', '15236.png', '21307.png', '13983.png', '14036.png', '19309.png', '19715.png', '10421.png', '15684.png', '15198.png', '12963.png', '17958.png', '01303.png', '13239.png', '11362.png', '01285.png', '09702.png', '21628.png', '22862.png', '24777.png', '04411.png', '00211.png', '07910.png', '15245.png', '20702.png', '21815.png', '17599.png', '23760.png', '20136.png', '06434.png', '14222.png', '09196.png', '20395.png', '11776.png', '03184.png', '00571.png', '01568.png', '18765.png', '11016.png', '04840.png', '21944.png', '08475.png', '05877.png', '09034.png', '12017.png', '10030.png', '16201.png', '24586.png', '01881.png', '19610.png', '23631.png', '11134.png', '20483.png', '08163.png', '06797.png', '04211.png', '11779.png', '04655.png', '17680.png', '12019.png', '08658.png', '21029.png', '11522.png', '24602.png', '04353.png', '23602.png', '07119.png', '21616.png', '09559.png', '13867.png', '08199.png', '21795.png', '14103.png', '16925.png', '09586.png', '06272.png', '24910.png', '00968.png', '02875.png', '14620.png', '03503.png', '24655.png', '11128.png', '17067.png', '06176.png', '17746.png', '18298.png', '17448.png', '16885.png', '10409.png', '00833.png', '11512.png', '15949.png', '11840.png', '11238.png', '24293.png', '23083.png', '11275.png', '19585.png', '17098.png', '16106.png', '01016.png', '11064.png', '22336.png', '16453.png', '00532.png', '17000.png', '16833.png', '03504.png', '13718.png', '08751.png', '22156.png', '15137.png', '03660.png', '08967.png', '03742.png', '19292.png', '00031.png', '16576.png', '15292.png', '22220.png', '09465.png', '12229.png', '19797.png', '00981.png', '00205.png', '18787.png', '16426.png', '16017.png', '14809.png', '19421.png', '18987.png', '15149.png', '17543.png', '23817.png', '01134.png', '18417.png', '12439.png', '21163.png', '07284.png', '06492.png', '06426.png', '16188.png', '14867.png', '13196.png', '22256.png', '14918.png', '07042.png', '22766.png', '23577.png', '22383.png', '21823.png', '17659.png', '19591.png', '05835.png', '14954.png', '22796.png', '09732.png', '02921.png', '14350.png', '03192.png', '22257.png', '17632.png', '07252.png', '10796.png', '09478.png', '21325.png', '01890.png', '19379.png', '18879.png', '16755.png', '04785.png', '12965.png', '08873.png', '06519.png', '12547.png', '05665.png', '18401.png', '08988.png', '13628.png', '06297.png', '14747.png', '24268.png', '17777.png', '10220.png', '02149.png', '24140.png', '05878.png', '01083.png', '09415.png', '04498.png', '17110.png', '23005.png', '17189.png', '03546.png', '11351.png', '03173.png', '19683.png', '19613.png', '02168.png', '15976.png', '01795.png', '13805.png', '02760.png', '08646.png', '13665.png', '17230.png', '20059.png', '05574.png', '17791.png', '14640.png', '02093.png', '15567.png', '21291.png', '05860.png', '14291.png', '04929.png', '13749.png', '18482.png', '08564.png', '16137.png', '15377.png', '13147.png', '11551.png', '24591.png', '02439.png', '15120.png', '15818.png', '04438.png', '13230.png', '02578.png', '14969.png', '07695.png', '13677.png', '22425.png', '15430.png', '01138.png', '09869.png', '04187.png', '11801.png', '24507.png', '17436.png', '15925.png', '23241.png', '03985.png', '02173.png', '24333.png', '01427.png', '17414.png', '21865.png', '24410.png', '06975.png', '06097.png', '06924.png', '08341.png', '22841.png', '12650.png', '21830.png', '18461.png', '05773.png', '10462.png', '22048.png', '02465.png', '23844.png', '24341.png', '06472.png', '18256.png', '17155.png', '18683.png', '24325.png', '14525.png', '17256.png', '06331.png', '14141.png', '01681.png', '04805.png', '22723.png', '12188.png', '06881.png', '03791.png', '12322.png', '07964.png', '06791.png', '00868.png', '21981.png', '17079.png', '16235.png', '11052.png', '13546.png', '10386.png', '13878.png', '11815.png', '03710.png', '16168.png', '18058.png', '19151.png', '03004.png', '21697.png', '07294.png', '00401.png', '20080.png', '22718.png', '03606.png', '05225.png', '05433.png', '01949.png', '22402.png', '24720.png', '13916.png', '21482.png', '19454.png', '11428.png', '12159.png', '07303.png', '21677.png', '16282.png', '05389.png', '21409.png', '00677.png', '13663.png', '00620.png', '09507.png', '19639.png', '05883.png', '01512.png', '20642.png', '15397.png', '20588.png', '02894.png', '00892.png', '21339.png', '20480.png', '14118.png', '21937.png', '23060.png', '15743.png', '16782.png', '01917.png', '11824.png', '13480.png', '21593.png', '19568.png', '21312.png', '07559.png', '16081.png', '15002.png', '05621.png', '15226.png', '15324.png', '21021.png', '15745.png', '14240.png', '15787.png', '00288.png', '06749.png', '01386.png', '18241.png', '23236.png', '19306.png', '09685.png', '13870.png', '05985.png', '02569.png', '02294.png', '04424.png', '17262.png', '10612.png', '09437.png', '09419.png', '09047.png', '23525.png', '24353.png', '10397.png', '10792.png', '01095.png', '11374.png', '03258.png', '16165.png', '09980.png', '20724.png', '04203.png', '01456.png', '08977.png', '21066.png', '10101.png', '12975.png', '05975.png', '21709.png', '04109.png', '18818.png', '13626.png', '18346.png', '07128.png', '10863.png', '03879.png', '13116.png', '23527.png', '02345.png', '11204.png', '11361.png', '15478.png', '08018.png', '01678.png', '22432.png', '12193.png', '02015.png', '17385.png', '13519.png', '02557.png', '13391.png', '09381.png', '19828.png', '02033.png', '22134.png', '06936.png', '17690.png', '15269.png', '06690.png', '15848.png', '24186.png', '04501.png', '01001.png', '24478.png', '08323.png', '17297.png', '15899.png', '11730.png', '13523.png', '00534.png', '01766.png', '14052.png', '13548.png', '09889.png', '00634.png', '01100.png', '14390.png', '07989.png', '20637.png', '12870.png', '01839.png', '14201.png', '07091.png', '11975.png', '04269.png', '12831.png', '13310.png', '12022.png', '20881.png', '16613.png', '24819.png', '10449.png', '20332.png', '11521.png', '04845.png', '16463.png', '11837.png', '06406.png', '20052.png', '22321.png', '06410.png', '23893.png', '21261.png', '18792.png', '22267.png', '05924.png', '20078.png', '17521.png', '03371.png', '05935.png', '08836.png', '24407.png', '13875.png', '08426.png', '24054.png', '04866.png', '05739.png', '11721.png', '19231.png', '13606.png', '10362.png', '17925.png', '03807.png', '21306.png', '12858.png', '21726.png', '21559.png', '02738.png', '08191.png', '17666.png', '02109.png', '16122.png', '24945.png', '01554.png', '20993.png', '09581.png', '19972.png', '22351.png', '24852.png', '08511.png', '23392.png', '11670.png', '10709.png', '20094.png', '04057.png', '10973.png', '19699.png', '24040.png', '03771.png', '08372.png', '00444.png', '02524.png', '09098.png', '22283.png', '06912.png', '22209.png', '15123.png', '17494.png', '11186.png', '22433.png', '06229.png', '23619.png', '16259.png', '22588.png', '02790.png', '12916.png', '04479.png', '04995.png', '22877.png', '22907.png', '04392.png', '09881.png', '01616.png', '18848.png', '03352.png', '17760.png', '07007.png', '01611.png', '21115.png', '23811.png', '22706.png', '00917.png', '12312.png', '10474.png', '01948.png', '18815.png', '14663.png', '17179.png', '02410.png', '13298.png', '10662.png', '00895.png', '18006.png', '11191.png', '03841.png', '16692.png', '09066.png', '16597.png', '18711.png', '24329.png', '06771.png', '18491.png', '10565.png', '11712.png', '16914.png', '15749.png', '08122.png', '24171.png', '10393.png', '09837.png', '19547.png', '01721.png', '09115.png', '13807.png', '00094.png', '12465.png', '04686.png', '04319.png', '06861.png', '01240.png', '01085.png', '15695.png', '02840.png', '02523.png', '16169.png', '21998.png', '17771.png', '17519.png', '23506.png', '13493.png', '15373.png', '23523.png', '07110.png', '10694.png', '10112.png', '18998.png', '16939.png', '18788.png', '12326.png', '11373.png', '00665.png', '02744.png', '14639.png', '13172.png', '20142.png', '22725.png', '21774.png', '09149.png', '07433.png', '22732.png', '13098.png', '20527.png', '19716.png', '07840.png', '12244.png', '14469.png', '11780.png', '03552.png', '05750.png', '13799.png', '19289.png', '08380.png', '24749.png', '00260.png', '05273.png', '03501.png', '03837.png', '11448.png', '19349.png', '00982.png', '04218.png', '07064.png', '17187.png', '16919.png', '02540.png', '04588.png', '03202.png', '14324.png', '19478.png', '06823.png', '06905.png', '21039.png', '12394.png', '16599.png', '02646.png', '02154.png', '14107.png', '11271.png', '00961.png', '12479.png', '20645.png', '12428.png', '23219.png', '13974.png', '19300.png', '09230.png', '24561.png', '00497.png', '13706.png', '22222.png', '13005.png', '03427.png', '15130.png', '13788.png', '19257.png', '05011.png', '14827.png', '20906.png', '01282.png', '19135.png', '05076.png', '20840.png', '12936.png', '09091.png', '07413.png', '06932.png', '05212.png', '24822.png', '12559.png', '20203.png', '01747.png', '16959.png', '18607.png', '18377.png', '12366.png', '09630.png', '21721.png', '10578.png', '18022.png', '02949.png', '14684.png', '04858.png', '04118.png', '06699.png', '23125.png', '06822.png', '03964.png', '21943.png', '03800.png', '08376.png', '16696.png', '05692.png', '04844.png', '10731.png', '18547.png', '17490.png', '17498.png', '01723.png', '23911.png', '11310.png', '14468.png', '08049.png', '08051.png', '21255.png', '07652.png', '07216.png', '11673.png', '19418.png', '13344.png', '02801.png', '15318.png', '20584.png', '18943.png', '01305.png', '02637.png', '15793.png', '10577.png', '07511.png', '22736.png', '09817.png', '08157.png', '12032.png', '22011.png', '00329.png', '09505.png', '24726.png', '20931.png', '12614.png', '13364.png', '01267.png', '22424.png', '05526.png', '04321.png', '24188.png', '05899.png', '05590.png', '22462.png', '03717.png', '01810.png', '16392.png', '22816.png', '19836.png', '14227.png', '13925.png', '04819.png', '06602.png', '08731.png', '00506.png', '09135.png', '15508.png', '22333.png', '23841.png', '02673.png', '15163.png', '11711.png', '06394.png', '16327.png', '11262.png', '17421.png', '08671.png', '07728.png', '23000.png', '19819.png', '01411.png', '02861.png', '03003.png', '20025.png', '21264.png', '02351.png', '13671.png', '24459.png', '12792.png', '18174.png', '06469.png', '04925.png', '07596.png', '15905.png', '19245.png', '03943.png', '14387.png', '03006.png', '00829.png', '19958.png', '19749.png', '15320.png', '21821.png', '17898.png', '06745.png', '01344.png', '13455.png', '00791.png', '09839.png', '08609.png', '08468.png', '05069.png', '16008.png', '20776.png', '04549.png', '05764.png', '10559.png', '23904.png', '08700.png', '22027.png', '20327.png', '02761.png', '13615.png', '06071.png', '13716.png', '03141.png', '05535.png', '03287.png', '07725.png', '13798.png', '02510.png', '01070.png', '18445.png', '05414.png', '06814.png', '03022.png', '23678.png', '22513.png', '02455.png', '19771.png', '00085.png', '15333.png', '02331.png', '17216.png', '02819.png', '09795.png', '13336.png', '02316.png', '08287.png', '12521.png', '11021.png', '14790.png', '19942.png', '05620.png', '09358.png', '06181.png', '12639.png', '08792.png', '13399.png', '05062.png', '22198.png', '10373.png', '13520.png', '11062.png', '16736.png', '05527.png', '11197.png', '00879.png', '06450.png', '18564.png', '07482.png', '08525.png', '23581.png', '15863.png', '23550.png', '24267.png', '12407.png', '03237.png', '13027.png', '18816.png', '10306.png', '01465.png', '01838.png', '10963.png', '21702.png', '23065.png', '14704.png', '13774.png', '14953.png', '12932.png', '07034.png', '08686.png', '04070.png', '22308.png', '04305.png', '13342.png', '18847.png', '16368.png', '12171.png', '15354.png', '01499.png', '05987.png', '15209.png', '00439.png', '23135.png', '17357.png', '13291.png', '24342.png', '01595.png', '13445.png', '07305.png', '00676.png', '24773.png', '06412.png', '23289.png', '13449.png', '12192.png', '17006.png', '03770.png', '13824.png', '21405.png', '24154.png', '14538.png', '00461.png', '10240.png', '18380.png', '17400.png', '18891.png', '19181.png', '21762.png', '23040.png', '19178.png', '23797.png', '20291.png', '15239.png', '10999.png', '23836.png', '07723.png', '09336.png', '18951.png', '20113.png', '18906.png', '16208.png', '16991.png', '12736.png', '00706.png', '04481.png', '01665.png', '24845.png', '16270.png', '08264.png', '09166.png', '08301.png', '07215.png', '15645.png', '12438.png', '17739.png', '18239.png', '03061.png', '13730.png', '03137.png', '10467.png', '05873.png', '21919.png', '12499.png', '18469.png', '00509.png', '02197.png', '01979.png', '08215.png', '17738.png', '04107.png', '09366.png', '12933.png', '16352.png', '07379.png', '23445.png', '14375.png', '06259.png', '05068.png', '11567.png', '24825.png', '21286.png', '00333.png', '19834.png', '07161.png', '21532.png', '15255.png', '09508.png', '21090.png', '24763.png', '05513.png', '01898.png', '21904.png', '18026.png', '24441.png', '01475.png', '11474.png', '03147.png', '04136.png', '13345.png', '05232.png', '24052.png', '23558.png', '11138.png', '13920.png', '08854.png', '06474.png', '24388.png', '20373.png', '14505.png', '10622.png', '18742.png', '23193.png', '01983.png', '16340.png', '23220.png', '15260.png', '05610.png', '18934.png', '17258.png', '21384.png', '11370.png', '11526.png', '10191.png', '08467.png', '15704.png', '00494.png', '13938.png', '14672.png', '03154.png', '00824.png', '06659.png', '10309.png', '16003.png', '07085.png', '21199.png', '00650.png', '20363.png', '10554.png', '04753.png', '17196.png', '11190.png', '24307.png', '21304.png', '07189.png', '09257.png', '03537.png', '02910.png', '17194.png', '06570.png', '18615.png', '00937.png', '07854.png', '01658.png', '08999.png', '12183.png', '01272.png', '03847.png', '00647.png', '05476.png', '09004.png', '18755.png', '15103.png', '12801.png', '05560.png', '17683.png', '00779.png', '16935.png', '19048.png', '18327.png', '11408.png', '22080.png', '14801.png', '17772.png', '06028.png', '04403.png', '10522.png', '13323.png', '09044.png', '06077.png', '07837.png', '03818.png', '04382.png', '00466.png', '09569.png', '14723.png', '00077.png', '15495.png', '00307.png', '20042.png', '05260.png', '18259.png', '02427.png', '03178.png', '21563.png', '19575.png', '07547.png', '22930.png', '02662.png', '24163.png', '03103.png', '04871.png', '20400.png', '06906.png', '16549.png', '06680.png', '03773.png', '10126.png', '07650.png', '09180.png', '03331.png', '01239.png', '18333.png', '19668.png', '11456.png', '10894.png', '00727.png', '00403.png', '02971.png', '19397.png', '17121.png', '18627.png', '13705.png', '02719.png', '19488.png', '12749.png', '05013.png', '11565.png', '02131.png', '03436.png', '01329.png', '23555.png', '23173.png', '13564.png', '22157.png', '10520.png', '16888.png', '10444.png', '05064.png', '08757.png', '04335.png', '15523.png', '08838.png', '16273.png', '07288.png', '22734.png', '19375.png', '09065.png', '11282.png', '17232.png', '03521.png', '03549.png', '03992.png', '10435.png', '20303.png', '03324.png', '17293.png', '02431.png', '03223.png', '14042.png', '11228.png', '24570.png', '17780.png', '17962.png', '20260.png', '08902.png', '19551.png', '17916.png', '01283.png', '18843.png', '04945.png', '07601.png', '12021.png', '18475.png', '06465.png', '19926.png', '00938.png', '01103.png', '09299.png', '07270.png', '13865.png', '07103.png', '17759.png', '05429.png', '02381.png', '06249.png', '19428.png', '21510.png', '05923.png', '12919.png', '10514.png', '19310.png', '06449.png', '14397.png', '03579.png', '08187.png', '18168.png', '21465.png', '09807.png', '10654.png', '16964.png', '10361.png', '23917.png', '08388.png', '08811.png', '05060.png', '22445.png', '22874.png', '07584.png', '20438.png', '14043.png', '12472.png', '11758.png', '24243.png', '00603.png', '01516.png', '23975.png', '20653.png', '12121.png', '04499.png', '22666.png', '11124.png', '03857.png', '04724.png', '10013.png', '16735.png', '22773.png', '00084.png', '08473.png', '07380.png', '09679.png', '19946.png', '17658.png', '06698.png', '16596.png', '19742.png', '18219.png', '24520.png', '16075.png', '13896.png', '17747.png', '02021.png', '06329.png', '21692.png', '23639.png', '23153.png', '07084.png', '11796.png', '21135.png', '05014.png', '08925.png', '11686.png', '14577.png', '10312.png', '24674.png', '21846.png', '15092.png', '04283.png', '03344.png', '11376.png', '16366.png', '10033.png', '01450.png', '12760.png', '00325.png', '18161.png', '18226.png', '22845.png', '19865.png', '11627.png', '02002.png', '08210.png', '12412.png', '11141.png', '21065.png', '08640.png', '14263.png', '00139.png', '06547.png', '00024.png', '15429.png', '16203.png', '07322.png', '20736.png', '14721.png', '00426.png', '19446.png', '16424.png', '15058.png', '20243.png', '01638.png', '09452.png', '05902.png', '10353.png', '02390.png', '24774.png', '04009.png', '24599.png', '16330.png', '08061.png', '03037.png', '03715.png', '23147.png', '12367.png', '18493.png', '17198.png', '02247.png', '20859.png', '04243.png', '23449.png', '01197.png', '11869.png', '14296.png', '15593.png', '06875.png', '14685.png', '16010.png', '08419.png', '09736.png', '10958.png', '17911.png', '23052.png', '23104.png', '14527.png', '07253.png', '22291.png', '22550.png', '00278.png', '23283.png', '23293.png', '10006.png', '03520.png', '14771.png', '21727.png', '01365.png', '07494.png', '22981.png', '03283.png', '21940.png', '08872.png', '16172.png', '21176.png', '01862.png', '14713.png', '23067.png', '13727.png', '15194.png', '13535.png', '04531.png', '23846.png', '11544.png', '20783.png', '18130.png', '01661.png', '23379.png', '04515.png', '02779.png', '09726.png', '22419.png', '24222.png', '05365.png', '23306.png', '03763.png', '19707.png', '09314.png', '07406.png', '16151.png', '23186.png', '02773.png', '08254.png', '03626.png', '16585.png', '05443.png', '17118.png', '00743.png', '23384.png', '13917.png', '23296.png', '02274.png', '15290.png', '10438.png', '16970.png', '14603.png', '17614.png', '00478.png', '08311.png', '19869.png', '23049.png', '09133.png', '23688.png', '11309.png', '06283.png', '12331.png', '01321.png', '16031.png', '16822.png', '16374.png', '09638.png', '03741.png', '09027.png', '09406.png', '23495.png', '24639.png', '04699.png', '13543.png', '09361.png', '10839.png', '12139.png', '17795.png', '15266.png', '15644.png', '11520.png', '24367.png', '08913.png', '09060.png', '13744.png', '11024.png', '23391.png', '10204.png', '17223.png', '15972.png', '04815.png', '08724.png', '11775.png', '15190.png', '16249.png', '23074.png', '03654.png', '08954.png', '24755.png', '02087.png', '15256.png', '01718.png', '01184.png', '18885.png', '12059.png', '17893.png', '00741.png', '04621.png', '24481.png', '21688.png', '07384.png', '18959.png', '14157.png', '01819.png', '03021.png', '22932.png', '06672.png', '17644.png', '23627.png', '07100.png', '17950.png', '11765.png', '15128.png', '17282.png', '10592.png', '10770.png', '19627.png', '10180.png', '01986.png', '08221.png', '22795.png', '14120.png', '06230.png', '05242.png', '07138.png', '04144.png', '04183.png', '04301.png', '01506.png', '08552.png', '05269.png', '15217.png', '14275.png', '24077.png', '11896.png', '02767.png', '07861.png', '15687.png', '18128.png', '04371.png', '23693.png', '17288.png', '12164.png', '17333.png', '00166.png', '16148.png', '06045.png', '11152.png', '15298.png', '06876.png', '19511.png', '11149.png', '22770.png', '11554.png', '04870.png', '19919.png', '19377.png', '03916.png', '15730.png', '09563.png', '02781.png', '13032.png', '20875.png', '20576.png', '04755.png', '01945.png', '15576.png', '22111.png', '01039.png', '04965.png', '20918.png', '13742.png', '17202.png', '14003.png', '24776.png', '18352.png', '03812.png', '14194.png', '23593.png', '21747.png', '05666.png', '13159.png', '11414.png', '18261.png', '06946.png', '06647.png', '03406.png', '14383.png', '02942.png', '03170.png', '16433.png', '09900.png', '13199.png', '01720.png', '04489.png', '17967.png', '13893.png', '16072.png', '23648.png', '11959.png', '02724.png', '11058.png', '22242.png', '07309.png', '19450.png', '13790.png', '13787.png', '06020.png', '06360.png', '20368.png', '23570.png', '01724.png', '22489.png', '06036.png', '10154.png', '14481.png', '22835.png', '07401.png', '06030.png', '03312.png', '17831.png', '11198.png', '02115.png', '17520.png', '03257.png', '05888.png', '18722.png', '01822.png', '22078.png', '19172.png', '03686.png', '20810.png', '15322.png', '09689.png', '13701.png', '04280.png', '18968.png', '11265.png', '15927.png', '13957.png', '23400.png', '12235.png', '11875.png', '16357.png', '01985.png', '01739.png', '09142.png', '24469.png', '18342.png', '23432.png', '01714.png', '07471.png', '01629.png', '24064.png', '00445.png', '12673.png', '09244.png', '15134.png', '20845.png', '22052.png', '00075.png', '05919.png', '14113.png', '18786.png', '03567.png', '17928.png', '12829.png', '10179.png', '23007.png', '20173.png', '14119.png', '05920.png', '05882.png', '00191.png', '01591.png', '08783.png', '20256.png', '15562.png', '15040.png', '03914.png', '19229.png', '02494.png', '11367.png', '06655.png', '11467.png', '08102.png', '18822.png', '23362.png', '24552.png', '15445.png', '17963.png', '15646.png', '08165.png', '17056.png', '06783.png', '17775.png', '14940.png', '16958.png', '21024.png', '04685.png', '24491.png', '05094.png', '10050.png', '06815.png', '07930.png', '08149.png', '19480.png', '23281.png', '04530.png', '21105.png', '22552.png', '09931.png', '13430.png', '18127.png', '18749.png', '02632.png', '20258.png', '24110.png', '05984.png', '04598.png', '10962.png', '05966.png', '08549.png', '24033.png', '06391.png', '09009.png', '06148.png', '18389.png', '08831.png', '08250.png', '19429.png', '02123.png', '05465.png', '21750.png', '21341.png', '08738.png', '13545.png', '19695.png', '20735.png', '14006.png', '22196.png', '07076.png', '24809.png', '11968.png', '02214.png', '10092.png', '20273.png', '01861.png', '02871.png', '12654.png', '11642.png', '02680.png', '10966.png', '23093.png', '02491.png', '24923.png', '04246.png', '23314.png', '24607.png', '05352.png', '04680.png', '17725.png', '06059.png', '07357.png', '00557.png', '19110.png', '14045.png', '17854.png', '19482.png', '17028.png', '23910.png']
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
        
        if i_iter == 291451:
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
