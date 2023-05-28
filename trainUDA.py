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
    a = ['07280.png', '20068.png', '06542.png', '11158.png', '18814.png', '03738.png', '05330.png', '19730.png', '01776.png', '17721.png', '14726.png', '04575.png', '18997.png', '09097.png', '17462.png', '21889.png', '20196.png', '02651.png', '05664.png', '09403.png', '20128.png', '02237.png', '22807.png', '24720.png', '23961.png', '09095.png', '08025.png', '20946.png', '03389.png', '06550.png', '06945.png', '10018.png', '15441.png', '09656.png', '06352.png', '00435.png', '14072.png', '06650.png', '14343.png', '13063.png', '09080.png', '24845.png', '15773.png', '13162.png', '22559.png', '02735.png', '21435.png', '02632.png', '10034.png', '15704.png', '19869.png', '05688.png', '03032.png', '02781.png', '16628.png', '02481.png', '10326.png', '19756.png', '18976.png', '09003.png', '17452.png', '04608.png', '24474.png', '20951.png', '21743.png', '16772.png', '16443.png', '12395.png', '21756.png', '07494.png', '05677.png', '11480.png', '00662.png', '15622.png', '07010.png', '09812.png', '18904.png', '17871.png', '04885.png', '04421.png', '15462.png', '06810.png', '01366.png', '11370.png', '21565.png', '06777.png', '05016.png', '24404.png', '20454.png', '03233.png', '12974.png', '05942.png', '18694.png', '20667.png', '05360.png', '24428.png', '10788.png', '15277.png', '08289.png', '10948.png', '12799.png', '19837.png', '24486.png', '22009.png', '15292.png', '03067.png', '01662.png', '12309.png', '24597.png', '19886.png', '21874.png', '09905.png', '06879.png', '08811.png', '03559.png', '22104.png', '23428.png', '21997.png', '04114.png', '19670.png', '24734.png', '16869.png', '08812.png', '04987.png', '19308.png', '22853.png', '15364.png', '13203.png', '05254.png', '08001.png', '13091.png', '06473.png', '03222.png', '20646.png', '09132.png', '11294.png', '07729.png', '14010.png', '20437.png', '21628.png', '19350.png', '01954.png', '17309.png', '14848.png', '16441.png', '18395.png', '17165.png', '02031.png', '12212.png', '04810.png', '13226.png', '12607.png', '01774.png', '21518.png', '15293.png', '17401.png', '00303.png', '09201.png', '19956.png', '03154.png', '16298.png', '06968.png', '20175.png', '03880.png', '17645.png', '05930.png', '21919.png', '17736.png', '12733.png', '02727.png', '10177.png', '22818.png', '08367.png', '08281.png', '20771.png', '23671.png', '12106.png', '04032.png', '18405.png', '06335.png', '03953.png', '24776.png', '08906.png', '16195.png', '01167.png', '12729.png', '16166.png', '06187.png', '16777.png', '14485.png', '04784.png', '17224.png', '12992.png', '16756.png', '19384.png', '14876.png', '07447.png', '24559.png', '08657.png', '20551.png', '19504.png', '16365.png', '24142.png', '18812.png', '02167.png', '04581.png', '01713.png', '14662.png', '24315.png', '15987.png', '13384.png', '16923.png', '00982.png', '00401.png', '04656.png', '08055.png', '02809.png', '17596.png', '02429.png', '06256.png', '04435.png', '00232.png', '22530.png', '11235.png', '06045.png', '12581.png', '20384.png', '04153.png', '00362.png', '19860.png', '12665.png', '05880.png', '09525.png', '00964.png', '21491.png', '20002.png', '17269.png', '23482.png', '24031.png', '14331.png', '02841.png', '19766.png', '01480.png', '12522.png', '22704.png', '14506.png', '03799.png', '16367.png', '08668.png', '11423.png', '03243.png', '09941.png', '11463.png', '20295.png', '13103.png', '07628.png', '15114.png', '04993.png', '13722.png', '09482.png', '23660.png', '20364.png', '01091.png', '09376.png', '00665.png', '05693.png', '20046.png', '13361.png', '24787.png', '24328.png', '22149.png', '06222.png', '15333.png', '10185.png', '07117.png', '08654.png', '11375.png', '16516.png', '04147.png', '20223.png', '24206.png', '16204.png', '16733.png', '04986.png', '02673.png', '23642.png', '10061.png', '22796.png', '23873.png', '12993.png', '02699.png', '08874.png', '22947.png', '04693.png', '19925.png', '09044.png', '14379.png', '16491.png', '23016.png', '21839.png', '06480.png', '07839.png', '12103.png', '09824.png', '18758.png', '21290.png', '06747.png', '18105.png', '00432.png', '01911.png', '19530.png', '10119.png', '21655.png', '06254.png', '07459.png', '10149.png', '00426.png', '21800.png', '10922.png', '13101.png', '00352.png', '01509.png', '21663.png', '21911.png', '21097.png', '21787.png', '20997.png', '02295.png', '08567.png', '18555.png', '20189.png', '17559.png', '24737.png', '08062.png', '19535.png', '07100.png', '10674.png', '16319.png', '01617.png', '22161.png', '15717.png', '04962.png', '14430.png', '18858.png', '09369.png', '19765.png', '24872.png', '08010.png', '13762.png', '21472.png', '21199.png', '11548.png', '03481.png', '14383.png', '08931.png', '13837.png', '12568.png', '18375.png', '12021.png', '17551.png', '09434.png', '20881.png', '20576.png', '02996.png', '11672.png', '11523.png', '15150.png', '24219.png', '18511.png', '01063.png', '24423.png', '17861.png', '21489.png', '04453.png', '10890.png', '10951.png', '10972.png', '00976.png', '20766.png', '12653.png', '17201.png', '04027.png', '08881.png', '16137.png', '21623.png', '10474.png', '14280.png', '04903.png', '15501.png', '21305.png', '18475.png', '16672.png', '18095.png', '01052.png', '07262.png', '06372.png', '18443.png', '19966.png', '14531.png', '17393.png', '19413.png', '07732.png', '16501.png', '04593.png', '09140.png', '22265.png', '13180.png', '02458.png', '09808.png', '07242.png', '24746.png', '17443.png', '24774.png', '15164.png', '20387.png', '03624.png', '14319.png', '17405.png', '22457.png', '22948.png', '24729.png', '16315.png', '08251.png', '01359.png', '13287.png', '20423.png', '18135.png', '18894.png', '17757.png', '24036.png', '19841.png', '12141.png', '05355.png', '11980.png', '15721.png', '04181.png', '06932.png', '24843.png', '15871.png', '12600.png', '18720.png', '20231.png', '03920.png', '20445.png', '06181.png', '14930.png', '06126.png', '13326.png', '04388.png', '12828.png', '24850.png', '17607.png', '24021.png', '19770.png', '18743.png', '01096.png', '15606.png', '15363.png', '23437.png', '20837.png', '08322.png', '21002.png', '03580.png', '24663.png', '01659.png', '14969.png', '10896.png', '07345.png', '03691.png', '04420.png', '11440.png', '12847.png', '11575.png', '02530.png', '24003.png', '13789.png', '14360.png', '22845.png', '18073.png', '07996.png', '07700.png', '01883.png', '15750.png', '04552.png', '07718.png', '21616.png', '01511.png', '04188.png', '11379.png', '06129.png', '05821.png', '02564.png', '04526.png', '24214.png', '13346.png', '06264.png', '22537.png', '01928.png', '19554.png', '14533.png', '22560.png', '12061.png', '22544.png', '16996.png', '22216.png', '07468.png', '18376.png', '22527.png', '12908.png', '15065.png', '12367.png', '19199.png', '01834.png', '08012.png', '12826.png', '08637.png', '01592.png', '09459.png', '18546.png', '01333.png', '06897.png', '19008.png', '17667.png', '09552.png', '23103.png', '14819.png', '04940.png', '09943.png', '10533.png', '03791.png', '00421.png', '10453.png', '15792.png', '23917.png', '02958.png', '20704.png', '15074.png', '12796.png', '05496.png', '19996.png', '17148.png', '09855.png', '14256.png', '15594.png', '24691.png', '12310.png', '20885.png', '21166.png', '24916.png', '07686.png', '17799.png', '19634.png', '21942.png', '18853.png', '07028.png', '14519.png', '01763.png', '15696.png', '02588.png', '08624.png', '16434.png', '10720.png', '18045.png', '06737.png', '04999.png', '20409.png', '05278.png', '06639.png', '11013.png', '11450.png', '04585.png', '00061.png', '09982.png', '21571.png', '07688.png', '09884.png', '24778.png', '07382.png', '00358.png', '20563.png', '24797.png', '01440.png', '12788.png', '04109.png', '05362.png', '05294.png', '14605.png', '09127.png', '10675.png', '04732.png', '13031.png', '10348.png', '09642.png', '17310.png', '08973.png', '24166.png', '01084.png', '13054.png', '23989.png', '00539.png', '13966.png', '00057.png', '05961.png', '06031.png', '18218.png', '07338.png', '23764.png', '23440.png', '21449.png', '00628.png', '16610.png', '06727.png', '08895.png', '09990.png', '03261.png', '20962.png', '17976.png', '02869.png', '10988.png', '17579.png', '03535.png', '23479.png', '02446.png', '24891.png', '22037.png', '08412.png', '18078.png', '01368.png', '14814.png', '21490.png', '11640.png', '12096.png', '20283.png', '06744.png', '07093.png', '20473.png', '21011.png', '15218.png', '03269.png', '19072.png', '01952.png', '06851.png', '20369.png', '15519.png', '11795.png', '14764.png', '23520.png', '01346.png', '19798.png', '21614.png', '02731.png', '10512.png', '05149.png', '17652.png', '19847.png', '19683.png', '07430.png', '22368.png', '03167.png', '16361.png', '16377.png', '16715.png', '18092.png', '24366.png', '16403.png', '08791.png', '05383.png', '18626.png', '04238.png', '09682.png', '08599.png', '21223.png', '08792.png', '23564.png', '05082.png', '24441.png', '21741.png', '03998.png', '15541.png', '12970.png', '05857.png', '07916.png', '22959.png', '15908.png', '06563.png', '00356.png', '11304.png', '07804.png', '11303.png', '23511.png', '07927.png', '12439.png', '11399.png', '18237.png', '14262.png', '10997.png', '16508.png', '10127.png', '15217.png', '21025.png', '03267.png', '15532.png', '22581.png', '02214.png', '13840.png', '03984.png', '00568.png', '10503.png', '07742.png', '14786.png', '04456.png', '11727.png', '22369.png', '06091.png', '12052.png', '07443.png', '13804.png', '00148.png', '22003.png', '11104.png', '04630.png', '13742.png', '04276.png', '15048.png', '24753.png', '08666.png', '17352.png', '06291.png', '05081.png', '11486.png', '10057.png', '02155.png', '15520.png', '09767.png', '19212.png', '06462.png', '08759.png', '19378.png', '13577.png', '13172.png', '22184.png', '14660.png', '11731.png', '08560.png', '21032.png', '19258.png', '07423.png', '04120.png', '07613.png', '19401.png', '07819.png', '06580.png', '22239.png', '20810.png', '16632.png', '16433.png', '11983.png', '13985.png', '00810.png', '15806.png', '21347.png', '02140.png', '08262.png', '20614.png', '22851.png', '08676.png', '19294.png', '21103.png', '13543.png', '14525.png', '16462.png', '08761.png', '04559.png', '16117.png', '05542.png', '19293.png', '07115.png', '08662.png', '10684.png', '23467.png', '16457.png', '24590.png', '16027.png', '19533.png', '15459.png', '17020.png', '12798.png', '04243.png', '06128.png', '05172.png', '10108.png', '23251.png', '20117.png', '19380.png', '14054.png', '15613.png', '13119.png', '19472.png', '02883.png', '14721.png', '17026.png', '24372.png', '10056.png', '20612.png', '23561.png', '20599.png', '06194.png', '16896.png', '01089.png', '22059.png', '08502.png', '19734.png', '09280.png', '22211.png', '10877.png', '19701.png', '00180.png', '22005.png', '04651.png', '14511.png', '08147.png', '01619.png', '06688.png', '24731.png', '08003.png', '12726.png', '07725.png', '01560.png', '09439.png', '09860.png', '16145.png', '17246.png', '24728.png', '07603.png', '16799.png', '14374.png', '12209.png', '19738.png', '12885.png', '19709.png', '21537.png', '21228.png', '11314.png', '21658.png', '06350.png', '12425.png', '14756.png', '20880.png', '11714.png', '19579.png', '22210.png', '24043.png', '22428.png', '18540.png', '19164.png', '03438.png', '02100.png', '21812.png', '13232.png', '06284.png', '19280.png', '17949.png', '14878.png', '23285.png', '02939.png', '22344.png', '16395.png', '04641.png', '07187.png', '19410.png', '23570.png', '09361.png', '21156.png', '08370.png', '20329.png', '18305.png', '23219.png', '16612.png', '23326.png', '22639.png', '14708.png', '20060.png', '09485.png', '07286.png', '08282.png', '16566.png', '16941.png', '01701.png', '20665.png', '00858.png', '16326.png', '23855.png', '01429.png', '07584.png', '04213.png', '08948.png', '03466.png', '20065.png', '16664.png', '07598.png', '17507.png', '04471.png', '09477.png', '14890.png', '18699.png', '24782.png', '15335.png', '17034.png', '03983.png', '10140.png', '16499.png', '08580.png', '12918.png', '06235.png', '08644.png', '00695.png', '21806.png', '15813.png', '19505.png', '18827.png', '05777.png', '04138.png', '10167.png', '11802.png', '08993.png', '17347.png', '24932.png', '16049.png', '08103.png', '12774.png', '03870.png', '09954.png', '04997.png', '21313.png', '02467.png', '00411.png', '10181.png', '21125.png', '22624.png', '01026.png', '17360.png', '18941.png', '21488.png', '05459.png', '21646.png', '02149.png', '22252.png', '12093.png', '16153.png', '12538.png', '09322.png', '07601.png', '09733.png', '24392.png', '10486.png', '22743.png', '23058.png', '12083.png', '01626.png', '17368.png', '22744.png', '11478.png', '00600.png', '01487.png', '20274.png', '06013.png', '18323.png', '21154.png', '22705.png', '12824.png', '19459.png', '21779.png', '16150.png', '20091.png', '01081.png', '08403.png', '11071.png', '03676.png', '18225.png', '04633.png', '03272.png', '20236.png', '19465.png', '10961.png', '23859.png', '02826.png', '06077.png', '16263.png', '07722.png', '15658.png', '24669.png', '03495.png', '07222.png', '11779.png', '16436.png', '09748.png', '17437.png', '01764.png', '00969.png', '09563.png', '12257.png', '19296.png', '15958.png', '22194.png', '14441.png', '23199.png', '02965.png', '00815.png', '13675.png', '13802.png', '22765.png', '15456.png', '08162.png', '13102.png', '05988.png', '23301.png', '00218.png', '21843.png', '21683.png', '11442.png', '04672.png', '11813.png', '00905.png', '04855.png', '05407.png', '10247.png', '01397.png', '20849.png', '11835.png', '02022.png', '18535.png', '24665.png', '21940.png', '23599.png', '11774.png', '04555.png', '14140.png', '18805.png', '14111.png', '18279.png', '22067.png', '11520.png', '01868.png', '14353.png', '16341.png', '12357.png', '07350.png', '04715.png', '14861.png', '07935.png', '14836.png', '19027.png', '00366.png', '21418.png', '22703.png', '06124.png', '02670.png', '17773.png', '04330.png', '01709.png', '24250.png', '06796.png', '11796.png', '02260.png', '03886.png', '20141.png', '22545.png', '13266.png', '05948.png', '15714.png', '11690.png', '02704.png', '22730.png', '11315.png', '14480.png', '22076.png', '22306.png', '02209.png', '11823.png', '00466.png', '08131.png', '12450.png', '10356.png', '18495.png', '23031.png', '23185.png', '12088.png', '05161.png', '11301.png', '07457.png', '23180.png', '09381.png', '14394.png', '01846.png', '01890.png', '01577.png', '24755.png', '21930.png', '22866.png', '19901.png', '06379.png', '20158.png', '09666.png', '05897.png', '00871.png', '18583.png', '19911.png', '16404.png', '03084.png', '01214.png', '22607.png', '23348.png', '21434.png', '01046.png', '00908.png', '18828.png', '10968.png', '03263.png', '01799.png', '09031.png', '11669.png', '18854.png', '02019.png', '16807.png', '19808.png', '14868.png', '24286.png', '00429.png', '17989.png', '18617.png', '02086.png', '24458.png', '15903.png', '19238.png', '15537.png', '01815.png', '00186.png', '17678.png', '19909.png', '23883.png', '00993.png', '14025.png', '00887.png', '17642.png', '09588.png', '19032.png', '15136.png', '02202.png', '04310.png', '08301.png', '21902.png', '21819.png', '17758.png', '16981.png', '06552.png', '08044.png', '15160.png', '03297.png', '18441.png', '03068.png', '07971.png', '01323.png', '23433.png', '18489.png', '23761.png', '22422.png', '05990.png', '04366.png', '24323.png', '07902.png', '09745.png', '09580.png', '01779.png', '10253.png', '03928.png', '02586.png', '16581.png', '08799.png', '05144.png', '15852.png', '16207.png', '10152.png', '14772.png', '08595.png', '02878.png', '24358.png', '01062.png', '04643.png', '12482.png', '15099.png', '11288.png', '05888.png', '08332.png', '18903.png', '19806.png', '10182.png', '14779.png', '21736.png', '24146.png', '21477.png', '15453.png', '18040.png', '03364.png', '12156.png', '23225.png', '03121.png', '07086.png', '21867.png', '07228.png', '09847.png', '01623.png', '03344.png', '10270.png', '20168.png', '01113.png', '06239.png', '20444.png', '24187.png', '13720.png', '19024.png', '19325.png', '15328.png', '07649.png', '13774.png', '00641.png', '02662.png', '17532.png', '03871.png', '13860.png', '15879.png', '08300.png', '00461.png', '09274.png', '08647.png', '18129.png', '22528.png', '16614.png', '24818.png', '03005.png', '07809.png', '19742.png', '06208.png', '19955.png', '16218.png', '11439.png', '11397.png', '18017.png', '16179.png', '12475.png', '11529.png', '21333.png', '00669.png', '11272.png', '22892.png', '20337.png', '15035.png', '03316.png', '01075.png', '00369.png', '13097.png', '00562.png', '08698.png', '21276.png', '05985.png', '16648.png', '15604.png', '08114.png', '13660.png', '13285.png', '17155.png', '14877.png', '04773.png', '00310.png', '10523.png', '15018.png', '12354.png', '23236.png', '06710.png', '21042.png', '23665.png', '22004.png', '10244.png', '18037.png', '07032.png', '01963.png', '09803.png', '00545.png', '21714.png', '05619.png', '02309.png', '03528.png', '14030.png', '15126.png', '21452.png', '23825.png', '00055.png', '08840.png', '13325.png', '12631.png', '20911.png', '04898.png', '07614.png', '02190.png', '16883.png', '15556.png', '20775.png', '20282.png', '10391.png', '15209.png', '01955.png', '11732.png', '09946.png', '08586.png', '13406.png', '24496.png', '22480.png', '11028.png', '08197.png', '07689.png', '13452.png', '24693.png', '14036.png', '20221.png', '05304.png', '03606.png', '17084.png', '03836.png', '06817.png', '03853.png', '15897.png', '10398.png', '24303.png', '00717.png', '15511.png', '10048.png', '15648.png', '01277.png', '16703.png', '24923.png', '09015.png', '03221.png', '13175.png', '20503.png', '06413.png', '17963.png', '23168.png', '24495.png', '16837.png', '08704.png', '00290.png', '23111.png', '11730.png', '13580.png', '20673.png', '01808.png', '02967.png', '12387.png', '14909.png', '09420.png', '15433.png', '01234.png', '23141.png', '06363.png', '08454.png', '17867.png', '07467.png', '15159.png', '09471.png', '19091.png', '13659.png', '18284.png', '16493.png', '23075.png', '20756.png', '11619.png', '07678.png', '03981.png', '24069.png', '20411.png', '03422.png', '13093.png', '11430.png', '21769.png', '06280.png', '02486.png', '22762.png', '20929.png', '05275.png', '19781.png', '14264.png', '03555.png', '11750.png', '23866.png', '19427.png', '20433.png', '16814.png', '07006.png', '20385.png', '14776.png', '19374.png', '18161.png', '06378.png', '22172.png', '20098.png', '22993.png', '02430.png', '07780.png', '03461.png', '09643.png', '07957.png', '18115.png', '05558.png', '19868.png', '20624.png', '00629.png', '09452.png', '00080.png', '22812.png', '03909.png', '11100.png', '10125.png', '03226.png', '19172.png', '15666.png', '19526.png', '15383.png', '16780.png', '23257.png', '19848.png', '23035.png', '11762.png', '12613.png', '01913.png', '09181.png', '11053.png', '21664.png', '24147.png', '03315.png', '01119.png', '01147.png', '20630.png', '24472.png', '01375.png', '21897.png', '13434.png', '18260.png', '09107.png', '15891.png', '18544.png', '13237.png', '06012.png', '00704.png', '24117.png', '01925.png', '15357.png', '08306.png', '00952.png', '18060.png', '05412.png', '00319.png', '01370.png', '15085.png', '02650.png', '12589.png', '16405.png', '23863.png', '10794.png', '22223.png', '01369.png', '05577.png', '10885.png', '17624.png', '12343.png', '06015.png', '04178.png', '03660.png', '17865.png', '17804.png', '14705.png', '08631.png', '07797.png', '13457.png', '23656.png', '04896.png', '05466.png', '13863.png', '17553.png', '24256.png', '24708.png', '01573.png', '09318.png', '05998.png', '18567.png', '14309.png', '11860.png', '17495.png', '09772.png', '19466.png', '21815.png', '01090.png', '17497.png', '00881.png', '22408.png', '13378.png', '08530.png', '13958.png', '21494.png', '21391.png', '05079.png', '12821.png', '03302.png', '21569.png', '04852.png', '04941.png', '13245.png', '09278.png', '15817.png', '00458.png', '02741.png', '05151.png', '04560.png', '05651.png', '24903.png', '06186.png', '07759.png', '18296.png', '16904.png', '20711.png', '06972.png', '00553.png', '24830.png', '05614.png', '11720.png', '22313.png', '01187.png', '18817.png', '16631.png', '21164.png', '06853.png', '13535.png', '20123.png', '18345.png', '22017.png', '07907.png', '23538.png', '02482.png', '02098.png', '10526.png', '14493.png', '12110.png', '24485.png', '24515.png', '00693.png', '03062.png', '18959.png', '11237.png', '03472.png', '17377.png', '12794.png', '24686.png', '21436.png', '16565.png', '08183.png', '16899.png', '03743.png', '06156.png', '24054.png', '16280.png', '12239.png', '17182.png', '14916.png', '20169.png', '08374.png', '15780.png', '06184.png', '04880.png', '20611.png', '11507.png', '07123.png', '02366.png', '00824.png', '15597.png', '12520.png', '23941.png', '10975.png', '13760.png', '13779.png', '07380.png', '02947.png', '12317.png', '11782.png', '20699.png', '19442.png', '21049.png', '18044.png', '00734.png', '22663.png', '24804.png', '07933.png', '03709.png', '08783.png', '12149.png', '12504.png', '16425.png', '03776.png', '13809.png', '07421.png', '01958.png', '19375.png', '03166.png', '09639.png', '19502.png', '18440.png', '09034.png', '19801.png', '20050.png', '01789.png', '05738.png', '07212.png', '10613.png', '06164.png', '14950.png', '06935.png', '08394.png', '23439.png', '10601.png', '21420.png', '17073.png', '00652.png', '06735.png', '12305.png', '15919.png', '18521.png', '09522.png', '09729.png', '12809.png', '21927.png', '17498.png', '23210.png', '14710.png', '23888.png', '08561.png', '02510.png', '24135.png', '10199.png', '20827.png', '07343.png', '14853.png', '15422.png', '16749.png', '10898.png', '08528.png', '20939.png', '00592.png', '20637.png', '08020.png', '10970.png', '02747.png', '05951.png', '22351.png', '19340.png', '21048.png', '21667.png', '17979.png', '21334.png', '15097.png', '00703.png', '03548.png', '14495.png', '19883.png', '07898.png', '05999.png', '17392.png', '18896.png', '21990.png', '00647.png', '20583.png', '13098.png', '19928.png', '07221.png', '24909.png', '02442.png', '04539.png', '09970.png', '08675.png', '10665.png', '02130.png', '03711.png', '01350.png', '20479.png', '07552.png', '22283.png', '17550.png', '21742.png', '03936.png', '22534.png', '04800.png', '12388.png', '24650.png', '03285.png', '08212.png', '02998.png', '02976.png', '02640.png', '14921.png', '21087.png', '03299.png', '14482.png', '08847.png', '19431.png', '16005.png', '02284.png', '22939.png', '06692.png', '11595.png', '24225.png', '01535.png', '24551.png', '14002.png', '09469.png', '02316.png', '09430.png', '21078.png', '17611.png', '16119.png', '20678.png', '09030.png', '00853.png', '23829.png', '19510.png', '13258.png', '02736.png', '10305.png', '19646.png', '23052.png', '01417.png', '17942.png', '01178.png', '13850.png', '09891.png', '11580.png', '13283.png', '10306.png', '20059.png', '21852.png', '15638.png', '10089.png', '13248.png', '05769.png', '04037.png', '04317.png', '17101.png', '11606.png', '09904.png', '10176.png', '17051.png', '04700.png', '14159.png', '00454.png', '00368.png', '17719.png', '16652.png', '01433.png', '07735.png', '17221.png', '11828.png', '01693.png', '08325.png', '04536.png', '11902.png', '23379.png', '13367.png', '05764.png', '05421.png', '01337.png', '03059.png', '15510.png', '09342.png', '07177.png', '00559.png', '05899.png', '03403.png', '24104.png', '04201.png', '00598.png', '20862.png', '13685.png', '00199.png', '15400.png', '07768.png', '03617.png', '14737.png', '22093.png', '14580.png', '18868.png', '04600.png', '00471.png', '15765.png', '00992.png', '08949.png', '07658.png', '13395.png', '10831.png', '23393.png', '05773.png', '19614.png', '11236.png', '17523.png', '23492.png', '13982.png', '03508.png', '07767.png', '24889.png', '19616.png', '10852.png', '03168.png', '19021.png', '12592.png', '09806.png', '16698.png', '15202.png', '02712.png', '13991.png', '16986.png', '14994.png', '09899.png', '06832.png', '07229.png', '24118.png', '11907.png', '12591.png', '01881.png', '12112.png', '00586.png', '15628.png', '03361.png', '04084.png', '09011.png', '16539.png', '17149.png', '08466.png', '06694.png', '01538.png', '06118.png', '18169.png', '22460.png', '18168.png', '03036.png', '19188.png', '06924.png', '13952.png', '20292.png', '05935.png', '00807.png', '10823.png', '00132.png', '23169.png', '16369.png', '12644.png', '04040.png', '02984.png', '22546.png', '21119.png', '00193.png', '06823.png', '18741.png', '24959.png', '18024.png', '11888.png', '08794.png', '11305.png', '22761.png', '22585.png', '22490.png', '01902.png', '17212.png', '12841.png', '21410.png', '02053.png', '11179.png', '08334.png', '24807.png', '03456.png', '18403.png', '15956.png', '08736.png', '01185.png', '05929.png', '08152.png', '11475.png', '05458.png', '04665.png', '07592.png', '00088.png', '00755.png', '05599.png', '21253.png', '12231.png', '21619.png', '06439.png', '00736.png', '21203.png', '01914.png', '13132.png', '18315.png', '08669.png', '05276.png', '23135.png', '05579.png', '08382.png', '03672.png', '23794.png', '04953.png', '07079.png', '20323.png', '18028.png', '00599.png', '01863.png', '14112.png', '07136.png', '20782.png', '00453.png', '11697.png', '17122.png', '17482.png', '10871.png', '20392.png', '17906.png', '16685.png', '20961.png', '20826.png', '15701.png', '20759.png', '06466.png', '13477.png', '15468.png', '04197.png', '10389.png', '06073.png', '06567.png', '11427.png', '15985.png', '14420.png', '20417.png', '17366.png', '12825.png', '18500.png', '03726.png', '07375.png', '09143.png', '04264.png', '21442.png', '13635.png', '20900.png', '11122.png', '04329.png', '20683.png', '16238.png', '05409.png', '20819.png', '13696.png', '05540.png', '13142.png', '10986.png', '05679.png', '09781.png', '09865.png', '24045.png', '07550.png', '14509.png', '15137.png', '16061.png', '09345.png', '14302.png', '23723.png', '07062.png', '07149.png', '19541.png', '01596.png', '04045.png', '06555.png', '03859.png', '22407.png', '01184.png', '19628.png', '20313.png', '14306.png', '10302.png', '16330.png', '04180.png', '13961.png', '07660.png', '22899.png', '23602.png', '05818.png', '16190.png', '01455.png', '04195.png', '12311.png', '02370.png', '18319.png', '06941.png', '09696.png', '07607.png', '05155.png', '23795.png', '00926.png', '19259.png', '19804.png', '03445.png', '12185.png', '10920.png', '18397.png', '24503.png', '02318.png', '11547.png', '05727.png', '19163.png', '00247.png', '03262.png', '02413.png', '20253.png', '20418.png', '22113.png', '24517.png', '22790.png', '06837.png', '07673.png', '06441.png', '01021.png', '04119.png', '09175.png', '15802.png', '08273.png', '11765.png', '08864.png', '01236.png', '18575.png', '17845.png', '07214.png', '02886.png', '03077.png', '22303.png', '21557.png', '04558.png', '16978.png', '13423.png', '23308.png', '13056.png', '08263.png', '20415.png', '01853.png', '18542.png', '17230.png', '15938.png', '21461.png', '00063.png', '13273.png', '05286.png', '02892.png', '22521.png', '11328.png', '11741.png', '21608.png', '22023.png', '14544.png', '19623.png', '20205.png', '15803.png', '16548.png', '17098.png', '23265.png', '21914.png', '24530.png', '05582.png', '06334.png', '05370.png', '14922.png', '03687.png', '21992.png', '07053.png', '14185.png', '11852.png', '09450.png', '05310.png', '13947.png', '02346.png', '12502.png', '06035.png', '10243.png', '21744.png', '16781.png', '14458.png', '23782.png', '06532.png', '04730.png', '17453.png', '22828.png', '23248.png', '18308.png', '11826.png', '02305.png', '01240.png', '20998.png', '09194.png', '18074.png', '10358.png', '16999.png', '24453.png', '14494.png', '04036.png', '11667.png', '21757.png', '20394.png', '12748.png', '21962.png', '24507.png', '01394.png', '17117.png', '07590.png', '23490.png', '12111.png', '18763.png', '21329.png', '04367.png', '06435.png', '21970.png', '13648.png', '05829.png', '08275.png', '22236.png', '01135.png', '09110.png', '00109.png', '07609.png', '09590.png', '17473.png', '23980.png', '19049.png', '04701.png', '06849.png', '02929.png', '10019.png', '22507.png', '16676.png', '11359.png', '21335.png', '23306.png', '04843.png', '15653.png', '15776.png', '24784.png', '03549.png', '10129.png', '06021.png', '22048.png', '16400.png', '09752.png', '06263.png', '20785.png', '20818.png', '19523.png', '05118.png', '21833.png', '24394.png', '05987.png', '20788.png', '21093.png', '21899.png', '08743.png', '23460.png', '19339.png', '04606.png', '11431.png', '20325.png', '16624.png', '24329.png', '03574.png', '14908.png', '09265.png', '13230.png', '10022.png', '23718.png', '01434.png', '04275.png', '10146.png', '23685.png', '06422.png', '13901.png', '21719.png', '14607.png', '12307.png', '15740.png', '03459.png', '18179.png', '13676.png', '07427.png', '03833.png', '23259.png', '11078.png', '02528.png', '16089.png', '10965.png', '18830.png', '00740.png', '07096.png', '14062.png', '04086.png', '16191.png', '08622.png', '01639.png', '15590.png', '19057.png', '22714.png', '06375.png', '15760.png', '16930.png', '02136.png', '18615.png', '13539.png', '00519.png', '17776.png', '05077.png', '24341.png', '17716.png', '20509.png', '06106.png', '24125.png', '22248.png', '01803.png', '22641.png', '24660.png', '21319.png', '11742.png', '18404.png', '10783.png', '04280.png', '10935.png', '21527.png', '08365.png', '06485.png', '19066.png', '04447.png', '04765.png', '24238.png', '12878.png', '22650.png', '04034.png', '21750.png', '19103.png', '00126.png', '23804.png', '09104.png', '11118.png', '00921.png', '13312.png', '24243.png', '11596.png', '14934.png', '00719.png', '06094.png', '16825.png', '13123.png', '24479.png', '21593.png', '08830.png', '11175.png', '22840.png', '12272.png', '13087.png', '23034.png', '05334.png', '23878.png', '18391.png', '24255.png', '11121.png', '03755.png', '21821.png', '14741.png', '15391.png', '18800.png', '17516.png', '23167.png', '16106.png', '01811.png', '14666.png', '20840.png', '20222.png', '10455.png', '13270.png', '00785.png', '17764.png', '01610.png', '15046.png', '11622.png', '01657.png', '06835.png', '14856.png', '02719.png', '19509.png', '00890.png', '20691.png', '03883.png', '09387.png', '24184.png', '08744.png', '06586.png', '02780.png', '06669.png', '11421.png', '17359.png', '24532.png', '23972.png', '04742.png', '13923.png', '05544.png', '24379.png', '03587.png', '11635.png', '24871.png', '17290.png', '00928.png', '09934.png', '22060.png', '09078.png', '06881.png', '10451.png', '12414.png', '02499.png', '24960.png', '01061.png', '10024.png', '06158.png', '08077.png', '16593.png', '10166.png', '10667.png', '16325.png', '04711.png', '07502.png', '20462.png', '14513.png', '06906.png', '08967.png', '02539.png', '01529.png', '21170.png', '23641.png', '24073.png', '02658.png', '13693.png', '16980.png', '21780.png', '16968.png', '11284.png', '21362.png', '10801.png', '00101.png', '04191.png', '19613.png', '13748.png', '17038.png', '09137.png', '15743.png', '20878.png', '22327.png', '14596.png', '19114.png', '15808.png', '18251.png', '13044.png', '06667.png', '10958.png', '15056.png', '23165.png', '12718.png', '05308.png', '05379.png', '15914.png', '13842.png', '03301.png', '04352.png', '03579.png', '21984.png', '08434.png', '12762.png', '03235.png', '20595.png', '21281.png', '19040.png', '07448.png', '04097.png', '08915.png', '15960.png', '04602.png', '21211.png', '17147.png', '15599.png', '15118.png', '04586.png', '15464.png', '24112.png', '01358.png', '00713.png', '06999.png', '22162.png', '17857.png', '07049.png', '21647.png', '02388.png', '16328.png', '20289.png', '22987.png', '11291.png', '16524.png', '04411.png', '21050.png', '08638.png', '04429.png', '16742.png', '00910.png', '20534.png', '22383.png', '03158.png', '01165.png', '22237.png', '12601.png', '08247.png', '15521.png', '10365.png', '08095.png', '13150.png', '19404.png', '10705.png', '09601.png', '22025.png', '07347.png', '06477.png', '19014.png', '22633.png', '16490.png', '10484.png', '04405.png', '02160.png', '00798.png', '08746.png', '11733.png', '12676.png', '23011.png', '18579.png', '12768.png', '08977.png', '21165.png', '06098.png', '23361.png', '13019.png', '24631.png', '17557.png', '09702.png', '01562.png', '13972.png', '04155.png', '21445.png', '10545.png', '20545.png', '06696.png', '02604.png', '15090.png', '10756.png', '14120.png', '00296.png', '16722.png', '05274.png', '13066.png', '15530.png', '18266.png', '21856.png', '08064.png', '09488.png', '19986.png', '21343.png', '13820.png', '15654.png', '22635.png', '00870.png', '01203.png', '19650.png', '07392.png', '05311.png', '22771.png', '05525.png', '03964.png', '09600.png', '21685.png', '22372.png', '02123.png', '18784.png', '13189.png', '15229.png', '21980.png', '20977.png', '09888.png', '10918.png', '18232.png', '19678.png', '02512.png', '20731.png', '22906.png', '08202.png', '01647.png', '08232.png', '13340.png', '09325.png', '06517.png', '21621.png', '20857.png', '09606.png', '11151.png', '22692.png', '00350.png', '23699.png', '02436.png', '14900.png', '21412.png', '05092.png', '12951.png', '02042.png', '06841.png', '20376.png', '24055.png', '11589.png', '04131.png', '18108.png', '10964.png', '21625.png', '03431.png', '09892.png', '20348.png', '06447.png', '04033.png', '24462.png', '22804.png', '16886.png', '05724.png', '02615.png', '10351.png', '08663.png', '00076.png', '13319.png', '04770.png', '07612.png', '11972.png', '17302.png', '08913.png', '17872.png', '14681.png', '00929.png', '00589.png', '19950.png', '18144.png', '18632.png', '07108.png', '16584.png', '23937.png', '03764.png', '24190.png', '23690.png', '12532.png', '15523.png', '10870.png', '13967.png', '01950.png', '06534.png', '12903.png', '08136.png', '21617.png', '17164.png', '00782.png', '21377.png', '19061.png', '04782.png', '22221.png', '21532.png', '10869.png', '15498.png', '06359.png', '13638.png', '03926.png', '09867.png', '06452.png', '16273.png', '16884.png', '14983.png', '23598.png', '17139.png', '16993.png', '16306.png', '21890.png', '20485.png', '02831.png', '14754.png', '00094.png', '05109.png', '24098.png', '16224.png', '04210.png', '19572.png', '01634.png', '19169.png', '01871.png', '21632.png', '22204.png', '19774.png', '09672.png', '01120.png', '18344.png', '07894.png', '00410.png', '01750.png', '02848.png', '02247.png', '04315.png', '15983.png', '00136.png', '23200.png', '17675.png', '05094.png', '24652.png', '09170.png', '23292.png', '11299.png', '02671.png', '24934.png', '12689.png', '06383.png', '03656.png', '12298.png', '17126.png', '18457.png', '09677.png', '00431.png', '23284.png', '12490.png', '20093.png', '16481.png', '10786.png', '05870.png', '11040.png', '11567.png', '24446.png', '09323.png', '10485.png', '03605.png', '23899.png', '13669.png', '24828.png', '15051.png', '03507.png', '16752.png', '10100.png', '00084.png', '06390.png', '13700.png', '22905.png', '17960.png', '16712.png', '24247.png', '19617.png', '00771.png', '20580.png', '24245.png', '24844.png', '20463.png', '20915.png', '06816.png', '06608.png', '11905.png', '22938.png', '22439.png', '22203.png', '03131.png', '20357.png', '07420.png', '23113.png', '03903.png', '06150.png', '14828.png', '17928.png', '12754.png', '02725.png', '10040.png', '00054.png', '18053.png', '08739.png', '24103.png', '15004.png', '23303.png', '22571.png', '19692.png', '02507.png', '00840.png', '08527.png', '04778.png', '10304.png', '13735.png', '20190.png', '09317.png', '12331.png', '20634.png', '13588.png', '12701.png', '05964.png', '03019.png', '06679.png', '17797.png', '18790.png', '14902.png', '05125.png', '12851.png', '23298.png', '07046.png', '14589.png', '03492.png', '09713.png', '15848.png', '09976.png', '17438.png', '09944.png', '06482.png', '01795.png', '23583.png', '13937.png', '14859.png', '15361.png', '13957.png', '04597.png', '08765.png', '21407.png', '10428.png', '17116.png', '02412.png', '19506.png', '10014.png', '00484.png', '23372.png', '10380.png', '19677.png', '13358.png', '24208.png', '00324.png', '16103.png', '24656.png', '19743.png', '00835.png', '15351.png', '17792.png', '11113.png', '10410.png', '23196.png', '14678.png', '20931.png', '06332.png', '14365.png', '17045.png', '06087.png', '02710.png', '04867.png', '23189.png', '09203.png', '10792.png', '06425.png', '06801.png', '24646.png', '07230.png', '18462.png', '11754.png', '09173.png', '11163.png', '05139.png', '11253.png', '03788.png', '20737.png', '05702.png', '21861.png', '13870.png', '22558.png', '14690.png', '05610.png', '23843.png', '09935.png', '09050.png', '20173.png', '01587.png', '07992.png', '14192.png', '06910.png', '08431.png', '14156.png', '18844.png', '10514.png', '02178.png', '00605.png', '21161.png', '21558.png', '19640.png', '00183.png', '14241.png', '11187.png', '03671.png', '06782.png', '11803.png', '20193.png', '23578.png', '22285.png', '05892.png', '05491.png', '15889.png', '12086.png', '17513.png', '06537.png', '21056.png', '00130.png', '06825.png', '03214.png', '02068.png', '23779.png', '22894.png', '13815.png', '02224.png', '03977.png', '17496.png', '23091.png', '07573.png', '05524.png', '19648.png', '04457.png', '08349.png', '01816.png', '08655.png', '16788.png', '06366.png', '17699.png', '18886.png', '23128.png', '12874.png', '04463.png', '24066.png', '14488.png', '10501.png', '03107.png', '22603.png', '12452.png', '19897.png', '03696.png', '13723.png', '14698.png', '11494.png', '18973.png', '12308.png', '00757.png', '13575.png', '16250.png', '19490.png', '24144.png', '16891.png', '12480.png', '14358.png', '11528.png', '00967.png', '01778.png', '04495.png', '16266.png', '09118.png', '05609.png', '08423.png', '12401.png', '22511.png', '24535.png', '09654.png', '17389.png', '19376.png', '15098.png', '18342.png', '14390.png', '03929.png', '12153.png', '24230.png', '12588.png', '09412.png', '09788.png', '06362.png', '00552.png', '05345.png', '23028.png', '18685.png', '00329.png', '16641.png', '13322.png', '07261.png', '23889.png', '24077.png', '15798.png', '05636.png', '20921.png', '22759.png', '21958.png', '15208.png', '02119.png', '02623.png', '20466.png', '06933.png', '00191.png', '21496.png', '18593.png', '18461.png', '08075.png', '21946.png', '21715.png', '03792.png', '20431.png', '10939.png', '08826.png', '14550.png', '23356.png', '23050.png', '06701.png', '05032.png', '01188.png', '06571.png', '06626.png', '20527.png', '19797.png', '17995.png', '23652.png', '04904.png', '24572.png', '18923.png', '24185.png', '05606.png', '13990.png', '12429.png', '20363.png', '15568.png', '01625.png', '11243.png', '01892.png', '16014.png', '06020.png', '15516.png', '11126.png', '20269.png', '17483.png', '17656.png', '02200.png', '11719.png', '08421.png', '07821.png', '02657.png', '18016.png', '19748.png', '11699.png', '07398.png', '11707.png', '00968.png', '13650.png', '17820.png', '24823.png', '18777.png', '09447.png', '09418.png', '16875.png', '22069.png', '12734.png', '18330.png', '16744.png', '06283.png', '08994.png', '07304.png', '08937.png', '14298.png', '21318.png', '15793.png', '17070.png', '19957.png', '21931.png', '10651.png', '03640.png', '10012.png', '08747.png', '09049.png', '00024.png', '05523.png', '00978.png', '14474.png', '17940.png', '13062.png', '20621.png', '12576.png', '01443.png', '17171.png', '05143.png', '06927.png', '22174.png', '01058.png', '01166.png', '09236.png', '14141.png', '04433.png', '11797.png', '05238.png', '01746.png', '18203.png', '00535.png', '11329.png', '03614.png', '06341.png', '11744.png', '09252.png', '00646.png', '14985.png', '09582.png', '15735.png', '12988.png', '07670.png', '04056.png', '00983.png', '13494.png', '00070.png', '08317.png', '17150.png', '05461.png', '19873.png', '05538.png', '04649.png', '07407.png', '10032.png', '01898.png', '18996.png', '15133.png', '08795.png', '08734.png', '08179.png', '11904.png', '13825.png', '14292.png', '17587.png', '05960.png', '01666.png', '06038.png', '10879.png', '03215.png', '23727.png', '04657.png', '12994.png', '08004.png', '04918.png', '08028.png', '09242.png', '00832.png', '09491.png', '11864.png', '08420.png', '07112.png', '08781.png', '17899.png', '14402.png', '19231.png', '15180.png', '13943.png', '00571.png', '22299.png', '07306.png', '21279.png', '15374.png', '08670.png', '15162.png', '09815.png', '02799.png', '11908.png', '21288.png', '01770.png', '13987.png', '15884.png', '02973.png', '22793.png', '01238.png', '05844.png', '12397.png', '18749.png', '16394.png', '10203.png', '05442.png', '06253.png', '21095.png', '15268.png', '20183.png', '08145.png', '20087.png', '18080.png', '17013.png', '23824.png', '21000.png', '09466.png', '13788.png', '12178.png', '17723.png', '05323.png', '21543.png', '18793.png', '14150.png', '23680.png', '18389.png', '16452.png', '10859.png', '03806.png', '10804.png', '00570.png', '22047.png', '18604.png', '07501.png', '14942.png', '21922.png', '13260.png', '04133.png', '07955.png', '05706.png', '11534.png', '10049.png', '14568.png', '10124.png', '17911.png', '09730.png', '22071.png', '07216.png', '10077.png', '02199.png', '01342.png', '05515.png', '19078.png', '22708.png', '14972.png', '17604.png', '14780.png', '06634.png', '12161.png', '14313.png', '07051.png', '18197.png', '24193.png', '22411.png', '10112.png', '24667.png', '00489.png', '02181.png', '20291.png', '24215.png', '10412.png', '10862.png', '04542.png', '04619.png', '04783.png', '12720.png', '16654.png', '22776.png', '01767.png', '10949.png', '01756.png', '06102.png', '07179.png', '21309.png', '06685.png', '18851.png', '17409.png', '06848.png', '15280.png', '18897.png', '14716.png', '21346.png', '01435.png', '04814.png', '15859.png', '12637.png', '00917.png', '14317.png', '08384.png', '07887.png', '00241.png', '07950.png', '06049.png', '20377.png', '24946.png', '18595.png', '20622.png', '14694.png', '06758.png', '23976.png', '02770.png', '00203.png', '15690.png', '24580.png', '11347.png', '16354.png', '09999.png', '12115.png', '04680.png', '09291.png', '09405.png', '04204.png', '10318.png', '12651.png', '01555.png', '07888.png', '03942.png', '03421.png', '03284.png', '02745.png', '05259.png', '17531.png', '24623.png', '15943.png', '18864.png', '24571.png', '22413.png', '01138.png', '23734.png', '00163.png', '21174.png', '08718.png', '08511.png', '09235.png', '18850.png', '18801.png', '24900.png', '12687.png', '08223.png', '22163.png', '04220.png', '02427.png', '01297.png', '10882.png', '18672.png', '11969.png', '19718.png', '16383.png', '13515.png', '05301.png', '17372.png', '13262.png', '19657.png', '17730.png', '08553.png', '03119.png', '08882.png', '10524.png', '20748.png', '18066.png', '08716.png', '18935.png', '22971.png', '08393.png', '11759.png', '18603.png', '03268.png', '13331.png', '15157.png', '15533.png', '02757.png', '04624.png', '19512.png', '07667.png', '15917.png', '12213.png', '24907.png', '00692.png', '05736.png', '24697.png', '14742.png', '13115.png', '08612.png', '00805.png', '03165.png', '11725.png', '16662.png', '14692.png', '22341.png', '05675.png', '21094.png', '10457.png', '04192.png', '02560.png', '16076.png', '15250.png', '00814.png', '07195.png', '17314.png', '01788.png', '11392.png', '06320.png', '01353.png', '06662.png', '01600.png', '17674.png', '12578.png', '03498.png', '11526.png', '18228.png', '02890.png', '11355.png', '01654.png', '11805.png', '07961.png', '03258.png', '02873.png', '08808.png', '07390.png', '18288.png', '20710.png', '23571.png', '24293.png', '24027.png', '11912.png', '10208.png', '15665.png', '04720.png', '05324.png', '03565.png', '08164.png', '13747.png', '06095.png', '08340.png', '13464.png', '12688.png', '14829.png', '04490.png', '24351.png', '21698.png', '03083.png', '06930.png', '06292.png', '11433.png', '14052.png', '20239.png', '23017.png', '06036.png', '03133.png', '19303.png', '03042.png', '15829.png', '02775.png', '10626.png', '01030.png', '19620.png', '05937.png', '20086.png', '23593.png', '11786.png', '24376.png', '18746.png', '13761.png', '21953.png', '06463.png', '14730.png', '17071.png', '17048.png', '01979.png', '05893.png', '14266.png', '12964.png', '20688.png', '18723.png', '16213.png', '16092.png', '01088.png', '15506.png', '23669.png', '07344.png', '17061.png', '02421.png', '05550.png', '16515.png', '13657.png', '10325.png', '06209.png', '11381.png', '24155.png', '22779.png', '18647.png', '17957.png', '05509.png', '03568.png', '02276.png', '13949.png', '11943.png', '03727.png', '14747.png', '08210.png', '22751.png', '00391.png', '00572.png', '16368.png', '21645.png', '22124.png', '23684.png', '20817.png', '22670.png', '05972.png', '05054.png', '07999.png', '18881.png', '11231.png', '13746.png', '22164.png', '24621.png', '13883.png', '08258.png', '17493.png', '21798.png', '15554.png', '01544.png', '05169.png', '23839.png', '16270.png', '22914.png', '20294.png', '09270.png', '14881.png', '18772.png', '15795.png', '22273.png', '12727.png', '21795.png', '06099.png', '04215.png', '20863.png', '22584.png', '03486.png', '11656.png', '02729.png', '23547.png', '05826.png', '24555.png', '14656.png', '17577.png', '24220.png', '16140.png', '12580.png', '07560.png', '23049.png', '11327.png', '01033.png', '17253.png', '04242.png', '18343.png', '02425.png', '00743.png', '17640.png', '22916.png', '07997.png', '10293.png', '18734.png', '13713.png', '00579.png', '05757.png', '00100.png', '06686.png', '02829.png', '20303.png', '12987.png', '19214.png', '21302.png', '08398.png', '13856.png', '02409.png', '12006.png', '06892.png', '11531.png', '04196.png', '20164.png', '06612.png', '13926.png', '20126.png', '17831.png', '07830.png', '04309.png', '20496.png', '13683.png', '20732.png', '11037.png', '19706.png', '10996.png', '09750.png', '09443.png', '23582.png', '00194.png', '03149.png', '12430.png', '10782.png', '16073.png', '20492.png', '01564.png', '23651.png', '08082.png', '02496.png', '14523.png', '01651.png', '17049.png', '19321.png', '12823.png', '24209.png', '10117.png', '02476.png', '14229.png', '17052.png', '21690.png', '13076.png', '18472.png', '11921.png', '10488.png', '10337.png', '00563.png', '19836.png', '17994.png', '18336.png', '14429.png', '17467.png', '07273.png', '11940.png', '06348.png', '13847.png', '10263.png', '12717.png', '00557.png', '16786.png', '01918.png', '13549.png', '17382.png', '20709.png', '05250.png', '04410.png', '15197.png', '02283.png', '12686.png', '03634.png', '02684.png', '18689.png', '22850.png', '10233.png', '21425.png', '08674.png', '03638.png', '21236.png', '20879.png', '00361.png', '15058.png', '17597.png', '00642.png', '09489.png', '12131.png', '07575.png', '01347.png', '10799.png', '07492.png', '11145.png', '15931.png', '20186.png', '05190.png', '20052.png', '13437.png', '00796.png', '07021.png', '20013.png', '24548.png', '17017.png', '02035.png', '05477.png', '08246.png', '10662.png', '08048.png', '10572.png', '22315.png', '04408.png', '16063.png', '03353.png', '06875.png', '01978.png', '03278.png', '06936.png', '22229.png', '00634.png', '04486.png', '13849.png', '19903.png', '04764.png', '04244.png', '00645.png', '15185.png', '17373.png', '14612.png', '04224.png', '01941.png', '15139.png', '05952.png', '17268.png', '00724.png', '23354.png', '21262.png', '21762.png', '07747.png', '00851.png', '15758.png', '17191.png', '22179.png', '09584.png', '20102.png', '20008.png', '10388.png', '08130.png', '17595.png', '14246.png', '01007.png', '06814.png', '16124.png', '09961.png', '18267.png', '23938.png', '11293.png', '14682.png', '13490.png', '20790.png', '07717.png', '17873.png', '03370.png', '23215.png', '01537.png', '01769.png', '17088.png', '20728.png', '16650.png', '11760.png', '17830.png', '15593.png', '00561.png', '10275.png', '24120.png', '09199.png', '04496.png', '13599.png', '02824.png', '22320.png', '24783.png', '14114.png', '21786.png', '05163.png', '07208.png', '20692.png', '20233.png', '06942.png', '09850.png', '23957.png', '16846.png', '21426.png', '20991.png', '17319.png', '01885.png', '11068.png', '18414.png', '19353.png', '04103.png', '20874.png', '18341.png', '03430.png', '21838.png', '13305.png', '17156.png', '11781.png', '03027.png', '11997.png', '12498.png', '22740.png', '03921.png', '01468.png', '19576.png', '10126.png', '17134.png', '20830.png', '02504.png', '22536.png', '05787.png', '16709.png', '00265.png', '10575.png', '17050.png', '23681.png', '09902.png', '06047.png', '05145.png', '02249.png', '18847.png', '20848.png', '21152.png', '00347.png', '20907.png', '03260.png', '03690.png', '11065.png', '16221.png', '20838.png', '24544.png', '11696.png', '07110.png', '12911.png', '14835.png', '09470.png', '01850.png', '14454.png', '14723.png', '22191.png', '01992.png', '16064.png', '15801.png', '11159.png', '24178.png', '09994.png', '17722.png', '17741.png', '02592.png', '00457.png', '08953.png', '14541.png', '04849.png', '17568.png', '13889.png', '10366.png', '06113.png', '15113.png', '09875.png', '10341.png', '06967.png', '00587.png', '18289.png', '15862.png', '10760.png', '06471.png', '19989.png', '14479.png', '12348.png', '14149.png', '15539.png', '03516.png', '10672.png', '16141.png', '00322.png', '11246.png', '02330.png', '19241.png', '19952.png', '08871.png', '19656.png', '10826.png', '07930.png', '21138.png', '19193.png', '06199.png', '21475.png', '06205.png', '07884.png', '01377.png', '16133.png', '24074.png', '08556.png', '15787.png', '11307.png', '10260.png', '06165.png', '07944.png', '00606.png', '22992.png', '22227.png', '00790.png', '18879.png', '22348.png', '12292.png', '13431.png', '16225.png', '05887.png', '03238.png', '04024.png', '00129.png', '04910.png', '06025.png', '23206.png', '23661.png', '18759.png', '12989.png', '22063.png', '10521.png', '00916.png', '07640.png', '05732.png', '21910.png', '16112.png', '05097.png', '19833.png', '16139.png', '14745.png', '14293.png', '21326.png', '17136.png', '00211.png', '07785.png', '18678.png', '13139.png', '13213.png', '07580.png', '01191.png', '20835.png', '13886.png', '14078.png', '09653.png', '24235.png', '19889.png', '00416.png', '15332.png', '19059.png', '19589.png', '06489.png', '01079.png', '13851.png', '02906.png', '00152.png', '24327.png', '23040.png', '03385.png', '06252.png', '13584.png', '10887.png', '11061.png', '13506.png', '05417.png', '16964.png', '04709.png', '03744.png', '06540.png', '19156.png', '06729.png', '03523.png', '22086.png', '13176.png', '21465.png', '00111.png', '04505.png', '17087.png', '16510.png', '23448.png', '15360.png', '11847.png', '09395.png', '06541.png', '12012.png', '19665.png', '00848.png', '12630.png', '11509.png', '09794.png', '13641.png', '02756.png', '21149.png', '01822.png', '21928.png', '05996.png', '19800.png', '07432.png', '13400.png', '13445.png', '21588.png', '14085.png', '17988.png', '20069.png', '15858.png', '03752.png', '20517.png', '19923.png', '21374.png', '09818.png', '24426.png', '16161.png', '16314.png', '21337.png', '05261.png', '22988.png', '24160.png', '00153.png', '05762.png', '00142.png', '00274.png', '17998.png', '20524.png', '01864.png', '11623.png', '20796.png', '01198.png', '23423.png', '12238.png', '08140.png', '03451.png', '14842.png', '02360.png', '03701.png', '05793.png', '06953.png', '23586.png', '05881.png', '17506.png', '01047.png', '21315.png', '05131.png', '06526.png', '14641.png', '11500.png', '00876.png', '21416.png', '16988.png', '03113.png', '12373.png', '11234.png', '06446.png', '22183.png', '23353.png', '10855.png', '21578.png', '09598.png', '16771.png', '17622.png', '24371.png', '04844.png', '05154.png', '17110.png', '01300.png', '13181.png', '16862.png', '18638.png', '24519.png', '05687.png', '03023.png', '20218.png', '03173.png', '02087.png', '09621.png', '11999.png', '07508.png', '06260.png', '22415.png', '07204.png', '03895.png', '05638.png', '07017.png', '16739.png', '14148.png', '16389.png', '18855.png', '00913.png', '24512.png', '02925.png', '03002.png', '09841.png', '10974.png', '04786.png', '09816.png', '11369.png', '19023.png', '00256.png', '10301.png', '04290.png', '23190.png', '06437.png', '04909.png', '20816.png', '05635.png', '20162.png', '09321.png', '01540.png', '00443.png', '11691.png', '24205.png', '06296.png', '24382.png', '17186.png', '17706.png', '09319.png', '22296.png', '09968.png', '20669.png', '19814.png', '09465.png', '01943.png', '02852.png', '06370.png', '12703.png', '16735.png', '10693.png', '08432.png', '22264.png', '22593.png', '18716.png', '06223.png', '04386.png', '05315.png', '18469.png', '17228.png', '16082.png', '24313.png', '04591.png', '07073.png', '14180.png', '03976.png', '08441.png', '15022.png', '09454.png', '12652.png', '08362.png', '20039.png', '12440.png', '03331.png', '02579.png', '09814.png', '20457.png', '03621.png', '21877.png', '22693.png', '17686.png', '17066.png', '19879.png', '23526.png', '07947.png', '20981.png', '11230.png', '12808.png', '13915.png', '13393.png', '16097.png', '13220.png', '04272.png', '13612.png', '04118.png', '12217.png', '00093.png', '15878.png', '17311.png', '24611.png', '22570.png', '09893.png', '01298.png', '08006.png', '04970.png', '00081.png', '09063.png', '14489.png', '08892.png', '09725.png', '00198.png', '18818.png', '19475.png', '10460.png', '06527.png', '15913.png', '07290.png', '01114.png', '03155.png', '20255.png', '04872.png', '06402.png', '01563.png', '16522.png', '17565.png', '15108.png', '09130.png', '20801.png', '13865.png', '14068.png', '20412.png', '19959.png', '05834.png', '08571.png', '08156.png', '03837.png', '02390.png', '06407.png', '05061.png', '06065.png', '15188.png', '13522.png', '24071.png', '08857.png', '20157.png', '09359.png', '17234.png', '02815.png', '05692.png', '14226.png', '05593.png', '16704.png', '12406.png', '18117.png', '05922.png', '19047.png', '16249.png', '06440.png', '05751.png', '16585.png', '09103.png', '08142.png', '20224.png', '01692.png', '06367.png', '21832.png', '00523.png', '03347.png', '04796.png', '12269.png', '21519.png', '13157.png', '00141.png', '09125.png', '14308.png', '06478.png', '19703.png', '19904.png', '00534.png', '16113.png', '04442.png', '24951.png', '03300.png', '03999.png', '04979.png', '02961.png', '00517.png', '12099.png', '12779.png', '03282.png', '05778.png', '09963.png', '16358.png', '14687.png', '23559.png', '18148.png', '19622.png', '18418.png', '16523.png', '00388.png', '12368.png', '06884.png', '12326.png', '12700.png', '03539.png', '24132.png', '08303.png', '06798.png', '11637.png', '18141.png', '20027.png', '21797.png', '01916.png', '06990.png', '00944.png', '22540.png', '16509.png', '09444.png', '07493.png', '10697.png', '08485.png', '09494.png', '20660.png', '00289.png', '09641.png', '02614.png', '14391.png', '23383.png', '19922.png', '12590.png', '13634.png', '04537.png', '08896.png', '00242.png', '06792.png', '09180.png', '17025.png', '15816.png', '08720.png', '00430.png', '20751.png', '00262.png', '19392.png', '08755.png', '19520.png', '03786.png', '11964.png', '12314.png', '06937.png', '14492.png', '10643.png', '10825.png', '12831.png', '01640.png', '11186.png', '20481.png', '08296.png', '18618.png', '11042.png', '24065.png', '00341.png', '19210.png', '24964.png', '07644.png', '05072.png', '19498.png', '20773.png', '12786.png', '15262.png', '16159.png', '09014.png', '02518.png', '06597.png', '15444.png', '22078.png', '15644.png', '02277.png', '24450.png', '19961.png', '12535.png', '04887.png', '10505.png', '00113.png', '06887.png', '09455.png', '15688.png', '06913.png', '16205.png', '22764.png', '18283.png', '24567.png', '23343.png', '19605.png', '20519.png', '14296.png', '20312.png', '22417.png', '07201.png', '05796.png', '16607.png', '16415.png', '16529.png', '03409.png', '16308.png', '12032.png', '12783.png', '10375.png', '18324.png', '14109.png', '04107.png', '24504.png', '09460.png', '00187.png', '10976.png', '05624.png', '22465.png', '15438.png', '15969.png', '09396.png', '06330.png', '08958.png', '14735.png', '06257.png', '14032.png', '18630.png', '24879.png', '11063.png', '09609.png', '04550.png', '19598.png', '00745.png', '15761.png', '07578.png', '03757.png', '23205.png', '00852.png', '18355.png', '08969.png', '21018.png', '14471.png', '24634.png', '21181.png', '10329.png', '11275.png', '15841.png', '07675.png', '15204.png', '12262.png', '06008.png', '16898.png', '07315.png', '17490.png', '06603.png', '21127.png', '10047.png', '20992.png', '23603.png', '08371.png', '14146.png', '06242.png', '12615.png', '12518.png', '04830.png', '19635.png', '02600.png', '19187.png', '23118.png', '15289.png', '20802.png', '24353.png', '22931.png', '19715.png', '01141.png', '03839.png', '10594.png', '14774.png', '12492.png', '12107.png', '11788.png', '24082.png', '10504.png', '16637.png', '08980.png', '12696.png', '23014.png', '03842.png', '00607.png', '10296.png', '03838.png', '13348.png', '17848.png', '20382.png', '22700.png', '19783.png', '17044.png', '06878.png', '10138.png', '23276.png', '12469.png', '13569.png', '13140.png', '18269.png', '03448.png', '07199.png', '14427.png', '14709.png', '14546.png', '02487.png', '17948.png', '23796.png', '22950.png', '07417.png', '18820.png', '16517.png', '11933.png', '04206.png', '00298.png', '04345.png', '05726.png', '16090.png', '13677.png', '17578.png', '18243.png', '06236.png', '07622.png', '23790.png', '15570.png', '00053.png', '09440.png', '16864.png', '24607.png', '24810.png', '19417.png', '07698.png', '07976.png', '20994.png', '16616.png', '21480.png', '18120.png', '20687.png', '24762.png', '21484.png', '08153.png', '00300.png', '17503.png', '16302.png', '00407.png', '18999.png', '01082.png', '09675.png', '18948.png', '20424.png', '24268.png', '21231.png', '17008.png', '05533.png', '12725.png', '10441.png', '08372.png', '09503.png', '15038.png', '20028.png', '21601.png', '09102.png', '19565.png', '13875.png', '02594.png', '07923.png', '06681.png', '15334.png', '05898.png', '24226.png', '18715.png', '12002.png', '11212.png', '02169.png', '06831.png', '06855.png', '17887.png', '21824.png', '08927.png', '07858.png', '08943.png', '14693.png', '22879.png', '10989.png', '13228.png', '20996.png', '01594.png', '18182.png', '16301.png', '04465.png', '21195.png', '10743.png', '23380.png', '14640.png', '07294.png', '01575.png', '13653.png', '21917.png', '13068.png', '03979.png', '10525.png', '06267.png', '04616.png', '14955.png', '20100.png', '10768.png', '03129.png', '23351.png', '02052.png', '02453.png', '18587.png', '04311.png', '20591.png', '09267.png', '17072.png', '13396.png', '08078.png', '04992.png', '04969.png', '05344.png', '01274.png', '03739.png', '24925.png', '03934.png', '06883.png', '10717.png', '12622.png', '00728.png', '04300.png', '17987.png', '21044.png', '19832.png', '13486.png', '09516.png', '24581.png', '15564.png', '08534.png', '24131.png', '12004.png', '19942.png', '00585.png', '05877.png', '00181.png', '09012.png', '15469.png', '16175.png', '04815.png', '23099.png', '08952.png', '20644.png', '07890.png', '21233.png', '03890.png', '00029.png', '08285.png', '04017.png', '03086.png', '11488.png', '23024.png', '16838.png', '13247.png', '21204.png', '16285.png', '21818.png', '17774.png', '23478.png', '20121.png', '11194.png', '21746.png', '22805.png', '18180.png', '08464.png', '01655.png', '04749.png', '00226.png', '03141.png', '21015.png', '00537.png', '16538.png', '11857.png', '07330.png', '00348.png', '21551.png', '12380.png', '03264.png', '19345.png', '04122.png', '08237.png', '05707.png', '13449.png', '24911.png', '07460.png', '17742.png', '11641.png', '04860.png', '16831.png', '10644.png', '06966.png', '00735.png', '05548.png', '13734.png', '24348.png', '16420.png', '04281.png', '10268.png', '08452.png', '00694.png', '11675.png', '15977.png', '10046.png', '23415.png', '05068.png', '01894.png', '03959.png', '03170.png', '20516.png', '11054.png', '06514.png', '14499.png', '19416.png', '17157.png', '14639.png', '06807.png', '04462.png', '06994.png', '17419.png', '03749.png', '13426.png', '10927.png', '14574.png', '10910.png', '15094.png', '11238.png', '05111.png', '14961.png', '13666.png', '13071.png', '13570.png', '17271.png', '23116.png', '22189.png', '20074.png', '14992.png', '19536.png', '13190.png', '19300.png', '01781.png', '06161.png', '14222.png', '21792.png', '11918.png', '15748.png', '22310.png', '15222.png', '18661.png', '11906.png', '05573.png', '13785.png', '16929.png', '24957.png', '02115.png', '09662.png', '12456.png', '01806.png', '07922.png', '22576.png', '17924.png', '16810.png', '06715.png', '14278.png', '18654.png', '22426.png', '05576.png', '08347.png', '09054.png', '13269.png', '02559.png', '08602.png', '07464.png', '20767.png', '23525.png', '14521.png', '15876.png', '07247.png', '23543.png', '06324.png', '16575.png', '05530.png', '19524.png', '24014.png', '04751.png', '05667.png', '04464.png', '02738.png', '00383.png', '20210.png', '14295.png', '01128.png', '23981.png', '19052.png', '15235.png', '02007.png', '16868.png', '00954.png', '17648.png', '20920.png', '13787.png', '03841.png', '17726.png', '00123.png', '23273.png', '13563.png', '10039.png', '20730.png', '07979.png', '07528.png', '12101.png', '03174.png', '01983.png', '15538.png', '10663.png', '10673.png', '06079.png', '02627.png', '00959.png', '13327.png', '15206.png', '15127.png', '13931.png', '21654.png', '13965.png', '23314.png', '11959.png', '03961.png', '14825.png', '00980.png', '11365.png', '02796.png', '07159.png', '00682.png', '21464.png', '12177.png', '04285.png', '08671.png', '14609.png', '13200.png', '22886.png', '17809.png', '23767.png', '20638.png', '05851.png', '18091.png', '04621.png', '19514.png', '12478.png', '03153.png', '08634.png', '21817.png', '08121.png', '04753.png', '23536.png', '19107.png', '16693.png', '18208.png', '00250.png', '21159.png', '18175.png', '22839.png', '19132.png', '09305.png', '04795.png', '18750.png', '07596.png', '08916.png', '05943.png', '21836.png', '04781.png', '09091.png', '06772.png', '12804.png', '17296.png', '04955.png', '10035.png', '03950.png', '23820.png', '24738.png', '18963.png', '16615.png', '19820.png', '08176.png', '17494.png', '24695.png', '05231.png', '13412.png', '13645.png', '04091.png', '10551.png', '12063.png', '16571.png', '11089.png', '17971.png', '03208.png', '22130.png', '18508.png', '11188.png', '06285.png', '00068.png', '19937.png', '07395.png', '10929.png', '14449.png', '08862.png', '14152.png', '13637.png', '05915.png', '11726.png', '06400.png', '07181.png', '24357.png', '03522.png', '24298.png', '06959.png', '11890.png', '17499.png', '00007.png', '19557.png', '13717.png', '06581.png', '12497.png', '10904.png', '10956.png', '12666.png', '24437.png', '02922.png', '17637.png', '22706.png', '20203.png', '01219.png', '02070.png', '19597.png', '05733.png', '15995.png', '16677.png', '04857.png', '21067.png', '10582.png', '12579.png', '11418.png', '21012.png', '22471.png', '19168.png', '02258.png', '17708.png', '05816.png', '13749.png', '07869.png', '02767.png', '17350.png', '04893.png', '24792.png', '23139.png', '16217.png', '01018.png', '00567.png', '17036.png', '22630.png', '20715.png', '00530.png', '10806.png', '06415.png', '12586.png', '13922.png', '16245.png', '07265.png', '07777.png', '18862.png', '03028.png', '21151.png', '12084.png', '02401.png', '00283.png', '12282.png', '01364.png', '08940.png', '04942.png', '12917.png', '17731.png', '00008.png', '23253.png', '07637.png', '10043.png', '13530.png', '03686.png', '17890.png', '19918.png', '08244.png', '19012.png', '06895.png', '16091.png', '03246.png', '24094.png', '07576.png', '02375.png', '16032.png', '09780.png', '14940.png', '23227.png', '12678.png', '17526.png', '15067.png', '07939.png', '15828.png', '12191.png', '20215.png', '01428.png', '13421.png', '24337.png', '17325.png', '07811.png', '05398.png', '18183.png', '12649.png', '17195.png', '10216.png', '07969.png', '10212.png', '10520.png', '23369.png', '13589.png', '16870.png', '09876.png', '06243.png', '00095.png', '07236.png', '16435.png', '11493.png', '05402.png', '06644.png', '07772.png', '18665.png', '02374.png', '15724.png', '21237.png', '05703.png', '04879.png', '10003.png', '06289.png', '04441.png', '24174.png', '06995.png', '15732.png', '13174.png', '20843.png', '16007.png', '24319.png', '20129.png', '11767.png', '20607.png', '01553.png', '01483.png', '15466.png', '06010.png', '08270.png', '01579.png', '08926.png', '23419.png', '23696.png', '04912.png', '12192.png', '24803.png', '04710.png', '16156.png', '03566.png', '10901.png', '20078.png', '13430.png', '07931.png', '07031.png', '10702.png', '06721.png', '04787.png', '13602.png', '17241.png', '13137.png', '23500.png', '24136.png', '05669.png', '01982.png', '20657.png', '19935.png', '13394.png', '10779.png', '12812.png', '09350.png', '05056.png', '11827.png', '15128.png', '21455.png', '12362.png', '10657.png', '05676.png', '03817.png', '11263.png', '10344.png', '15285.png', '21799.png', '13202.png', '02554.png', '07522.png', '17921.png', '02934.png', '09246.png', '04966.png', '07030.png', '14066.png', '02040.png', '09355.png', '08015.png', '12042.png', '04532.png', '10413.png', '12556.png', '01551.png', '20455.png', '09664.png', '15559.png', '05189.png', '13216.png', '08288.png', '07125.png', '23499.png', '18257.png', '00812.png', '15577.png', '22038.png', '13630.png', '19365.png', '05052.png', '23861.png', '17301.png', '01787.png', '12724.png', '10639.png', '14042.png', '09835.png', '21993.png', '08319.png', '09298.png', '10522.png', '05327.png', '04382.png', '18339.png', '18958.png', '03103.png', '02966.png', '21185.png', '16998.png', '10335.png', '19976.png', '21217.png', '21252.png', '09149.png', '00034.png', '02768.png', '07844.png', '05083.png', '19371.png', '18992.png', '13911.png', '04758.png', '17353.png', '14937.png', '06059.png', '21059.png', '24812.png', '08378.png', '20974.png', '02034.png', '12795.png', '00614.png', '19278.png', '24749.png', '05408.png', '19447.png', '00089.png', '19960.png', '07604.png', '15258.png', '17602.png', '07141.png', '05959.png', '14973.png', '20655.png', '21194.png', '01946.png', '23704.png', '02919.png', '19762.png', '13843.png', '11884.png', '14364.png', '12382.png', '20698.png', '03287.png', '03774.png', '14748.png', '15447.png', '02184.png', '01409.png', '20088.png', '12584.png', '14462.png', '20797.png', '12058.png', '24901.png', '22767.png', '23299.png', '07519.png', '14613.png', '21921.png', '12643.png', '20508.png', '22028.png', '18191.png', '05251.png', '17298.png', '19569.png', '16833.png', '23818.png', '05797.png', '13833.png', '18054.png', '10744.png', '15741.png', '15680.png', '08585.png', '21515.png', '14424.png', '24004.png', '10849.png', '12623.png', '17124.png', '06436.png', '08115.png', '23463.png', '02964.png', '11093.png', '09055.png', '22258.png', '06137.png', '21069.png', '04230.png', '10187.png', '15401.png', '22781.png', '08773.png', '09596.png', '06465.png', '09866.png', '12807.png', '20293.png', '19645.png', '04686.png', '10917.png', '01094.png', '05487.png', '10399.png', '12996.png', '20831.png', '07711.png', '04868.png', '16692.png', '13745.png', '15463.png', '03852.png', '19876.png', '12436.png', '11807.png', '09222.png', '24384.png', '00178.png', '20368.png', '08376.png', '00405.png', '11986.png', '02520.png', '16177.png', '23919.png', '23331.png', '01671.png', '21512.png', '18504.png', '02636.png', '08962.png', '08520.png', '13775.png', '14616.png', '13857.png', '23990.png', '01879.png', '06237.png', '11055.png', '19534.png', '06476.png', '02358.png', '00357.png', '11859.png', '07915.png', '20903.png', '03681.png', '14108.png', '07616.png', '22794.png', '06461.png', '02668.png', '07626.png', '04519.png', '08352.png', '23363.png', '08459.png', '19987.png', '07001.png', '19758.png', '13428.png', '02945.png', '15184.png', '06421.png', '10878.png', '19819.png', '06175.png', '15377.png', '23241.png', '21215.png', '20938.png', '15567.png', '01279.png', '19549.png', '10213.png', '21006.png', '07170.png', '20912.png', '21145.png', '20159.png', '22784.png', '20365.png', '15302.png', '10659.png', '01839.png', '18762.png', '07441.png', '00935.png', '19999.png', '23476.png', '07657.png', '22300.png', '18892.png', '04662.png', '21245.png', '13288.png', '17174.png', '22898.png', '13671.png', '17543.png', '24333.png', '02701.png', '22862.png', '06950.png', '23071.png', '15266.png', '11110.png', '19388.png', '00428.png', '16766.png', '14648.png', '01797.png', '07982.png', '10234.png', '16876.png', '04363.png', '09101.png', '24664.png', '20770.png', '18852.png', '15739.png', '01507.png', '00173.png', '21580.png', '00097.png', '14064.png', '05517.png', '19385.png', '07659.png', '02327.png', '20380.png', '14023.png', '13282.png', '11477.png', '15966.png', '06731.png', '24299.png', '10692.png', '14490.png', '00844.png', '01515.png', '05537.png', '05775.png', '01960.png', '14584.png', '00134.png', '05001.png', '10884.png', '13194.png', '15057.png', '00934.png', '12330.png', '08925.png', '03505.png', '02351.png', '18936.png', '19647.png', '17594.png', '00012.png', '14719.png', '12731.png', '21387.png', '03049.png', '04104.png', '14183.png', '19452.png', '08724.png', '16134.png', '22449.png', '13649.png', '21234.png', '09828.png', '22073.png', '13919.png', '22269.png', '24028.png', '18211.png', '10597.png', '06385.png', '08045.png', '06948.png', '06488.png', '01074.png', '03055.png', '22594.png', '20185.png', '20084.png', '03519.png', '14847.png', '05495.png', '13725.png', '24726.png', '16042.png', '08712.png', '13976.png', '01780.png', '06177.png', '21104.png', '17671.png', '00491.png', '19390.png', '13811.png', '14083.png', '07433.png', '10440.png', '07641.png', '04981.png', '15794.png', '24756.png', '16215.png', '12114.png', '19230.png', '12499.png', '13781.png', '14067.png', '03304.png', '04384.png', '04166.png', '23263.png', '09346.png', '04452.png', '23166.png', '21478.png', '00092.png', '00021.png', '02395.png', '03771.png', '07063.png', '11899.png', '19484.png', '20914.png', '06116.png', '09109.png', '21123.png', '01196.png', '03329.png', '00495.png', '08172.png', '09227.png', '17575.png', '00985.png', '07050.png', '03171.png', '17194.png', '04850.png', '03123.png', '00456.png', '21860.png', '17540.png', '14084.png', '06993.png', '02653.png', '09165.png', '21956.png', '06830.png', '18362.png', '19055.png', '15607.png', '09372.png', '17655.png', '23344.png', '24666.png', '12757.png', '06207.png', '03826.png', '24152.png', '05394.png', '22254.png', '08753.png', '24223.png', '08032.png', '18902.png', '21595.png', '13199.png', '17170.png', '10530.png', '12915.png', '06615.png', '19145.png', '18396.png', '08741.png', '19366.png', '13868.png', '01813.png', '18458.png', '00786.png', '09328.png', '09927.png', '04025.png', '23659.png', '10845.png', '06711.png', '12208.png', '23920.png', '15382.png', '02812.png', '22282.png', '18367.png', '19468.png', '02538.png', '21118.png', '11794.png', '18842.png', '15177.png', '13211.png', '06261.png', '08031.png', '16695.png', '15957.png', '13100.png', '12925.png', '13039.png', '13108.png', '24744.png', '08656.png', '23964.png', '24948.png', '16990.png', '09427.png', '22696.png', '01267.png', '00892.png', '01652.png', '12432.png', '24908.png', '10029.png', '22854.png', '17849.png', '01718.png', '01748.png', '11875.png', '02709.png', '07760.png', '04173.png', '23143.png', '16070.png', '19450.png', '04139.png', '12932.png', '02760.png', '09100.png', '05133.png', '24883.png', '23596.png', '18113.png', '09176.png', '08642.png', '24521.png', '22813.png', '10934.png', '14006.png', '11173.png', '23700.png', '14250.png', '14810.png', '22557.png', '05566.png', '17487.png', '21675.png', '19001.png', '18485.png', '19863.png', '18263.png', '03425.png', '06470.png', '07090.png', '24347.png', '06640.png', '10190.png', '19115.png', '05782.png', '24042.png', '19938.png', '08291.png', '14410.png', '08813.png', '12742.png', '13777.png', '06588.png', '03295.png', '05934.png', '04472.png', '23850.png', '03993.png', '06880.png', '19290.png', '22024.png', '05022.png', '11377.png', '00259.png', '09442.png', '19714.png', '15664.png', '21073.png', '10614.png', '24047.png', '05740.png', '23965.png', '22159.png', '07566.png', '11424.png', '15627.png', '08146.png', '06125.png', '14753.png', '11270.png', '11620.png', '13050.png', '21520.png', '06432.png', '00041.png', '20258.png', '23953.png', '04297.png', '06201.png', '19277.png', '04576.png', '06718.png', '19155.png', '18005.png', '03069.png', '07551.png', '01121.png', '00680.png', '00315.png', '14879.png', '09106.png', '14314.png', '05600.png', '08780.png', '00654.png', '21246.png', '03876.png', '16237.png', '07376.png', '15153.png', '18454.png', '12983.png', '14137.png', '19408.png', '04257.png', '03223.png', '15000.png', '10371.png', '22625.png', '01728.png', '00997.png', '03137.png', '05456.png', '11737.png', '19444.png', '12013.png', '11260.png', '00202.png', '05507.png', '15011.png', '22551.png', '12174.png', '24432.png', '01737.png', '18157.png', '18177.png', '10661.png', '21356.png', '03862.png', '04286.png', '08659.png', '23442.png', '15544.png', '03980.png', '06011.png', '00437.png', '23898.png', '03332.png', '13664.png', '24594.png', '16482.png', '12965.png', '23808.png', '19676.png', '14405.png', '22564.png', '08665.png', '11497.png', '20235.png', '00684.png', '22198.png', '09001.png', '06491.png', '22934.png', '09089.png', '05328.png', '24710.png', '12514.png', '15800.png', '04390.png', '06393.png', '24192.png', '16052.png', '16789.png', '14917.png', '12117.png', '03889.png', '09392.png', '04897.png', '16254.png', '16025.png', '22656.png', '02593.png', '05858.png', '01476.png', '23422.png', '01800.png', '13446.png', '06323.png', '14169.png', '11525.png', '06781.png', '10372.png', '22022.png', '19315.png', '05476.png', '11661.png', '08111.png', '01845.png', '17927.png', '17479.png', '06111.png', '23391.png', '17777.png', '10718.png', '24922.png', '18029.png', '17756.png', '10115.png', '17080.png', '05176.png', '16407.png', '07287.png', '12291.png', '05073.png', '14894.png', '03491.png', '02522.png', '08627.png', '15982.png', '07697.png', '20340.png', '00488.png', '24287.png', '09836.png', '23592.png', '19712.png', '17610.png', '19025.png', '20661.png', '23096.png', '12890.png', '03663.png', '16342.png', '08901.png', '07189.png', '18928.png', '18668.png', '06297.png', '22121.png', '17724.png', '24958.png', '24589.png', '23529.png', '01696.png', '13703.png', '22016.png', '18649.png', '18160.png', '06548.png', '02488.png', '11487.png', '16371.png', '20913.png', '22632.png', '21716.png', '12628.png', '14186.png', '19947.png', '07143.png', '07643.png', '11746.png', '13354.png', '09272.png', '14797.png', '15237.png', '00060.png', '00436.png', '19672.png', '23540.png', '18142.png', '14943.png', '13467.png', '21863.png', '22773.png', '07623.png', '11048.png', '04339.png', '09768.png', '01070.png', '01861.png', '22126.png', '03700.png', '16975.png', '08965.png', '07977.png', '13552.png', '03846.png', '20135.png', '24281.png', '15625.png', '08549.png', '14542.png', '08601.png', '04053.png', '03393.png', '09791.png', '19323.png', '12228.png', '09210.png', '24401.png', '19422.png', '05806.png', '12332.png', '14034.png', '19777.png', '02082.png', '10123.png', '18832.png', '06636.png', '16066.png', '02516.png', '12045.png', '19757.png', '23683.png', '04930.png', '17450.png', '24459.png', '16827.png', '03536.png', '23614.png', '05186.png', '06336.png', '00506.png', '23747.png', '05411.png', '13626.png', '03328.png', '11643.png', '19105.png', '11755.png', '20114.png', '14384.png', '05450.png', '15286.png', '03357.png', '03065.png', '12516.png', '22903.png', '23458.png', '00039.png', '24302.png', '15112.png', '17806.png', '04217.png', '03800.png', '04728.png', '12598.png', '06014.png', '19476.png', '16914.png', '00965.png', '19682.png', '16101.png', '01980.png', '03653.png', '01862.png', '23418.png', '23163.png', '12610.png', '06431.png', '01499.png', '15951.png', '14475.png', '05152.png', '20786.png', '12082.png', '23502.png', '04951.png', '20772.png', '21415.png', '10338.png', '11897.png', '18456.png', '24086.png', '23272.png', '11457.png', '22150.png', '21265.png', '01932.png', '14562.png', '20297.png', '01478.png', '19007.png', '03198.png', '21689.png', '09744.png', '15378.png', '18303.png', '07444.png', '07625.png', '07662.png', '04938.png', '11265.png', '01149.png', '21243.png', '21003.png', '06169.png', '02251.png', '09877.png', '14838.png', '12792.png', '04491.png', '01997.png', '09887.png', '10991.png', '13154.png', '21372.png', '12363.png', '13295.png', '01695.png', '12249.png', '16829.png', '00033.png', '06401.png', '07981.png', '18466.png', '07936.png', '02937.png', '13903.png', '18752.png', '15922.png', '17288.png', '10353.png', '19547.png', '08070.png', '12945.png', '15763.png', '13622.png', '15485.png', '13750.png', '00440.png', '17901.png', '14966.png', '10647.png', '24586.png', '18498.png', '01962.png', '23287.png', '20846.png', '05680.png', '08944.png', '13038.png', '18366.png', '24307.png', '03227.png', '08979.png', '04327.png', '21242.png', '20355.png', '14907.png', '18954.png', '03162.png', '14578.png', '19139.png', '10241.png', '15314.png', '23609.png', '03150.png', '17355.png', '03731.png', '01525.png', '24724.png', '10079.png', '07054.png', '22746.png', '12837.png', '18889.png', '16269.png', '22388.png', '06877.png', '03828.png', '00079.png', '03602.png', '22707.png', '23184.png', '03452.png', '14932.png', '06189.png', '10312.png', '17703.png', '03172.png', '00121.png', '15053.png', '07224.png', '06469.png', '09678.png', '18810.png', '19085.png', '10979.png', '12865.png', '09698.png', '09651.png', '13410.png', '16144.png', '14626.png', '21671.png', '21707.png', '24369.png', '11636.png', '02320.png', '14100.png', '20894.png', '10866.png', '00059.png', '15399.png', '14512.png', '10546.png', '00390.png', '06663.png', '06724.png', '23821.png', '08260.png', '04900.png', '19152.png', '12914.png', '00970.png', '06893.png', '18244.png', '06765.png', '23800.png', '17250.png', '14941.png', '23416.png', '15836.png', '24183.png', '10540.png', '02241.png', '03971.png', '00252.png', '07895.png', '02508.png', '11120.png', '02570.png', '02236.png', '13356.png', '15012.png', '16450.png', '10171.png', '07106.png', '20023.png', '20053.png', '15608.png', '01173.png', '01901.png', '08507.png', '02666.png', '14284.png', '23330.png', '17701.png', '08468.png', '15864.png', '02464.png', '20452.png', '22518.png', '14817.png', '01473.png', '13902.png', '14597.png', '22880.png', '23395.png', '08336.png', '15372.png', '10467.png', '13711.png', '24492.png', '09171.png', '20111.png', '23742.png', '22400.png', '11130.png', '01112.png', '07959.png', '18143.png', '20277.png', '19120.png', '00520.png', '18582.png', '19722.png', '19310.png', '20429.png', '15278.png', '01395.png', '16085.png', '06717.png', '13209.png', '08406.png', '24063.png', '04360.png', '07327.png', '20010.png', '21705.png', '03010.png', '23124.png', '17874.png', '14758.png', '23461.png', '22631.png', '10897.png', '00125.png', '22119.png', '00238.png', '23581.png', '11975.png', '24198.png', '09138.png', '17573.png', '17982.png', '09690.png', '00661.png', '20044.png', '10977.png', '08166.png', '20170.png', '11261.png', '02404.png', '19492.png', '21875.png', '14661.png', '20641.png', '07055.png', '19508.png', '02901.png', '08478.png', '12028.png', '14063.png', '08224.png', '20085.png', '18721.png', '13435.png', '24955.png', '03473.png', '00826.png', '15373.png', '02296.png', '08057.png', '05272.png', '14231.png', '22811.png', '17709.png', '09587.png', '05983.png', '18663.png', '17522.png', '04041.png', '24342.png', '13826.png', '07240.png', '02761.png', '11632.png', '11368.png', '06281.png', '04331.png', '10984.png', '17179.png', '19354.png', '17202.png', '02843.png', '15120.png', '06706.png', '01543.png', '17236.png', '07964.png', '01530.png', '05912.png', '21227.png', '14811.png', '22691.png', '20512.png', '06965.png', '05313.png', '12554.png', '03631.png', '13891.png', '16129.png', '14791.png', '20658.png', '04908.png', '24893.png', '15324.png', '05698.png', '23947.png', '10470.png', '09873.png', '21790.png', '07963.png', '01705.png', '16636.png', '15970.png', '01772.png', '15945.png', '20916.png', '24748.png', '20112.png', '05766.png', '03866.png', '07909.png', '05977.png', '22665.png', '00273.png', '21542.png', '24794.png', '18349.png', '10252.png', '08126.png', '24516.png', '09930.png', '01973.png', '04482.png', '15634.png', '22271.png', '11443.png', '05901.png', '15596.png', '08822.png', '05431.png', '10532.png', '16044.png', '21948.png', '03748.png', '09539.png', '18378.png', '09147.png', '15744.png', '22644.png', '22136.png', '02842.png', '12018.png', '10282.png', '23758.png', '05240.png', '00098.png', '19543.png', '02840.png', '13125.png', '03337.png', '05603.png', '13600.png', '15769.png', '14065.png', '02742.png', '13912.png', '03952.png', '03225.png', '08191.png', '13629.png', '01817.png', '19337.png', '12968.png', '07256.png', '05700.png', '20095.png', '20379.png', '07250.png', '18229.png', '07033.png', '09232.png', '21572.png', '24270.png', '17859.png', '04883.png', '19239.png', '23746.png', '00137.png', '21080.png', '04540.png', '23101.png', '23556.png', '01827.png', '10851.png', '20353.png', '09757.png', '09779.png', '15199.png', '00548.png', '18893.png', '17711.png', '00460.png', '06233.png', '09083.png', '03457.png', '13130.png', '02016.png', '06987.png', '16214.png', '17546.png', '05013.png', '09594.png', '04054.png', '14945.png', '04545.png', '10729.png', '12120.png', '00266.png', '18365.png', '14437.png', '14158.png', '14637.png', '02854.png', '06764.png', '04610.png', '10422.png', '20432.png', '23933.png', '01836.png', '17163.png', '01152.png', '16020.png', '13658.png', '20581.png', '24024.png', '24188.png', '21808.png', '02509.png', '11791.png', '09997.png', '12337.png', '05157.png', '11249.png', '05347.png', '03931.png', '18900.png', '23193.png', '08575.png', '20872.png', '03830.png', '17175.png', '06589.png', '09545.png', '24952.png', '22598.png', '23802.png', '09255.png', '13822.png', '00650.png', '16625.png', '08827.png', '19125.png', '17973.png', '08719.png', '19269.png', '24761.png', '18186.png', '00281.png', '13201.png', '16157.png', '08911.png', '08119.png', '02090.png', '10219.png', '23378.png', '03204.png', '02993.png', '18913.png', '14418.png', '11872.png', '14147.png', '18444.png', '10283.png', '08292.png', '19696.png', '10483.png', '13109.png', '23086.png', '07257.png', '06743.png', '20115.png', '21041.png', '12313.png', '17307.png', '01624.png', '01053.png', '14399.png', '03110.png', '18352.png', '13975.png', '18088.png', '21225.png', '21144.png', '17016.png', '09250.png', '14818.png', '24869.png', '16410.png', '13632.png', '02394.png', '15782.png', '03135.png', '14202.png', '03351.png', '06356.png', '09633.png', '05607.png', '13960.png', '03917.png', '23399.png', '22485.png', '05404.png', '07026.png', '02243.png', '02250.png', '13619.png', '01321.png', '10886.png', '12765.png', '03335.png', '11462.png', '00184.png', '15921.png', '21074.png', '17690.png', '00042.png', '01926.png', '24451.png', '02477.png', '14771.png', '20947.png', '13956.png', '06752.png', '09218.png', '12747.png', '10038.png', '13294.png', '02096.png', '21358.png', '08400.png', '24542.png', '20061.png', '20510.png', '08214.png', '22406.png', '21541.png', '24381.png', '01835.png', '13335.png', '17819.png', '03402.png', '00972.png', '08436.png', '11136.png', '24929.png', '11676.png', '04278.png', '07862.png', '03372.png', '17844.png', '00335.png', '02519.png', '14468.png', '21511.png', '04777.png', '02314.png', '09179.png', '15103.png', '23055.png', '01851.png', '20901.png', '05076.png', '03053.png', '13978.png', '18596.png', '15967.png', '04130.png', '04799.png', '01886.png', '11678.png', '16924.png', '04936.png', '19046.png', '18526.png', '15073.png', '09428.png', '24657.png', '09093.png', '17349.png', '09264.png', '15227.png', '20575.png', '11610.png', '05381.png', '09205.png', '08934.png', '05366.png', '21878.png', '15381.png', '24165.png', '14777.png', '13479.png', '20148.png', '07988.png', '14270.png', '08852.png', '02463.png', '10531.png', '04982.png', '23595.png', '07482.png', '07779.png', '00698.png', '17470.png', '05592.png', '02101.png', '06274.png', '00418.png', '11208.png', '18379.png', '07810.png', '08299.png', '23632.png', '05048.png', '07075.png', '15021.png', '03120.png', '11590.png', '03463.png', '05225.png', '02297.png', '12322.png', '03322.png', '11642.png', '20822.png', '16520.png', '21085.png', '13738.png', '02085.png', '03355.png', '06565.png', '06839.png', '01073.png', '15032.png', '22002.png', '12468.png', '07029.png', '09652.png', '17934.png', '08816.png', '17440.png', '15380.png', '23903.png', '20099.png', '04070.png', '03275.png', '04468.png', '00894.png', '08523.png', '14603.png', '09627.png', '18920.png', '13936.png', '10168.png', '09200.png', '15601.png', '21525.png', '03502.png', '18124.png', '14301.png', '07397.png', '21604.png', '13686.png', '08777.png', '03863.png', '00882.png', '14074.png', '05455.png', '01773.png', '11957.png', '03697.png', '01233.png', '17996.png', '21638.png', '07367.png', '06080.png', '14750.png', '16041.png', '08087.png', '09072.png', '06902.png', '19958.png', '01566.png', '08727.png', '08388.png', '24954.png', '23431.png', '22688.png', '02762.png', '11748.png', '08504.png', '22711.png', '07899.png', '01194.png', '23793.png', '01720.png', '17728.png', '19567.png', '02242.png', '20090.png', '10641.png', '21191.png', '18783.png', '19441.png', '09671.png', '21906.png', '21635.png', '16790.png', '20234.png', '20238.png', '06844.png', '23494.png', '09186.png', '24266.png', '20896.png', '07593.png', '11391.png', '11449.png', '05309.png', '19123.png', '18556.png', '08249.png', '24831.png', '21168.png', '07105.png', '03823.png', '21661.png', '02659.png', '11947.png', '14366.png', '20381.png', '04921.png', '23797.png', '04444.png', '07904.png', '20556.png', '23673.png', '04005.png', '13672.png', '22226.png', '01387.png', '13766.png', '08918.png', '00353.png', '18782.png', '11573.png', '20458.png', '24335.png', '17837.png', '18481.png', '22620.png', '01676.png', '21641.png', '03703.png', '24380.png', '24110.png', '19740.png', '22257.png', '04675.png', '14334.png', '01137.png', '02344.png', '01649.png', '06317.png', '11215.png', '04963.png', '19700.png', '03674.png', '17091.png', '13391.png', '13191.png', '08709.png', '19184.png', '24038.png', '13198.png', '00866.png', '22072.png', '23047.png', '23350.png', '07825.png', '09837.png', '08914.png', '18452.png', '14003.png', '01569.png', '21296.png', '00783.png', '13959.png', '10854.png', '04074.png', '09712.png', '00075.png', '06525.png', '21996.png', '01519.png', '10872.png', '05318.png', '09622.png', '15403.png', '05991.png', '18374.png', '11099.png', '08276.png', '09647.png', '21390.png', '16936.png', '18285.png', '21840.png', '04028.png', '21100.png', '02321.png', '12143.png', '02749.png', '21380.png', '14543.png', '16440.png', '13161.png', '13376.png', '05153.png', '07928.png', '13981.png', '19827.png', '04380.png', '04146.png', '13141.png', '17650.png', '17460.png', '23221.png', '12840.png', '15647.png', '09225.png', '22808.png', '09966.png', '21200.png', '19723.png', '07841.png', '06142.png', '18101.png', '22555.png', '24288.png', '15467.png', '05923.png', '09762.png', '23178.png', '12486.png', '21857.png', '02679.png', '12755.png', '22365.png', '07080.png', '21424.png', '12402.png', '14130.png', '23649.png', '07215.png', '11360.png', '10334.png', '09410.png', '24878.png', '07631.png', '22701.png', '21278.png', '01176.png', '11809.png', '13313.png', '18623.png', '10534.png', '03104.png', '18718.png', '17892.png', '00494.png', '18399.png', '16253.png', '16834.png', '11569.png', '24933.png', '13899.png', '11129.png', '15337.png', '00839.png', '11400.png', '10390.png', '09504.png', '23282.png', '15910.png', '16055.png', '15591.png', '19821.png', '01777.png', '02700.png', '24856.png', '15427.png', '22778.png', '23497.png', '13383.png', '02280.png', '20795.png', '05121.png', '23515.png', '20809.png', '18506.png', '11798.png', '23847.png', '15639.png', '08596.png', '20442.png', '16679.png', '24084.png', '17538.png', '04847.png', '15727.png', '06127.png', '16108.png', '16418.png', '16059.png', '16592.png', '08337.png', '12297.png', '14935.png', '08331.png', '12879.png', '24842.png', '04319.png', '15635.png', '15629.png', '06075.png', '24476.png', '08109.png', '18725.png', '00667.png', '01041.png', '03542.png', '21369.png', '15603.png', '23121.png', '24258.png', '21173.png', '24785.png', '15947.png', '23266.png', '05508.png', '05485.png', '17426.png', '15276.png', '07237.png', '18410.png', '19919.png', '24035.png', '13500.png', '16116.png', '18998.png', '05295.png', '03094.png', '00502.png', '08435.png', '05415.png', '11316.png', '01037.png', '14826.png', '02855.png', '02902.png', '13878.png', '22157.png', '05696.png', '04617.png', '20150.png', '15429.png', '14050.png', '08402.png', '10298.png', '12559.png', '05828.png', '22803.png', '02898.png', '05101.png', '16136.png', '20618.png', '05708.png', '11157.png', '04406.png', '02193.png', '20336.png', '16045.png', '20165.png', '07153.png', '21492.png', '05058.png', '01971.png', '00374.png', '17033.png', '18979.png', '04554.png', '02553.png', '10816.png', '22010.png', '02806.png', '09502.png', '24538.png', '19382.png', '12441.png', '22326.png', '01953.png', '09703.png', '21232.png', '21079.png', '09184.png', '15175.png', '06123.png', '07269.png', '02026.png', '08151.png', '14646.png', '03060.png', '05268.png', '09680.png', '00010.png', '12982.png', '02793.png', '04754.png', '16805.png', '09727.png', '24651.png', '09738.png', '13759.png', '11973.png', '01691.png', '02795.png', '10555.png', '13740.png', '01646.png', '06759.png', '18860.png', '07078.png', '11686.png', '12242.png', '07259.png', '02056.png', '19840.png', '00633.png', '20366.png', '05135.png', '04308.png', '03652.png', '24207.png', '05647.png', '01237.png', '05464.png', '24202.png', '09880.png', '03071.png', '19699.png', '24157.png', '18839.png', '03610.png', '00546.png', '16114.png', '13537.png', '05478.png', '09635.png', '20493.png', '06593.png', '05199.png', '12842.png', '21961.png', '07045.png', '04713.png', '01416.png', '21522.png', '13481.png', '12280.png', '05553.png', '07943.png', '05196.png', '21354.png', '09859.png', '12039.png', '17232.png', '17383.png', '01690.png', '05894.png', '23495.png', '18383.png', '03381.png', '07316.png', '03279.png', '03432.png', '17005.png', '21915.png', '20824.png', '08754.png', '01430.png', '03901.png', '11216.png', '20734.png', '22682.png', '03644.png', '01605.png', '19610.png', '18215.png', '11527.png', '22402.png', '07838.png', '03462.png', '11372.png', '12460.png', '24509.png', '19972.png', '18981.png', '17851.png', '19499.png', '00751.png', '03905.png', '03503.png', '15455.png', '14731.png', '02106.png', '04792.png', '15558.png', '02609.png', '21038.png', '19906.png', '24169.png', '12420.png', '14044.png', '17662.png', '09526.png', '12969.png', '03633.png', '15904.png', '20557.png', '09215.png', '23722.png', '09517.png', '20733.png', '11107.png', '19965.png', '21508.png', '09370.png', '13192.png', '19208.png', '02956.png', '03406.png', '19639.png', '14119.png', '23934.png', '04944.png', '01586.png', '11247.png', '05678.png', '13772.png', '08433.png', '15089.png', '08508.png', '22357.png', '21072.png', '00213.png', '14614.png', '21310.png', '12829.png', '03843.png', '23255.png', '16775.png', '15281.png', '16009.png', '14095.png', '14381.png', '03115.png', '16347.png', '16488.png', '10591.png', '13478.png', '01518.png', '11960.png', '14392.png', '18883.png', '10122.png', '11102.png', '15259.png', '14487.png', '07083.png', '17943.png', '05556.png', '18307.png', '09062.png', '13184.png', '07627.png', '24502.png', '13462.png', '13559.png', '10321.png', '01708.png', '21538.png', '04964.png', '04927.png', '04398.png', '01759.png', '04419.png', '12872.png', '23608.png', '22011.png', '20016.png', '00681.png', '02935.png', '02692.png', '22261.png', '03649.png', '09008.png', '11812.png', '07234.png', '03789.png', '03783.png', '03469.png', '13890.png', '07672.png', '12956.png', '07359.png', '04984.png', '00591.png', '01275.png', '03418.png', '04001.png', '14933.png', '21868.png', '16038.png', '06110.png', '10732.png', '02173.png', '10449.png', '07196.png', '17287.png', '09623.png', '07875.png', '07800.png', '13052.png', '00270.png', '18547.png', '02547.png', '05970.png', '00427.png', '02183.png', '11004.png', '10169.png', '13935.png', '24575.png', '10940.png', '03201.png', '16591.png', '09832.png', '15901.png', '19843.png', '15655.png', '12488.png', '00889.png', '02960.png', '02146.png', '18382.png', '01268.png', '13186.png', '22438.png', '24715.png', '17707.png', '04448.png', '11435.png', '07694.png', '18089.png', '22501.png', '17011.png', '13196.png', '18181.png', '20552.png', '23019.png', '20515.png', '01162.png', '05967.png', '01758.png', '23066.png', '11874.png', '13298.png', '16445.png', '15062.png', '16419.png', '16422.png', '17966.png', '01947.png', '14051.png', '03257.png', '04099.png', '15937.png', '09437.png', '19456.png', '23557.png', '04004.png', '23995.png', '17365.png', '03342.png', '09527.png', '10565.png', '10994.png', '10250.png', '14892.png', '11692.png', '24221.png', '23390.png', '15631.png', '20142.png', '20636.png', '04915.png', '10448.png', '00332.png', '20110.png', '03669.png', '01908.png', '24635.png', '01786.png', '13864.png', '12869.png', '24540.png', '02888.png', '21440.png', '00351.png', '20781.png', '15720.png', '10911.png', '11296.png', '08138.png', '13682.png', '18222.png', '04629.png', '18933.png', '19899.png', '18006.png', '11164.png', '11150.png', '07765.png', '18234.png', '00384.png', '06265.png', '02703.png', '10629.png', '17852.png', '23646.png', '00179.png', '18373.png', '11604.png', '15824.png', '10238.png', '04819.png', '12740.png', '22384.png', '13208.png', '13296.png', '15579.png', '01325.png', '24026.png', '22068.png', '11653.png', '06543.png', '01477.png', '18019.png', '10384.png', '18771.png', '22403.png', '11550.png', '10074.png', '13892.png', '13934.png', '04925.png', '05372.png', '23239.png', '24711.png', '19342.png', '01991.png', '17058.png', '07733.png', '15515.png', '17879.png', '23627.png', '08056.png', '17759.png', '04003.png', '01445.png', '09862.png', '16714.png', '02164.png', '05918.png', '02644.png', '00409.png', '02451.png', '19905.png', '08063.png', '05124.png', '14653.png', '12527.png', '22095.png', '08227.png', '10680.png', '12722.png', '17427.png', '12473.png', '15019.png', '22412.png', '08252.png', '23968.png', '04239.png', '14136.png', '07583.png', '03695.png', '16468.png', '14899.png', '20761.png', '22266.png', '19400.png', '11056.png', '07680.png', '10153.png', '23740.png', '13373.png', '09543.png', '18039.png', '01790.png', '21271.png', '03628.png', '08633.png', '06426.png', '17346.png', '01738.png', '04601.png', '05218.png', '13921.png', '23653.png', '16937.png', '08610.png', '07249.png', '06032.png', '08684.png', '09261.png', '19081.png', '09640.png', '15442.png', '16094.png', '00524.png', '06808.png', '05634.png', '05808.png', '22331.png', '11206.png', '22667.png', '07424.png', '05031.png', '01724.png', '06151.png', '02117.png', '09042.png', '00761.png', '13072.png', '12620.png', '04888.png', '22481.png', '01638.png', '02556.png', '00309.png', '03061.png', '05060.png', '04827.png', '12633.png', '00670.png', '14351.png', '03543.png', '23709.png', '13034.png', '15548.png', '00217.png', '09206.png', '00195.png', '15989.png', '03530.png', '16598.png', '05590.png', '08313.png', '21061.png', '01460.png', '00555.png', '23286.png', '16858.png', '00953.png', '10763.png', '06230.png', '10640.png', '23720.png', '05441.png', '13146.png', '11416.png', '09807.png', '08803.png', '03906.png', '23617.png', '24838.png', '17884.png', '05814.png', '18574.png', '09917.png', '15167.png', '13565.png', '02569.png', '08833.png', '00763.png', '21084.png', '22498.png', '16859.png', '14015.png', '17639.png', '21188.png', '13646.png', '02223.png', '04661.png', '00521.png', '23384.png', '03013.png', '07303.png', '00709.png', '12443.png', '02385.png', '09569.png', '22542.png', '14059.png', '09509.png', '01761.png', '15306.png', '02441.png', '02813.png', '03176.png', '23568.png', '05978.png', '05181.png', '21129.png', '21866.png', '09397.png', '13333.png', '00003.png', '07020.png', '04038.png', '18300.png', '09320.png', '17762.png', '24626.png', '18688.png', '15640.png', '10340.png', '10255.png', '04695.png', '02610.png', '06980.png', '06683.png', '17739.png', '24228.png', '02774.png', '15317.png', '08218.png', '05224.png', '00355.png', '04083.png', '04417.png', '02400.png', '12853.png', '02417.png', '19071.png', '07913.png', '23677.png', '12007.png', '13411.png', '07296.png', '06568.png', '10741.png', '05837.png', '11025.png', '07567.png', '09374.png', '01792.png', '06691.png', '09732.png', '11627.png', '02452.png', '02091.png', '13414.png', '12391.png', '24640.png', '02759.png', '08037.png', '02944.png', '12900.png', '11671.png', '03209.png', '09528.png', '09379.png', '02323.png', '22909.png', '23697.png', '08326.png', '24508.png', '14652.png', '07941.png', '02307.png', '21322.png', '03029.png', '22081.png', '19336.png', '01837.png', '12664.png', '05443.png', '15872.png', '04761.png', '15959.png', '21344.png', '02909.png', '14988.png', '05237.png', '11345.png', '07723.png', '02045.png', '19295.png', '14077.png', '00120.png', '22841.png', '20439.png', '11658.png', '23364.png', '05391.png', '13366.png', '12526.png', '07542.png', '08307.png', '23510.png', '03095.png', '18084.png', '10880.png', '05514.png', '12194.png', '00920.png', '15612.png', '12400.png', '01752.png', '00946.png', '15303.png', '23895.png', '05674.png', '01383.png', '10584.png', '08844.png', '00150.png', '09557.png', '02828.png', '17945.png', '03940.png', '24811.png', '06182.png', '06659.png', '04002.png', '11022.png', '14151.png', '07685.png', '08442.png', '22116.png', '12246.png', '03814.png', '18969.png', '06190.png', '21630.png', '23146.png', '12124.png', '21239.png', '23505.png', '00264.png', '16332.png', '19346.png', '10008.png', '05884.png', '00171.png', '15855.png', '10605.png', '21163.png', '09665.png', '05872.png', '10189.png', '08205.png', '04835.png', '06773.png', '06594.png', '24234.png', '16892.png', '13021.png', '11925.png', '19662.png', '01004.png', '21250.png', '24596.png', '19529.png', '14047.png', '18714.png', '01987.png', '22970.png', '05029.png', '12507.png', '03802.png', '06096.png', '13227.png', '19062.png', '18625.png', '13429.png', '06523.png', '23737.png', '22207.png', '17093.png', '10588.png', '21999.png', '04876.png', '08955.png', '09490.png', '19600.png', '17160.png', '04227.png', '07349.png', '19084.png', '21392.png', '08853.png', '24317.png', '11840.png', '19262.png', '19016.png', '04793.png', '22735.png', '02681.png', '08194.png', '21923.png', '21573.png', '16765.png', '24339.png', '15771.png', '00677.png', '24005.png', '13885.png', '15213.png', '18381.png', '23911.png', '04177.png', '03713.png', '07816.png', '21585.png', '10346.png', '14213.png', '20005.png', '00481.png', '19494.png', '15290.png', '05429.png', '10196.png', '19263.png', '01830.png', '13425.png', '12981.png', '00995.png', '16189.png', '10224.png', '11618.png', '09462.png', '08265.png', '17249.png', '01935.png', '09486.png', '23438.png', '11717.png', '19736.png', '01457.png', '02229.png', '03320.png', '24890.png', '14091.png', '15225.png', '02989.png', '04574.png', '02733.png', '08706.png', '21487.png', '17902.png', '06070.png', '09191.png', '15651.png', '08959.png', '03724.png', '00638.png', '15387.png', '00915.png', '16663.png', '16597.png', '08648.png', '10350.png', '24240.png', '12979.png', '06539.png', '23140.png', '23611.png', '04250.png', '04254.png', '05605.png', '23312.png', '08132.png', '09457.png', '08691.png', '21791.png', '24786.png', '01118.png', '10728.png', '23230.png', '08184.png', '23191.png', '07313.png', '06033.png', '13534.png', '10361.png', '12483.png', '07152.png', '01933.png', '21579.png', '23553.png', '06562.png', '06319.png', '15512.png', '17390.png', '01865.png', '03767.png', '07833.png', '03126.png', '13642.png', '24321.png', '20187.png', '18644.png', '19497.png', '01278.png', '24537.png', '10848.png', '06273.png', '05920.png', '20631.png', '02818.png', '22153.png', '07013.png', '16362.png', '19440.png', '07320.png', '05637.png', '06894.png', '15291.png', '01682.png', '10746.png', '10320.png', '16460.png', '17240.png', '14664.png', '00636.png', '12558.png', '01588.png', '04628.png', '23177.png', '08939.png', '13168.png', '09090.png', '21972.png', '17747.png', '24080.png', '02355.png', '03897.png', '14643.png', '12470.png', '19752.png', '13610.png', '05916.png', '07547.png', '04341.png', '12151.png', '01250.png', '03241.png', '04092.png', '01282.png', '23572.png', '09950.png', '05003.png', '09777.png', '22533.png', '14373.png', '22177.png', '00860.png', '18165.png', '14855.png', '22964.png', '00911.png', '19573.png', '05876.png', '14831.png', '05025.png', '04049.png', '10136.png', '09306.png', '14547.png', '03008.png', '02845.png', '16555.png', '08049.png', '15343.png', '16067.png', '24672.png', '18509.png', '21888.png', '11468.png', '10941.png', '19307.png', '05422.png', '14128.png', '00903.png', '06921.png', '23823.png', '19636.png', '10528.png', '15257.png', '15313.png', '10210.png', '00792.png', '03919.png', '20926.png', '03973.png', '17615.png', '22158.png', '23828.png', '13472.png', '07399.png', '14339.png', '23527.png', '07260.png', '22120.png', '10272.png', '16547.png', '09120.png', '05472.png', '02206.png', '07863.png', '06508.png', '12975.png', '10751.png', '17583.png', '06227.png', '11946.png', '12127.png', '05672.png', '03847.png', '00343.png', '10637.png', '23867.png', '08679.png', '14701.png', '02860.png', '18435.png', '18449.png', '18027.png', '21618.png', '14425.png', '05015.png', '07764.png', '16543.png', '18906.png', '05162.png', '01396.png', '17002.png', '11939.png', '19226.png', '00941.png', '14712.png', '23506.png', '10703.png', '00048.png', '15636.png', '08458.png', '12147.png', '12466.png', '23852.png', '01015.png', '02405.png', '17351.png', '19387.png', '11335.png', '19233.png', '15020.png', '09608.png', '13205.png', '18442.png', '13941.png', '10416.png', '21212.png', '12910.png', '20582.png', '01907.png', '07185.png', '18765.png', '19668.png', '05932.png', '10076.png', '16494.png', '11465.png', '06143.png', '24790.png', '02715.png', '11683.png', '07393.png', '18622.png', '13964.png', '01707.png', '02342.png', '12744.png', '14201.png', '16313.png', '24702.png', '11097.png', '21887.png', '06750.png', '11460.png', '21053.png', '18093.png', '06528.png', '13375.png', '17676.png', '12978.png', '14622.png', '12205.png', '11073.png', '13493.png', '18402.png', '12572.png', '11970.png', '17669.png', '16188.png', '04291.png', '21534.png', '03404.png', '04031.png', '15200.png', '15623.png', '00452.png', '22515.png', '19992.png', '24939.png', '05874.png', '15436.png', '24030.png', '12138.png', '21932.png', '18597.png', '17793.png', '18034.png', '22698.png', '17364.png', '07241.png', '14863.png', '14959.png', '01677.png', '16028.png', '01505.png', '11248.png', '09817.png', '21583.png', '02558.png', '03187.png', '23532.png', '05447.png', '08681.png', '20627.png', '10102.png', '24514.png', '16056.png', '19760.png', '19405.png', '09027.png', '19669.png', '23195.png', '09226.png', '10944.png', '04660.png', '18635.png', '07211.png', '19861.png', '23073.png', '06725.png', '16707.png', '21912.png', '03290.png', '09597.png', '16630.png', '13834.png', '18732.png', '20867.png', '04047.png', '18911.png', '15621.png', '06904.png', '08513.png', '00067.png', '08723.png', '03211.png', '23523.png', '06423.png', '23569.png', '02460.png', '23153.png', '22824.png', '18301.png', '09422.png', '07650.png', '20079.png', '14645.png', '00373.png', '01542.png', '10622.png', '01969.png', '17608.png', '04174.png', '19191.png', '23432.png', '06628.png', '11233.png', '13968.png', '15971.png', '07539.png', '03326.png', '07796.png', '09669.png', '12883.png', '09765.png', '20668.png', '11998.png', '22737.png', '12950.png', '05234.png', '05689.png', '14560.png', '17512.png', '24528.png', '22358.png', '13980.png', '15207.png', '09022.png', '21640.png', '24833.png', '12485.png', '13238.png', '05503.png', '21550.png', '08308.png', '11317.png', '08752.png', '16504.png', '12732.png', '08333.png', '18420.png', '02304.png', '17219.png', '08226.png', '21549.png', '07428.png', '09800.png', '15385.png', '04469.png', '13173.png', '20225.png', '14965.png', '10707.png', '12533.png', '09188.png', '15470.png', '18525.png', '19403.png', '05499.png', '20979.png', '05683.png', '01025.png', '07538.png', '19681.png', '23078.png', '20507.png', '11639.png', '11470.png', '16336.png', '05436.png', '18529.png', '20214.png', '04802.png', '08501.png', '20188.png', '15344.png', '18287.png', '14677.png', '23534.png', '00157.png', '23914.png', '08891.png', '23813.png', '03769.png', '03231.png', '12835.png', '04779.png', '22955.png', '19759.png', '18823.png', '19041.png', '16583.png', '18325.png', '16472.png', '24926.png', '21409.png', '14990.png', '15832.png', '23949.png', '20671.png', '18947.png', '11095.png', '21802.png', '04261.png', '18841.png', '11549.png', '09347.png', '11799.png', '05522.png', '04305.png', '13613.png', '02262.png', '19270.png', '05754.png', '22583.png', '20856.png', '18940.png', '22381.png', '03100.png', '18776.png', '15620.png', '13502.png', '00444.png', '00632.png', '07656.png', '10352.png', '12916.png', '15875.png', '20984.png', '23268.png', '11603.png', '01557.png', '10328.png', '18590.png', '09759.png', '16820.png', '19839.png', '16531.png', '20891.png', '20488.png', '05307.png', '12608.png', '19369.png', '16946.png', '12773.png', '18363.png', '05608.png', '22187.png', '16246.png', '24470.png', '06410.png', '13219.png', '03885.png', '17644.png', '08932.png', '20043.png', '21834.png', '02378.png', '04886.png', '01611.png', '03152.png', '07934.png', '10713.png', '02009.png', '16411.png', '12736.png', '05483.png', '12076.png', '12991.png', '22289.png', '24963.png', '00083.png', '16486.png', '07145.png', '22514.png', '24578.png', '02282.png', '09586.png', '13821.png', '04231.png', '00901.png', '13357.png', '19140.png', '20642.png', '13004.png', '14938.png', '08630.png', '00327.png', '08863.png', '20144.png', '10075.png', '05419.png', '21530.png', '11984.png', '23840.png', '13359.png', '08100.png', '00902.png', '19687.png', '02790.png', '16537.png', '10690.png', '18697.png', '23277.png', '02494.png', '23674.png', '16240.png', '20808.png', '04587.png', '12344.png', '12399.png', '00386.png', '17653.png', '21394.png', '10437.png', '01222.png', '12962.png', '21841.png', '19469.png', '01124.png', '05904.png', '17770.png', '01976.png', '18413.png', '08354.png', '14486.png', '20714.png', '23310.png', '08036.png', '05201.png', '01040.png', '24148.png', '12655.png', '19437.png', '13409.png', '07671.png', '08531.png', '22591.png', '22263.png', '22925.png', '16680.png', '24001.png', '23815.png', '20967.png', '12771.png', '17689.png', '19887.png', '23136.png', '20262.png', '02416.png', '07621.png', '18856.png', '21712.png', '20199.png', '02630.png', '07405.png', '14620.png', '11046.png', '06803.png', '22082.png', '21190.png', '13505.png', '05350.png', '17704.png', '11853.png', '08572.png', '10037.png', '07739.png', '04889.png', '22972.png', '02769.png', '06023.png', '07577.png', '15811.png', '03577.png', '07066.png', '01461.png', '23838.png', '04673.png', '11050.png', '24579.png', '05947.png', '16502.png', '12204.png', '03255.png', '09057.png', '06017.png', '12674.png', '21540.png', '16618.png', '20419.png', '08407.png', '14684.png', '24758.png', '17196.png', '13743.png', '14173.png', '11273.png', '24370.png', '05709.png', '03922.png', '13906.png', '03323.png', '13562.png', '17997.png', '24827.png', '15425.png', '07235.png', '01213.png', '07967.png', '12697.png', '24263.png', '06531.png', '20697.png', '21459.png', '12955.png', '15746.png', '00396.png', '11728.png', '11199.png', '10930.png', '01101.png', '18944.png', '10443.png', '04737.png', '20995.png', '17599.png', '22789.png', '20877.png', '00919.png', '05289.png', '09515.png', '17414.png', '18735.png', '01719.png', '05966.png', '22686.png', '12290.png', '12448.png', '07990.png', '06051.png', '15618.png', '11414.png', '21311.png', '14225.png', '00872.png', '09217.png', '20520.png', '04790.png', '00145.png', '15749.png', '19111.png', '14276.png', '23774.png', '18082.png', '18656.png', '00615.png', '18840.png', '11780.png', '11098.png', '23267.png', '02001.png', '00744.png', '11420.png', '00653.png', '04932.png', '00312.png', '09302.png', '22587.png', '03556.png', '02338.png', '04364.png', '07587.png', '09161.png', '21393.png', '02064.png', '00906.png', '03194.png', '01432.png', '10666.png', '08539.png', '22553.png', '09394.png', '05612.png', '06451.png', '06061.png', '20643.png', '14552.png', '04596.png', '23394.png', '01446.png', '00104.png', '14431.png', '09687.png', '21463.png', '05043.png', '17312.png', '13880.png', '20670.png', '20309.png', '17081.png', '18166.png', '06392.png', '01177.png', '09683.png', '10400.png', '23238.png', '09938.png', '16431.png', '08351.png', '10280.png', '24518.png', '18196.png', '16373.png', '07469.png', '23355.png', '09400.png', '14820.png', '15611.png', '22437.png', '01905.png', '09458.png', '10553.png', '23246.png', '22033.png', '01679.png', '08978.png', '07750.png', '17520.png', '05627.png', '05649.png', '22026.png', '06813.png', '07896.png', '21112.png', '07332.png', '03793.png', '03369.png', '14501.png', '08521.png', '07782.png', '01054.png', '06064.png', '15450.png', '04351.png', '22645.png', '15282.png', '01995.png', '06232.png', '17912.png', '17177.png', '10781.png', '05210.png', '00441.png', '12501.png', '11165.png', '17647.png', '07746.png', '06624.png', '07737.png', '21348.png', '20571.png', '09445.png', '17129.png', '14981.png', '09957.png', '23421.png', '16750.png', '05869.png', '20529.png', '02626.png', '02963.png', '15264.png', '08635.png', '04475.png', '23485.png', '22967.png', '11425.png', '24085.png', '15583.png', '12711.png', '23784.png', '07543.png', '06092.png', '01287.png', '14004.png', '04048.png', '16971.png', '12218.png', '11011.png', '07646.png', '11384.png', '01159.png', '19273.png', '07059.png', '24153.png', '07039.png', '22477.png', '08690.png', '10045.png', '17227.png', '10020.png', '21052.png', '23831.png', '11625.png', '03117.png', '13754.png', '10836.png', '01180.png', '03093.png', '20712.png', '07310.png', '04259.png', '05204.png', '09207.png', '14058.png', '18128.png', '14303.png', '20413.png', '21881.png', '19726.png', '04375.png', '13805.png', '21864.png', '18769.png', '13882.png', '16353.png', '11472.png', '11521.png', '20386.png', '20922.png', '04167.png', '13165.png', '11313.png', '11366.png', '10424.png', '14978.png', '20198.png', '18825.png', '12458.png', '22562.png', '13601.png', '07720.png', '11530.png', '10688.png', '09572.png', '24914.png', '08090.png', '00011.png', '15255.png', '03081.png', '09595.png', '05768.png', '07986.png', '16259.png', '09010.png', '21117.png', '01896.png', '23094.png', '19268.png', '15632.png', '06991.png', '07411.png', '15426.png', '02212.png', '17021.png', '02820.png', '07526.png', '08190.png', '21682.png', '12251.png', '08395.png', '13712.png', '18460.png', '21778.png', '23801.png', '18989.png', '06587.png', '21913.png', '09705.png', '07161.png', '24029.png', '09481.png', '17962.png', '10630.png', '17735.png', '23250.png', '15775.png', '23837.png', '22319.png', '07914.png', '18309.png', '09041.png', '05479.png', '16855.png', '19690.png', '15261.png', '24109.png', '17588.png', '05283.png', '00221.png', '15061.png', '02187.png', '14846.png', '03550.png', '09233.png', '03733.png', '23010.png', '16912.png', '10105.png', '21763.png', '11909.png', '21182.png', '17569.png', '01151.png', '06427.png', '22874.png', '04478.png', '17371.png', '12866.png', '08089.png', '23334.png', '07232.png', '07162.png', '05164.png', '23971.png', '09204.png', '04209.png', '13532.png', '00701.png', '04569.png', '13929.png', '23978.png', '21433.png', '08169.png', '08692.png', '18076.png', '15565.png', '24605.png', '11581.png', '06103.png', '15955.png', '11633.png', '00564.png', '18189.png', '19391.png', '04841.png', '14029.png', '06951.png', '19485.png', '02982.png', '14012.png', '23231.png', '17362.png', '16193.png', '19109.png', '12137.png', '09826.png', '09911.png', '14628.png', '01867.png', '21090.png', '03775.png', '00526.png', '17672.png', '17140.png', '00475.png', '01945.png', '19875.png', '19680.png', '11749.png', '10489.png', '19845.png', '11814.png', '20500.png', '12952.png', '03742.png', '05586.png', '07788.png', '05014.png', '09230.png', '10829.png', '15514.png', '01226.png', '16839.png', '16920.png', '10239.png', '01345.png', '07058.png', '16822.png', '13839.png', '24280.png', '14695.png', '12065.png', '04098.png', '20883.png', '15930.png', '18145.png', '13401.png', '05288.png', '02050.png', '15318.png', '23868.png', '21733.png', '00403.png', '03078.png', '06171.png', '00482.png', '12949.png', '09626.png', '24439.png', '17947.png', '23560.png', '07784.png', '22677.png', '09996.png', '19421.png', '07669.png', '24610.png', '04506.png', '16427.png', '05589.png', '15415.png', '02027.png', '06203.png', '13271.png', '21816.png', '21920.png', '09646.png', '14875.png', '12880.png', '17378.png', '18298.png', '23125.png', '16003.png', '08106.png', '07814.png', '21197.png', '14798.png', '15145.png', '16972.png', '24798.png', '00750.png', '06602.png', '00716.png', '00975.png', '17270.png', '13484.png', '02010.png', '03145.png', '12279.png', '15271.png', '10773.png', '12358.png', '07091.png', '01314.png', '02543.png', '16503.png', '22433.png', '05393.png', '14638.png', '18930.png', '10359.png', '13995.png', '14200.png', '07067.png', '14680.png', '00394.png', '01072.png', '02893.png', '06869.png', '11932.png', '18560.png', '23518.png', '06395.png', '03975.png', '18984.png', '16323.png', '10990.png', '14526.png', '16267.png', '03191.png', '16456.png', '22284.png', '17805.png', '16181.png', '06460.png', '21810.png', '12597.png', '09923.png', '22830.png', '22579.png', '00895.png', '11166.png', '19985.png', '06276.png', '01419.png', '00756.png', '20718.png', '06871.png', '13904.png', '15093.png', '15472.png', '17941.png', '09603.png', '24870.png', '02544.png', '01354.png', '01448.png', '24718.png', '18837.png', '17158.png', '05096.png', '19710.png', '04816.png', '09029.png', '20976.png', '23320.png', '04985.png', '17054.png', '18888.png', '22090.png', '16568.png', '16309.png', '24683.png', '09340.png', '00699.png', '17447.png', '07932.png', '01869.png', '10638.png', '11409.png', '00045.png', '03440.png', '09882.png', '24090.png', '03611.png', '14457.png', '11479.png', '06119.png', '12672.png', '07337.png', '14026.png', '17634.png', '21676.png', '15649.png', '00207.png', '13518.png', '14089.png', '05717.png', '18598.png', '12335.png', '17152.png', '23382.png', '04831.png', '20566.png', '21713.png', '11091.png', '03954.png', '03296.png', '00334.png', '19301.png', '05684.png', '10225.png', '06558.png', '11654.png', '18848.png', '13058.png', '10218.png', '18551.png', '20905.png', '15500.png', '17263.png', '15465.png', '07713.png', '14967.png', '06820.png', '16429.png', '24680.png', '15689.png', '03374.png', '03834.png', '09419.png', '20330.png', '22611.png', '03544.png', '10937.png', '05817.png', '19721.png', '07368.png', '05165.png', '07873.png', '11252.png', '03527.png', '06610.png', '14840.png', '03795.png', '13249.png', '17923.png', '17910.png', '00463.png', '03914.png', '21446.png', '01688.png', '03849.png', '21057.png', '05018.png', '15016.png', '06042.png', '20654.png', '12222.png', '03030.png', '04123.png', '19655.png', '20930.png', '18064.png', '05435.png', '07881.png', '05114.png', '11849.png', '22526.png', '22748.png', '17430.png', '06520.png', '21373.png', '23370.png', '24168.png', '14570.png', '13243.png', '10493.png', '11035.png', '10078.png', '02576.png', '03079.png', '00779.png', '14477.png', '22910.png', '07070.png', '01915.png', '22755.png', '01271.png', '09853.png', '10170.png', '12334.png', '22410.png', '07883.png', '15284.png', '13377.png', '08583.png', '19253.png', '17560.png', '21482.png', '05735.png', '17096.png', '19745.png', '14865.png', '16012.png', '07774.png', '21240.png', '05486.png', '11181.png', '12091.png', '15116.png', '18986.png', '06641.png', '05613.png', '21844.png', '12698.png', '16168.png', '06996.png', '23905.png', '12663.png', '01636.png', '22139.png', '22347.png', '11445.png', '02977.png', '23138.png', '18067.png', '09105.png', '20743.png', '12472.png', '14411.png', '03650.png', '07365.png', '24167.png', '20792.png', '14711.png', '08729.png', '10550.png', '16885.png', '11140.png', '14936.png', '09709.png', '23053.png', '00254.png', '02590.png', '00850.png', '10001.png', '00933.png', '16928.png', '18753.png', '08584.png', '23590.png', '16908.png', '17326.png', '14545.png', '11277.png', '03125.png', '10787.png', '01129.png', '21611.png', '22294.png', '05080.png', '00864.png', '04718.png', '15143.png', '05927.png', '00046.png', '01245.png', '20464.png', '15329.png', '07135.png', '12553.png', '15043.png', '11886.png', '19807.png', '03586.png', '01329.png', '14209.png', '22827.png', '15325.png', '15988.png', '18216.png', '12446.png', '22616.png', '04269.png', '19633.png', '23420.png', '03780.png', '00514.png', '17842.png', '12906.png', '03011.png', '08542.png', '07248.png']
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
        
        if i_iter == 105577:
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
