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
    a = 	['14489.png', '07855.png', '04754.png', '15545.png', '13663.png', '11395.png', '17066.png', '24619.png', '12799.png', '21496.png', '15331.png', '07317.png', '08015.png', '17481.png', '08812.png', '07459.png', '09488.png', '23853.png', '12419.png', '12127.png', '18255.png', '24455.png', '15920.png', '11958.png', '14462.png', '19524.png', '09316.png', '04678.png', '15559.png', '01030.png', '06184.png', '17215.png', '00196.png', '05417.png', '04456.png', '06725.png', '21530.png', '16011.png', '19409.png', '18806.png', '01503.png', '02716.png', '21989.png', '14307.png', '09705.png', '20197.png', '13213.png', '03928.png', '04227.png', '09724.png', '23413.png', '07743.png', '17834.png', '04553.png', '16916.png', '11320.png', '19364.png', '13380.png', '17903.png', '05535.png', '03940.png', '19350.png', '17246.png', '01347.png', '23708.png', '22364.png', '22003.png', '08709.png', '09732.png', '07169.png', '14160.png', '11262.png', '00698.png', '16251.png', '14676.png', '15878.png', '01463.png', '00128.png', '00011.png', '00847.png', '19345.png', '19278.png', '07746.png', '12092.png', '16610.png', '17895.png', '20582.png', '19706.png', '23748.png', '18700.png', '11514.png', '20889.png', '19511.png', '05277.png', '11807.png', '23375.png', '13922.png', '03621.png', '10423.png', '17542.png', '08027.png', '17728.png', '04186.png', '12961.png', '00345.png', '05988.png', '20302.png', '17162.png', '20419.png', '21406.png', '22313.png', '17997.png', '20853.png', '08627.png', '21004.png', '12383.png', '17993.png', '24616.png', '21041.png', '11465.png', '20944.png', '23388.png', '20511.png', '07450.png', '21015.png', '15103.png', '12090.png', '15981.png', '08926.png', '13583.png', '12117.png', '13425.png', '06233.png', '06178.png', '17297.png', '16905.png', '03059.png', '05858.png', '05853.png', '08500.png', '07816.png', '08666.png', '10411.png', '09657.png', '00462.png', '08589.png', '01677.png', '24530.png', '08047.png', '21163.png', '03987.png', '23218.png', '17228.png', '16692.png', '14140.png', '12437.png', '18511.png', '00520.png', '24470.png', '00023.png', '08553.png', '10546.png', '01757.png', '14034.png', '21226.png', '17816.png', '22847.png', '20652.png', '13094.png', '14105.png', '15622.png', '07975.png', '18796.png', '18336.png', '09591.png', '06765.png', '19628.png', '01292.png', '23876.png', '05275.png', '15254.png', '11359.png', '14579.png', '09572.png', '13198.png', '02780.png', '19674.png', '08250.png', '14077.png', '17840.png', '06985.png', '12136.png', '13855.png', '01513.png', '01649.png', '05935.png', '11215.png', '11864.png', '24487.png', '02679.png', '13736.png', '18478.png', '04357.png', '23730.png', '13560.png', '10246.png', '16423.png', '05713.png', '21516.png', '24451.png', '05245.png', '08079.png', '03179.png', '11752.png', '01532.png', '17500.png', '23230.png', '06136.png', '10494.png', '05951.png', '19728.png', '03632.png', '09285.png', '03087.png', '13329.png', '22519.png', '19357.png', '24819.png', '24184.png', '04008.png', '18330.png', '05510.png', '08139.png', '15206.png', '00942.png', '03457.png', '23649.png', '15880.png', '20718.png', '04019.png', '04090.png', '00828.png', '10861.png', '09610.png', '22352.png', '13035.png', '06457.png', '20762.png', '23949.png', '13400.png', '09178.png', '14963.png', '18984.png', '18959.png', '09423.png', '08229.png', '24483.png', '08054.png', '13351.png', '21146.png', '15307.png', '20360.png', '20615.png', '24119.png', '05180.png', '12114.png', '01262.png', '24357.png', '05904.png', '21855.png', '11120.png', '06285.png', '20439.png', '11490.png', '08117.png', '21625.png', '06270.png', '19893.png', '15565.png', '01867.png', '15059.png', '18837.png', '01612.png', '11294.png', '05663.png', '17862.png', '09486.png', '22613.png', '20317.png', '13067.png', '08761.png', '17324.png', '09173.png', '10364.png', '18922.png', '08269.png', '17953.png', '18669.png', '23836.png', '01338.png', '22417.png', '00553.png', '03532.png', '14871.png', '08758.png', '00682.png', '03439.png', '17181.png', '13726.png', '24745.png', '04592.png', '23768.png', '24346.png', '10892.png', '16086.png', '11101.png', '03520.png', '03070.png', '12620.png', '11466.png', '23269.png', '12826.png', '21878.png', '20175.png', '08893.png', '18446.png', '10926.png', '16940.png', '14062.png', '20907.png', '22689.png', '05314.png', '17694.png', '04662.png', '21672.png', '11151.png', '18146.png', '08362.png', '02504.png', '06956.png', '11604.png', '06816.png', '04206.png', '18440.png', '00800.png', '24965.png', '15609.png', '05102.png', '13821.png', '16210.png', '17185.png', '23779.png', '20844.png', '08730.png', '05481.png', '15178.png', '08257.png', '12682.png', '10046.png', '00835.png', '06188.png', '03806.png', '16325.png', '03696.png', '24177.png', '24133.png', '13750.png', '18004.png', '13048.png', '07989.png', '08479.png', '15426.png', '09541.png', '22506.png', '12744.png', '12389.png', '22934.png', '17725.png', '12793.png', '05146.png', '15210.png', '00855.png', '03573.png', '22392.png', '20267.png', '17890.png', '01848.png', '18282.png', '12699.png', '08125.png', '14174.png', '06829.png', '08332.png', '01312.png', '01567.png', '04224.png', '13138.png', '24839.png', '01437.png', '19647.png', '23758.png', '24714.png', '00547.png', '12770.png', '22562.png', '18589.png', '19478.png', '05441.png', '11040.png', '07359.png', '18904.png', '09421.png', '10652.png', '24009.png', '01435.png', '08974.png', '17843.png', '15827.png', '13395.png', '07495.png', '04166.png', '10059.png', '21084.png', '15713.png', '00109.png', '20227.png', '20075.png', '08975.png', '18020.png', '00278.png', '06480.png', '21352.png', '12310.png', '16590.png', '20144.png', '02782.png', '08910.png', '11671.png', '13326.png', '12386.png', '19092.png', '04343.png', '10853.png', '11090.png', '17265.png', '21573.png', '06870.png', '13937.png', '18434.png', '19015.png', '17282.png', '15417.png', '22939.png', '15498.png', '14376.png', '06305.png', '14334.png', '12193.png', '09447.png', '10131.png', '08693.png', '18157.png', '23186.png', '07918.png', '16876.png', '11157.png', '19609.png', '16993.png', '20311.png', '11400.png', '20352.png', '12711.png', '23159.png', '06466.png', '17861.png', '14840.png', '18123.png', '04046.png', '01377.png', '19322.png', '04715.png', '23481.png', '18656.png', '19733.png', '02149.png', '15998.png', '09687.png', '19368.png', '01479.png', '23216.png', '17094.png', '01539.png', '19584.png', '00274.png', '21605.png', '09259.png', '17848.png', '10888.png', '09509.png', '15506.png', '22594.png', '10018.png', '17597.png', '19025.png', '16079.png', '19171.png', '07053.png', '05030.png', '14192.png', '11970.png', '05288.png', '14366.png', '20092.png', '08806.png', '18485.png', '21345.png', '06994.png', '10164.png', '22057.png', '00567.png', '22673.png', '13974.png', '10830.png', '17795.png', '07083.png', '11542.png', '22715.png', '18756.png', '11256.png', '00550.png', '08601.png', '21332.png', '06637.png', '17175.png', '14110.png', '00863.png', '08412.png', '07465.png', '15061.png', '13002.png', '15995.png', '10041.png', '17878.png', '14758.png', '10288.png', '14001.png', '10710.png', '18905.png', '11068.png', '10672.png', '09822.png', '15129.png', '09147.png', '19575.png', '13054.png', '07817.png', '09940.png', '17140.png', '03145.png', '16314.png', '23784.png', '18493.png', '11475.png', '09187.png', '04733.png', '17047.png', '09832.png', '12973.png', '06729.png', '03314.png', '16416.png', '21131.png', '06104.png', '18800.png', '06999.png', '01333.png', '22977.png', '15841.png', '05536.png', '07829.png', '05745.png', '14533.png', '12666.png', '16138.png', '01274.png', '08321.png', '21893.png', '20777.png', '07835.png', '22992.png', '14161.png', '11913.png', '23731.png', '14449.png', '04379.png', '06553.png', '16159.png', '24318.png', '05118.png', '12523.png', '15582.png', '17643.png', '11125.png', '05334.png', '08323.png', '16324.png', '12455.png', '01526.png', '21115.png', '01143.png', '03836.png', '09671.png', '07619.png', '01244.png', '01707.png', '05474.png', '16464.png', '07907.png', '07714.png', '05430.png', '22214.png', '08056.png', '02307.png', '22621.png', '21018.png', '15372.png', '19864.png', '01461.png', '16470.png', '05852.png', '19314.png', '19617.png', '06427.png', '13519.png', '01708.png', '08530.png', '01318.png', '05765.png', '11365.png', '21986.png', '15487.png', '10941.png', '06855.png', '08566.png', '03275.png', '08294.png', '08822.png', '02743.png', '17226.png', '23922.png', '19983.png', '24864.png', '12201.png', '15978.png', '06407.png', '23787.png', '07577.png', '16456.png', '10040.png', '07909.png', '00702.png', '11870.png', '21811.png', '19311.png', '19990.png', '07976.png', '08095.png', '07704.png', '21565.png', '18889.png', '20377.png', '07152.png', '04879.png', '09416.png', '07946.png', '06249.png', '17255.png', '04735.png', '16492.png', '17717.png', '16878.png', '00181.png', '02629.png', '24779.png', '22370.png', '03224.png', '10741.png', '17532.png', '13718.png', '13921.png', '11332.png', '22073.png', '11993.png', '18257.png', '03101.png', '15328.png', '24505.png', '15136.png', '07654.png', '13923.png', '18830.png', '14319.png', '01838.png', '08669.png', '14342.png', '22008.png', '10333.png', '00248.png', '00058.png', '05812.png', '13755.png', '05638.png', '07125.png', '19555.png', '10363.png', '14137.png', '14740.png', '05514.png', '04417.png', '18514.png', '17729.png', '03325.png', '06181.png', '18406.png', '20739.png', '24187.png', '14873.png', '16681.png', '22360.png', '01650.png', '15794.png', '19944.png', '00925.png', '10173.png', '17515.png', '11528.png', '22811.png', '02924.png', '02527.png', '04980.png', '15480.png', '18897.png', '05403.png', '20231.png', '00868.png', '08737.png', '16856.png', '14288.png', '12805.png', '04474.png', '24268.png', '21340.png', '00508.png', '12316.png', '19640.png', '19787.png', '02265.png', '18189.png', '04872.png', '20191.png', '16561.png', '20658.png', '00450.png', '12380.png', '18624.png', '00104.png', '20038.png', '02720.png', '06761.png', '23297.png', '04627.png', '00982.png', '15802.png', '06322.png', '15110.png', '16092.png', '10051.png', '23141.png', '24372.png', '23239.png', '24770.png', '14331.png', '10842.png', '10607.png', '08442.png', '22731.png', '10229.png', '06878.png', '23860.png', '05376.png', '02552.png', '10196.png', '04651.png', '11382.png', '07185.png', '06794.png', '00662.png', '24010.png', '05041.png', '20110.png', '04863.png', '18795.png', '21246.png', '02771.png', '03130.png', '15151.png', '08235.png', '23601.png', '08801.png', '09571.png', '13493.png', '12146.png', '24543.png', '07133.png', '03985.png', '09105.png', '05554.png', '21860.png', '16516.png', '05649.png', '03048.png', '14361.png', '15247.png', '01026.png', '14643.png', '09993.png', '09330.png', '13559.png', '16749.png', '06644.png', '22923.png', '14967.png', '00269.png', '10389.png', '12921.png', '01620.png', '15145.png', '14096.png', '23418.png', '08680.png', '18802.png', '20882.png', '17355.png', '11267.png', '21087.png', '23752.png', '05863.png', '21028.png', '02158.png', '05398.png', '23479.png', '00063.png', '11772.png', '12429.png', '17252.png', '15541.png', '13335.png', '07310.png', '15130.png', '22375.png', '13435.png', '07039.png', '11344.png', '07167.png', '20105.png', '23158.png', '22039.png', '01270.png', '03045.png', '20912.png', '13515.png', '22681.png', '11204.png', '12290.png', '23415.png', '03003.png', '16763.png', '20034.png', '23878.png', '16693.png', '13173.png', '22345.png', '10324.png', '01120.png', '06596.png', '22582.png', '17365.png', '01353.png', '24431.png', '12850.png', '10964.png', '11717.png', '18340.png', '24432.png', '03333.png', '11557.png', '00758.png', '19886.png', '16941.png', '08280.png', '15167.png', '19829.png', '01615.png', '23912.png', '20494.png', '06850.png', '18988.png', '19044.png', '06983.png', '03082.png', '17453.png', '04695.png', '01834.png', '17477.png', '21130.png', '24148.png', '11025.png', '14500.png', '08082.png', '09309.png', '15515.png', '05990.png', '10397.png', '19049.png', '16963.png', '23902.png', '20964.png', '24966.png', '05575.png', '01061.png', '09191.png', '03528.png', '05856.png', '18691.png', '12152.png', '04857.png', '13956.png', '05072.png', '09934.png', '23429.png', '11003.png', '19445.png', '09583.png', '04809.png', '04235.png', '20009.png', '24757.png', '15557.png', '16774.png', '17599.png', '22935.png', '09271.png', '01313.png', '02618.png', '17289.png', '22604.png', '14690.png', '17403.png', '19161.png', '14673.png', '09209.png', '06435.png', '11517.png', '13863.png', '13381.png', '22147.png', '21386.png', '24639.png', '06141.png', '07124.png', '17682.png', '13878.png', '09226.png', '22860.png', '05974.png', '02722.png', '19399.png', '12192.png', '00727.png', '12778.png', '02335.png', '21302.png', '22445.png', '23343.png', '01439.png', '20262.png', '22953.png', '13096.png', '03180.png', '16349.png', '08104.png', '15792.png', '06661.png', '06811.png', '04926.png', '16362.png', '20288.png', '01305.png', '10452.png', '13426.png', '16766.png', '23006.png', '16654.png', '15468.png', '00261.png', '09445.png', '08328.png', '05625.png', '16025.png', '15896.png', '07532.png', '15918.png', '06951.png', '04472.png', '09720.png', '13456.png', '16072.png', '17401.png', '14148.png', '19776.png', '06716.png', '22469.png', '00404.png', '15702.png', '14128.png', '23339.png', '03956.png', '20565.png', '12342.png', '18559.png', '18021.png', '02290.png', '20178.png', '05943.png', '17572.png', '12259.png', '19182.png', '16132.png', '19926.png', '04424.png', '02146.png', '07811.png', '09660.png', '20557.png', '06145.png', '16798.png', '13246.png', '18677.png', '13542.png', '10599.png', '02469.png', '01686.png', '15467.png', '19817.png', '13212.png', '21500.png', '05715.png', '00246.png', '20131.png', '01216.png', '11972.png', '02422.png', '11988.png', '18699.png', '03404.png', '14929.png', '21759.png', '05386.png', '14216.png', '04355.png', '06045.png', '08576.png', '15319.png', '13427.png', '15026.png', '00304.png', '01326.png', '09347.png', '00289.png', '05268.png', '15709.png', '13122.png', '16429.png', '22844.png', '11640.png', '23072.png', '09152.png', '19732.png', '17009.png', '14346.png', '01010.png', '15439.png', '10746.png', '18384.png', '09725.png', '14611.png', '13009.png', '08631.png', '05224.png', '04373.png', '22969.png', '00368.png', '02526.png', '01218.png', '18659.png', '21162.png', '00341.png', '19193.png', '24155.png', '18743.png', '12842.png', '21391.png', '21398.png', '07749.png', '09111.png', '03590.png', '15654.png', '05545.png', '00538.png', '12338.png', '04544.png', '11413.png', '18534.png', '07902.png', '11002.png', '15412.png', '10784.png', '06698.png', '11305.png', '02122.png', '17089.png', '08414.png', '03382.png', '12748.png', '17170.png', '04526.png', '01954.png', '22855.png', '21225.png', '22162.png', '21804.png', '04496.png', '01323.png', '18479.png', '11442.png', '08136.png', '22383.png', '20123.png', '03995.png', '05719.png', '09375.png', '10062.png', '04389.png', '23593.png', '10552.png', '18643.png', '18998.png', '09460.png', '06925.png', '15320.png', '23788.png', '24591.png', '23002.png', '13488.png', '05994.png', '01151.png', '01055.png', '18022.png', '24061.png', '05501.png', '13979.png', '11207.png', '12543.png', '20730.png', '18555.png', '05585.png', '01469.png', '01342.png', '24203.png', '03792.png', '18845.png', '05684.png', '05194.png', '22098.png', '18544.png', '14794.png', '18519.png', '15402.png', '21589.png', '10481.png', '07184.png', '22536.png', '08194.png', '12460.png', '21185.png', '10390.png', '14824.png', '24620.png', '02468.png', '19082.png', '01689.png', '11874.png', '15540.png', '11627.png', '00202.png', '11219.png', '19875.png', '22995.png', '10872.png', '20543.png', '15280.png', '21890.png', '21818.png', '10883.png', '12925.png', '19386.png', '07097.png', '04938.png', '11578.png', '14040.png', '00523.png', '07115.png', '04605.png', '11399.png', '18054.png', '23745.png', '03146.png', '07006.png', '22396.png', '02147.png', '09718.png', '16974.png', '16047.png', '00878.png', '24806.png', '04572.png', '11062.png', '17767.png', '23911.png', '03032.png', '05523.png', '10283.png', '00565.png', '21872.png', '09715.png', '01331.png', '11688.png', '07804.png', '05577.png', '15912.png', '16345.png', '07327.png', '07486.png', '18133.png', '21985.png', '21762.png', '10231.png', '21067.png', '05958.png', '16756.png', '11915.png', '14316.png', '20989.png', '17072.png', '15960.png', '01849.png', '14133.png', '21105.png', '21025.png', '06563.png', '11509.png', '03613.png', '13695.png', '11786.png', '23572.png', '23057.png', '23632.png', '16144.png', '09751.png', '20970.png', '11073.png', '20708.png', '11560.png', '15591.png', '05940.png', '12670.png', '06319.png', '22096.png', '19668.png', '21510.png', '20633.png', '03183.png', '10201.png', '13633.png', '07558.png', '23610.png', '06475.png', '04660.png', '13379.png', '18731.png', '21093.png', '12831.png', '01167.png', '13428.png', '19312.png', '23568.png', '17828.png', '19958.png', '22479.png', '22755.png', '02408.png', '09004.png', '21706.png', '04067.png', '16650.png', '02702.png', '19270.png', '14256.png', '12679.png', '00326.png', '10703.png', '18329.png', '05260.png', '06283.png', '13933.png', '19253.png', '00695.png', '17809.png', '04261.png', '00036.png', '00510.png', '10090.png', '02776.png', '08212.png', '01303.png', '00441.png', '06395.png', '15983.png', '22988.png', '09122.png', '10343.png', '21182.png', '14791.png', '03201.png', '12231.png', '23166.png', '18805.png', '16861.png', '24800.png', '16715.png', '17488.png', '00335.png', '05380.png', '02952.png', '12317.png', '06042.png', '16426.png', '10465.png', '20370.png', '04645.png', '23786.png', '03465.png', '12939.png', '10150.png', '11130.png', '21708.png', '22888.png', '09629.png', '15997.png', '04959.png', '19069.png', '04547.png', '19594.png', '07660.png', '08030.png', '14586.png', '10920.png', '15717.png', '18958.png', '10202.png', '17087.png', '18614.png', '18410.png', '11830.png', '18031.png', '13779.png', '17436.png', '15782.png', '04995.png', '19855.png', '06562.png', '14784.png', '24734.png', '17589.png', '15589.png', '17546.png', '01209.png', '21334.png', '07165.png', '13433.png', '01624.png', '23813.png', '07777.png', '17509.png', '21852.png', '01721.png', '12681.png', '08327.png', '16400.png', '12431.png', '04421.png', '11236.png', '18647.png', '05212.png', '15187.png', '17172.png', '08547.png', '10597.png', '20654.png', '04674.png', '05779.png', '08013.png', '12336.png', '02709.png', '10212.png', '07612.png', '12110.png', '19455.png', '00832.png', '22211.png', '21828.png', '11145.png', '06882.png', '02017.png', '14437.png', '14138.png', '20908.png', '23207.png', '03412.png', '04751.png', '05547.png', '03118.png', '01286.png', '09789.png', '20871.png', '18576.png', '21546.png', '07822.png', '12519.png', '20007.png', '10039.png', '06630.png', '18537.png', '21262.png', '14786.png', '02489.png', '10922.png', '08606.png', '11949.png', '11186.png', '02164.png', '14320.png', '15894.png', '08315.png', '10440.png', '10446.png', '10251.png', '11233.png', '24522.png', '04944.png', '09723.png', '12563.png', '15031.png', '01267.png', '00004.png', '00586.png', '04832.png', '09957.png', '21660.png', '21525.png', '11299.png', '08514.png', '10857.png', '17998.png', '07731.png', '17825.png', '10081.png', '19680.png', '06758.png', '05872.png', '16391.png', '06449.png', '02194.png', '07054.png', '16099.png', '15756.png', '01381.png', '22265.png', '05811.png', '23210.png', '23695.png', '15474.png', '00130.png', '16737.png', '18290.png', '03298.png', '14468.png', '07945.png', '08586.png', '13596.png', '02544.png', '17652.png', '06006.png', '05183.png', '22405.png', '20641.png', '13613.png', '11583.png', '04518.png', '03387.png', '07437.png', '12226.png', '12211.png', '22052.png', '20181.png', '07803.png', '01346.png', '17486.png', '10570.png', '19917.png', '06401.png', '06886.png', '05320.png', '24444.png', '09466.png', '00905.png', '22143.png', '03411.png', '24210.png', '00488.png', '08842.png', '21679.png', '10381.png', '12687.png', '16004.png', '16685.png', '15758.png', '03606.png', '12690.png', '07196.png', '13294.png', '04760.png', '24960.png', '04345.png', '08106.png', '24421.png', '24374.png', '15371.png', '14039.png', '17346.png', '22735.png', '17580.png', '11140.png', '17018.png', '19063.png', '09840.png', '21494.png', '11177.png', '19553.png', '14416.png', '20971.png', '23875.png', '13131.png', '00047.png', '16885.png', '05108.png', '10373.png', '09805.png', '08626.png', '07593.png', '10316.png', '06554.png', '08753.png', '12060.png', '13489.png', '22652.png', '12724.png', '19140.png', '05039.png', '21900.png', '21832.png', '21972.png', '12817.png', '15585.png', '11214.png', '23142.png', '20925.png', '13459.png', '10722.png', '07422.png', '05808.png', '04026.png', '03522.png', '08042.png', '10896.png', '03015.png', '10482.png', '14787.png', '19256.png', '11836.png', '04268.png', '24569.png', '16275.png', '08653.png', '13780.png', '11209.png', '19836.png', '12720.png', '18199.png', '22393.png', '06358.png', '12746.png', '24622.png', '10438.png', '08923.png', '20962.png', '09842.png', '03284.png', '17506.png', '20850.png', '00491.png', '13022.png', '01050.png', '23526.png', '04165.png', '14383.png', '06962.png', '15595.png', '16743.png', '16475.png', '14780.png', '16458.png', '08499.png', '10764.png', '00125.png', '08338.png', '14602.png', '13437.png', '03389.png', '12248.png', '10317.png', '20196.png', '10720.png', '17755.png', '16773.png', '05616.png', '14822.png', '06324.png', '17536.png', '07891.png', '19913.png', '13760.png', '22132.png', '21633.png', '07136.png', '16333.png', '19414.png', '21887.png', '07338.png', '17951.png', '06963.png', '09095.png', '15288.png', '15489.png', '24173.png', '19393.png', '15443.png', '12546.png', '09737.png', '13158.png', '03523.png', '13001.png', '09478.png', '16980.png', '07728.png', '07597.png', '05226.png', '07676.png', '00558.png', '20519.png', '07087.png', '01584.png', '08629.png', '23488.png', '01796.png', '01927.png', '07915.png', '18890.png', '01199.png', '09886.png', '11680.png', '21721.png', '02280.png', '00102.png', '21742.png', '16042.png', '09292.png', '06810.png', '00111.png', '14671.png', '18132.png', '10493.png', '03709.png', '19931.png', '08667.png', '20506.png', '16616.png', '24798.png', '21808.png', '23189.png', '01028.png', '18413.png', '08595.png', '07823.png', '15043.png', '04296.png', '24022.png', '05628.png', '03536.png', '23073.png', '16365.png', '07848.png', '08341.png', '15682.png', '23603.png', '07493.png', '08302.png', '16330.png', '23830.png', '20973.png', '23859.png', '04069.png', '18846.png', '09981.png', '04713.png', '24457.png', '02803.png', '12070.png', '23367.png', '20429.png', '24860.png', '06029.png', '22212.png', '06513.png', '03680.png', '17480.png', '10694.png', '09713.png', '16967.png', '21949.png', '08597.png', '11349.png', '09382.png', '23892.png', '05198.png', '06583.png', '16791.png', '03655.png', '19532.png', '02747.png', '16013.png', '04219.png', '22491.png', '15667.png', '20016.png', '24227.png', '14418.png', '18449.png', '05877.png', '18169.png', '04358.png', '22471.png', '04837.png', '06387.png', '15286.png', '11102.png', '03617.png', '16990.png', '20618.png', '02321.png', '01495.png', '14801.png', '03663.png', '14213.png', '07198.png', '09834.png', '08816.png', '05699.png', '10909.png', '01525.png', '21453.png', '11653.png', '06888.png', '01107.png', '15701.png', '22468.png', '02015.png', '03508.png', '11804.png', '03969.png', '01250.png', '12545.png', '18873.png', '11021.png', '00257.png', '10610.png', '02528.png', '22131.png', '03310.png', '05560.png', '13473.png', '20441.png', '14118.png', '03568.png', '14855.png', '10646.png', '13656.png', '09282.png', '17432.png', '19923.png', '04351.png', '15345.png', '21574.png', '00851.png', '06121.png', '04773.png', '01141.png', '11652.png', '17196.png', '12224.png', '04826.png', '20515.png', '11415.png', '06276.png', '03268.png', '07830.png', '12669.png', '14263.png', '14091.png', '18706.png', '04978.png', '09858.png', '12177.png', '02833.png', '18241.png', '10962.png', '07210.png', '00644.png', '07289.png', '20328.png', '20082.png', '17992.png', '10893.png', '05025.png', '08700.png', '24001.png', '12257.png', '17158.png', '16924.png', '08413.png', '11798.png', '23920.png', '21981.png', '18149.png', '15597.png', '24378.png', '12180.png', '22861.png', '21894.png', '17090.png', '08688.png', '22231.png', '05665.png', '02985.png', '08355.png', '05897.png', '14230.png', '18649.png', '21801.png', '02923.png', '12254.png', '09593.png', '13634.png', '06997.png', '02624.png', '08523.png', '21800.png', '11420.png', '01260.png', '09319.png', '14566.png', '04571.png', '08188.png', '21288.png', '15038.png', '06946.png', '19334.png', '00397.png', '08664.png', '22353.png', '21478.png', '17517.png', '20662.png', '05522.png', '23993.png', '03759.png', '12428.png', '23224.png', '03869.png', '23637.png', '08544.png', '17234.png', '10548.png', '20974.png', '04367.png', '24763.png', '13890.png', '19309.png', '23191.png', '03784.png', '14807.png', '15116.png', '00206.png', '22863.png', '17128.png', '01816.png', '05340.png', '04444.png', '20835.png', '13676.png', '08258.png', '14061.png', '08175.png', '05262.png', '17668.png', '02192.png', '04775.png', '24211.png', '15644.png', '09798.png', '00237.png', '06706.png', '11053.png', '06201.png', '24024.png', '20449.png', '04481.png', '22578.png', '24593.png', '14723.png', '17568.png', '22356.png', '20359.png', '10079.png', '21003.png', '11722.png', '21835.png', '10413.png', '05491.png', '07761.png', '13475.png', '02622.png', '13690.png', '06031.png', '09196.png', '14836.png', '15281.png', '16921.png', '02565.png', '01573.png', '17197.png', '21842.png', '03487.png', '10620.png', '03791.png', '21951.png', '10419.png', '18711.png', '13413.png', '00881.png', '13134.png', '21370.png', '11504.png', '03602.png', '21269.png', '07357.png', '02579.png', '06880.png', '23453.png', '08985.png', '19896.png', '08654.png', '02201.png', '22769.png', '21948.png', '14268.png', '11310.png', '23775.png', '20069.png', '02195.png', '24564.png', '19347.png', '22296.png', '21463.png', '18327.png', '11539.png', '00116.png', '14670.png', '11014.png', '06472.png', '13237.png', '01291.png', '06756.png', '06370.png', '11389.png', '12289.png', '17698.png', '00922.png', '18069.png', '08840.png', '04302.png', '22931.png', '24815.png', '14828.png', '07132.png', '08497.png', '03385.png', '23070.png', '10414.png', '04476.png', '12294.png', '09241.png', '10405.png', '06923.png', '13775.png', '11544.png', '07634.png', '24418.png', '04657.png', '04005.png', '11039.png', '01222.png', '12872.png', '15034.png', '19637.png', '08274.png', '12821.png', '21318.png', '22727.png', '01423.png', '19670.png', '07552.png', '03307.png', '17787.png', '10756.png', '20246.png', '14524.png', '14373.png', '17360.png', '04750.png', '04521.png', '16829.png', '14880.png', '17917.png', '12414.png', '13806.png', '10085.png', '03096.png', '04101.png', '18858.png', '01158.png', '03823.png', '11132.png', '17685.png', '15563.png', '06591.png', '19966.png', '14747.png', '14011.png', '14379.png', '01972.png', '13506.png', '16511.png', '09307.png', '12557.png', '16519.png', '17007.png', '22692.png', '02288.png', '13145.png', '15719.png', '02682.png', '16964.png', '24517.png', '00577.png', '18621.png', '14104.png', '02247.png', '05301.png', '13904.png', '17084.png', '14753.png', '09777.png', '24329.png', '03521.png', '22851.png', '20205.png', '24182.png', '12043.png', '23292.png', '04161.png', '00227.png', '17264.png', '08059.png', '20218.png', '01824.png', '14428.png', '23838.png', '16111.png', '16777.png', '04350.png', '11082.png', '06515.png', '11376.png', '23144.png', '13626.png', '23659.png', '19888.png', '16028.png', '19865.png', '11026.png', '13338.png', '23724.png', '20445.png', '11692.png', '13823.png', '05173.png', '22812.png', '02338.png', '20735.png', '11245.png', '00574.png', '21466.png', '02820.png', '17621.png', '14848.png', '18828.png', '08527.png', '19056.png', '01839.png', '12411.png', '08611.png', '15616.png', '19254.png', '15974.png', '08752.png', '19741.png', '17362.png', '04930.png', '13234.png', '16722.png', '08202.png', '08233.png', '21545.png', '07336.png', '16101.png', '06988.png', '05142.png', '10421.png', '20715.png', '08063.png', '02788.png', '00789.png', '20054.png', '13202.png', '15201.png', '00853.png', '05643.png', '15449.png', '11731.png', '05807.png', '15851.png', '14082.png', '00898.png', '01047.png', '03267.png', '17526.png', '02706.png', '04694.png', '04727.png', '08516.png', '23208.png', '07724.png', '08349.png', '19005.png', '07949.png', '09859.png', '06656.png', '01436.png', '10479.png', '10868.png', '07675.png', '17606.png', '22449.png', '05862.png', '19620.png', '15757.png', '24682.png', '24204.png', '10185.png', '08008.png', '24398.png', '09294.png', '21486.png', '14340.png', '07916.png', '01687.png', '19791.png', '23496.png', '23155.png', '04781.png', '17482.png', '01334.png', '19953.png', '23131.png', '09338.png', '01575.png', '21680.png', '09092.png', '13180.png', '06595.png', '07733.png', '10566.png', '16127.png', '15945.png', '18924.png', '17452.png', '09962.png', '02956.png', '17222.png', '23094.png', '21988.png', '11869.png', '05874.png', '14647.png', '04617.png', '19036.png', '21600.png', '19554.png', '12481.png', '04914.png', '05929.png', '01501.png', '21841.png', '22560.png', '02988.png', '24590.png', '13080.png', '00038.png', '16024.png', '18946.png', '23753.png', '17146.png', '24495.png', '05438.png', '02109.png', '09110.png', '21255.png', '23160.png', '00941.png', '04833.png', '11497.png', '13507.png', '05418.png', '15440.png', '14388.png', '17653.png', '03414.png', '15343.png', '04126.png', '13088.png', '03339.png', '21979.png', '03371.png', '05298.png', '24655.png', '20608.png', '23416.png', '04144.png', '08943.png', '02975.png', '13072.png', '06495.png', '22610.png', '00051.png', '21612.png', '08917.png', '19960.png', '21978.png', '16862.png', '01896.png', '04079.png', '00280.png', '23645.png', '04726.png', '10616.png', '23370.png', '00545.png', '17817.png', '22401.png', '17068.png', '17877.png', '17193.png', '19466.png', '12433.png', '10790.png', '11827.png', '03790.png', '13533.png', '03530.png', '08724.png', '13129.png', '21655.png', '12128.png', '22900.png', '18173.png', '02435.png', '18635.png', '17675.png', '17963.png', '06017.png', '23404.png', '11333.png', '22425.png', '17501.png', '09370.png', '00656.png', '02909.png', '10146.png', '01458.png', '07534.png', '23516.png', '16562.png', '10950.png', '08143.png', '14355.png', '22314.png', '04432.png', '09417.png', '11235.png', '01466.png', '07454.png', '02023.png', '22348.png', '03228.png', '20336.png', '24942.png', '16658.png', '22979.png', '01787.png', '09846.png', '23396.png', '12585.png', '00010.png', '09618.png', '17245.png', '02902.png', '19874.png', '09819.png', '08703.png', '10516.png', '05646.png', '03072.png', '07931.png', '21198.png', '24883.png', '05670.png', '03092.png', '02061.png', '03638.png', '00939.png', '03189.png', '15448.png', '21793.png', '16994.png', '11341.png', '03690.png', '05101.png', '18192.png', '09645.png', '13787.png', '03303.png', '09596.png', '08646.png', '21001.png', '00029.png', '12323.png', '11282.png', '06606.png', '05804.png', '03196.png', '01587.png', '06800.png', '00753.png', '06615.png', '17057.png', '04298.png', '20960.png', '22650.png', '11008.png', '01125.png', '24871.png', '11614.png', '16334.png', '22433.png', '13209.png', '24936.png', '01493.png', '20665.png', '12814.png', '00376.png', '22982.png', '04961.png', '24053.png', '13518.png', '12379.png', '10742.png', '22005.png', '24719.png', '21809.png', '09263.png', '11651.png', '22463.png', '20062.png', '20101.png', '19750.png', '18538.png', '04344.png', '06701.png', '04777.png', '14010.png', '05103.png', '19868.png', '12375.png', '03125.png', '19226.png', '02378.png', '09380.png', '00897.png', '03852.png', '11110.png', '24642.png', '03991.png', '17372.png', '02932.png', '03403.png', '15285.png', '05197.png', '01470.png', '10042.png', '05832.png', '06826.png', '15893.png', '06573.png', '01720.png', '02801.png', '15862.png', '07791.png', '21856.png', '21166.png', '19154.png', '08891.png', '21014.png', '02318.png', '13673.png', '10969.png', '22461.png', '22282.png', '12633.png', '11295.png', '21421.png', '14522.png', '10768.png', '12927.png', '24752.png', '21563.png', '14021.png', '10312.png', '21016.png', '07815.png', '00732.png', '22694.png', '20928.png', '08150.png', '08756.png', '11817.png', '06517.png', '23522.png', '23955.png', '05627.png', '06558.png', '19823.png', '16598.png', '12479.png', '06734.png', '15933.png', '21635.png', '22209.png', '04684.png', '21932.png', '08308.png', '19031.png', '14947.png', '14402.png', '00357.png', '03136.png', '00425.png', '05784.png', '03841.png', '15627.png', '01809.png', '17628.png', '14942.png', '24215.png', '24865.png', '04215.png', '00641.png', '17458.png', '12644.png', '02060.png', '13078.png', '04006.png', '20475.png', '05035.png', '18266.png', '00045.png', '09967.png', '00771.png', '23458.png', '23512.png', '12525.png', '11377.png', '16032.png', '24278.png', '09950.png', '19363.png', '02865.png', '23455.png', '24926.png', '09532.png', '23485.png', '22489.png', '06497.png', '03100.png', '13928.png', '24556.png', '20303.png', '13140.png', '01371.png', '09893.png', '05549.png', '18470.png', '16230.png', '21946.png', '20826.png', '06803.png', '13443.png', '08261.png', '11458.png', '17336.png', '22455.png', '01079.png', '11228.png', '11241.png', '00175.png', '01799.png', '17242.png', '17314.png', '24442.png', '20576.png', '01840.png', '16084.png', '03282.png', '03388.png', '03548.png', '04979.png', '16064.png', '11103.png', '11036.png', '18598.png', '08345.png', '24108.png', '22334.png', '05171.png', '04168.png', '17335.png', '10303.png', '18676.png', '15610.png', '07471.png', '10071.png', '18431.png', '23687.png', '03539.png', '18667.png', '20120.png', '03833.png', '04340.png', '02979.png', '13701.png', '09811.png', '17910.png', '13611.png', '14333.png', '00830.png', '08860.png', '04029.png', '08224.png', '21779.png', '00414.png', '08658.png', '05134.png', '07106.png', '16223.png', '10650.png', '10507.png', '23010.png', '02363.png', '16559.png', '16726.png', '24363.png', '16711.png', '24199.png', '16250.png', '18960.png', '03497.png', '13320.png', '01183.png', '18832.png', '22845.png', '06346.png', '11566.png', '19782.png', '24264.png', '08152.png', '24506.png', '22165.png', '04068.png', '15936.png', '14368.png', '12385.png', '00516.png', '17820.png', '19497.png', '21789.png', '12677.png', '17587.png', '00892.png', '09794.png', '20258.png', '01402.png', '08113.png', '10211.png', '20108.png', '11041.png', '13985.png', '21567.png', '12707.png', '10734.png', '14344.png', '19230.png', '22605.png', '16545.png', '11805.png', '01407.png', '08936.png', '02687.png', '23668.png', '16706.png', '21168.png', '15674.png', '04401.png', '10077.png', '05462.png', '19395.png', '11791.png', '06091.png', '03502.png', '09322.png', '05202.png', '09469.png', '09912.png', '21781.png', '01086.png', '04942.png', '07758.png', '07166.png', '15437.png', '07725.png', '24573.png', '08831.png', '06371.png', '05617.png', '14365.png', '10016.png', '03698.png', '23058.png', '14369.png', '22641.png', '04513.png', '03962.png', '24039.png', '10278.png', '14637.png', '12627.png', '00544.png', '04937.png', '01681.png', '22207.png', '06654.png', '20170.png', '05936.png', '20986.png', '07249.png', '17960.png', '11903.png', '15754.png', '00449.png', '07765.png', '01519.png', '08578.png', '24812.png', '17913.png', '05884.png', '05785.png', '13267.png', '15675.png', '09155.png', '01340.png', '04285.png', '00055.png', '09515.png', '12862.png', '21017.png', '10450.png', '14348.png', '03464.png', '11105.png', '04228.png', '03574.png', '10747.png', '02316.png', '21498.png', '01187.png', '22175.png', '04871.png', '15246.png', '00593.png', '06416.png', '23916.png', '03271.png', '16704.png', '13424.png', '07237.png', '05137.png', '07993.png', '09518.png', '01140.png', '01946.png', '11107.png', '19094.png', '05854.png', '17863.png', '15447.png', '03935.png', '12468.png', '13785.png', '24728.png', '11853.png', '01960.png', '01772.png', '06742.png', '08088.png', '08777.png', '00527.png', '18717.png', '14249.png', '13590.png', '14167.png', '23805.png', '16857.png', '10538.png', '14701.png', '06935.png', '15214.png', '10833.png', '03147.png', '22793.png', '10819.png', '05249.png', '16528.png', '14664.png', '15118.png', '08873.png', '04347.png', '05647.png', '03509.png', '07859.png', '17882.png', '17358.png', '07555.png', '09210.png', '08389.png', '04643.png', '20555.png', '07475.png', '01401.png', '19076.png', '20833.png', '07499.png', '21803.png', '19351.png', '01794.png', '12188.png', '06780.png', '02978.png', '20883.png', '24524.png', '20972.png', '19378.png', '08897.png', '05448.png', '18746.png', '17916.png', '16217.png', '19979.png', '03216.png', '03654.png', '13431.png', '16682.png', '02203.png', '00594.png', '10091.png', '04211.png', '07717.png', '05771.png', '23504.png', '02889.png', '06207.png', '06836.png', '16007.png', '15562.png', '08265.png', '07953.png', '19289.png', '11996.png', '13862.png', '02179.png', '08069.png', '22603.png', '12507.png', '07065.png', '22951.png', '03265.png', '07944.png', '16372.png', '17144.png', '14103.png', '11899.png', '20326.png', '00168.png', '14783.png', '23187.png', '16595.png', '09996.png', '06240.png', '09982.png', '15971.png', '19891.png', '17329.png', '09089.png', '22936.png', '10286.png', '02080.png', '17793.png', '05694.png', '09873.png', '12063.png', '07431.png', '06621.png', '19186.png', '04442.png', '20690.png', '05158.png', '19533.png', '01866.png', '20659.png', '01640.png', '19552.png', '07478.png', '07747.png', '22867.png', '19783.png', '01712.png', '19420.png', '07225.png', '11434.png', '16834.png', '06026.png', '24613.png', '02339.png', '18385.png', '05997.png', '17521.png', '05328.png', '07663.png', '17841.png', '13508.png', '12787.png', '22060.png', '24786.png', '17190.png', '19726.png', '03030.png', '01400.png', '09482.png', '15925.png', '08796.png', '16801.png', '07419.png', '13193.png', '08678.png', '21447.png', '19906.png', '21244.png', '00276.png', '11892.png', '19274.png', '07030.png', '03830.png', '15700.png', '14501.png', '22958.png', '07775.png', '17757.png', '09754.png', '02673.png', '17002.png', '22989.png', '14904.png', '05891.png', '06412.png', '13769.png', '16975.png', '09362.png', '00944.png', '19493.png', '19104.png', '11199.png', '01399.png', '15747.png', '13157.png', '18064.png', '05954.png', '20323.png', '07163.png', '08727.png', '20869.png', '15268.png', '03968.png', '15416.png', '20307.png', '15509.png', '15394.png', '24954.png', '03225.png', '08512.png', '17476.png', '02502.png', '04203.png', '00383.png', '02465.png', '11587.png', '22424.png', '03825.png', '11111.png', '03594.png', '16630.png', '11109.png', '14162.png', '04020.png', '09134.png', '16367.png', '08217.png', '11562.png', '02784.png', '14520.png', '01389.png', '18190.png', '06959.png', '09678.png', '06010.png', '04911.png', '08607.png', '14868.png', '11404.png', '02875.png', '10254.png', '18228.png', '13630.png', '01309.png', '07889.png', '06841.png', '16606.png', '20408.png', '00386.png', '21689.png', '24340.png', '19201.png', '19189.png', '04886.png', '10636.png', '20772.png', '16640.png', '10947.png', '19307.png', '19671.png', '12653.png', '21194.png', '08563.png', '22760.png', '12032.png', '21376.png', '00013.png', '11520.png', '06891.png', '10972.png', '23551.png', '03780.png', '13018.png', '05567.png', '04598.png', '03337.png', '02007.png', '07669.png', '06924.png', '17921.png', '18059.png', '13754.png', '02690.png', '09995.png', '12397.png', '10336.png', '02737.png', '22210.png', '06360.png', '23591.png', '10574.png', '23533.png', '23163.png', '20460.png', '01179.png', '02497.png', '02031.png', '02494.png', '24284.png', '02464.png', '15434.png', '09180.png', '05932.png', '04506.png', '18429.png', '14727.png', '03449.png', '21408.png', '18642.png', '11250.png', '15493.png', '04550.png', '13150.png', '06536.png', '17459.png', '21178.png', '14502.png', '22999.png', '20545.png', '06821.png', '10435.png', '17974.png', '05927.png', '09440.png', '09217.png', '18733.png', '18568.png', '04250.png', '11242.png', '15857.png', '17699.png', '24540.png', '18586.png', '01232.png', '14179.png', '21303.png', '02066.png', '16248.png', '07947.png', '16966.png', '18625.png', '15575.png', '17704.png', '13223.png', '16191.png', '13609.png', '14679.png', '00582.png', '02749.png', '24536.png', '13712.png', '07287.png', '00388.png', '05829.png', '11211.png', '15341.png', '17033.png', '06163.png', '22781.png', '20550.png', '15815.png', '06232.png', '00827.png', '12606.png', '04492.png', '13190.png', '00139.png', '06627.png', '20125.png', '07550.png', '02659.png', '04065.png', '02215.png', '17829.png', '09237.png', '01248.png', '23128.png', '09680.png', '14532.png', '06867.png', '02358.png', '18920.png', '01247.png', '15099.png', '22658.png', '20462.png', '12636.png', '20898.png', '00564.png', '24339.png', '14210.png', '23119.png', '05923.png', '21049.png', '12185.png', '00466.png', '07742.png', '04013.png', '07265.png', '20388.png', '05714.png', '06254.png', '23183.png', '13744.png', '13336.png', '09143.png', '00480.png', '05375.png', '03997.png', '08086.png', '14508.png', '14117.png', '13662.png', '07104.png', '05016.png', '22086.png', '22470.png', '15976.png', '18850.png', '00837.png', '20394.png', '23255.png', '17830.png', '11540.png', '12696.png', '07103.png', '03679.png', '11594.png', '24415.png', '19457.png', '18545.png', '07602.png', '03435.png', '24006.png', '02003.png', '22589.png', '13910.png', '00963.png', '12496.png', '21515.png', '24130.png', '13568.png', '09772.png', '10155.png', '12022.png', '19492.png', '09860.png', '07277.png', '07762.png', '22600.png', '12875.png', '18278.png', '20726.png', '05207.png', '03703.png', '18609.png', '12346.png', '19516.png', '08937.png', '07344.png', '20683.png', '11519.png', '03919.png', '17524.png', '22273.png', '17952.png', '16858.png', '09793.png', '12625.png', '06309.png', '24235.png', '11856.png', '16612.png', '23666.png', '17494.png', '16306.png', '18638.png', '13291.png', '21005.png', '14965.png', '20252.png', '13822.png', '11896.png', '00665.png', '00043.png', '15224.png', '08620.png', '18480.png', '16238.png', '10487.png', '09229.png', '09227.png', '21891.png', '15347.png', '22018.png', '23581.png', '05222.png', '17361.png', '19165.png', '21864.png', '18793.png', '11455.png', '15195.png', '11127.png', '00264.png', '01507.png', '08251.png', '21321.png', '23721.png', '11324.png', '09970.png', '12490.png', '11266.png', '05544.png', '16198.png', '15814.png', '01446.png', '03144.png', '23338.png', '23702.png', '13793.png', '17718.png', '17307.png', '22558.png', '12650.png', '00815.png', '11323.png', '22180.png', '22928.png', '15000.png', '14918.png', '02598.png', '03645.png', '22389.png', '01629.png', '18055.png', '19017.png', '01096.png', '16274.png', '12363.png', '20863.png', '22420.png', '01397.png', '20236.png', '05703.png', '17833.png', '17706.png', '14821.png', '01449.png', '18841.png', '14674.png', '15262.png', '20274.png', '00418.png', '07827.png', '02777.png', '02148.png', '24152.png', '20586.png', '05185.png', '08706.png', '16396.png', '16078.png', '03612.png', '21992.png', '24889.png', '04208.png', '08564.png', '05001.png', '06255.png', '21688.png', '08712.png', '11507.png', '15429.png', '16928.png', '09493.png', '12923.png', '24128.png', '08271.png', '00494.png', '23703.png', '11927.png', '19002.png', '22525.png', '00114.png', '19954.png', '10879.png', '03893.png', '04546.png', '15047.png', '17325.png', '05200.png', '00020.png', '04714.png', '23288.png', '19784.png', '00929.png', '15904.png', '18333.png', '14211.png', '20219.png', '04397.png', '04245.png', '19890.png', '07149.png', '08118.png', '10899.png', '04982.png', '05399.png', '11147.png', '13932.png', '17224.png', '20093.png', '00065.png', '23206.png', '10407.png', '20595.png', '21656.png', '16817.png', '19080.png', '20312.png', '13607.png', '13930.png', '03425.png', '14420.png', '17422.png', '16182.png', '12702.png', '15534.png', '08603.png', '16760.png', '04410.png', '00263.png', '08356.png', '05741.png', '15732.png', '19595.png', '20876.png', '13600.png', '17424.png', '19398.png', '24515.png', '24288.png', '03214.png', '02989.png', '04654.png', '01046.png', '11476.png', '01182.png', '04677.png', '16778.png', '04706.png', '05760.png', '11274.png', '04913.png', '21153.png', '22184.png', '09403.png', '02155.png', '24947.png', '15135.png', '21599.png', '07934.png', '16739.png', '10777.png', '08074.png', '21368.png', '22063.png', '22791.png', '09668.png', '16771.png', '06476.png', '23112.png', '21866.png', '02846.png', '19910.png', '19683.png', '02297.png', '00700.png', '21258.png', '17639.png', '21667.png', '16262.png', '18122.png', '24420.png', '06710.png', '00172.png', '18839.png', '01483.png', '21763.png', '10916.png', '07363.png', '11119.png', '09279.png', '17837.png', '03770.png', '18168.png', '02596.png', '07927.png', '13957.png', '05515.png', '16779.png', '24716.png', '08885.png', '04231.png', '09908.png', '09691.png', '06278.png', '21317.png', '22120.png', '22823.png', '13245.png', '19429.png', '11230.png', '07264.png', '11128.png', '00131.png', '19219.png', '14180.png', '13109.png', '20854.png', '05827.png', '19952.png', '12888.png', '15182.png', '18838.png', '23220.png', '05214.png', '02968.png', '19797.png', '19335.png', '09408.png', '10981.png', '22559.png', '06259.png', '17266.png', '19508.png', '04794.png', '07349.png', '20632.png', '02100.png', '24542.png', '24553.png', '19034.png', '10823.png', '20128.png', '15239.png', '04996.png', '14158.png', '17465.png', '04539.png', '09854.png', '01468.png', '18720.png', '18903.png', '07027.png', '01888.png', '06843.png', '23959.png', '10005.png', '16750.png', '13044.png', '20464.png', '00857.png', '20118.png', '16216.png', '17865.png', '23018.png', '08431.png', '11912.png', '16484.png', '18757.png', '18840.png', '10681.png', '24436.png', '16441.png', '18310.png', '08865.png', '21434.png', '22019.png', '23147.png', '13041.png', '09058.png', '16502.png', '01492.png', '11210.png', '03483.png', '06072.png', '09317.png', '04259.png', '08509.png', '20771.png', '17445.png', '06943.png', '00637.png', '18295.png', '16859.png', '12908.png', '23954.png', '09929.png', '05316.png', '02731.png', '23761.png', '23076.png', '05057.png', '14217.png', '16738.png', '09287.png', '20005.png', '14157.png', '08963.png', '24724.png', '17894.png', '20830.png', '06004.png', '19982.png', '16524.png', '01872.png', '11771.png', '04099.png', '08408.png', '22676.png', '08022.png', '11141.png', '20292.png', '17005.png', '13490.png', '03327.png', '00363.png', '03848.png', '12560.png', '16501.png', '17008.png', '13500.png', '22684.png', '03563.png', '00726.png', '02827.png', '04218.png', '05240.png', '17037.png', '02977.png', '14065.png', '06148.png', '22294.png', '18372.png', '16575.png', '01212.png', '11201.png', '11171.png', '00083.png', '18467.png', '07699.png', '04589.png', '23257.png', '18127.png', '03887.png', '03954.png', '09582.png', '17274.png', '04196.png', '15153.png', '05166.png', '17269.png', '21389.png', '21284.png', '14227.png', '14902.png', '00699.png', '09979.png', '06504.png', '17076.png', '00413.png', '23846.png', '10590.png', '09972.png', '01565.png', '10804.png', '05129.png', '04585.png', '11770.png', '02415.png', '11106.png', '14864.png', '18600.png', '23951.png', '12467.png', '23750.png', '20829.png', '15629.png', '07878.png', '21377.png', '18883.png', '08000.png', '12828.png', '21309.png', '14739.png', '21822.png', '24759.png', '03263.png', '22084.png', '21455.png', '01070.png', '01644.png', '15155.png', '14252.png', '12732.png', '18527.png', '00103.png', '01592.png', '04286.png', '23286.png', '20313.png', '07492.png', '15881.png', '12369.png', '20688.png', '09086.png', '00171.png', '22371.png', '12532.png', '14621.png', '04106.png', '23741.png', '12951.png', '20410.png', '13772.png', '13142.png', '14626.png', '00519.png', '10387.png', '21319.png', '16452.png', '14447.png', '09627.png', '08383.png', '24878.png', '04999.png', '07283.png', '23906.png', '07273.png', '02270.png', '18560.png', '23790.png', '19190.png', '13942.png', '24051.png', '12200.png', '02127.png', '11740.png', '05564.png', '16403.png', '14610.png', '15676.png', '05911.png', '22181.png', '12859.png', '22238.png', '19168.png', '19870.png', '21203.png', '07157.png', '01726.png', '16622.png', '15137.png', '15215.png', '12612.png', '15160.png', '23781.png', '11216.png', '24413.png', '00932.png', '00736.png', '07218.png', '16550.png', '00566.png', '16886.png', '04741.png', '22706.png', '14565.png', '12613.png', '22482.png', '20887.png', '04658.png', '20947.png', '03896.png', '24692.png', '11662.png', '17864.png', '23381.png', '04399.png', '21858.png', '12361.png', '02478.png', '09358.png', '11720.png', '07841.png', '03847.png', '12566.png', '13529.png', '02366.png', '06576.png', '11248.png', '23377.png', '23594.png', '01691.png', '08001.png', '20631.png', '12509.png', '22484.png', '03061.png', '18716.png', '16369.png', '21628.png', '24722.png', '04586.png', '03277.png', '15276.png', '18137.png', '06502.png', '14715.png', '05449.png', '19426.png', '14451.png', '05892.png', '20363.png', '12992.png', '20770.png', '08744.png', '11868.png', '04180.png', '16877.png', '19991.png', '14164.png', '24425.png', '21110.png', '08880.png', '02818.png', '09273.png', '03373.png', '24805.png', '20753.png', '16108.png', '17893.png', '06011.png', '16503.png', '15204.png', '12608.png', '00559.png', '00958.png', '17980.png', '12083.png', '16626.png', '07452.png', '14985.png', '11319.png', '05123.png', '11747.png', '10517.png', '11574.png', '03565.png', '04986.png', '23508.png', '18919.png', '24460.png', '02809.png', '13872.png', '04672.png', '06768.png', '19359.png', '00913.png', '21253.png', '05876.png', '04704.png', '07458.png', '12409.png', '00773.png', '06685.png', '00511.png', '05928.png', '24944.png', '15014.png', '21109.png', '02946.png', '09682.png', '12387.png', '17565.png', '18754.png', '07688.png', '11723.png', '19712.png', '16893.png', '17487.png', '18088.png', '18986.png', '04131.png', '24578.png', '07904.png', '12894.png', '21920.png', '22457.png', '17178.png', '08881.png', '03641.png', '19925.png', '01281.png', '21571.png', '24190.png', '23633.png', '14452.png', '20251.png', '12312.png', '21188.png', '13055.png', '11664.png', '14528.png', '15098.png', '20899.png', '12301.png', '20459.png', '12610.png', '07040.png', '07643.png', '23301.png', '08755.png', '15951.png', '19238.png', '11024.png', '06933.png', '22493.png', '13908.png', '04052.png', '20371.png', '21772.png', '19777.png', '10449.png', '09686.png', '14603.png', '23035.png', '05966.png', '08495.png', '22862.png', '02515.png', '02983.png', '01889.png', '09186.png', '23965.png', '02921.png', '03972.png', '10614.png', '18763.png', '23129.png', '22397.png', '20606.png', '07504.png', '18457.png', '14236.png', '11656.png', '07394.png', '21716.png', '16166.png', '19813.png', '17792.png', '02608.png', '22055.png', '20731.png', '19951.png', '15544.png', '11730.png', '20955.png', '13732.png', '12643.png', '14996.png', '12062.png', '08797.png', '17348.png', '22148.png', '21766.png', '18741.png', '19211.png', '06025.png', '03076.png', '21381.png', '07837.png', '13955.png', '13615.png', '11882.png', '05724.png', '00654.png', '17988.png', '18160.png', '21554.png', '08861.png', '10430.png', '18424.png', '03222.png', '13383.png', '13710.png', '22206.png', '12695.png', '01156.png', '12327.png', '00295.png', '15883.png', '13642.png', '19927.png', '01278.png', '12458.png', '03105.png', '02800.png', '14925.png', '01746.png', '04644.png', '13536.png', '22252.png', '15985.png', '22924.png', '04470.png', '10075.png', '09053.png', '07370.png', '24535.png', '07644.png', '20520.png', '02374.png', '14584.png', '06038.png', '19157.png', '13810.png', '02077.png', '24718.png', '12073.png', '09223.png', '13795.png', '24583.png', '13588.png', '19272.png', '12165.png', '09584.png', '17231.png', '13905.png', '14717.png', '04669.png', '05242.png', '22767.png', '14689.png', '11701.png', '04987.png', '06027.png', '22010.png', '18578.png', '19696.png', '23501.png', '08886.png', '11831.png', '03334.png', '04939.png', '05655.png', '04732.png', '09595.png', '13683.png', '06712.png', '08105.png', '07838.png', '15711.png', '06856.png', '16097.png', '02315.png', '12768.png', '17683.png', '24464.png', '13624.png', '20458.png', '04948.png', '06620.png', '02353.png', '13438.png', '09474.png', '08504.png', '04851.png', '03493.png', '23909.png', '23016.png', '16264.png', '11552.png', '03347.png', '06774.png', '16319.png', '07952.png', '13239.png', '01683.png', '10998.png', '18158.png', '21231.png', '20870.png', '06382.png', '21504.png', '18483.png', '23136.png', '17109.png', '02857.png', '24904.png', '22197.png', '04717.png', '17615.png', '01073.png', '20454.png', '14659.png', '22943.png', '07351.png', '06222.png', '00591.png', '05840.png', '11142.png', '03300.png', '00367.png', '17239.png', '02660.png', '20239.png', '05235.png', '04374.png', '07479.png', '13482.png', '08720.png', '09895.png', '22874.png', '07752.png', '11097.png', '23358.png', '01806.png', '03406.png', '22878.png', '01133.png', '07199.png', '09806.png', '24143.png', '12857.png', '15138.png', '16781.png', '23182.png', '02097.png', '11682.png', '08481.png', '00463.png', '22374.png', '00086.png', '02461.png', '23855.png', '04384.png', '04404.png', '12527.png', '23995.png', '10454.png', '00988.png', '20858.png', '12328.png', '06094.png', '00783.png', '06914.png', '09688.png', '16577.png', '16944.png', '01842.png', '24670.png', '01214.png', '24915.png', '21912.png', '24091.png', '03296.png', '01835.png', '10827.png', '02030.png', '18547.png', '20167.png', '11675.png', '09702.png', '03248.png', '11657.png', '19857.png', '18637.png', '06635.png', '00446.png', '15030.png', '18617.png', '04017.png', '00475.png', '00834.png', '22121.png', '09419.png', '14080.png', '05483.png', '19362.png', '03103.png', '00902.png', '06146.png', '05814.png', '09040.png', '18419.png', '01780.png', '21385.png', '09374.png', '15304.png', '12085.png', '15671.png', '10753.png', '10306.png', '10514.png', '09514.png', '23026.png', '02559.png', '09994.png', '12615.png', '12183.png', '10133.png', '23726.png', '16003.png', '06077.png', '07716.png', '01450.png', '10200.png', '14327.png', '06391.png', '12915.png', '14390.png', '22478.png', '08366.png', '18129.png', '17063.png', '12938.png', '12731.png', '09624.png', '24092.png', '01224.png', '19024.png', '13214.png', '07274.png', '02866.png', '14730.png', '01665.png', '10182.png', '15087.png', '10512.png', '23168.png', '19361.png', '09248.png', '10805.png', '03591.png', '08824.png', '01642.png', '12432.png', '11123.png', '14704.png', '10664.png', '13709.png', '03658.png', '15192.png', '02059.png', '02482.png', '03379.png', '12881.png', '03524.png', '16361.png', '18567.png', '19986.png', '19757.png', '00839.png', '11848.png', '02475.png', '02364.png', '19403.png', '02402.png', '12791.png', '14095.png', '04673.png', '08918.png', '00312.png', '15934.png', '12583.png', '07611.png', '19117.png', '16908.png', '21950.png', '08560.png', '08571.png', '20327.png', '15078.png', '18302.png', '21776.png', '03950.png', '08227.png', '11416.png', '12822.png', '07642.png', '06098.png', '03026.png', '16268.png', '09523.png', '16001.png', '20897.png', '19856.png', '20785.png', '04869.png', '07736.png', '02692.png', '20124.png', '04490.png', '20141.png', '01756.png', '14214.png', '16952.png', '18164.png', '22302.png', '17665.png', '05835.png', '15197.png', '20309.png', '03735.png', '15049.png', '23473.png', '02141.png', '22960.png', '06109.png', '05060.png', '09587.png', '16510.png', '18471.png', '20362.png', '12843.png', '23861.png', '18387.png', '01136.png', '04295.png', '11030.png', '15897.png', '06965.png', '05283.png', '02721.png', '17419.png', '14829.png', '09100.png', '24886.png', '07073.png', '21707.png', '15948.png', '09312.png', '10793.png', '19290.png', '14007.png', '13994.png', '09334.png', '17119.png', '13657.png', '20321.png', '19941.png', '11227.png', '07570.png', '16906.png', '23514.png', '21543.png', '05070.png', '22595.png', '02772.png', '14656.png', '16206.png', '16618.png', '22790.png', '22158.png', '09320.png', '16273.png', '17326.png', '18751.png', '23025.png', '22764.png', '07399.png', '17922.png', '04094.png', '24230.png', '17945.png', '20499.png', '12673.png', '22226.png', '11630.png', '07983.png', '00297.png', '14792.png', '03033.png', '13670.png', '01508.png', '07521.png', '08941.png', '10009.png', '12683.png', '07616.png', '09311.png', '18393.png', '12528.png', '09326.png', '22521.png', '11331.png', '22443.png', '08710.png', '21249.png', '08747.png', '12359.png', '24726.png', '05938.png', '23882.png', '09350.png', '20453.png', '12987.png', '24019.png', '19832.png', '04786.png', '02209.png', '19831.png', '13124.png', '24165.png', '16214.png', '12659.png', '24631.png', '03732.png', '20481.png', '24697.png', '20021.png', '10178.png', '10175.png', '02161.png', '22426.png', '23822.png', '08659.png', '24956.png', '00178.png', '20073.png', '06353.png', '14511.png', '17948.png', '01174.png', '16611.png', '21202.png', '07522.png', '02286.png', '07233.png', '08033.png', '18708.png', '18167.png', '13604.png', '20022.png', '06594.png', '14816.png', '08517.png', '22729.png', '19914.png', '23627.png', '08849.png', '04057.png', '23536.png', '11534.png', '07645.png', '04862.png', '21298.png', '01763.png', '19197.png', '05844.png', '20104.png', '01404.png', '09558.png', '01170.png', '18807.png', '06226.png', '19353.png', '19150.png', '04062.png', '22205.png', '15023.png', '17534.png', '21913.png', '06797.png', '18450.png', '17635.png', '03012.png', '10770.png', '13535.png', '03234.png', '09066.png', '20548.png', '11183.png', '13199.png', '07278.png', '17024.png', '04148.png', '05608.png', '15872.png', '18196.png', '13587.png', '04583.png', '06949.png', '03135.png', '13449.png', '05338.png', '16674.png', '00161.png', '05153.png', '17538.png', '15538.png', '17732.png', '21805.png', '22759.png', '12752.png', '05937.png', '16316.png', '07012.png', '06409.png', '16689.png', '01684.png', '06421.png', '06505.png', '04582.png', '07867.png', '20644.png', '22498.png', '12450.png', '09426.png', '07342.png', '09381.png', '18686.png', '05414.png', '03884.png', '07751.png', '19657.png', '02267.png', '10895.png', '15015.png', '01284.png', '11180.png', '22507.png', '05081.png', '06338.png', '11654.png', '17115.png', '15024.png', '02293.png', '05774.png', '20825.png', '03584.png', '04304.png', '21239.png', '07354.png', '03169.png', '21904.png', '24645.png', '00241.png', '19612.png', '16642.png', '07029.png', '14721.png', '14750.png', '21702.png', '16922.png', '04328.png', '16385.png', '13276.png', '24934.png', '23489.png', '20133.png', '12295.png', '14707.png', '13043.png', '07664.png', '12569.png', '09219.png', '24496.png', '01351.png', '20920.png', '16694.png', '02668.png', '01497.png', '17869.png', '11393.png', '20958.png', '11732.png', '09526.png', '05749.png', '24276.png', '12575.png', '18710.png', '16339.png', '04756.png', '13549.png', '05004.png', '17854.png', '23478.png', '14898.png', '01639.png', '16488.png', '17285.png', '24443.png', '08438.png', '04921.png', '21499.png', '12674.png', '03261.png', '05304.png', '07886.png', '19018.png', '08743.png', '14054.png', '09260.png', '04616.png', '24843.png', '06195.png', '15008.png', '02665.png', '05559.png', '13847.png', '00943.png', '10335.png', '22973.png', '18091.png', '14766.png', '05493.png', '22549.png', '11100.png', '20369.png', '18509.png', '06775.png', '02094.png', '04908.png', '14165.png', '20861.png', '00768.png', '01862.png', '16930.png', '07930.png', '21968.png', '05864.png', '08400.png', '24598.png', '12413.png', '05506.png', '15577.png', '17516.png', '19710.png', '10103.png', '05791.png', '06289.png', '07595.png', '19250.png', '08610.png', '14663.png', '05980.png', '11273.png', '08887.png', '00461.png', '08898.png', '17884.png', '10275.png', '02413.png', '04896.png', '22092.png', '04855.png', '06522.png', '23980.png', '00871.png', '18620.png', '18762.png', '07545.png', '14719.png', '24004.png', '04066.png', '24389.png', '16336.png', '16380.png', '11917.png', '05772.png', '02457.png', '11590.png', '01670.png', '23619.png', '01626.png', '03148.png', '16405.png', '02218.png', '17312.png', '07236.png', '20295.png', '14692.png', '06846.png', '17803.png', '16508.png', '09022.png', '03581.png', '20776.png', '00689.png', '20567.png', '07776.png', '09774.png', '24109.png', '14246.png', '14811.png', '17769.png', '00238.png', '22253.png', '22571.png', '17673.png', '08723.png', '23963.png', '21471.png', '14712.png', '22203.png', '14773.png', '00975.png', '10656.png', '13520.png', '24098.png', '02881.png', '13093.png', '13906.png', '09498.png', '17504.png', '24662.png', '12812.png', '22971.png', '17503.png', '23087.png', '04084.png', '24813.png', '22651.png', '10925.png', '19207.png', '06272.png', '13103.png', '03095.png', '20797.png', '03859.png', '20539.png', '14126.png', '04816.png', '09159.png', '10885.png', '06314.png', '21796.png', '15613.png', '07630.png', '23154.png', '05710.png', '01511.png', '10693.png', '09213.png', '16266.png', '03600.png', '15330.png', '10277.png', '12135.png', '16623.png', '19973.png', '12297.png', '18285.png', '06043.png', '21129.png', '08189.png', '10732.png', '17612.png', '14233.png', '00464.png', '15016.png', '08677.png', '19567.png', '11335.png', '06423.png', '00755.png', '08112.png', '24243.png', '20613.png', '16280.png', '19847.png', '24447.png', '09781.png', '05751.png', '01478.png', '13728.png', '11881.png', '23201.png', '10604.png', '07228.png', '05040.png', '20423.png', '03466.png', '11387.png', '23692.png', '08513.png', '07143.png', '06939.png', '22997.png', '16110.png', '13962.png', '18985.png', '13244.png', '13895.png', '10901.png', '22985.png', '04167.png', '10611.png', '00375.png', '21468.png', '07487.png', '24750.png', '01159.png', '22657.png', '22756.png', '02770.png', '10873.png', '10968.png', '07615.png', '23561.png', '02013.png', '02972.png', '01500.png', '02252.png', '17614.png', '22929.png', '09874.png', '10321.png', '18750.png', '07216.png', '09897.png', '15913.png', '20556.png', '19903.png', '11254.png', '02471.png', '00273.png', '23700.png', '04121.png', '03827.png', '20640.png', '10473.png', '11842.png', '23879.png', '23689.png', '14771.png', '07302.png', '20570.png', '15765.png', '10798.png', '02417.png', '10287.png', '08428.png', '06646.png', '00635.png', '13888.png', '08455.png', '09901.png', '13999.png', '06696.png', '00882.png', '22023.png', '00571.png', '03801.png', '11435.png', '14769.png', '23907.png', '17991.png', '13697.png', '04169.png', '19826.png', '17610.png', '23694.png', '08635.png', '03324.png', '02603.png', '10386.png', '13967.png', '04133.png', '14097.png', '01748.png', '13411.png', '06087.png', '24906.png', '04993.png', '02607.png', '21829.png', '03038.png', '22025.png', '17056.png', '20867.png', '16615.png', '17147.png', '11548.png', '13972.png', '23908.png', '06155.png', '23872.png', '08303.png', '18274.png', '06218.png', '11303.png', '01951.png', '00803.png', '17205.png', '22492.png', '06981.png', '12943.png', '03768.png', '23675.png', '21980.png', '12501.png', '01706.png', '12266.png', '19747.png', '05770.png', '00521.png', '03670.png', '10905.png', '02929.png', '07862.png', '24008.png', '02969.png', '11154.png', '01901.png', '07912.png', '11964.png', '09314.png', '04479.png', '19474.png', '06664.png', '11762.png', '10832.png', '21282.png', '09921.png', '14810.png', '06444.png', '02583.png', '11945.png', '22691.png', '24076.png', '21927.png', '00653.png', '07866.png', '13419.png', '16383.png', '06727.png', '18033.png', '10412.png', '12947.png', '13838.png', '13655.png', '12742.png', '19424.png', '23368.png', '05017.png', '08804.png', '03223.png', '17571.png', '03281.png', '20051.png', '03659.png', '10102.png', '07569.png', '17712.png', '08166.png', '21880.png', '13330.png', '19245.png', '18197.png', '13616.png', '17973.png', '07852.png', '07882.png', '01717.png', '10934.png', '19802.png', '23194.png', '08765.png', '03299.png', '18476.png', '00980.png', '02196.png', '15139.png', '09395.png', '24416.png', '14078.png', '03160.png', '17661.png', '14796.png', '20444.png', '01607.png', '16195.png', '01537.png', '15431.png', '12054.png', '15567.png', '00616.png', '17794.png', '24207.png', '13185.png', '10388.png', '24392.png', '15910.png', '12406.png', '12607.png', '09190.png', '10569.png', '14006.png', '01165.png', '20465.png', '21999.png', '08521.png', '21027.png', '21956.png', '18144.png', '05837.png', '09986.png', '15952.png', '12736.png', '06708.png', '02399.png', '19596.png', '18135.png', '04447.png', '02585.png', '01962.png', '24580.png', '15322.png', '17969.png', '11760.png', '05653.png', '05128.png', '22254.png', '23569.png', '15249.png', '17710.png', '10365.png', '21791.png', '21767.png', '02935.png', '10840.png', '18736.png', '12520.png', '19383.png', '05248.png', '19850.png', '18989.png', '07508.png', '24849.png', '09911.png', '09036.png', '04136.png', '10195.png', '19199.png', '08687.png', '18972.png', '07091.png', '14049.png', '03176.png', '03143.png', '17603.png', '21429.png', '00315.png', '11276.png', '19833.png', '10593.png', '10549.png', '09044.png', '13541.png', '17662.png', '06366.png', '13133.png', '10824.png', '16074.png', '21768.png', '20090.png', '21730.png', '20901.png', '07638.png', '16647.png', '13324.png', '06067.png', '22262.png', '13398.png', '15596.png', '09728.png', '18877.png', '03631.png', '20661.png', '21053.png', '19678.png', '24704.png', '20564.png', '22611.png', '22534.png', '11488.png', '08211.png', '10532.png', '07810.png', '08482.png', '13684.png', '02279.png', '15598.png', '20089.png', '01551.png', '15039.png', '04303.png', '04634.png', '11468.png', '07168.png', '00734.png', '19023.png', '18495.png', '04924.png', '11375.png', '18207.png', '08367.png', '14916.png', '14185.png', '22524.png', '00994.png', '21327.png', '17629.png', '21103.png', '22913.png', '12396.png', '07759.png', '14073.png', '13162.png', '06328.png', '13062.png', '09203.png', '05324.png', '17208.png', '13704.png', '21834.png', '06003.png', '11108.png', '03559.png', '19801.png', '02245.png', '06837.png', '01374.png', '24953.png', '24862.png', '14937.png', '08004.png', '05331.png', '09633.png', '08760.png', '18011.png', '19296.png', '17886.png', '06901.png', '22188.png', '04372.png', '07379.png', '20607.png', '18868.png', '09249.png', '11297.png', '01722.png', '02855.png', '24033.png', '22553.png', '00930.png', '12329.png', '04001.png', '18944.png', '02669.png', '08862.png', '09007.png', '09091.png', '20098.png', '22128.png', '24765.png', '17220.png', '20150.png', '24869.png', '17758.png', '06822.png', '17547.png', '06589.png', '19129.png', '22166.png', '18066.png', '15397.png', '07204.png', '16434.png', '10613.png', '15923.png', '00610.png', '18623.png', '12948.png', '04530.png', '06008.png', '15342.png', '22320.png', '20722.png', '19872.png', '24403.png', '18412.png', '01818.png', '06179.png', '06363.png', '05154.png', '14708.png', '00190.png', '03185.png', '09303.png', '20755.png', '04719.png', '13123.png', '18050.png', '10820.png', '08757.png', '18683.png', '05420.png', '16979.png', '03562.png', '05381.png', '18758.png', '13597.png', '24868.png', '00005.png', '14645.png', '01101.png', '09542.png', '23541.png', '15907.png', '15069.png', '14859.png', '18945.png', '10952.png', '20208.png', '19264.png', '15469.png', '01693.png', '13629.png', '24758.png', '20216.png', '09090.png', '06325.png', '19072.png', '08144.png', '11406.png', '23352.png', '15705.png', '07740.png', '18202.png', '04534.png', '10012.png', '13460.png', '23139.png', '19461.png', '08291.png', '15414.png', '14736.png', '20366.png', '08447.png', '16568.png', '15392.png', '19066.png', '16557.png', '10771.png', '14657.png', '11643.png', '02761.png', '10337.png', '03060.png', '22015.png', '06349.png', '11223.png', '00374.png', '16062.png', '08179.png', '24831.png', '08053.png', '13487.png', '07305.png', '14861.png', '08449.png', '08992.png', '04843.png', '03066.png', '16241.png', '01754.png', '11967.png', '02707.png', '05257.png', '04464.png', '24145.png', '12313.png', '22830.png', '20287.png', '01020.png', '13694.png', '06123.png', '01910.png', '12119.png', '22122.png', '21223.png', '07769.png', '07485.png', '16170.png', '09759.png', '10471.png', '05695.png', '17120.png', '08337.png', '08077.png', '01388.png', '21710.png', '16148.png', '24931.png', '11932.png', '05111.png', '06928.png', '20961.png', '10598.png', '01445.png', '19123.png', '13071.png', '23977.png', '05898.png', '01938.png', '19163.png', '17302.png', '12145.png', '19690.png', '16976.png', '19662.png', '06915.png', '11325.png', '04844.png', '13691.png', '09234.png', '22915.png', '22103.png', '12868.png', '12403.png', '24814.png', '24485.png', '10556.png', '19228.png', '23385.png', '14283.png', '07928.png', '02523.png', '08872.png', '20780.png', '14761.png', '04769.png', '15079.png', '20023.png', '17108.png', '07684.png', '10629.png', '17475.png', '09608.png', '03605.png', '24592.png', '13917.png', '09483.png', '11575.png', '00390.png', '17423.png', '20229.png', '06860.png', '18687.png', '06150.png', '21196.png', '24715.png', '22322.png', '07181.png', '24218.png', '23722.png', '05007.png', '23454.png', '07674.png', '02125.png', '00153.png', '03319.png', '15063.png', '08738.png', '14580.png', '19818.png', '04092.png', '04780.png', '20498.png', '08584.png', '10519.png', '16129.png', '06588.png', '24074.png', '13053.png', '16840.png', '20346.png', '14927.png', '22949.png', '18648.png', '19141.png', '19871.png', '13595.png', '11841.png', '00573.png', '14805.png', '12195.png', '16977.png', '14208.png', '15303.png', '23744.png', '22287.png', '07245.png', '14023.png', '10495.png', '01285.png', '05847.png', '10050.png', '08451.png', '11889.png', '09225.png', '19327.png', '12055.png', '00061.png', '03864.png', '22136.png', '24537.png', '03467.png', '18240.png', '10289.png', '14030.png', '01950.png', '13936.png', '23824.png', '24303.png', '08133.png', '00067.png', '06128.png', '22683.png', '07637.png', '11738.png', '04396.png', '07142.png', '12789.png', '02986.png', '20534.png', '07662.png', '01098.png', '21807.png', '23437.png', '01068.png', '18138.png', '13874.png', '12661.png', '02954.png', '12408.png', '04392.png', '03561.png', '01623.png', '01897.png', '04767.png', '06552.png', '09480.png', '05689.png', '04823.png', '06048.png', '18090.png', '09849.png', '11409.png', '24214.png', '03257.png', '20904.png', '24888.png', '10176.png', '08115.png', '12698.png', '00456.png', '24717.png', '05490.png', '18268.png', '15293.png', '01562.png', '14459.png', '03761.png', '23793.png', '21143.png', '01441.png', '10356.png', '18012.png', '07219.png', '13983.png', '23690.png', '04531.png', '13429.png', '21230.png', '04949.png', '18180.png', '09552.png', '05861.png', '03438.png', '06339.png', '13975.png', '12572.png', '01766.png', '11570.png', '21320.png', '16130.png', '23785.png', '00298.png', '11985.png', '14835.png', '23809.png', '20473.png', '03866.png', '06952.png', '04618.png', '12061.png', '13356.png', '06912.png', '03961.png', '02238.png', '02700.png', '13589.png', '17511.png', '23256.png', '06375.png', '21300.png', '14842.png', '05996.png', '08587.png', '02546.png', '19089.png', '21405.png', '05767.png', '01472.png', '14115.png', '15253.png', '09672.png', '05165.png', '22116.png', '05439.png', '00117.png', '07194.png', '07438.png', '23661.png', '24181.png', '05345.png', '21518.png', '20967.png', '15829.png', '22354.png', '00646.png', '24380.png', '03006.png', '15618.png', '22716.png', '02186.png', '22806.png', '24189.png', '11800.png', '16442.png', '04870.png', '22616.png', '01091.png', '01747.png', '11083.png', '19536.png', '17754.png', '01321.png', '08829.png', '04156.png', '20159.png', '04258.png', '20050.png', '10518.png', '21426.png', '10554.png', '16667.png', '06819.png', '02765.png', '00796.png', '06668.png', '22832.png', '19574.png', '08532.png', '23641.png', '17457.png', '12931.png', '12808.png', '21076.png', '22466.png', '05889.png', '09160.png', '15621.png', '17118.png', '01171.png', '01092.png', '18120.png', '04366.png', '10424.png', '12713.png', '04045.png', '24911.png', '11931.png', '22139.png', '15588.png', '00361.png', '22515.png', '07202.png', '20194.png', '08439.png', '14595.png', '09550.png', '12688.png', '11029.png', '22436.png', '19999.png', '07818.png', '09871.png', '21739.png', '11370.png', '07256.png', '13299.png', '07994.png', '00990.png', '11911.png', '01975.png', '07951.png', '13945.png', '00035.png', '21712.png', '05556.png', '04532.png', '00515.png', '09038.png', '00318.png', '11736.png', '13641.png', '04795.png', '09116.png', '05011.png', '07964.png', '24365.png', '03446.png', '17896.png', '09280.png', '14514.png', '17006.png', '20117.png', '08057.png', '01264.png', '04039.png', '11582.png', '22648.png', '01632.png', '07385.png', '02873.png', '20752.png', '08371.png', '04313.png', '00201.png', '22202.png', '01465.png', '20807.png', '05003.png', '14243.png', '10340.png', '17189.png', '19131.png', '23502.png', '09620.png', '03512.png', '00652.png', '00997.png', '15425.png', '00879.png', '01882.png', '13943.png', '03448.png', '09123.png', '05419.png', '09920.png', '14380.png', '16720.png', '16468.png', '21566.png', '11268.png', '17677.png', '01051.png', '17627.png', '07490.png', '00687.png', '09894.png', '18678.png', '00437.png', '23062.png', '23685.png', '04724.png', '07032.png', '18070.png', '01166.png', '13738.png', '20162.png', '16633.png', '20217.png', '06760.png', '11529.png', '04063.png', '12291.png', '08577.png', '19649.png', '21135.png', '06344.png', '23113.png', '00788.png', '01581.png', '13286.png', '09101.png', '12740.png', '21621.png', '14398.png', '15742.png', '00066.png', '17508.png', '02384.png', '13794.png', '10781.png', '15956.png', '13266.png', '08375.png', '07588.png', '17230.png', '08444.png', '11712.png', '03699.png', '20573.png', '01239.png', '05945.png', '23304.png', '01123.png', '07345.png', '24122.png', '21640.png', '01935.png', '03044.png', '14467.png', '15233.png', '08837.png', '11602.png', '15452.png', '10587.png', '20265.png', '17345.png', '03353.png', '20549.png', '03140.png', '21308.png', '14101.png', '13606.png', '17339.png', '10932.png', '12475.png', '19697.png', '16309.png', '24633.png', '19630.png', '21544.png', '18276.png', '15245.png', '12781.png', '16814.png', '22667.png', '21056.png', '23743.png', '21764.png', '02409.png', '17533.png', '14219.png', '20560.png', '05319.png', '02566.png', '11940.png', '16051.png', '07879.png', '12057.png', '02828.png', '04074.png', '20100.png', '21418.png', '20409.png', '04414.png', '15165.png', '24699.png', '08040.png', '12604.png', '18230.png', '23100.png', '23762.png', '14408.png', '05476.png', '03944.png', '00162.png', '11846.png', '21072.png', '06194.png', '12269.png', '07750.png', '19749.png', '01190.png', '06570.png', '03256.png', '17243.png', '16307.png', '06750.png', '19122.png', '03064.png', '01240.png', '14624.png', '23960.png', '09683.png', '03110.png', '13476.png', '14756.png', '18654.png', '01223.png', '17742.png', '04398.png', '18532.png', '09148.png', '24618.png', '22279.png', '09676.png', '09425.png', '15296.png', '07343.png', '09944.png', '06288.png', '17180.png', '13243.png', '20220.png', '22256.png', '17978.png', '06171.png', '16377.png', '02897.png', '12735.png', '08460.png', '04885.png', '14337.png', '15780.png', '21091.png', '22879.png', '08778.png', '15272.png', '12883.png', '10913.png', '12953.png', '05626.png', '18498.png', '05027.png', '15477.png', '15693.png', '09067.png', '06032.png', '09780.png', '06922.png', '24924.png', '18473.png', '15365.png', '22615.png', '10416.png', '17636.png', '21983.png', '12656.png', '04960.png', '10461.png', '10810.png', '13567.png', '01484.png', '17689.png', '18328.png', '08433.png', '23024.png', '09640.png', '11737.png', '15113.png', '02310.png', '18969.png', '18103.png', '13522.png', '10199.png', '05492.png', '16161.png', '12199.png', '18112.png', '13360.png', '21119.png', '22596.png', '03171.png', '22885.png', '18068.png', '24859.png', '05018.png', '22223.png', '21831.png', '22263.png', '19369.png', '07111.png', '08264.png', '15816.png', '08825.png', '03158.png', '12442.png', '08665.png', '13944.png', '19384.png', '14401.png', '24233.png', '11260.png', '24928.png', '21484.png', '07207.png', '19534.png', '07903.png', '14977.png', '00728.png', '13724.png', '10055.png', '18926.png', '16240.png', '20886.png', '00957.png', '02619.png', '12189.png', '21138.png', '10235.png', '07019.png', '04120.png', '23394.png', '17872.png', '04846.png', '20790.png', '10023.png', '13344.png', '06385.png', '04841.png', '00408.png', '09251.png', '13564.png', '23565.png', '06064.png', '16120.png', '02797.png', '15875.png', '02590.png', '15703.png', '03172.png', '05143.png', '07690.png', '24025.png', '12855.png', '03971.png', '06028.png', '16709.png', '10726.png', '20784.png', '15092.png', '16203.png', '14593.png', '07107.png', '19713.png', '18318.png', '09601.png', '08038.png', '09140.png', '11928.png', '13671.png', '00991.png', '15633.png', '20146.png', '02087.png', '15270.png', '04808.png', '09077.png', '13382.png', '19212.png', '01226.png', '20020.png', '09351.png', '16212.png', '04525.png', '18400.png', '03601.png', '19428.png', '17647.png', '01316.png', '16425.png', '08694.png', '15035.png', '02620.png', '11631.png', '03328.png', '13875.png', '12878.png', '17905.png', '12167.png', '21080.png', '16982.png', '24014.png', '13841.png', '22509.png', '08536.png', '17062.png', '13747.png', '18843.png', '19557.png', '09539.png', '01111.png', '14650.png', '22292.png', '18978.png', '21533.png', '16017.png', '23411.png', '17074.png', '02757.png', '21044.png', '03611.png', '03571.png', '20143.png', '09807.png', '09333.png', '17730.png', '24605.png', '03737.png', '10765.png', '19449.png', '18672.png', '01599.png', '10475.png', '05763.png', '15212.png', '00474.png', '13598.png', '22177.png', '15789.png', '12596.png', '08394.png', '24827.png', '15691.png', '24448.png', '24256.png', '22012.png', '06074.png', '01999.png', '24007.png', '23639.png', '21517.png', '06820.png', '18779.png', '11808.png', '13210.png', '04091.png', '17835.png', '10818.png', '10754.png', '24595.png', '24265.png', '02840.png', '00862.png', '05983.png', '19795.png', '12398.png', '11001.png', '22144.png', '10739.png', '15252.png', '22020.png', '13702.png', '12635.png', '20530.png', '11168.png', '04594.png', '12628.png', '14748.png', '06257.png', '22856.png', '23326.png', '15017.png', '17751.png', '02329.png', '15473.png', '14251.png', '12841.png', '04329.png', '04699.png', '11231.png', '18910.png', '24366.png', '21392.png', '17870.png', '18921.png', '21033.png', '24898.png', '05918.png', '22267.png', '22049.png', '06024.png', '03802.png', '01383.png', '24808.png', '10409.png', '17839.png', '06845.png', '22717.png', '00660.png', '01425.png', '19839.png', '02439.png', '06494.png', '17337.png', '17454.png', '07428.png', '00947.png', '10986.png', '21420.png', '23704.png', '07395.png', '16083.png', '10054.png', '18452.png', '20628.png', '04142.png', '12747.png', '19305.png', '10654.png', '01814.png', '10313.png', '17350.png', '19275.png', '13484.png', '15783.png', '14409.png', '21741.png', '22530.png', '22386.png', '24509.png', '16666.png', '23635.png', '06618.png', '13496.png', '10444.png', '12283.png', '13494.png', '21106.png', '12059.png', '07180.png', '19743.png', '08695.png', '20604.png', '22245.png', '06410.png', '15697.png', '08464.png', '02484.png', '02880.png', '05475.png', '10428.png', '18823.png', '15048.png', '15843.png', '03326.png', '21451.png', '12985.png', '05079.png', '05675.png', '15735.png', '15406.png', '23923.png', '23967.png', '12473.png', '20651.png', '20516.png', '16645.png', '13685.png', '00808.png', '20263.png', '05620.png', '02128.png', '18610.png', '19840.png', '21341.png', '11625.png', '23556.png', '24055.png', '19887.png', '07539.png', '23613.png', '14658.png', '20917.png', '13307.png', '00843.png', '19724.png', '02971.png', '00455.png', '24832.png', '14487.png', '19573.png', '08348.png', '11070.png', '11345.png', '18704.png', '16038.png', '03803.png', '23369.png', '16045.png', '22976.png', '06428.png', '16698.png', '13357.png', '10568.png', '16607.png', '11232.png', '10869.png', '16912.png', '21586.png', '06078.png', '11774.png', '23852.png', '17968.png', '22669.png', '06287.png', '01907.png', '01498.png', '06070.png', '14332.png', '07795.png', '21684.png', '06411.png', '21960.png', '18309.png', '24854.png', '03800.png', '18003.png', '04712.png', '17979.png', '07559.png', '11779.png', '06873.png', '02273.png', '06910.png', '08296.png', '13777.png', '03715.png', '10680.png', '17490.png', '14429.png', '00378.png', '18271.png', '14448.png', '00321.png', '18316.png', '05325.png', '11716.png', '23862.png', '08314.png', '21291.png', '17256.png', '10078.png', '06738.png', '15363.png', '24795.png', '04510.png', '09043.png', '01330.png', '00353.png', '18557.png', '24756.png', '19694.png', '21875.png', '16027.png', '10163.png', '05157.png', '14870.png', '23402.png', '12096.png', '10152.png', '03416.png', '18166.png', '24753.png', '08324.png', '05992.png', '16054.png', '04637.png', '24115.png', '07778.png', '02967.png', '22685.png', '06944.png', '21548.png', '18468.png', '11597.png', '21425.png', '23889.png', '22266.png', '18917.png', '22520.png', '04800.png', '22377.png', '06720.png', '00099.png', '11414.png', '18494.png', '00054.png', '21073.png', '09648.png', '20061.png', '08803.png', '04584.png', '11286.png', '17846.png', '05711.png', '20620.png', '16831.png', '07813.png', '02381.png', '13721.png', '04234.png', '16534.png', '06040.png', '06806.png', '11989.png', '00254.png', '13183.png', '05621.png', '02392.png', '06486.png', '11353.png', '21023.png', '20250.png', '06102.png', '10940.png', '00833.png', '19658.png', '20705.png', '19415.png', '10982.png', '05297.png', '05205.png', '11363.png', '16687.png', '19981.png', '20592.png', '00658.png', '06264.png', '10576.png', '10248.png', '23085.png', '07779.png', '10951.png', '17999.png', '16690.png', '20645.png', '02204.png', '11270.png', '13076.png', '11218.png', '16871.png', '20559.png', '02269.png', '11586.png', '19288.png', '03793.png', '13958.png', '10404.png', '06714.png', '17619.png', '15198.png', '00333.png', '23042.png', '09062.png', '12597.png', '22332.png', '00407.png', '21886.png', '08651.png', '09121.png', '06969.png', '21817.png', '07410.png', '02134.png', '04059.png', '21152.png', '14052.png', '05565.png', '18605.png', '16518.png', '20748.png', '01464.png', '15325.png', '02915.png', '15194.png', '12267.png', '14486.png', '18427.png', '16467.png', '24837.png', '13952.png', '12020.png', '03990.png', '23656.png', '05572.png', '24400.png', '05006.png', '15686.png', '02712.png', '03384.png', '03018.png', '06942.png', '21116.png', '08180.png', '16073.png', '15669.png', '18435.png', '22065.png', '13792.png', '23350.png', '16999.png', '16632.png', '20112.png', '18472.png', '23710.png', '09507.png', '05550.png', '12203.png', '21305.png', '17707.png', '07802.png', '22786.png', '20452.png', '19481.png', '22805.png', '06248.png', '22034.png', '09621.png', '16609.png', '06670.png', '16478.png', '16116.png', '14959.png', '19116.png', '20122.png', '18029.png', '19753.png', '04242.png', '12537.png', '24095.png', '20041.png', '12007.png', '09656.png', '08916.png', '22813.png', '04348.png', '15184.png', '24921.png', '09131.png', '06649.png', '00290.png', '09535.png', '16485.png', '08572.png', '10713.png', '11939.png', '03106.png', '14755.png', '02519.png', '12689.png', '15831.png', '24081.png', '05019.png', '09299.png', '10266.png', '24238.png', '14981.png', '13555.png', '12932.png', '14668.png', '08748.png', '01540.png', '01853.png', '09889.png', '22777.png', '20467.png', '05582.png', '03053.png', '21440.png', '05110.png', '01891.png', '04898.png', '04287.png', '22699.png', '07162.png', '01242.png', '03320.png', '16228.png', '12058.png', '14854.png', '11277.png', '05797.png', '03746.png', '11470.png', '18728.png', '21331.png', '04902.png', '16117.png', '22418.png', '08154.png', '06611.png', '17745.png', '08546.png', '18325.png', '15710.png', '10379.png', '00764.png', '07341.png', '08159.png', '17726.png', '20510.png', '18650.png', '23596.png', '21959.png', '10949.png', '00824.png', '01405.png', '12994.png', '16382.png', '22637.png', '17582.png', '14029.png', '03302.png', '20233.png', '04976.png', '09742.png', '06229.png', '10015.png', '04141.png', '24769.png', '22569.png', '02150.png', '12101.png', '08716.png', '23517.png', '12268.png', '16537.png', '03428.png', '20235.png', '08792.png', '15542.png', '14745.png', '07512.png', '06717.png', '10980.png', '05048.png', '21390.png', '05457.png', '21745.png', '14154.png', '24630.png', '14775.png', '23777.png', '17168.png', '02521.png', '18256.png', '11525.png', '11016.png', '19473.png', '03564.png', '22236.png', '03767.png', '17435.png', '18655.png', '02998.png', '03019.png', '09340.png', '17061.png', '06986.png', '09646.png', '20695.png', '10315.png', '20200.png', '12372.png', '09887.png', '05664.png', '13727.png', '07415.png', '21204.png', '19770.png', '03842.png', '10117.png', '10515.png', '23866.png', '20999.png', '15314.png', '02295.png', '01564.png', '22485.png', '24037.png', '03427.png', '23452.png', '17260.png', '11057.png', '04708.png', '10844.png', '20060.png', '21703.png', '20603.png', '13842.png', '22295.png', '23576.png', '19320.png', '09816.png', '00625.png', '15115.png', '01412.png', '07118.png', '08387.png', '04023.png', '20026.png', '13008.png', '05825.png', '13233.png', '17148.png', '02499.png', '21121.png', '11031.png', '03029.png', '23791.png', '17327.png', '14308.png', '13457.png', '13474.png', '10677.png', '24917.png', '24654.png', '16068.png', '11403.png', '21313.png', '00078.png', '20668.png', '22726.png', '20834.png', '13623.png', '05052.png', '03027.png', '00470.png', '07470.png', '00394.png', '01660.png', '09658.png', '19203.png', '15879.png', '24018.png', '05307.png', '15860.png', '20428.png', '24563.png', '00359.png', '01153.png', '19771.png', '06937.png', '23520.png', '11394.png', '18860.png', '22394.png', '06110.png', '00949.png', '22400.png', '11502.png', '23456.png', '12471.png', '02428.png', '16128.png', '07524.png', '11050.png', '22221.png', '17410.png', '16927.png', '02577.png', '09228.png', '22298.png', '01082.png', '15823.png', '10109.png', '08832.png', '09179.png', '11963.png', '04093.png', '17079.png', '09756.png', '10670.png', '03208.png', '14107.png', '09659.png', '00706.png', '09766.png', '07850.png', '05024.png', '22901.png', '08062.png', '14492.png', '12995.png', '04320.png', '19525.png', '17701.png', '10013.png', '04810.png', '16608.png', '18674.png', '06778.png', '07958.png', '18409.png', '15638.png', '04426.png', '11803.png', '24465.png', '15095.png', '12758.png', '21644.png', '20052.png', '03490.png', '09157.png', '16006.png', '02355.png', '23925.png', '16107.png', '07705.png', '07074.png', '00025.png', '01454.png', '20240.png', '23571.png', '04709.png', '06175.png', '16710.png', '07652.png', '19589.png', '21780.png', '12416.png', '24414.png', '22311.png', '19949.png', '13660.png', '24401.png', '18259.png', '14442.png', '24225.png', '10167.png', '06662.png', '19366.png', '18294.png', '03341.png', '02939.png', '22101.png', '01926.png', '01545.png', '03895.png', '03496.png', '11966.png', '06835.png', '00537.png', '20885.png', '11010.png', '06356.png', '20605.png', '00950.png', '22779.png', '19987.png', '03981.png', '15840.png', '24952.png', '16942.png', '23305.png', '24835.png', '21506.png', '09079.png', '11357.png', '02259.png', '09927.png', '09003.png', '03057.png', '22810.png', '09458.png', '17439.png', '23510.png', '19834.png', '08283.png', '15500.png', '17866.png', '18968.png', '06302.png', '09415.png', '07587.png', '12212.png', '08422.png', '21481.png', '19078.png', '00729.png', '17177.png', '06555.png', '00136.png', '02104.png', '14652.png', '08421.png', '13028.png', '17081.png', '18909.png', '03979.png', '00228.png', '08741.png', '07212.png', '19093.png', '01859.png', '14971.png', '23046.png', '19013.png', '06876.png', '24581.png']
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
        
        if i_iter == 254276:
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
