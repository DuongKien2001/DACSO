import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

from umap import UMAP
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d
from utils.helpers import colorize_mask
from torchvision import transforms
import utils.palette as palette

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()

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

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(data_list, class_num, dataset, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
    return aveJ

def evaluate(model, dataset, ignore_label=250, save_output_images=False, save_dir=None, input_size=(512,1024), model1=None):

    if dataset == 'cityscapes':
        num_classes = 19
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        ignore_label = 250

    elif dataset == 'gta':
        num_classes = 19
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        test_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', img_size=(1280,720), mean=IMG_MEAN)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)
        ignore_label = 255

    print('Evaluating, found ' + str(len(testloader)) + ' images.')

    data_list = []
    colorize = VOCColorize()
    colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta", "violet", "lime", "darkred", "springgreen", "darkcyan", "tan"])
    """class_names = ["road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle"]"""

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        size = size[0]
        #if index > 500:
        #    break
        with torch.no_grad():
            output, feature  = model(Variable(image).cuda())
            output = interp(output)
            output1 = output
            #output1_o  = model1(Variable(image).cuda())[0]
            #output1_o = interp(output1_o)
            #output1_o = output1_o
            
            label_cuda = Variable(label.long()).cuda()
            criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??
            loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()
            feature = feature.cpu().data[0].numpy()
            feature = feature.transpose(1,2,0)
            a,b,_ = feature.shape
            feature = feature.reshape([-1, 2048])

            #umap2d = UMAP(init='random', random_state=0)

            #proj_2d = umap2d.fit_transform(feature)
            label1 = F.interpolate(label.unsqueeze(1).float(), size=(a,b), mode='nearest').squeeze(1).long()
            label1 = label1.cpu().data[0].numpy()
            label1 = label1.reshape(-1)
            
            #   plt.scatter(proj[:,0], proj[:,1], color = colors[i])

            #plt.show()
            #plt.savefig('dacs/'+str(index)+'.png')
            #plt.figure().clear()

            if dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)
            elif dataset == 'gta':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)

            
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)

            data_list.append([gt.flatten(), output.flatten()])
            """
            if index < 500:  
                
                save_image(image[0].cpu(),index,'_input',palette.CityScpates_palette)
                _, pred_u_s = torch.max(output1, dim=1)
                #_, pred = torch.max(output1_o, dim=1)
                save_image(pred_u_s[0].cpu(),index,'_pred',palette.CityScpates_palette)
                #save_image(pred[0].cpu(),index,'_pred_o',palette.CityScpates_palette)
                save_image(label[0].cpu(), index,'_label',palette.CityScpates_palette)
                
                ll = [1,0,13,14]
                lf = [0,0,0,0]
                for i in range(4):
                    m = label1 == ll[i]
                    f = feature[m]
                    if lf[i] == 0:
                        lf[i] = f
                    else:
                        lf[i] = np.append(lf[i], f, axis = 0)

            if index == 480:
                u = np.append(lf[0], lf[1], axis = 0)
                u = np.append(u, lf[2], axis = 0)
                u = np.append(u, lf[3], axis = 0)
                umap2d = UMAP(init='random', random_state=0)

                proj_2d = umap2d.fit_transform(u)
                plt.scatter(proj_2d[0:lf[0].shape[0],0], proj_2d[0:lf[0].shape[0]:,1], color = colors[0])
                v = lf[0].shape[0]
                plt.scatter(proj_2d[v:v+lf[1].shape[0],0], proj_2d[v:v+lf[1].shape[0]:,1], color = colors[1])
                v = v+lf[1].shape[0]
                plt.scatter(proj_2d[v:v+lf[2].shape[0],0], proj_2d[v:v+lf[2].shape[0]:,1], color = colors[2])
                v = v+lf[2].shape[0]
                plt.scatter(proj_2d[v:v+lf[3].shape[0],0], proj_2d[v:v+lf[3].shape[0]:,1], color = colors[3])
                plt.savefig('dacs/'+'a.png')
                plt.figure().clear()
                print('save success')
            """      
            
        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    mIoU = get_iou(data_list, num_classes, dataset, filename)
    loss = np.mean(total_loss)
    return mIoU, loss

def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #model = torch.nn.DataParallel(Res_Deeplab(num_classes=num_classes), device_ids=args.gpu)
    model = Res_Deeplab(num_classes=num_classes)
    #model1 = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    #checkpoint1 = torch.load(args.model_path_o)
    try:
        model.load_state_dict(checkpoint['ema_model'])
        #model1.load_state_dict(checkpoint1['ema_model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['ema_model'])

    model.cuda()
    model.eval()
    #model1.cuda()
    #model1.eval()

    evaluate(model, dataset, ignore_label=ignore_label, save_output_images=args.save_output_images, save_dir=save_dir, input_size=input_size)


if __name__ == '__main__':
    args = get_arguments()

    config = torch.load(args.model_path)['config']

    dataset = config['dataset']
    #dataset = 'cityscapes'
    if dataset == 'cityscapes':
        num_classes = 19
        input_size = (512,1024)
    if dataset == 'gta':
        num_classes = 19
        input_size = (1280,720)

    ignore_label = config['ignore_label']
    save_dir = os.path.join(*args.model_path.split('/')[:-1])

    main()
