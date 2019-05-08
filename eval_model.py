import argparse
import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
from pathlib import Path
import glob

from time import time

# PyTorch includes
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
import networks.deeplab_resnet as resnet
from layers.loss import class_balanced_cross_entropy_loss
from dataloaders.helpers import *

from networks.deeplab_resnet import ClassifierModule, PSPModule

from tqdm import tqdm

def test(net, loader, save_dir_res):
    # Setting parameters
    relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
    zero_pad_crop = True  # Insert zero padding when cropping the image
    # Main Testing Loop
    for sample_batched in tqdm(loader, leave=False, desc='Building Results'):
        inputs = sample_batched['concat']
        gts = sample_batched['gt']
        metas = sample_batched['meta']

        # Forward of the mini-batch
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear',
                           align_corners=True)
        output_sigmoid = torch.sigmoid(outputs).cpu().numpy()
        
        #for jj in range(int(inputs.size()[0])):
        for jj, (img, obj) in enumerate(zip(metas['image'], metas['object'])):
            pred = output_sigmoid
            pred = np.transpose(pred[jj, ...], (1, 2, 0))
            pred = np.squeeze(pred)
            
            gt = tens2image(gts[jj, ...])
            bbox = get_bbox(gt, pad=relax_crop, zero_pad=zero_pad_crop)
            result = crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop,
                                   relax=relax_crop)

            # Save the result, attention to the index jj
            image_name = '{}-{}.png'.format(img, obj)
            sm.imsave(str(save_dir_res / image_name), result)

def eval_model(net, save_dir, batch_size=10):
    # Setting parameters
    relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
    zero_pad_crop = True  # Insert zero padding when cropping the image

    net.eval()
    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        tr.ToImage(norm_elem='extreme_points'),
        tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.ToTensor()])
    db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts, retname=True)
    testloader = DataLoader(db_test, batch_size=1,
                            shuffle=False, num_workers=2)

    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        test(net, testloader, save_dir)

if __name__ == '__main__':

    # Setting parameters
    use_sbd = False
    nEpochs = 100  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume

    p = OrderedDict()  # Parameters to include in report
    classifier = 'psp'  # Head classifier to use
    p['trainBatch'] = 12  # Training batch size
    testBatch = 12  # Testing batch size
    useTest = 1  # See evolution of the test set when training?
    nTestInterval = 10  # Run on test set every nTestInterval epochs
    snapshot = 20  # Store a model every snapshot epochs
    relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
    nInputChannels = 4  # Number of input channels (RGB + heatmap of extreme points)
    zero_pad_crop = True  # Insert zero padding when cropping the image
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = 1e-8  # Learning rate
    p['wd'] = 0.0005  # Weight decay
    p['momentum'] = 0.9  # Momentum

    device = torch.device('cuda:1')
    if False:
        net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
        net.load_state_dict(torch.load('run_-1/models/dextr_pascal-sbd.pth',
                                   map_location=device))
    else:
        if True:
            densenet = models.densenet121(pretrained=True)
            densenet.features[0] = nn.Conv2d(nInputChannels, 64, kernel_size=7,
                                             stride=2, padding=3, bias=False)
            classifier = PSPModule(512, sizes=(1,2,3,6,12))
            first = densenet.features[0]
            net = nn.Sequential(densenet.features[:-5], classifier)
        else:
            net = nn.Sequential(nn.Conv2d(nInputChannels, 3, kernel_size=2, stride=1,
                                          padding=1, bias=False),
                                models.densenet121(pretrained=True).features,
                                nn.Conv2d(1024, 1, kernel_size=2, stride=1, padding=1))
        net.load_state_dict(torch.load('run_9/models/dextr_pascal_epoch-299.pth',
                                       map_location=device))
    net.to(device)
    net.eval()
    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        tr.ToImage(norm_elem='extreme_points'),
        tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.ToTensor()])
    db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts, retname=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    save_dir = 'run_9'
    save_dir_res = os.path.join(save_dir, 'Results')
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    print('Testing Network')
    ious = [[] for _ in range(10)]
    with torch.no_grad():
        # Main Testing Loop
        for ii, sample_batched in enumerate(tqdm(testloader)):

            inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']

            # Forward of the mini-batch
            inputs = inputs.to(device)
            begin = time()
            outputs = net.forward(inputs)
            #print('Forward: ', time() - begin)
            outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
            #print('Upsample: ', time() - begin)
            #outputs = outputs.to(torch.device('cpu'))
            output_sigmoid = torch.sigmoid(outputs).cpu().numpy()
            for i, iou in enumerate(ious):
                pred = (np.squeeze(output_sigmoid) > (0.5 + 0.05*i))
                val = sample_batched['crop_gt'].numpy().astype(np.int)
                union = np.logical_or(pred, val).sum()
                isect = np.logical_and(pred, val).sum()
                #print('IOU: ', isect / union, ' U: ', union, ' IS: ', isect)
                iou.append(isect / union)
            #loss = class_balanced_cross_entropy_loss(outputs, sample_batched['crop_gt'].to('cpu'),
            #                                         size_average=False,
            #                                         batch_average=True)
            #print(loss)
            if True:
                for jj in range(int(inputs.size()[0])):
                    pred = torch.sigmoid(outputs)
                    pred = np.transpose(pred.cpu().numpy()[jj, ...], (1, 2, 0))
                    pred = np.squeeze(pred)
                    
                    gt = tens2image(gts[jj, :, :, :])
                    bbox = get_bbox(gt, pad=relax_crop, zero_pad=zero_pad_crop)
                    result = crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop, relax=relax_crop)

                    # Save the result, attention to the index jj
                    sm.imsave(os.path.join(save_dir_res, metas['image'][jj] + '-' + metas['object'][jj] + '.png'), result)
    print('Average IOU: ', np.mean(iou))
