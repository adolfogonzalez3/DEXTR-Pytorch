
import argparse
import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
from pathlib import Path
import glob

# PyTorch includes
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

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

from nets import create_densenet, create_squeezenet, create_shufflenet

from tqdm import trange, tqdm

def create_transforms(relax_crop, zero_crop):
    # Preparation of the data loaders
    first = [tr.CropFromMask(crop_elems=('image', 'gt'),
                             relax=relax_crop, zero_pad=zero_crop),
             tr.FixedResize(resolutions={'crop_image': (512, 512),
                                         'crop_gt': (512, 512)})]
    second = [tr.ToImage(norm_elem='extreme_points'),
              tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
              tr.ToTensor()]
    train_tf = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        *first, tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'), *second])
    test_tf = transforms.Compose([
        *first, tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'), *second])
    return train_tf, test_tf

def test(net, testloader, device):
    running_loss = 0
    net.eval()
    with torch.no_grad():
        for ii, sample_batched in enumerate(tqdm(testloader, leave=False)):
            inputs = sample_batched['concat'].to(device)
            gts = sample_batched['crop_gt'].to(device)

            # Forward pass of the mini-batch
            output = net.forward(inputs)
            output = interpolate(output, size=(512, 512), mode='bilinear',
                                 align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts,
                                                     size_average=False)
            running_loss += loss.item()
    return running_loss / len(testloader)

def load_model(model_name, nInputChannels):
    if model_name == 'densenet':
        return create_densenet(nInputChannels)
    elif model_name == 'squeezenet':
        return create_squeezenet(nInputChannels)
    elif model_name == 'shufflenet':
        return create_shufflenet(nInputChannels)

def train(model_name, gpu_id, learning_rate):
    # Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        tqdm.write('Using GPU: {} '.format(gpu_id))

    # Setting parameters
    use_sbd = False
    nEpochs = 100  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume

    p = OrderedDict()  # Parameters to include in report
    classifier = 'psp'  # Head classifier to use
    p['trainBatch'] = 5  # Training batch size
    testBatch = 5  # Testing batch size
    useTest = True  # See evolution of the test set when training?
    nTestInterval = 10  # Run on test set every nTestInterval epochs
    snapshot = 10  # Store a model every snapshot epochs
    relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
    nInputChannels = 4  # Number of input channels (RGB + heatmap of extreme points)
    zero_pad_crop = True  # Insert zero padding when cropping the image
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = learning_rate #1e-4  # Learning rate
    p['wd'] = 0.0005  # Weight decay
    p['momentum'] = 0.9  # Momentum

    # Results and model directories (a new directory is generated for every run)
    package_path = Path(__file__).resolve()
    folder_name = 'runs-{}-{:f}'.format(model_name, learning_rate)
    save_dir_root = package_path.parent / 'RUNS'
    exp_name = model_name
    save_dir = save_dir_root / folder_name
    (save_dir / 'models').mkdir(parents=True, exist_ok=True)
    with (save_dir / 'log.csv').open('wt') as csv:
        csv.write('train_loss,test_loss\n')
    tqdm.write(str(save_dir))
    

    # Network definition
    modelName = model_name
    net = load_model(model_name, nInputChannels)
    #if resume_epoch == 0:
    #    print("Initializing from pretrained Deeplab-v2 model")
    #else:
    #    weights_path = save_dir / 'models'
    #    weights_path /= '%s_epoch-%d.pth' % (modelName, resume_epoch-1)
    #    print("Initializing weights from: ", weights_path)
    #    net.load_state_dict(torch.load(weights_path,
    #                                   map_location=lambda s, _: s))
    train_params = [{'params': net.parameters(), 'lr': p['lr']}]

    net.to(device)

    # Training the network
    if resume_epoch != nEpochs:
        # Logging into Tensorboard
        time_now = datetime.now().strftime('%b%d_%H-%M-%S')
        hostname = socket.gethostname()
        log_dir = save_dir / 'models' / '{}_{}'.format(time_now, hostname)
        writer = SummaryWriter(log_dir=str(log_dir))

        # Use the following optimizer
        #optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
        optimizer = optim.Adam(train_params, lr=p['lr'], weight_decay=p['wd'])
        p['optimizer'] = str(optimizer)

        # Preparation of the data loaders
        train_tf, test_tf = create_transforms(relax_crop, zero_pad_crop)
        voc_train = pascal.VOCSegmentation(split='train', download=True,
                                           transform=train_tf)
        voc_val = pascal.VOCSegmentation(split='val', download=True,
                                         transform=test_tf)

        if use_sbd:
            sbd = sbd.SBDSegmentation(split=['train', 'val'], retname=True,
                                      transform=train_tf)
            db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
        else:
            db_train = voc_train

        p['dataset_train'] = str(db_train)
        p['transformations_train'] = [str(t) for t in train_tf.transforms]
        p['dataset_test'] = str(db_train)
        p['transformations_test'] = [str(t) for t in test_tf.transforms]

        trainloader = DataLoader(db_train, batch_size=p['trainBatch'],
                                 shuffle=True, num_workers=2)
        testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False,
                                num_workers=2)
        generate_param_report((save_dir / exp_name).with_suffix('.txt'), p)

        # Train variables
        num_img_tr = len(trainloader)
        num_img_ts = len(testloader)
        running_loss_tr = 0.0
        running_loss_ts = 0.0
        aveGrad = 0
        #print("Training Network")
        # Main Training and Testing Loop
        for epoch in trange(resume_epoch, nEpochs):
            start_time = timeit.default_timer()

            net.train()
            for ii, sample_batched in enumerate(tqdm(trainloader, leave=False)):
                inputs = sample_batched['concat'].to(device)
                gts = sample_batched['crop_gt'].to(device)

                # Forward-Backward of the mini-batch
                inputs.requires_grad_()
                
                output = net.forward(inputs)#.cpu()
                #print(output.shape)
                #exit()
                output = interpolate(output, size=(512, 512), mode='bilinear',
                                     align_corners=True).to(device)
                # Compute the losses, side outputs and fuse
                loss = class_balanced_cross_entropy_loss(output, gts,
                                                         size_average=False,
                                                         batch_average=True)
                running_loss_tr += loss.item()
                #print(loss.item())

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(),
                                      ii + num_img_tr * epoch)
                    optimizer.step()
                    optimizer.zero_grad()
                    aveGrad = 0                

            # Save the model
            if (epoch + 1) % snapshot == 0:
                weights_path = save_dir / 'models'
                weights_path /= '{}_epoch-{:d}.pth'.format(modelName, epoch)
                torch.save(net.state_dict(), weights_path)

            # One testing epoch
            if useTest and (epoch + 1) % nTestInterval == 0:
                msg = 'Test Loss: {:.3f}'
                test_loss = test(net, testloader, device)
                tqdm.write(msg.format(test_loss))
                running_loss_tr = running_loss_tr / num_img_tr
                with (save_dir / 'log.csv').open('at') as csv:
                    csv.write(','.join([str(running_loss_tr),
                                        str(test_loss)]))
                    csv.write('\n')
                writer.add_scalar('data/total_loss_epoch', running_loss_tr,
                                    epoch)
                num_images = ii*p['trainBatch']+inputs.data.shape[0]
                msg = '[Epoch: {:d}, numImages: {:5d}]'
                tqdm.write(msg.format(epoch, num_images))
                tqdm.write('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                msg = "Execution time: {:.3f}"
                tqdm.write(msg.format(stop_time - start_time))

        writer.close()

if __name__ == '__main__':
    for lr in [1e-5, 1e-6, 1e-7]:
        train('squeezenet', 0, lr)
