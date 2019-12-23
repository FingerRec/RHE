import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse
from utils import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--save_model', type=str, default='checkpoints/')
parser.add_argument('--dataset',default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something_something_v1'])
parser.add_argument('--root', type=str, default="")
parser.add_argument('--arch',default='i3d', type=str, choices=['i3d', 'mpi3d', 'nli3d', 'mfnet', 'fli3d', 'mpi3d_pt', 'i3dpt'])
parser.add_argument('--train_list', default='data/kinetics_rgb_train_list.txt', type=str)
parser.add_argument('--val_list', default='data/kinetics_rgb_val_list.txt', type=str)
parser.add_argument('--cluster_list', default='data/kinetics_rgb_cluster_train_list.txt', type=str)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--cluster_train', type=int, default=0)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--stride', default=1, type=int,help='stride of temporal image')
parser.add_argument('--weights', default="", type=str,help='checkpoints')
##learing stragety
parser.add_argument('--dropout', '--do', default=0.64, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--mixup', type=int, help ='if use mixup do data augmentation', default=0)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-7)')
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 20, 25, 30, 35, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

#=====================Runtime Config
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision import transforms
import videotransforms
from utils import *

import numpy as np

from net.i3d_1fc import InceptionI3d
from net.multi_path_i3d import MultiPathI3d
from net.fl_i3d import FullLengthI3d
from net.basic_models.label_smooth import SLSloss
#from charades_dataset import Charades as Dataset
from util.eda import VideoEda

from tensorboardX import SummaryWriter
import shutil
import datetime
best_prec1 = 0
torch.manual_seed(1)
date_time = datetime.datetime.today().strftime('%m-%d-%H%M')


def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print(weight_dict)
    model_dict.update(weight_dict)
    return model_dict


def main():
    global best_prec1, args
    # setup dataset
    if args.dataset == 'something_something_v1':
        train_transforms = transforms.Compose([
                                               videotransforms.RandomCrop(112)
        ])
        test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])
    else:
        train_transforms = transforms.Compose([
                                               videotransforms.RandomCrop(224),
                                               videotransforms.RandomHorizontalFlip(),
        ])
        test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    if args.arch == 'fli3d':
        from dataset.fl_ucf101_dataset import I3dDataSet
        segments = 2 # 3 x 64 input
    else:
        from dataset.ucf101_dataset import I3dDataSet
        segments = 1
    if args.dataset == 'ucf101':
        num_class = 101
        data_length = 64
        image_tmpl = "frame{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        data_length = 64
        image_tmpl = "img_{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        data_length = 64
        image_tmpl = "img_{:05d}.jpg"
    elif args.dataset == 'something_something_v1':
        num_class = 174
        data_length = 8
        image_tmpl = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    dataset = I3dDataSet(args.root, args.train_list, num_segments=segments,
                   new_length=data_length,
                   stride=args.stride,
                   modality=args.mode,
                   dataset = args.dataset,
                   test_mode=False,
                   image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    if args.cluster_train:
        cluster_train_dataset = I3dDataSet(args.root, args.cluster_list, num_segments=segments,
                       new_length=data_length,
                       stride = args.stride,
                       modality=args.mode,
                       dataset = args.dataset,
                       image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                       transform=train_transforms)
        cluster_dataloader = torch.utils.data.DataLoader(cluster_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                                 pin_memory=True)
    val_dataset = I3dDataSet(args.root, args.val_list, num_segments=segments,
                   new_length=data_length,
                   stride= args.stride,
                   modality=args.mode,
                   test_mode=True,
                   dataset = args.dataset,
                   image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_logger = Logger('logs/' + args.dataset + '/val.log', ['epoch', 'acc'])
    # setup the model
    if args.mode == 'flow':
        if args.arch == 'i3d':
            i3d = InceptionI3d(num_classes=num_class, in_channels=2)
        elif args.arch == 'mpi3d':
            i3d = MultiPathI3d(num_classes=num_class, in_channels=2, dropout_prob=args.dropout)
        elif args.arch == 'mpi3d_pt':
            from net.multi_path_i3d_pt import I3D
            i3d = I3D(num_classes=num_class,modality='flow', dropout_prob=args.dropout)
        elif args.arch == 'fli3d':
            i3d = FullLengthI3d(num_classes=num_class, in_channels=2, dropout_prob=args.dropout)
        elif args.arch == 'i3dpt':
            from net.i3dpt import I3D
            i3d = I3D(num_classes=num_class,modality='flow', dropout_prob=args.dropout)
        else:
            print("unsupport arch now")
        if args.weights=="":
            if args.arch == 'mpi3d_pt' or args.arch == 'i3dpt':
                pretrain_dict = torch.load('pretrained_models/model_flow.pth')
            else:
                pretrain_dict = torch.load('pretrained_models/flow_imagenet.pt')
            model_dict = i3d.state_dict()
            model_dict = weight_transform(model_dict, pretrain_dict)
            i3d.load_state_dict(model_dict)
        else:
            checkpoint = torch.load(args.weights)
            print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
            # base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            # i3d.load_state_dict(base_dict)
            pretrain_dict =  {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            model_dict = i3d.state_dict()
            model_dict = weight_transform(model_dict, pretrain_dict)
            i3d.load_state_dict(model_dict)
    else:
        if args.arch == 'i3d':
            i3d = InceptionI3d(num_classes=num_class, in_channels=3, dropout_prob=args.dropout)
        elif args.arch == 'mpi3d':
            i3d = MultiPathI3d(num_classes=num_class, in_channels=3, dropout_prob=args.dropout)
        elif args.arch == 'mpi3d_pt':
            from net.multi_path_i3d_pt import I3D
            i3d = I3D(num_classes=num_class,modality='rgb', dropout_prob=args.dropout)
        elif args.arch == 'i3dpt':
            from net.i3dpt import I3D
            i3d = I3D(num_classes=num_class,modality='rgb', dropout_prob=args.dropout)

        elif args.arch == 'fli3d':
            i3d = FullLengthI3d(num_classes=num_class, in_channels=3, dropout_prob=args.dropout)
        else:
            print("unsupport arch now")
        if args.weights=="":
            if args.arch == 'mpi3d_pt' or args.arch == 'i3dpt':
                pretrain_dict = torch.load('pretrained_models/model_rgb.pth')
            else:
                pretrain_dict = torch.load('pretrained_models/rgb_imagenet.pt')
            model_dict = i3d.state_dict()
            model_dict = weight_transform(model_dict, pretrain_dict)
            i3d.load_state_dict(model_dict)
        else:
            checkpoint = torch.load(args.weights)
            print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
            # base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            # i3d.load_state_dict(base_dict)
            pretrain_dict =  {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            model_dict = i3d.state_dict()
            model_dict = weight_transform(model_dict, pretrain_dict)
            i3d.load_state_dict(model_dict)

    i3d = nn.DataParallel(i3d).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            i3d.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {}) best_prec1 {}"
                  .format(args.evaluate, checkpoint['epoch'], best_prec1)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    if args.arch == 'mpi3d_pt' or args.arch == 'i3dpt':
        criterion = torch.nn.NLLLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()#we use transfer_learning hrer?
    # ===========================add for label smooth========================
    # add smooth loss with 0.1
    # criterion = SLSloss()
    # parameters =  transfer_learning_stragety(i3d, args.lr)
    if args.arch == 'mpi3d_pt':
        # from net.multi_path_i3d_pt import get_fine_tuning_parameters
        # parameters = get_fine_tuning_parameters(i3d, args.lr)
        parameters = i3d.parameters()
    else:
        parameters = i3d.parameters()
    optimizer = optim.SGD(parameters,
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    writer_1 = SummaryWriter('log/' + args.dataset + '/' + date_time + '/plot_1') #create log folders for plot
    writer_2 = SummaryWriter('log/' + args.dataset + '/' + date_time + '/plot_2')
    with open('logs/' + args.dataset + '/' + args.arch + '_'  + args.mode + '_' + args.dataset + '_gpu' + args.gpus + '_validation.txt', 'a') as f:
        f.write("begin_time:{} ".format(str(time.time())))
        f.write("dataset: {} ".format(args.dataset))
        f.write("arch: {} ".format(args.arch))
        f.write("lr: {} ".format(args.lr))
        f.write("dropout: {} ".format(args.dropout))
        f.write("weight_decay {} ".format(args.weight_decay))
        f.write('\n')
        f.write('*'*50)
        f.write('\n')
    timer = Timer()
    for epoch in range(args.start_epoch, args.epochs):
        timer.tic()
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        alpha = 0.5
        if epoch % 3 == 0 and args.cluster_train:
            train_prec1, train_loss = train(cluster_dataloader, i3d, criterion, optimizer, epoch, alpha)
        else:
            train_prec1, train_loss = train(train_dataloader, i3d, criterion, optimizer, epoch, alpha)

            writer_1.add_scalar('Train/Accu', train_prec1, epoch)
            writer_1.add_scalar('Train/Loss', train_loss, epoch)
            #writer_1.flush()
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_loss = validate(val_dataloader, i3d, criterion, (epoch + 1) * len(train_dataloader), alpha)
            writer_2.add_scalar('Val/Accu', prec1, epoch)
            writer_2.add_scalar('Val/Loss', val_loss, epoch)
            #writer_2.flush()
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "i3d",
                'state_dict': i3d.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, best_prec1)
            val_logger.log({
                'epoch': epoch,
                'acc': prec1
            })
        timer.toc()
        left_time = timer.average_time * (args.epochs - epoch)
        print("best_prec1 is: {}".format(best_prec1))
        print("left time is: {}".format(timer.format(left_time)))
        with open('logs/' + args.dataset + '/' + args.arch + '_' + args.mode + '_' + args.dataset + '_gpu' + args.gpus + '_validation.txt', 'a') as f:
            f.write(str(epoch))
            f.write(" ")
            f.write(str(train_prec1))
            f.write(" ")
            f.write(str(prec1))
            f.write(" ")
            f.write(timer.format(timer.diff))
            f.write('\n')


def train(train_loader, model, criterion, optimizer, epoch, alpha=0.5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    MIXUP = args.mixup
    # mixup = CombineMix(1)
    # mixup = GridMix(1)
    mixup = SpatialMixup(1)
    # mixup = CutMix(1.0, 1.0)
    # mixup = Cut(1.0, 0.4)
    # mixup = PartMix(1.0)
    # mixup = VideoShake(1.0)
    # videoEda = VideoEda(alpha_sr=0, alpha_ri=0, alpha_rs=0.1, p_rd=0.1, num_aug=1)
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        # loss = criterion(output, target_var)
        if MIXUP:
            input_var = input.cuda()
            target_var = target.cuda()
            input_var = torch.autograd.Variable(input_var)
            target_var = torch.autograd.Variable(target_var)
            inputs = mixup.mixup_data(input_var)
            # inputs = videoEda.eda(input_var)
            # inputs, target_a, target_b, lam = mixup.mixup_data(input_var, target_var)
            # inputs, target_a, target_b = map(Variable, (inputs, target_a, target_b))
            output = model(inputs)
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            # loss = mixup.mixup_criterion(criterion, output, target_a, target_b, lam)
            # prec1, prec5 = accuracy_mixup(output.data, target_var, target_a, target_b, lam, topk=(1, 5))
        else:
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            # ===========================add for multi label loss==================
            if args.arch == 'mpi3d':
                main_path, path_s, path_m, path_l = model(input_var)
                # main_path, path_m, path_l = model(input_var)
                loss = criterion(main_path, target_var)
                path_s = criterion(path_s, target_var)
                path_m = criterion(path_m, target_var)
                path_l = criterion(path_l, target_var)
                loss += 0.2 * path_s + 0.5 * path_m + 0.1 * path_l
                # loss += 0.5*path_m + 0.1*path_l
                output = main_path
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            elif args.arch == 'mpi3d_pt':

                main_path, path_s, path_m, path_l = model(input_var)
                # main_path, path_m, path_l = model(input_var)
                loss = criterion(main_path, target_var)
                # path_s_1 = criterion(path_s_1, target_var)
                path_s = criterion(path_s, target_var)
                path_m = criterion(path_m, target_var)
                path_l = criterion(path_l, target_var)
                # loss = loss + path_s + path_m + path_l
                rate = epoch / 45
                # loss += 0.2 * path_s + 0.5 * path_m + 0.1 * path_l
                # loss +=  (0.5*rate) * path_s + (0.5 - 0.3*rate) * path_m + (1.01 - rate) * path_l
                import math
                loss += 0.3 * math.log(1 + (math.exp(1) - 1) * rate) * path_s + (
                            0.5 - 0.5 * math.log(1 + (math.exp(1) - 1) * rate)) * path_m + (
                                1.01 - math.log(1 + (math.exp(1) - 1) * rate)) * path_l
                output = path_s + path_m + path_l + main_path
                # output = main_path + 0.2 * path_s + 0.5 * path_m + 0.1 * path_l
                '''
                loss += (1 - math.log(1 + (math.exp(1) - 1) * rate)) * path_s + (
                             0.5 * math.log(1 + (math.exp(1) - 1) * rate)) * path_m + (
                                 math.log(1 + (math.exp(1) - 1) * rate)) * path_l
                '''
                # loss += 0.5*path_m + 0.1*path_l
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                # adaptively training, for each sample, select a suitable path to optimize, should begin form epoch 10
                # or use progressive?
                '''
                main_path, path_s, path_m, path_l = model(input_var)
                b, c = main_path.size()
                output = main_path.data # initalize
                rate = epoch/45
                thresh = -1
                if rate > thresh:
                    loss = None
                    for j in range(b):
                        if loss is None:
                            loss = min(criterion(main_path[j, :].view(1, c), target_var[j].view(1)),
                                             criterion(path_s[j, :].view(1, c), target_var[j].view(1)),
                                             criterion(path_m[j, :].view(1, c), target_var[j].view(1)),
                                             criterion(path_l[j, :].view(1, c), target_var[j].view(1)))
                        else:
                            loss += min(criterion(main_path[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_s[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_m[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_l[j, :].view(1, c), target_var[j].view(1)))
                        values = np.array((criterion(main_path[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_s[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_m[j, :].view(1, c), target_var[j].view(1)),
                                              criterion(path_l[j, :].view(1, c), target_var[j].view(1))))
                        index = np.argmin(values)
                        if index == 0:
                            output[j, :] = main_path[j, :]
                        elif index == 1:
                            output[j, :] = path_s[j, :]
                        elif index == 2:
                            output[j, :] = path_m[j, :]
                        else:
                            output[j, :] = path_l[j, :]
                else:
                    output = path_s + path_m + path_l + main_path
                    loss = criterion(main_path, target_var)
                    path_s = criterion(path_s, target_var)
                    path_m = criterion(path_m, target_var)
                    path_l = criterion(path_l, target_var)
                    rate = epoch / 45
                    import math
                    loss += 0.3 * math.log(1 + (math.exp(1) - 1) * rate) * path_s + (
                            0.5 - 0.5 * math.log(1 + (math.exp(1) - 1) * rate)) * path_m + (
                                    1.01 - math.log(1 + (math.exp(1) - 1) * rate)) * path_l
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                '''
            else:
                output = model(input_var)
                loss = criterion(output, target_var)
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # ========================add for dense path loss========================
            '''
            output, others_path = model(input_var)
            loss = criterion(output, target_var)
            total_path = len(others_path)
            for j, path in enumerate(others_path):
                alpha = (j+1)/(total_path+1)
                loss += alpha * criterion(path, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            '''
            '''
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            '''
        # for param in optimizer.param_groups:
        #     print(param['lr'])
        # for k, v in model.named_parameters():
        #     print(k)
        # print(optimizer.param_groups)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, iter, logger=None, alpha=0.5):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            # ==================add for multi path loss===========================
            if args.arch == 'mpi3d':
                output,_,_,_ = model(input_var)
            elif args.arch == 'mpi3d_pt':
                main_path, path_s, path_m, path_l = model(input_var)
                # ====================add for dense loss==============================
                # output, _ = model(input_var)
                # output = model(input_var, alpha)

                # output = main_path
                # b, c = main_path.size()
                # for j in range(b):
                #     values = np.array((max(main_path[j, :]), max(path_s[j, :]), max(path_m[j, :]), max(path_l[j, :])))
                #     index = np.argmax(values)
                #     if index == 0:
                #         output[j, :] = main_path[j, :]
                #     elif index == 1:
                #         output[j, :] = path_s[j, :]
                #     elif index == 2:
                #         output[j, :] = path_m[j, :]
                #     else:
                #         output[j, :] = path_l[j, :]

                output = main_path + path_m + path_l # don't use path_l here
                # output = main_path + 0.5 * path_m + 0.1 * path_l
            else:
                output = model(input_var)
            # ====================add for dense loss==============================
            # output, _ = model(input_var)
            # output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
               .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output: 16(batch_size) x 101
    target: 16 x 1
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 5 x 16
    # print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.3 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    # decay = args.weight_decay
    for param_group in optimizer.param_groups:
        # param_group['lr'] = decay * param_group['lr']
        param_group['lr'] = lr


def save_checkpoint(state, is_best, prec1, filename='_checkpoint.pth.tar'):
    filename = args.save_model + args.dataset  + '/' + args.arch + '_' + args.mode + filename
    torch.save(state, filename)
    if is_best and prec1 > 40:
        best_name = args.save_model + args.dataset + '/' + str(prec1)[0:6] + '_' + args.arch + '_' + args.mode + '_model_best.pth.tar'
        shutil.copyfile(filename, best_name)


def accuracy_mixup(output, targets, target_a, target_b, lam, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() #5 x 20
    #print(pred)
    #print(target_a.data)
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_1 = pred.eq(target_a.data.view(1, -1).expand_as(pred)) 
    correct_2 = pred.eq(target_b.data.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = lam * correct_1[:k].view(-1).float().sum(0) + (1-lam) * correct_2[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
