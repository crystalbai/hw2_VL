import argparse
import os
import shutil
import time
import sys
import math
sys.path.insert(0,'/home/spurushw/reps/hw-wsddn-sol/faster_rcnn')
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import _init_paths
from logger import Logger
from datasets.factory import get_imdb
from custom import *
from eval import compute_map,compute_curve
import visdom
import scipy.misc

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--vis',action='store_true')

best_prec1 = 0

# def multilabel_loss(fx,y):
#     loss = 0.0
#     batch = fx.size(0)
#     tmp =fx.data.cpu().numpy()
    
#     print "loss_pred_{0}_max_{1}_min".format(tmp.max(), tmp.min())

#     cls = fx.size(1)
# #     print "{0}_batchs_{1}_cls".format(batch, cls)
#     for b in range(batch):
#         for k in range(cls):
#             loss += torch.log(1+ torch.exp(-y[b,k]*fx[b,k]))
#     loss = loss/batch
#     return loss
def multilabel_loss(fx,y):
    batch = fx.size(0)
    loss = torch.log(1+ torch.exp(-y*fx)).sum(0).sum(0)/batch
    return loss

vis = visdom.Visdom(server='http://localhost',port='8097')
def main():
    np.random.seed(5)
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch=='localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch=='localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    criterion = multilabel_loss
    params = list(model.parameters())
#     optimizer = torch.optim.SGD(params, args.lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(params[10:], args.lr,
                                weight_decay=args.weight_decay)





    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    logger = Logger('./tfboard2', name='freeloc-vis')
    if args.evaluate:
        validate(val_loader, model, criterion, logger,0)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    







    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,logger)

        # evaluate on validation set
        if epoch%args.eval_freq==0 or epoch==args.epochs-1:
            m1, m2 = validate(val_loader, model, criterion, logger,epoch)
            score = m1*m2
            # remember best prec@1 and save checkpoint
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


#TODO: You can add input arguments if you wish
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for tensor in input:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return input
def train(train_loader, model, criterion, optimizer, epoch,logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
#         print np.unique(target)
#         while 1:
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)
        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output

        imoutput = model(input_var)
        cls = imoutput.size(1)
        batch = imoutput.size(0)
        imoutput = imoutput.view(batch, cls)
        loss = criterion(imoutput, target_var)
        
#         print "loss test value{0}".format(loss)

#             print target
#             print imoutput.data

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))
        # TODO: 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            tmp =imoutput.data.cpu().numpy()
            print "loss_pred_{0}_max_{1}_min".format(tmp.max(), tmp.min())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, avg_m1=avg_m1,
                   avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        logger.model_param_histo_summary(model, epoch*(len(train_loader))+i)
        logger.scalar_summary('loss', loss, epoch*(len(train_loader))+i)
        logger.scalar_summary('m1', m1, epoch*(len(train_loader))+i)
        logger.scalar_summary('m2', m2, epoch*(len(train_loader))+i)
        if i % np.floor(len(train_loader)/4) == 0:
            unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            vis_tensor = unorm(input)
            print "{0} image in this batch".format(input.shape[0])
            for batch_i in range(input.shape[0]):
#                 print("target shape{0}".format(target.shape))
                # visual images
                cap_tf = "iter{0}_batch{1}".format(i, batch_i)
#                 print(input[[batch_i],:,:,:].shape)
                logger.image_summary("{0}_image_visual".format(cap_tf), input[[batch_i],:,:,:], epoch*(len(train_loader))+i)
                f_map = model.get_featuremap().data.cpu().numpy()
                for i_cls in range(cls):
                    if target[batch_i,i_cls] == 1:
                        f_map_idx = f_map[batch_i, i_cls, :,:]
                        f_map_idx = scipy.misc.imresize(f_map_idx, (512, 512))
                        f_map_idx = f_map_idx[np.newaxis,:,:]
                        logger.image_summary("{0}_feature_map_{1}".format(cap_tf, train_loader.dataset.classes[i_cls]), f_map_idx, epoch*(len(train_loader))+i)
                #visual uisng visdom
                if epoch%2 == 0:
#                 if 1:
                    vis_tensor_batch = vis_tensor[[batch_i],:,:,:]
                    title_name = "{0}_{1}_{2}_image".format(epoch, epoch*(len(train_loader))+i, batch_i)
                    vis.images(vis_tensor_batch, opts=dict(title= title_name))

#                     f_map = model.get_featuremap().data.cpu().numpy()
                    gt_classes = target
                    for idx in range(gt_classes.shape[1]):
                        if target[batch_i, idx] == 1:
                            tmp = f_map[batch_i,idx, :,:]
                            cur_cls = scipy.misc.imresize(tmp, (512, 512))
                            title_name = "{0}_{1}_{2}_heatmap_{3}".format(epoch, epoch*(len(train_loader))+i, batch_i, train_loader.dataset.classes[idx])
                            vis.images(cur_cls,opts=dict(title= title_name))
#                             print title_name
                

def validate(val_loader, model, criterion, logger, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        imoutput = model(input_var)
        cls = imoutput.size(1)
        batch = imoutput.size(0)
        imoutput = imoutput.view(batch, cls)
        loss = criterion(imoutput, target_var)



        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   avg_m1=avg_m1, avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        vis_tensor = unorm(input)
        if epoch == 29 and i == 10:
            print "{0} image in this batch".format(input.shape[0])
            for batch_i in range(input.shape[0]):
                vis_tensor_batch = vis_tensor[[batch_i],:,:,:]
                title_name = "{0}_{1}_imageVal".format(epoch, batch_i)
                vis.images(vis_tensor_batch, opts=dict(title= title_name))

                f_map = model.get_featuremap().data.cpu().numpy()
                gt_classes = target
                for idx in range(gt_classes.shape[1]):
                    if target[batch_i, idx] == 1:
                        tmp = f_map[batch_i,idx, :,:]
                        cur_cls = scipy.misc.imresize(tmp, (512, 512))
                        title_name = "{0}_{1}_heatmapVal_{2}".format(epoch, batch_i, val_loader.dataset.classes[idx])
                        vis.images(cur_cls,opts=dict(title= title_name))
                        print title_name
  


    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))
    logger.scalar_summary('val_loss', losses.avg, epoch)
    logger.scalar_summary('val_m1', avg_m1.avg, epoch)
    logger.scalar_summary('val_m2', avg_m2.avg, epoch)

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def metric1(output, target):
#     # TODO: Ignore for now - proceed till instructed
#     output = output.cpu().numpy()
#     target = target.cpu().numpy()
#     AP = compute_map(target, output, average=None)
#     for i in range(len(AP)):
#         if math.isnan(AP[i]):
#             AP[i] = 0         
#     mAP = np.mean(AP)
# #     print('Obtained {} mAP'.format(mAP))
#     return mAP

def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    #transfer fx to p(k|x)
    thres = 0.5
    output = output.cpu().numpy()

    p = np.zeros(target.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            p[i][j] = ( (1.0/(1+np.exp(-output[i][j]))))
            target[i][j] = int(target[i][j] == 1)

    AP = compute_map(target, p, average=None)
    for i in range(len(AP)):
        if math.isnan(AP[i]):
            AP[i] = 0  
    mAP = np.mean(AP)
#     print('Obtained probability {} mAP'.format(mAP))
    return mAP

def metric2(output, target):
    # since this dataset is highly unbalanced, we only penalize the false positive 
        # TODO: Ignore for now - proceed till instructed
    #transfer fx to p(k|x)
    thres = 0.5
    output = output.cpu().numpy()
    p = np.zeros(target.shape, dtype = np.int32)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            tmp = ( (1.0/(1+np.exp(-output[i][j]))))
            p[i][j] = int(tmp> thres)
            target[i][j] = int(target[i][j] == 1)
    AP = compute_curve(target, p, average=None)
    for i in range(len(AP)):
        if math.isnan(AP[i]):
            AP[i] = 0  
    mAP = np.mean(AP)
#     print('Obtained curve {}'.format(mAP))
    return mAP

if __name__ == '__main__':
    main()
