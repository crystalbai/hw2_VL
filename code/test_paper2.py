from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os,shutil
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import cv2
import cPickle as pkl
import network
from wsddn import WSDDN
from logger import Logger
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from free_loc.main import metric1, metric2,AverageMeter
import torch.nn
from test import test_net
import math
import pdb
try:
    from termcolor import cprint
except ImportError:
    cprint = None
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
test_imdb_name = 'voc_2007_test'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 5000
epoch_loss = 20
start_step = 0
end_step = 50000
lr_decay_steps = {150000}
lr_decay = 1./10
thresh = 0.005
max_per_image = 300
rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = False
log_grads = False
resume = True

remove_all_log = not resume   # remove all historical experiments in TensorBoard
exp_name = "test_paper2" # the previous experiment name in TensorBoard
# ------------
if remove_all_log == True and os.path.exists(os.path.join("./tboard/", exp_name)):
    shutil.rmtree(os.path.join("./tboard/", exp_name))
logger = Logger('./tboard', name=exp_name)


if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS


#load test imdb
test_imdb = get_imdb(test_imdb_name)
rdl_roidb.prepare_roidb(test_imdb)
test_roidb = test_imdb.roidb
data_layer_test = RoIDataLayer(test_roidb, test_imdb.num_classes)
epoch = 0
# Create network and initialize
net = WSDDN(classes=test_imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.0001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
#     pret_net = pkl.load(open('./models/saved_model/wsdnn_50000.h5','r'))
        
else:
    pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue
resume_file = "./wsddn_test_checkpoint_adam"
if resume and os.path.isfile(resume_file):
    print("=> loading checkpoint '{}'".format(resume_file))
    checkpoint = torch.load(resume_file)
    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume_file, checkpoint['epoch']))
# Move model to GPU and set train mode
net.cuda()
net.eval()
save_name = '{}_{}'

aps = test_net(save_name, net, test_imdb, 
                   max_per_image, thresh=thresh, visualize=visualize,logger = logger, step = epoch)
print("here's the aps")
print(aps)


