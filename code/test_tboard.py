import _init_paths
from datasets.factory import get_imdb
import cv2
import visdom
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from logger import Logger
import roi_data_layer.roidb as rdl_roidb
from torch.autograd import Variable
# vis = visdom.Visdom(server='http://localhost',port='8097')
imdb = get_imdb('voc_2007_test')
roidb = imdb.roidb
print len(roidb)
logger = Logger('./tboard', name='test')
img = cv2.imread(imdb.image_path_at(1))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
v = torch.from_numpy(img).type(torch.FloatTensor)
v = torch.unsqueeze(v, 0)
print v.size()
logger.image_summary(imdb.image_path_at(1), v, 0)