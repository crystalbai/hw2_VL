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

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from free_loc.main import metric1, metric2,AverageMeter

try:
    from termcolor import cprint
except ImportError:
    cprint = None

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
epoch_loss = 500
start_step = 0
end_step = 50000
lr_decay_steps = {150000}
lr_decay = 1./10
thresh = 0.0001
rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = False
log_grads = False

remove_all_log = True   # remove all historical experiments in TensorBoard
exp_name = "paper2" # the previous experiment name in TensorBoard
# ------------
if remove_all_log == True and os.path.exists(os.path.join("./tboard/", exp_name)):
    shutil.rmtree(os.path.join("./tboard/", exp_name))
logger = Logger('./tboard', name=exp_name)
def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image, rois):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes // exclude
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    rois = np.hstack((np.zeros((rois.shape[0],1)),rois*im_scales[0]))
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    cls_prob = net(im_data, rois, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS


# load imdb and create data later
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

#load test imdb
test_imdb = get_imdb(test_imdb_name)
rdl_roidb.prepare_roidb(test_imdb)
test_roidb = test_imdb.roidb
data_layer_test = RoIDataLayer(test_roidb, test_imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
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

# Move model to GPU and set train mode
net.cuda()
net.train()

id_cls = {idx:c for idx, c in enumerate(imdb.classes)}
# Create optimizer for network parameters
params = list(net.parameters())
optimizer = torch.optim.SGD(params[2:], lr=lr, 
                            momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']

#     cls_tag = ""
#     _,gt_print_idx =np.where(gt_vec==1)
#     print(gt_print_idx)
#     for cls in gt_print_idx:
#         cls_tag += id_cls[cls]
#     print(cls_tag)
#     logger.image_summary(cls_tag, im_data, step)
    # forward
    
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % epoch_loss == 0:
        logger.scalar_summary('loss', loss, step)
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True
    #TODO: evaluate the model every N iterations (N defined in handout)
    if step%vis_interval==0:
        net.eval()
        losses = AverageMeter()
#         for i_test in range(len(test_roidb)):
        #just select 200 image
        data_layer_test._cur = 0
        for i_test in range(20):
            blobs = data_layer_test.forward()
            im_data = blobs['data']
            rois = blobs['rois']
            im_info = blobs['im_info']
            gt_vec = blobs['labels']
            pred = net(im_data, rois, im_info, gt_vec)
            loss = net.loss
            losses.update(loss.data[0])
        logger.scalar_summary('test_loss', losses.avg, step)
        #TODO: Perform all visualizations here
        log_text = "Testing stage at step {0} with loss {1}".format(step, losses.avg)
        log_print(log_text)
        #You can define other interval variable if you want (this is just an
        #example)
        #The intervals for different things are defined in the handout
        if visualize and step%vis_interval==0:
            #TODO: Create required visualizations
            if use_tensorboard:
                print('Logging to Tensorboard')
                for i_test in range(20):
                    i_test_idx = data_layer_test._perm[i_test]
                    im = cv2.imread(imdb.image_path_at(i_test_idx))
                    rois = imdb.roidb[i_test_idx]['boxes']
                    scores, boxes = im_detect(net, im, rois)
                    im2show = np.copy(im)

                    # skip j = 0, because it's the background class
                    for j in xrange(0, imdb.num_classes):
                        newj = j
                        inds = np.where(scores[:, newj] > thresh)[0]
                        cls_scores = scores[inds, newj]
                        cls_boxes = boxes[inds, newj * 4:(newj + 1) * 4]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                            .astype(np.float32, copy=False)
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep, :]
                        if visualize:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets)
                        all_boxes[j][i] = cls_dets
                    logger.image_summary(imdb.image_path_at(i_test_idx),im2show, step)
                    
                    
                
                

            if use_visdom:
                print('Logging to visdom')
        net.train()

    
    # Save model occasionally 
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX,step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

