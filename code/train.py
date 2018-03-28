from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os, shutil
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
from free_loc.main import metric1, metric2, AverageMeter
import torch.nn
import math
import visdom
from test import test_net
try:
    from termcolor import cprint
except ImportError:
    cprint = None

save_name = '{}_{}'
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
visual_hist = 2000
epoch_loss = 500
start_step = 0
end_step = 50000
lr_decay_steps = {150000}
lr_decay = 1. / 10
thresh = 0.005
rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False
resume = False
max_per_image = 300

remove_all_log = not resume  # remove all historical experiments in TensorBoard
exp_name = "paper2"  # the previous experiment name in TensorBoard
# ------------
if remove_all_log == True and os.path.exists(os.path.join("./tfboard2/", exp_name)):
    shutil.rmtree(os.path.join("./tfboard2/", exp_name))
logger = Logger('./tfboard2', name=exp_name)
vis = visdom.Visdom(server='http://localhost',port='7097')

# def vis_detections(im, class_name, dets, thresh=0.8):
#     """Visual debugging of detections."""
#     for i in range(np.minimum(10, dets.shape[0])):
#         bbox = tuple(int(np.round(x)) for x in dets[i, :4])
#         score = dets[i, -1]
#         if score > thresh:
#             cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
#             cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
#                         1.0, (0, 0, 255), thickness=1)
#     return im


# def im_detect(net, image, rois):
#     """Detect object classes in an image given object proposals.
#     Returns:
#         scores (ndarray): R x K array of object class scores (K includes // exclude
#             background as object category 0)
#         boxes (ndarray): R x (4*K) array of predicted bounding boxes
#     """

#     im_data, im_scales = net.get_image_blob(image)
#     rois = np.hstack((np.zeros((rois.shape[0], 1)), rois * im_scales[0]))
#     im_info = np.array(
#         [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
#         dtype=np.float32)

#     cls_prob = net(im_data, rois, im_info)
#     scores = cls_prob.data.cpu().numpy()
#     boxes = rois[:, 1:5] / im_info[0][2]

#     if cfg.TEST.BBOX_REG:
#         # Apply bounding-box regression deltas
#         box_deltas = bbox_pred.data.cpu().numpy()
#         pred_boxes = bbox_transform_inv(boxes, box_deltas)
#         pred_boxes = clip_boxes(pred_boxes, image.shape)
#     else:
#         # Simply repeat the boxes, once for each class
#         pred_boxes = np.tile(boxes, (1, scores.shape[1]))

#     return scores, pred_boxes


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

# load test imdb
test_imdb = get_imdb(test_imdb_name)
rdl_roidb.prepare_roidb(test_imdb)
test_roidb = test_imdb.roidb
data_layer_test = RoIDataLayer(test_roidb, test_imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.0001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'r'))
#     pret_net = pkl.load(open('./models/saved_model/wsdnn_50000.h5','r'))

else:
    pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
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
if resume and os.path.isfile("./wsddn_test_checkpoint"):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
# Move model to GPU and set train mode
net.cuda()
net.train()

id_cls = {idx: c for idx, c in enumerate(imdb.classes)}
# Create optimizer for network parameters
params = list(net.parameters())
# optimizer = torch.optim.SGD(params[2:], lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params[2:],lr = lr, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
win = vis.line(
    X=np.arange(0,1),
    Y=np.arange(0,1),
    opts=dict(title= "train loss", caption = "train_los")
)
mAP_win = vis.line(
    X=np.arange(0,1),
    Y=np.arange(0,1),
    opts=dict(title= "validation aps", caption = "val_aps")
)
for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    print(blobs['im_name'])

    #     cls_tag = ""
    #     _,gt_print_idx =np.where(gt_vec==1)
    #     print(gt_print_idx)
    #     for cls in gt_print_idx:
    #         cls_tag += id_cls[cls]
    #     print(cls_tag)
    #     logger.image_summary(cls_tag, im_data, step)
    # forward
    pred = net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, cur_loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, loss.data[0], fps, 1. / fps, lr, momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True
#         print("pred and target {0} and {1}".format(pred, gt_vec))


    #The intervals for different things are defined in the handout
    #TODO: Create required visualizations for training
    if use_tensorboard:
        if step % epoch_loss == 0:
            logger.scalar_summary('loss', train_loss / step_cnt, step)
        if step % visual_hist == 0:
            logger.model_param_histo_summary(net, step)
    if use_visdom:
        if step % epoch_loss == 0:
            cap = "train loss"
            vis.line(
                X=(np.asarray([ step])),
                Y=(np.asarray([ train_loss / step_cnt])),
                win=win,
                update='append'
            )
    # TODO: evaluate the model every N iterations (N defined in handout)

    if visualize and step%vis_interval==0:
        net.eval()
        aps = test_net(save_name, net, test_imdb,
                       max_per_image, thresh=thresh, visualize=visualize, logger=logger, step=step)
        net.train()
        #TODO: Create required visualizations
        if use_tensorboard:
            for i_cls in range(test_imdb.num_classes):
                logger.scalar_summary('AP_{0}'.format(test_imdb._classes[i_cls]), aps[i_cls], step)
                print('Logging to Tensorboard')
        if use_visdom:
            vis.line(
                X=(np.asarray([step])),
                Y=(np.asarray([np.mean(aps)])),
                win=mAP_win,
                update='append'
            )

            print('Logging to visdom')



#     if step % vis_interval == 0:
#         net.eval()

#         # You can define other interval variable if you want (this is just an
#         # example)
#         # The intervals for different things are defined in the handout
#         if visualize and step % vis_interval == 0:
#             # TODO: Create required visualizations
#             if use_tensorboard:
#                 print('Logging to Tensorboard')
#                 for i_test in range(20):
#                     i_test_idx = data_layer_test._perm[i_test]
#                     im = cv2.imread(imdb.image_path_at(i_test_idx))
#                     rois = imdb.roidb[i_test_idx]['boxes']
#                     scores, boxes = im_detect(net, im, rois)
#                     im2show = np.copy(im)
#                     #                     print(scores)
#                     # skip j = 0, because it's the background class
#                     #                     print("log for detection")
#                     for j in xrange(0, imdb.num_classes):
#                         newj = j
#                         inds = np.where(scores[:, newj] > thresh)[0]
#                         cls_scores = scores[inds, newj]
#                         cls_boxes = boxes[inds, newj * 4:(newj + 1) * 4]
#                         cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
#                             .astype(np.float32, copy=False)
#                         keep = nms(cls_dets, cfg.TEST.NMS)
#                         cls_dets = cls_dets[keep, :]
#                         #                         print("cls_dets's shape {0}".format(cls_dets.shape))
#                         if visualize:
#                             im2show = vis_detections(im2show, imdb.classes[j], cls_dets, thresh)
#                     im2show = cv2.cvtColor(im2show, cv2.COLOR_RGB2BGR)
#                     v = torch.from_numpy(im2show).type(torch.FloatTensor)
#                     v = torch.unsqueeze(v, 0)
#                     logger.image_summary(imdb.image_path_at(i_test_idx), v, step)

#             if use_visdom:
#                 vis.line(
#                     X=(np.asarray([step])),
#                     Y=(np.asarray([ train_loss / step_cnt])),
#                     win=win,
#                     update='append'
#                 )
#                 print('Logging to visdom')
#         net.train()

    # Save model occasionally
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        #         save_name = os.path.join(output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX,step))
        #         network.save_net(save_name, net)
        #         print('Saved model to {}'.format(save_name))
        save_checkpoint({
            'epoch': step,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "wsddn_test_checkpoint")

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

