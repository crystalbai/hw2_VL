import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob, prep_im_for_blob
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
from roi_pooling.modules.roi_pool import RoIPool
from torch.autograd import Variable
from matplotlib import pyplot as plt
from free_loc.main import UnNormalize
def debug_draw(img, rois):
#     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#     img = unorm(img)
    img = img.permute(0, 2, 3, 1)
    img = torch.squeeze(img)
    print img.shape
    img = img.data.cpu().numpy()
    rois = rois.data.cpu().numpy()
    for i in range(10):
        box = rois[i, 1:]
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),3)
    plt.imshow(img)
    plt.show()    
def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]




class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, classes=None, debug=False, training=True):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)
        
        #TODO: Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.roi_pool = RoIPool(7, 7, 1.0 / 34)
        self.classifier_share = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
#             nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier_c1 = nn.Sequential(
            nn.Linear(4096, self.n_classes)
        )
        self.classifier_c2 = nn.Sequential(
            nn.Softmax(dim = 1)
        )
        self.classifier_d1 = nn.Sequential(
            nn.Linear(4096, self.n_classes)
        )
        self.classifier_d2 = nn.Sequential(
            nn.Softmax(dim =0)
        )

        
        
        # loss
        self.cross_entropy = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy
	
    def forward(self, im_data, rois, im_info, gt_vec=None,
                gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        if rois.shape[0] > 256 and (self.training != True):
            rois = rois[:256,:]
#             print("after clip{0}".format(rois.shape))
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
#         print im_data.shape
        #TODO: Use im_data and rois as input
        # compute cls_prob which are N_roi X 20 scores
        # Checkout faster_rcnn.py for inspiration
        feature = self.features(im_data)
        rois = network.np_to_variable(rois, is_cuda=True)
#         print("feature {0}".format(feature))
#         debug_draw(im_data, rois)
#         print("rois_shape{0}".format(rois.shape))
        pooled_features = self.roi_pool(feature, rois)
#         print("pooled_features range {0}-{1}".format(pooled_features.min(),pooled_features.max()))
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.classifier_share(x)
#         print("classifier_share_min {0}-{1}".format(x.min(), x.max()))
        output_c1 = self.classifier_c1(x)
        output_c2 = self.classifier_c2(output_c1)
#         print("output_c1_range {0}-{1}".format(output_c1.min(), output_c1.max()))
#         print("output_c2_range {0}-{1}".format(output_c2.min(), output_c2.max()))
        output_d1 = self.classifier_d1(x)
        output_d2 = self.classifier_d2(output_d1)
#         print("output_d1_range {0}-{1}".format(output_d1.min(), output_d1.max()))
#         print("output_d2_range {0}-{1}".format(output_d2.min(), output_d2.max()))
        pred_score = output_c2*output_d2
        cls_prob = pred_score.sum(0)
#         print("shape of out put {0} and {1}".format(output_c1.size(), output_d1.size()))
        if cls_prob.data.cpu().max() > 1.0-1e-10 or output_c2.data.cpu().max()> 1.0-1e-10 or output_d2.data.cpu().max()> 1.0-1e-10:
            txt_out1 = output_c2.squeeze().data.cpu().numpy()
            txt_out2 = output_d2.squeeze().data.cpu().numpy()
            txt_d = output_d1.squeeze().data.cpu().numpy()
            np.savetxt('d_test.out', txt_d, delimiter=', ',fmt='%.4f') 
            np.savetxt('test.out', txt_out1, delimiter=', ',fmt='%.4f') 
            np.savetxt('test2.out', txt_out2, delimiter=', ',fmt='%.4f') 
            exit()

        
        assert cls_prob.max().data.cpu().numpy() <= 1
        if self.training:
            label_vec = network.np_to_variable(gt_vec, is_cuda=True)
            label_vec = label_vec.squeeze()
            self.cross_entropy = -self.build_loss(cls_prob, label_vec)
        return cls_prob, pred_score
    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector 
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called
        ## alert, the image number should be 1 per batch
#         print("cls_prob {0}".format(cls_prob))
#         print("label_vec {0}".format(label_vec))
        
        loss =torch.log(label_vec*(cls_prob-0.5)+0.5).sum(0)
#         print("loss {0}".format(loss))
        return loss

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        im_orig = im.astype(np.float32, copy=True)/255.0
        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []
        mean=np.array([[[0.485, 0.456, 0.406]]])
        std=np.array([[[0.229, 0.224, 0.225]]])
        for target_size in self.SCALES:
            im, im_scale = prep_im_for_blob(im_orig, target_size,
                                            self.MAX_SIZE,
                                            mean=mean,
                                            std=std)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def load_from_npz(self, params):
        self.features.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 
                 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

