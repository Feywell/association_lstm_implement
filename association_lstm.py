import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import vid
import os
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.utils.net_utils import _affine_grid_gen
from utils.viz import plot_bbox, plot_image
from matplotlib import pyplot as plt
from bnlstm import BNLSTM
from layers.modules import MultiProjectLoss

class association_lstm(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, cfg, size, base, extras, head, num_classes):
        super(association_lstm, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = vid
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.roi_pool = _RoIPooling(self.cfg['POOLING_SIZE'], self.cfg['POOLING_SIZE'], 1.0 / 16.0)
        self.roi_align = RoIAlignAvg(self.cfg['POOLING_SIZE'], self.cfg['POOLING_SIZE'], 1.0 / 16.0)

        self.grid_size = self.cfg['POOLING_SIZE'] * 2 if self.cfg['CROP_RESIZE_WITH_MAX_POOL'] else self.cfg['POOLING_SIZE']
        self.roi_crop = _RoICrop()
        self.img_shape = (self.cfg['min_dim'],self.cfg['min_dim'])
        self.tensor_len = 4+self.num_classes+49
        self.bnlstm1 = BNLSTM(input_size=84, hidden_size=150, batch_first=False, bidirectional=False)
        self.bnlstm2 = BNLSTM(input_size=150, hidden_size=300, batch_first=False, bidirectional=False)
        self.cls_pred = nn.Linear(300, self.num_classes)
        self.bbox_pred = nn.Linear(300, 4)
        self.association_pred = nn.Linear(300, 49)
        self.MultiProjectLoss = MultiProjectLoss(self.num_classes, 0, True, 3, 0.5 )
        if phase == 'vid_train':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Trnsform_target(num_classes, 200, 0.5, 0.01, 0.45)
            self.detect = train_target(num_classes, 200, 0.5, 0.01, 0.45)
    def forward(self, x, targets):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        batch_size = x.data.size(0)
        print('input image size: ',x.size())
        display_img = x[0].clone().cpu().numpy().transpose((1,2,0))
        print('display_img size: ',display_img.shape)
        
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        roi_feat = sources[1]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "vid_train":
            """
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                #conf.view(conf.size(0), -1, self.num_classes),
                self.priors.type(type(x.data)),                 # default boxes
                targets
            )
            """
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                #conf.view(conf.size(0), -1, self.num_classes),
                self.priors.type(type(x.data)),                 # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            # print('output size；', output.size())
            # print('output value: ', output[0], output[1])
            return output                            # 临时返回

        #rois, loc, conf, loc_t, conf_t = output  # rois size: batchsize, top_k, 5 
        rois, loc, conf, priors = output
       # print('after transform conf_t: ',conf_t)
        
     #   print('display rois values: \n',rois[0,0:4,:])
        img_scale = torch.Tensor([self.img_shape[1], self.img_shape[0],
                          self.img_shape[1], self.img_shape[0]])
        
        #bboxes = rois[0,:,1:].clone()*img_scale
      #  print('display scaled rois values: \n',bboxes[0:4,:])
      #  ax1 = None
      #  ax2 = None
      #  plot_image(display_img, ax=ax1, reverse_rgb=False)
      #  plot_bbox(display_img, bboxes[:5,:].cpu(), ax=ax2)
      #  plt.show()
        rois[:,:,1:] = rois[:,:,1:]*img_scale
       # print('display scaled rois values: \n',rois.size(),rois[:,0:4,:])
        if self.cfg['POOLING_MODE'] == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), roi_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.roi_crop(roi_feat, Variable(grid_yx).detach())
            if self.cfg['CROP_RESIZE_WITH_MAX_POOL']:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif self.cfg['POOLING_MODE'] == 'align':
            pooled_feat = self.roi_align(roi_feat, rois.view(-1, 5))
        elif self.cfg['POOLING_MODE'] == 'pool':
            pooled_feat = self.roi_pool(roi_feat, rois.view(-1,5))

      #  print('after roi feature size: ',pooled_feat.size())
        Con1_1 = nn.Conv2d(1024, 1, kernel_size=1, padding=0, dilation=1)
        pooled_feat = Con1_1(pooled_feat)
      #  print('after conv1_1 feature size: ',pooled_feat.size())
        scale = L2Norm(1, 20)
        normlize_feat = scale(pooled_feat)
      #  print('normlize feature size: ', normlize_feat.size())
        feat = normlize_feat.view(normlize_feat.size(0), normlize_feat.size(1), -1)
        #feat = pooled_feat.view(pooled_feat.size(0), pooled_feat.size(1), -1)
      #  print('after reshape feat size: ',feat.size())
        feat = feat.squeeze().view(batch_size, feat.size(0)/batch_size, -1)
      #  print('after slice feat size: ',feat.size())
        
     #   print('loc size: ',loc.size(), '\nconf size: ',conf.size())
        stacked_tensor = torch.cat((conf,loc,feat),-1)
      #  print('stacked_tensor size: ',stacked_tensor.size(),'\n',stacked_tensor[:,:2,:])
        o1, _ = self.bnlstm1(stacked_tensor)
        print('output1 size: ',o1.size())
        #print('hidden1 size: ',len(h1),h1[0].size(),h1[1].size())
        o2, _ = self.bnlstm2(o1)
        print('output2 size: ',o2.size())
        #print('hidden2 size: ',len(h2),h2[0].size(),h2[1].size())
        cls_pred = self.cls_pred(o2)
        print('cls_pred size: ',cls_pred.size())
        bbox_pred = self.bbox_pred(o2)
        print('bbox_pred size: ',bbox_pred.size())
        association_pred = self.association_pred(o2)
        print('association_pred size: ',association_pred.size())
        #loc_t, conf_t
        #print('loc_t size: ', loc_t.size())
       # print('conf_t size: ', conf_t.size())
        print('conf size: ',conf.size())
       # loc_loss, cls_loss = self.MultiProjectLoss(cls_pred, bbox_pred, association_pred, loc_t, conf_t)
       ## print('loc_loss size: ',loc_loss.size())
       ## print('cls_loss size: ',cls_loss.size())
     #   pooled_feat = pooled_feat.view(pooled_feat.size(0), pooled_feat.size(1), -1)
        print('output priors size: ',priors.size())
        return bbox_pred, cls_pred, self.priors

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_association_lstm(phase, cfg, size=300, num_classes=21):
    if phase != "test" and phase != "vid_train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return association_lstm(phase, cfg, size, base_, extras_, head_, num_classes)
