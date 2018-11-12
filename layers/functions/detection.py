# -*- coding: utf-8 -*-
import torch
from torch.autograd import Function
from ..box_utils import match, decode, nms
from data import voc as cfg
import numpy as np
import torch.nn as nn

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class Trnsform_target(nn.Module):
    """
    调用时不应该对 conf_data 采用softmax
    """
    def __init__(self, num_classes, top_k, overlap_thresh,conf_thresh, nms_thresh, use_gpu=True):
        super(Trnsform_target, self).__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.threshold = overlap_thresh
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.use_gpu = use_gpu

    def forward(self, loc_data, conf_data, prior_data, targets):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        return：
            rois：(tensor) rois after decoded loc data and nms
                Shape: [batch, top_k, 5]
            loc_pred: (tensor) loc after nms
                Shape: [batch, top_k, 4]
            cls_pred: (tensor) conf_data after nms
                Shape: [batch, top_k, num_classes]
        """

        priors = prior_data
        batch = loc_data.size(0)   # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch, num_priors, 4)
        conf_t = torch.Tensor(batch, num_priors)
        for idx in range(batch):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)    # 输出 loc_t, conf_t

        conf_t = conf_t.reshape(batch, num_priors,1)
        result = torch.zeros(batch, self.top_k, 1 + num_classes + 3* 4 + 1)

        conf_preds = conf_data.view(batch, num_priors,
                                    num_classes).transpose(2, 1)  # conf_preds size(num,num_classes,num_pirors)
        conf_data = conf_data.view(batch, num_priors,
                                   num_classes)

        decoded_box = loc_data.new(loc_data.size(0), loc_data.size(1), loc_data.size(2)).zero_()
        for i in range(batch):  # 对每个batch分别处理
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # box解码
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # conf_scores 为batch内容
            loc_keep = loc_data[i].clone()
            conf_keep = conf_data[i].clone()
            loc_t_keep = loc_t[i].clone()
            conf_t_keep = conf_t[i].clone()

            decoded_box[i] = decoded_boxes
            output = []
            for cl in range(1, num_classes):  # 对每个类目分别处理

                c_mask = conf_scores[cl].gt(self.conf_thresh)  # 将
               # print('conf_thresh: ',self.conf_thresh)
               # print('c_mask size: ',c_mask.size())
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0 or scores.size() == torch.Size([0]) :
                    print('scores dim: ',scores.dim(),scores.size())
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                s_mask = c_mask.unsqueeze(1).expand_as(conf_keep)
                t_mask = c_mask.unsqueeze(1).expand_as(conf_t_keep)

                boxes = decoded_boxes[l_mask].view(-1, 4)
               # print('boxes size: ',boxes.size())
              #  print('scores.size()',scores.size(),scores.dim())
              #  print('decoded_boxes size: ',decoded_boxes.size())
              #  print(' conf_scores size: ',conf_scores.size())
                
                loc = loc_keep[l_mask].view(-1, 4)
                loc_t_res = loc_t_keep[l_mask].view(-1,4)

                conf = conf_keep[s_mask].view(-1, num_classes)
                conf_t_res = conf_t_keep[t_mask].view(-1,1)

                #print(conf.size())

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output.append(
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]], loc[ids[:count]], conf[ids[:count]], loc_t_res[ids[:count]],
                               conf_t_res[ids[:count]]), 1)
                )

            #            print(type(output))
            #            print(len(output))
            #            print(output)

            #            print(output[0].size())
            #            print(output[1].size())

            res = output[0]
            for j in range(len(output) - 1):
                res = torch.cat((res, output[j + 1]), dim=0)
            #
            #            print('index: ',i,'type',type(res))
            #            print('res size: ',res.size())
            #            print('res',res)

            # 按照置信度排序
            sort_conf = res[:, 0].clone()
            res[:, 0] = i

            #  去除重复的框 采用numpy处理方式
            res = res.cpu().detach().numpy()
            b = np.ascontiguousarray(res).view(np.dtype((np.void, res.dtype.itemsize * res.shape[1])))
            _, idx = np.unique(b, return_index=True)
            keep_res = torch.from_numpy(res[idx])

            sort_val = sort_conf[idx].view(-1, 1)
            _, indices = sort_val[:, 0].sort(0, descending=True)
            # 保证输出框不大于top_k
            res_sel = keep_res[indices][:self.top_k]
            #            print(res_sel.size())
            #            print(result[i].size())
            result[i][:res_sel.size(0)] = res_sel

            #            print('result size: ',result.size())
            #            print('result: ',result)
            # 分片索引 选出指定列
            index1 = torch.tensor(range(0, 5))
            index2 = torch.tensor(range(5, 9))
            index3 = torch.tensor(range(9, 9 + num_classes))
            index4 = torch.tensor(range(9 + num_classes, 9+num_classes+4))
            index5 = torch.tensor(range(9 + num_classes + 4, 9+num_classes+4+1))
            # rois 为前5列
            rois = torch.index_select(result, -1, index1)
            #            print('rois: ',rois.size())
            # loc 为中间4列
            loc = torch.index_select(result, -1, index2)
            #            print('loc: ',loc.size())
            # cls 为最后num_classes列
            cls = torch.index_select(result, -1, index3)
        #            print('cls: ',cls)
            loc_truth = torch.index_select(result, -1, index4)
            conf_truth = torch.index_select(result, -1, index5)

        return rois, loc, cls, loc_truth, conf_truth

class test_target(Function):
    def __init__(self, num_classes, top_k, overlap_thresh, conf_thresh, nms_thresh, use_gpu=True):
        self.num_classes = num_classes
        self.top_k = top_k
        self.threshold = overlap_thresh
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.use_gpu = use_gpu

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        return：
            rois：(tensor) rois after decoded loc data and nms
                Shape: [batch, top_k, 5]
            loc_pred: (tensor) loc after nms
                Shape: [batch, top_k, 4]
            cls_pred: (tensor) conf_data after nms
                Shape: [batch, top_k, num_classes]
        """

        priors = prior_data
        batch = loc_data.size(0)  # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        result = torch.zeros(batch, self.top_k, 1 + 2 * 4 +num_classes)

        conf_preds = conf_data.view(batch, num_priors,
                                    num_classes).transpose(2, 1)  # conf_preds size(num,num_classes,num_pirors)
        conf_data = conf_data.view(batch, num_priors,
                                   num_classes)

        decoded_box = loc_data.new(loc_data.size(0), loc_data.size(1), loc_data.size(2)).zero_()
        for i in range(batch):  # 对每个batch分别处理
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # box解码
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # conf_scores 为batch内容
            loc_keep = loc_data[i].clone()
            conf_keep = conf_data[i].clone()

            decoded_box[i] = decoded_boxes
            output = []
            for cl in range(1, num_classes):  # 对每个类目分别处理

                c_mask = conf_scores[cl].gt(self.conf_thresh)  # 将
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0 or scores.size() == torch.Size([0]):
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                s_mask = c_mask.unsqueeze(1).expand_as(conf_keep)

                boxes = decoded_boxes[l_mask].view(-1, 4)
                loc = loc_keep[l_mask].view(-1, 4)

                conf = conf_keep[s_mask].view(-1, num_classes)

                # print(conf.size())

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output.append(
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]], loc[ids[:count]], conf[ids[:count]] ), 1)
                )

            res = output[0]
            for j in range(len(output) - 1):
                res = torch.cat((res, output[j + 1]), dim=0)
            #
            #            print('index: ',i,'type',type(res))
            #            print('res size: ',res.size())
            #            print('res',res)

            # 按照置信度排序
            # _, indices = res[:, 0].sort(0, descending=True)
            # 将第一列改为 rois 第一列格式
            sort_conf = res[:, 0].clone()
            res[:, 0] = i

            #  去除重复的框 采用numpy处理方式
            res = res.cpu().detach().numpy()
            b = np.ascontiguousarray(res).view(np.dtype((np.void, res.dtype.itemsize * res.shape[1])))
            _, idx = np.unique(b, return_index=True)
            keep_res = torch.from_numpy(res[idx])

            sort_val = sort_conf[idx].view(-1, 1)
            _, indices = sort_val[:, 0].sort(0, descending=True)
            # 保证输出框不大于top_k
            res_sel = keep_res[indices][:self.top_k]

            result[i][:res_sel.size(0)] = res_sel

            # 分片索引 选出指定列
            index1 = torch.tensor(range(0, 5))
            index2 = torch.tensor(range(5, 9))
            index3 = torch.tensor(range(9, 9 + num_classes))

            # rois 为前5列
            rois = torch.index_select(result, -1, index1)
            #            print('rois: ',rois.size())
            # loc 为中间4列
            loc = torch.index_select(result, -1, index2)
            #            print('loc: ',loc.size())
            # cls 为最后num_classes列
            cls = torch.index_select(result, -1, index3)

        return rois, loc, cls