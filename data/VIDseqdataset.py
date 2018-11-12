# -*- coding: UTF-8 -*-
"""VID Dataset Classes

"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VID_CLASSES = (  # always index 0
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle', 'dog',
    'domestic_cat', 'elephant', 'fox', 'giant_panda',
    'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake',
    'squirrel', 'tiger', 'train',
    'turtle', 'watercraft', 'whale', 'zebra')

VID_NAMEID = (
    'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061',
    'n02924116', 'n02958343', 'n02402425', 'n02084071', 'n02121808',
    'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451',
    'n02129165', 'n01674464', 'n02484322', 'n03790512', 'n02324045',
    'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604',
    'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',
)
# note: if you used our download scripts, this should be right
VID_ROOT = "/home/liyang/data/ILSVRC2015"

class VIDseqAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, name_to_ind=None, keep_difficult=False):
        self.name_to_ind = name_to_ind or dict(
            zip(VID_NAMEID, range(len(VID_NAMEID))))

        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            # 忽略difficult选项
            # difficult = int(obj.find('difficult').text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.name_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VIDseqDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, seq_len=24,phase='train',
                 image_sets=osp.join('ImageSets', 'VID'),
                 transform=None, target_transform=VIDseqAnnotationTransform(),
                 dataset_name='VID2015'):
        self.seq_len = seq_len
        self.root = root
        self.phase = phase
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', 'VID', self.phase, '%s.xml')        ## os.path as osp
        self._imgpath = osp.join('%s', 'Data', 'VID', self.phase, '%s.JPEG')
        self.ids = list()
        self.frame_id = list()
        self.seg_id = list()
        self.seg_len = list()
        self.num = 0

        listpath = osp.join(self.root, self.image_set, self.phase+'.txt')
        for line in open(listpath):
            self.ids.append((self.root, line.strip().split(' ')[0]))
            self.frame_id.append(int(line.strip().split(' ')[1]))
            self.seg_id.append(int(line.strip().split(' ')[2]))
            self.seg_len.append(int(line.strip().split(' ')[3]))
        # print('init ids_path:', self.ids[0])
        max_len = max(self.seg_len)
        min_len = min(self.seg_len)
        print('max_seg_len',max_len,'min_seg_len',min_len)

    def __getitem__(self, index):

        im, gt = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def read_anno(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_item(self, index):
        self.num += 1
        pre_k = self.seq_len//2-1
        aft_k = self.seq_len//2

        img_id = self.ids[index]
        frame_id = self.frame_id[index]
        seg_id = self.seg_id[index]
        seg_len = self.seg_len[index]

        start_index = index-pre_k
        end_index = index+aft_k
        if start_index<0 or self.seg_id[start_index] != seg_id:
            start_index = index
            end_index = index+self.seq_len-1
        elif end_index>=len(self.ids) or self.seg_id[end_index] != seg_id:
            start_index = index - self.seq_len +1
            end_index = index

        targets = []
        imgs = []
        for idx in range(start_index,end_index+1):
            im,gt,height,weight = self.read_anno(idx)
            imgs.append(im)
            targets.append(torch.FloatTensor(gt))

        return torch.stack(imgs, 0), targets


        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
