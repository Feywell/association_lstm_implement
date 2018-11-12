from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .VIDdataset import VIDetection, VIDAnnotationTransform, VID_NAMEID, VID_ROOT
from .VIDseqdataset import VIDseqDetection, VIDseqAnnotationTransform, VID_CLASSES, VID_ROOT
# from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    #print(type(batch))
    #print(len(batch))
    for sample in batch:
       # print(type(sample))
       # print(type(sample[0]))
       # print(sample[0])
       # print(sample[0].shape)
      #  print(type(sample[1]))
      #  print(sample[1])
      #  print(sample[1].shape)
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
     
   # imgs = torch.stack(imgs, 0)
    #print(type(imgs))
    #print(imgs.shape)
   # print(type(targets))
    #print(len(targets))
    #print(targets)
    return torch.stack(imgs, 0), targets


def vid_detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
 #   targets = []
 #   imgs = []
 #   print(type(batch))      # list
 #   print(len(batch))       # len
 #   for sample in batch:
 #       print(type(sample))      # tuple
 #       print(type(sample[0]))    
  #      print(sample[0])
 #       print(sample[0].shape)
  #      print(type(sample[1]))
 #       print(len(sample[1]))
  #      print(sample[1])

    return batch[0][0], batch[0][1]


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
