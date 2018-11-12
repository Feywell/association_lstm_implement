from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from test_association import bulid_test_association_lstm
import cv2
from data import *
from utils.viz import plot_bbox, plot_image, show_bbox, show_image
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
        test_img_path = '/home/liyang/experiment/ssd.pytorch-master/data/dog_fast2.JPEG'
        img = cv2.imread(test_img_path)
        print('img orignal size: ',img.shape)
        
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        rois = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        print('display rois values: \n',rois[0,0:4,:])        
        bboxes = rois[0,:,1:].clone()*scale
        print('bboxes values: ',bboxes[:4])

        fig = plt.figure(figsize=(32,32))
        show_image(img,fig, n=1, reverse_rgb=False)
        for i in range(1,5):
            show_bbox(img, bboxes[:i].cpu(), fig, n=i+1)
            
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  
        plt.subplots_adjust()
        plt.savefig('dog_fast2_new.png', dpi=300,bbox_inches ='tight')
        #ax2.savefig('test2.png', dpi=100)
        plt.show()

def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = bulid_test_association_lstm('vid_test', voc, 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print(net)
    print('Finished loading model!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
