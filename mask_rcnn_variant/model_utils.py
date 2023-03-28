import os
import torch
from torch.nn import *
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.models.detection.mask_rcnn as mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from collections import defaultdict

class Conv_Block(Module):
  '''
  This block integrates three types of convolutions
  1. Normal
  2. Separable
  3. Transposed

  It also includes 4 types of activations, ie, relu, sigmoid, softamx and softmax2d
  '''
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation='relu', mode='normal'):
    super(Conv_Block, self).__init__()

    self.mode_dict= ModuleDict({
        'normal': Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
        'separable': Sequential(Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels),
                                                     Conv2d(in_channels, out_channels, kernel_size=1)),
        
        'transposed': ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    })

    self.activations_dict = ModuleDict({
        'relu': ReLU(),
        'softmax': Softmax(),
        'softmax2d': Softmax2d(),
        'sigmoid': Sigmoid()
    })

  
    self.conv= self.mode_dict[mode]
    self.bn = BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
    self.activation= self.activations_dict[activation]

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.activation(x)

    return x
    

class RedRCNN_mask_head(Module):

  '''
  Mask_head for Red RCNN.
  '''

  def __init__(self, in_features, connection_type='t1'):

    super(RedRCNN_mask_head, self).__init__()

    self.conv_layer= Conv_Block(in_features, in_features, kernel_size=3, padding=1)
    self.conv_transpose_layer= Conv_Block(in_features, in_features, kernel_size=2, stride=2, mode='transposed')

    connectors={}
    connectors['t1']=[0, 0, 2, 1]
    connectors['t2']=[1, 2, 3, 4]
    connectors['t3']=[0, 1, 0, 3]
    self.connectors= connectors[connection_type]
    
  def forward(self, x):

    layer_outs=[torch.zeros_like(x), x]
    for stage_idx in self.connectors:
      layer_outs.append(layer_outs[stage_idx] + self.conv_layer(layer_outs[-1]))
    
    return self.conv_transpose_layer(layer_outs[-1])
    

class RED_RCNN(Module):
  def __init__(self, num_classes, connection_type='t1'):
    super(RED_RCNN, self).__init__()
    
    self.base_model = maskrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True, trainable_backbone_layers=None)
    self.in_features_box = self.base_model.roi_heads.box_predictor.cls_score.in_features
    self.in_features_mask = self.base_model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    self.hidden_layers = 256

    self.base_model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features_box, num_classes)
    self.base_model.roi_heads.mask_head = RedRCNN_mask_head(self.in_features_mask, connection_type)
    self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask, self.hidden_layers, num_classes)

  def forward(self, x, y=None, mode='train'):
    if mode=='train':
      self.train()
      return self.base_model(x, y)
    elif mode=='validation':
      self.eval()
      return self.base_model(x)
    else:
        raise ValueError(f"Invalid mode. Expecting 'train' or 'validation' but got '{mode}'")

class Classification_Branch(Module):
  def __init__(self, in_channels, num_classes=2):
    super(Classification_Branch, self).__init__()
    self.global_pool= AdaptiveAvgPool2d(output_size=(1,1))
    self.fc= Linear(in_features=in_channels, out_features=num_classes)
  
  def forward(self, x):
    return self.fc(torch.squeeze(self.global_pool(x)))


def dice_coef(scores, targets):
  smooth=1e-3
  iou= []
  score_masks= scores['masks']
  target_masks= targets['masks']
  classes= scores['labels']
  for i, single_set_masks in enumerate(score_masks):
    for j, score_mask in enumerate(single_set_masks):
      y_true= target_masks[classes[i][j]-1]
      y_pred= score_mask[0]
      numerator= (y_true*y_pred).sum()
      denominator= (y_true+y_pred).sum()
      iou.append((numerator+smooth)/(denominator+smooth))
  return [torch.mean(torch.Tensor(iou))]
  
# class IOU_LOSS(Module):
#   def __init__(self, smooth=1e-3):
#     super(IOU_LOSS, self).__init__()
#     self.smooth= smooth
  
#   def forward(self, scores, targets):
#     iou= []
#     score_masks= scores['masks']
#     target_masks= targets['masks']
#     classes= scores['labels']
#     for i, single_set_masks in enumerate(score_masks):
#       for j, score_mask in enumerate(single_set_masks):
#         y_true= target_masks[classes[i][j]-1]
#         y_pred= score_mask[0]
#         numerator= (y_true*y_pred).sum()
#         denominator= (y_true+y_pred-y_true*y_pred).sum()
#         iou.append((numerator+self.smooth)/(denominator+self.smooth))
#     return 1-torch.mean(torch.Tensor(iou))

# def merge_targets_to_device(targets, batch_size, device):
#   return [{key : value[i].to(device=device) for key, value in targets.items()} for i in range(batch_size)]

# def list_dicts_to_dict(list_dicts):
#   res = defaultdict(list)
#   for sub in list_dicts:
#       for key in sub:
#           res[key].append(sub[key]) 
#   return res

def merge_samples_to_batches(targets, batch_size):
  new_targets = []
  for i in range(batch_size):
    new_targets.append({})
    for k in targets:
      new_targets[i][k] = targets[k][i]
  return new_targets

def IOU_LOSS(scores, targets, smooth=1e-5):
    iou = []
    t_labels = targets['labels']
    t_masks = targets['masks']
    s_labels = scores['labels']
    s_masks = scores['masks']
    t_masks = torch.flatten(t_masks, start_dim=1, end_dim=2) #(2, 512*512)
    s_masks = torch.flatten(s_masks, start_dim=1, end_dim=3) #(100, 512*512)
    # print(t_masks.shape, s_masks.shape) #(2,262144), (100,262144)

    for i in range(len(s_labels)):
      intersection = (s_masks[i]*t_masks[s_labels[i]-1]).sum()
      union = (s_masks[i] + t_masks[s_labels[i]-1] - s_masks[i]*t_masks[s_labels[i]-1]).sum()
      iou.append(1 - ((intersection)/(union + smooth))) #IoU Loss
    
    a = torch.sum(torch.Tensor(iou))
    return a


def process_data(images, targets, batch_size, device):
  images = images/255.0
  targets = merge_samples_to_batches(targets, batch_size)
  images_list = list(image.to(device) for image in images)
  targets_dict = [{k: v.to(device) for k, v in t.items()} for t in targets]
  return images_list, targets_dict

def save_checkpoint(model, optimizer, file_name):

  checkpoint= {'state_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
  torch.save(checkpoint,file_name)

def load_checkpoint(model, optimizer, file_name, device):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  optimizer.load_state_dict(check_pt['optimizer_dict'])

  return model, optimizer
