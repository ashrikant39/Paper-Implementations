import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import *
import  albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Red_RCNN_Dataset(Dataset):

  def __init__(self, main_dir,csv_file_path, img_shape, transforms=None):
    self.main_dir= main_dir
    self.transforms = transforms
    self.img_shape= img_shape
    self.df= pd.read_csv(csv_file_path)
    self.classes = np.array([255, 128, 0])
    
  def process_masks(self, mask):
    obj_ids = np.unique(mask)
    masks = mask <= obj_ids[:-1, None, None]
    masks= np.uint8(masks)

    boxes = []
    for mask in masks:
      x,y,w,h= cv2.boundingRect(mask) 
      boxes.append(np.array([x,y,x+w,y+h]))

    return np.transpose(masks, (1, 2, 0)), np.array(boxes)
    
  def __getitem__(self, idx):

    img_path= os.path.join(self.main_dir, self.df.iloc[idx]['Image'])
    mask_path= os.path.join(self.main_dir, self.df.iloc[idx]['Mask'])
    class_label= self.df.iloc[idx]['Class label']
    
    img =cv2.resize(cv2.imread(img_path)[:,:,::-1], self.img_shape, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(cv2.imread(mask_path, 0), self.img_shape, interpolation=cv2.INTER_NEAREST)

    masks, boxes= self.process_masks(mask)
    num_objs = len(self.classes[1:])
    # print(boxes)

    target = {}
    target["boxes"] = boxes
    target["labels"]= [1, 2]
    target['class_label']= class_label
    target["masks"] = masks
    target["image_id"] = idx
    target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
    
    if self.transforms is not None:
      transformed = self.transforms(image=img, mask=target['masks'], bboxes=target['boxes'], bbox_classes=target['labels'])
      
      image = transformed['image']/255.
      target['masks'] = transformed['mask'].permute(2, 0, 1)
      target['boxes'] = np.array(transformed['bboxes'], dtype=np.int64)
      boxes_t = np.array(transformed['bboxes'])
      target['labels'] = np.array(transformed['bbox_classes'], dtype=int)
      target['area'] = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

    return image, target

  def __len__(self):
      return len(self.df)


def visualize_bbox(img, bbox, class_name, color_list=[(0,255,0),(0,0,255)], thickness=2):
    
    x_min, y_min, x_max, y_max = bbox
    if class_name == 'Cup':
      color = color_list[0]
    else:
      color = color_list[1]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color= (255,255,255),
        lineType=cv2.LINE_AA,
    )
    return img
    
    