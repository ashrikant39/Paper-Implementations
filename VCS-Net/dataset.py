import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm
from utils import *
import argparse, json


class Frame_Dataset(Dataset):

    def __init__(self, csv_path, BLOCK_SIZE, NUM_FRAMES, data_percent = 0.05):
        """
        Final Expected shape :(batch, frames, channel, height, width)
        """
        self.full_df = pd.read_csv(csv_path)
        self.block_size = BLOCK_SIZE
        self.num_frames = NUM_FRAMES
        self.total_examples = int(data_percent*len(self.full_df))
        self.use_df = self.full_df[:self.total_examples]
        
    def __len__(self):
        return self.total_examples
    
    def __getitem__(self, index):

        frames = []
        load_dir = self.use_df['dir_names'].iloc[index]
        filenames = sorted(os.listdir(load_dir))
        
        assert len(filenames) >= self.num_frames
        
        
        for img_filename in filenames[:self.num_frames]:
            image = torchvision.io.read_image(os.path.join(load_dir, img_filename))
            c, h, w = image.shape
            image = normalize(image[:,:(h - h%self.block_size),:(w - w%self.block_size)])
            frames.append(image)
        
        return torch.stack(frames, dim=0)
    
    def get_dataloader(self, BATCH_SIZE):
        return DataLoader(self, batch_size=BATCH_SIZE)

    def show_random_sequence(self):

        plt.figure(figsize=(20, 20))
        loader = self.get_dataloader(1)
        images = next(iter(loader))
        grid = make_grid(torch.permute(images[0], dims=[-1, 0, 1, 2]), nrow=3)
        show(grid)        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help = "path to parameter file")
    args = parser.parse_args()
    json_path = args.json_path
    
    with open(json_path, 'r') as fp:
        param_dict = json.load(fp)
    
    train_dataset = Frame_Dataset(param_dict["TRAIN_CSV_PATH"], param_dict["BLOCK_SIZE"], param_dict["NUM_FRAMES"], param_dict["DATA_PERCENT"])
    dataloader = train_dataset.get_dataloader(1)

    for batch in dataloader:
        print(f"Shape = {batch.shape}")
