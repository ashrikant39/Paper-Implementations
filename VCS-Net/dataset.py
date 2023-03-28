import os, shutil
from IPython.core.pylabtools import figsize
import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize = (40,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        
def segregate_data(main_dir, save_data_dir):

    filenames = os.listdir(main_dir)
    dirnames = []
    
    if os.path.exists(save_data_dir) is False:
        os.mkdir(save_data_dir)
        
    for filename in tqdm(filenames, total=len(filenames)):
    
        foldname, _ = filename.split("_")
        
        source = os.path.join(data_dir, filename)
        dest = os.path.join(save_data_dir, foldname, filename)
    
        if foldname not in dirnames:
            os.mkdir(os.path.join(save_data_dir, foldname))
            dirnames.append(foldname)
        
        shutil.copyfile(source, dest)
        
    return 

class Frame_Dataset(Dataset):

    def __init__(self, csv_path, BLOCK_SIZE, NUM_FRAMES):

        self.df = pd.read_csv(csv_path)
        self.block_size = BLOCK_SIZE
        self.num_frames = NUM_FRAMES
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        frames = []
        load_dir = self.df.iloc[index]['0']
        filenames = sorted(os.listdir(load_dir))
        
        assert len(filenames) >= self.num_frames
        
        
        for img_filename in filenames[:self.num_frames]:
            image = torchvision.io.read_image(os.path.join(load_dir, img_filename))/255.
            c, h, w = image.shape
            image = image[:,:(h - h%self.block_size),:(w - w%self.block_size)]
            frames.append(image)
        
        return torch.stack(frames, dim=-1)
    
    def get_dataloader(self, BATCH_SIZE):
        return DataLoader(self, batch_size=BATCH_SIZE)

    def show_random_sequence(self):

        plt.figure(figsize=(10,10))
        loader = self.get_dataloader(1)
        images = next(iter(loader))
        grid = make_grid(torch.permute(images[0], dims=[-1, 0, 1, 2]), nrow=7)
        show(grid)
        

if __name__=="__main__":
    print("Not written")