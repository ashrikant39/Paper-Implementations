import numpy as np
import torch, os, shutil
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
import shutil

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize = (40,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_batch(batch):
    grid = make_grid(batch[0], nrow=7)
    show(grid)
        
        
def segregate_data(main_dir, save_data_dir):

    filenames = os.listdir(main_dir)
    dirnames = []
    
    if os.path.exists(save_data_dir) is False:
        os.mkdir(save_data_dir)
        
    for filename in tqdm(filenames, total=len(filenames)):
    
        foldname, _ = filename.split("_")
        
        source = os.path.join(main_dir, filename)
        dest = os.path.join(save_data_dir, foldname, filename)
    
        if foldname not in dirnames:
            os.mkdir(os.path.join(save_data_dir, foldname))
            dirnames.append(foldname)
        
        shutil.copyfile(source, dest)
        
    return True

def normalize(x):
    return (x - x.min())/(x.max() - x.min())

    
def PSNR(input1, input2):
    psnr = peak_signal_noise_ratio(normalize(input1), normalize(input2))
    return psnr

def SSIM(input1, input2):
    """
    input_shape = (batch, frames, channel, height, width) -> (batch*frames, height, width, channel)
    """
    
    ip1 = np.transpose(normalize(input1).reshape(-1, *input1.shape[2:]), axes=[0, 2, 3, 1])
    ip2 = np.transpose(normalize(input2).reshape(-1, *input1.shape[2:]), axes=[0, 2, 3, 1])

    ssim_vals = []

    for i in range(len(ip1)):
        ssim_vals.append(structural_similarity(ip1[i], ip2[i], multichannel=True))

    # import pdb; pdb.set_trace()

    return np.mean(ssim_vals)
    
    
def test_images(images: torch.FloatTensor, keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder, keyframe_idxs):

    if images.max() != 1.0:
        images /= images.max()

    non_keyframe_idxs = [i for i in range(images.shape[1]) if i not in keyframe_idxs]
    keyframes = images[:,keyframe_idxs]
    non_keyframes = images[:,non_keyframe_idxs]

    keyframe_encoder.cpu().eval()
    non_keyframe_encoder.cpu().eval()
    keyframe_initial_decoder.cpu().eval()
    non_keyframe_initial_decoder.cpu().eval()
    deep_decoder.cpu().eval()

    with torch.no_grad():
        keyframe_measurements = keyframe_encoder(keyframes)
        non_keyframe_measurements = non_keyframe_encoder(non_keyframes)
        keyframe_initial_reconstruction = torch.reshape(keyframe_initial_decoder(keyframe_measurements), images[:,keyframe_idxs].shape)
        non_keyframe_initial_reconstruction = torch.reshape(non_keyframe_initial_decoder(non_keyframe_measurements), images[:,non_keyframe_idxs].shape)

        keyframe_deep_reconstruction, non_keyframe_deep_reconstruction = deep_decoder(keyframe_initial_reconstruction, non_keyframe_initial_reconstruction)
        deep_reconstruction = torch.zeros_like(images)
        deep_reconstruction[:,keyframe_idxs] = keyframe_deep_reconstruction
        deep_reconstruction[:, non_keyframe_idxs] = non_keyframe_deep_reconstruction
    
    show_batch(deep_reconstruction.cpu().detach())   
    return deep_reconstruction       