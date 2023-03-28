import time, copy
import itertools

import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm


from dataset import *
from models import *

def PSNR(input1, input2, max_val = 1.0):

    mse = F.mse_loss(input1, input2)
    psnr = 20*torch.log10(max_val/torch.sqrt(mse))
    return psnr
    

def train_model(keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder, dataloaders, criterion, optimizer, save_dir = None, num_epochs=10, model_name='VCSNet', device= "cpu"):
    """
    Args:
        model: The NN to train
        dataloaders: A dictionary containing at least the keys 
                    'train','val' that maps to Pytorch data loaders for the dataset
        criterion: The Loss function
        optimizer: Pytroch optimizer. The algorithm to update weights 
        num_epochs: How many epochs to train for
        save_dir: Where to save the best model weights that are found. Using None will not write anything to disk.

    Returns:
        model: The trained NN
        tr_acc_history: list, training accuracy history. Recording freq: one epoch.
        val_acc_history: list, validation accuracy history. Recording freq: one epoch.
    """

    val_psnr_history = []
    tr_psnr_history = []
    

    keyframe_encoder = keyframe_encoder.to(device)
    non_keyframe_encoder = non_keyframe_encoder.to(device)
    keyframe_initial_decoder = keyframe_initial_decoder.to(device)
    non_keyframe_initial_decoder = non_keyframe_initial_decoder.to(device)

    deep_decoder = deep_decoder.to(device)


    best_keyframe_encoder_wts = copy.deepcopy(keyframe_encoder.state_dict())
    best_non_keyframe_encoder_wts = copy.deepcopy(non_keyframe_encoder.state_dict())
    best_keyframe_init_decoder_wts = copy.deepcopy(keyframe_initial_decoder.state_dict())
    best_non_keyframe_init_decoder_wts = copy.deepcopy(non_keyframe_initial_decoder.state_dict())
    best_deep_decoder_wts = copy.deepcopy(deep_decoder.state_dict())
    
    best_psnr = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                keyframe_encoder.train()  # Set model to training mode
                non_keyframe_encoder.train()
                keyframe_initial_decoder.train()
                non_keyframe_initial_decoder.train()
                deep_decoder.train()
            else:
                keyframe_encoder.eval()   # Set model to evaluate mode
                non_keyframe_encoder.eval()
                keyframe_initial_decoder.eval()
                non_keyframe_initial_decoder.eval()
                deep_decoder.eval()

            # loss and number of correct prediction for the current batch
            running_loss = 0.0
            running_psnr = 0.0

            # Iterate over data.
            # TQDM has nice progress bars

            with tqdm(dataloaders[phase], unit="batch") as tepoch:

                for inputs in tepoch:

                    inputs = inputs.to(device)
                    
                    tepoch.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()


                    KEYFRAME_IDXS = [0, 10]
                    NON_KEYFRAME_IDXS = [i for i in range(NUM_FRAMES) if i not in KEYFRAME_IDXS]

                    keyframe_measurements = keyframe_encoder(inputs[...,KEYFRAME_IDXS])
                    non_keyframe_measurements = non_keyframe_encoder(inputs[...,NON_KEYFRAME_IDXS])

                    

                    keyframe_init_reconstruction = torch.reshape(keyframe_initial_decoder(keyframe_measurements), inputs[...,KEYFRAME_IDXS].shape)
                    non_keyframe_init_reconstruction = torch.reshape(non_keyframe_initial_decoder(non_keyframe_measurements), inputs[...,NON_KEYFRAME_IDXS].shape)

                    deep_recon_keyframes, deep_recon_non_keyframes = deep_decoder(keyframe_init_reconstruction, non_keyframe_init_reconstruction)
                    deep_reconstruction = torch.zeros_like(inputs)
                    deep_reconstruction[...,KEYFRAME_IDXS] = deep_recon_keyframes
                    deep_reconstruction[...,NON_KEYFRAME_IDXS] = deep_recon_non_keyframes

                    # import pdb; pdb.set_trace()

                    keyframe_init_recon_loss = criterion(keyframe_init_reconstruction, inputs[...,KEYFRAME_IDXS])
                    non_keyframe_init_recon_loss = criterion(non_keyframe_init_reconstruction, inputs[...,NON_KEYFRAME_IDXS])

                    deep_recon_loss = criterion(deep_reconstruction, inputs)

                    total_loss = keyframe_init_recon_loss + non_keyframe_init_recon_loss + deep_recon_loss

                    if phase == "train":
                        total_loss.backward()
                        optimizer.step()

                    psnr = PSNR(deep_reconstruction, inputs)
                    running_loss += (total_loss.item() * inputs.size(0))
                    running_psnr += psnr


                    tepoch.set_postfix(loss=total_loss.item(), psnr = psnr.item())
                    time.sleep(0.1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_psnr = running_psnr / len(dataloaders[phase].dataset)

            # deep copy the model
            if phase == 'val' and epoch_psnr > best_psnr:
                best_psnr = epoch_psnr
                best_keyframe_encoder_wts = copy.deepcopy(keyframe_encoder.state_dict())
                best_non_keyframe_encoder_wts = copy.deepcopy(non_keyframe_encoder.state_dict())
                best_keyframe_init_decoder_wts = copy.deepcopy(keyframe_initial_decoder.state_dict())
                best_non_keyframe_init_decoder_wts = copy.deepcopy(non_keyframe_initial_decoder.state_dict())
                best_deep_decoder_wts = copy.deepcopy(deep_decoder.state_dict())


                if save_dir:
                    torch.save(best_keyframe_encoder_wts, os.path.join(save_dir, model_name + '_keyframe_encoder.pth'))
                    torch.save(best_non_keyframe_encoder_wts, os.path.join(save_dir, model_name + '_non_keyframe_encoder.pth'))
                    torch.save(best_keyframe_init_decoder_wts, os.path.join(save_dir, model_name + '_keyframe_init_decoder.pth'))
                    torch.save(best_non_keyframe_init_decoder_wts, os.path.join(save_dir, model_name + '_non_keyframe_init_decoder.pth'))
                    torch.save(best_deep_decoder_wts, os.path.join(save_dir, model_name + '_deep_decoder.pth'))
                    

            # record the train/val accuracies
            if phase == 'val':
                val_psnr_history.append(epoch_psnr)
            else:
                tr_psnr_history.append(epoch_psnr)
                
    print('Best val PSNR: {:4f}'.format(best_psnr))

    return keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder, tr_psnr_history, val_psnr_history
    


if __name__=="__main__":
    
    IMG_CHANNELS = 3
    NUM_FRAMES = 21
    NUM_KEYFRAMES = 2
    NUM_NON_KEYFRAMES = NUM_FRAMES - NUM_KEYFRAMES
    KEYFRAME_SAMPLING_RATIO = 2**(-2)
    NON_KEYFRAME_SAMPLING_RATIO = 2**(-4)
    BLOCK_SIZE = 32
    BATCH_SIZE = 1
    
    train_dataset = Frame_Dataset("data/train.csv", BLOCK_SIZE, NUM_FRAMES)
    val_dataset = Frame_Dataset("data/val.csv", BLOCK_SIZE, NUM_FRAMES)
    dataloaders = {"train": train_dataset.get_dataloader(BATCH_SIZE),
                   "val": val_dataset.get_dataloader(BATCH_SIZE)}
                   
    
    keyframe_encoder = VCSNet_Encoder(image_channels=IMG_CHANNELS,
                             block_size = BLOCK_SIZE, 
                             subsampling_ratio = KEYFRAME_SAMPLING_RATIO,
                             num_frames = NUM_KEYFRAMES)
    
    non_keyframe_encoder = VCSNet_Encoder(image_channels=IMG_CHANNELS,
                             block_size = BLOCK_SIZE, 
                             subsampling_ratio = NON_KEYFRAME_SAMPLING_RATIO,
                             num_frames = NUM_NON_KEYFRAMES)
    
    keyframe_initial_decoder = VCSNet_Initial_Decoder(image_channels=IMG_CHANNELS,
                                block_size = BLOCK_SIZE, 
                                subsampling_ratio = KEYFRAME_SAMPLING_RATIO,
                                num_frames = NUM_KEYFRAMES)
    
    non_keyframe_initial_decoder = VCSNet_Initial_Decoder(image_channels=IMG_CHANNELS,
                                    block_size = BLOCK_SIZE, 
                                    subsampling_ratio = NON_KEYFRAME_SAMPLING_RATIO,
                                    num_frames = NUM_NON_KEYFRAMES)
            
    deep_decoder = VCSNet_Keyframe_Decoder(image_channels=IMG_CHANNELS,
                                      conv_channels= 2,
                                      kernel_size = 3,
                                      stride = 1,
                                      padding = 1,
                                      total_convs = 5,
                                      num_frames = NUM_FRAMES)
                                    
    save_dir = "weights"
    optimizer = optim.SGD(params = list(keyframe_encoder.parameters()) + list(non_keyframe_encoder.parameters()) +list(keyframe_initial_decoder.parameters()) + list(non_keyframe_initial_decoder.parameters())+list(deep_decoder.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    
    encoder_trained, init_decoder_trained, deep_decoder_trained, train_psnr_history, val_psnr_history = train_model(keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder, dataloaders, criterion, optimizer, save_dir, device=device)
