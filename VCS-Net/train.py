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

def train_model(keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder, dataloaders, criterion, optimizer, scheduler, save_dir = None, num_epochs=10, model_name='VCSNet', device= "cpu"):
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

                    ###############################################################
                    # TODO:                                                       #
                    # Please read all the inputs carefully!                       #
                    # For "train" phase:                                          #
                    # (i)   Compute the outputs using the model                   #
                    #       Also, use the  outputs to calculate the class         #
                    #       predicted by the model,                               #
                    #       Store the predicted class in 'preds'                  #
                    #       (Think: argmax of outputs across a dimension)         #
                    #       torch.max() might help!                               #
                    # (ii)  Use criterion to store the loss in 'loss'             # 
                    # (iii) Update the model parameters                           #
                    # Notes:                                                      #
                    # - Don't forget to zero the gradients before beginning the   #
                    # loop!                                                       #
                    # - "val" phase is the same as train, but without backprop    #
                    # - Compute the outputs (Same as "train", calculate 'preds'   #
                    # too),                                                       #
                    # - Calculate the loss and store it in 'loss'                 #
                    ###############################################################
                    optimizer.zero_grad()


                    keyframe_idxs = deep_decoder.keyframe_idxs
                    non_keyframe_idxs = deep_decoder.non_keyframe_idxs



                    keyframe_measurements = keyframe_encoder(inputs[:,keyframe_idxs])
                    non_keyframe_measurements = non_keyframe_encoder(inputs[:,non_keyframe_idxs])

                    

                    keyframe_init_reconstruction = keyframe_initial_decoder(keyframe_measurements)
                    non_keyframe_init_reconstruction = non_keyframe_initial_decoder(non_keyframe_measurements)

                    keyframe_deep_reconstruction, non_keyframe_deep_reconstruction = deep_decoder(keyframe_init_reconstruction, non_keyframe_init_reconstruction)
    
                    keyframe_init_recon_loss = criterion(keyframe_init_reconstruction, inputs[:,keyframe_idxs])
                    non_keyframe_init_recon_loss = criterion(non_keyframe_init_reconstruction, inputs[:,non_keyframe_idxs])

                    keyframe_deep_recon_loss = criterion(keyframe_deep_reconstruction, inputs[:,keyframe_idxs])
                    non_keyframe_deep_recon_loss = criterion(non_keyframe_deep_reconstruction, inputs[:,non_keyframe_idxs])

                    total_loss =  keyframe_deep_recon_loss + non_keyframe_deep_recon_loss + keyframe_init_recon_loss + non_keyframe_init_recon_loss

                    deep_reconstruction = np.zeros(list(inputs.shape), dtype = np.float32)
                    deep_reconstruction[:,keyframe_idxs] = keyframe_deep_reconstruction.cpu().detach().numpy()
                    deep_reconstruction[:, non_keyframe_idxs] = non_keyframe_deep_reconstruction.cpu().detach().numpy()

                    if phase == "train":
                        total_loss.backward()
                        optimizer.step()

                    psnr = PSNR(deep_reconstruction, inputs.cpu().detach().numpy())
                    ssim = SSIM(deep_reconstruction, inputs.cpu().detach().numpy())

                    running_loss += (total_loss.item() * inputs.size(0))
                    running_psnr += psnr.item()


                    tepoch.set_postfix(loss=total_loss.item(), psnr = psnr.item(), ssim=ssim.item())
                    time.sleep(0.1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_psnr = running_psnr / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_psnr))

            # deep copy the model
            if phase == 'val' and epoch_psnr > best_psnr:
                best_psnr = epoch_psnr
                best_keyframe_encoder_wts = copy.deepcopy(keyframe_encoder.state_dict())
                best_non_keyframe_encoder_wts = copy.deepcopy(non_keyframe_encoder.state_dict())
                best_keyframe_init_decoder_wts = copy.deepcopy(keyframe_initial_decoder.state_dict())
                best_non_keyframe_init_decoder_wts = copy.deepcopy(non_keyframe_initial_decoder.state_dict())
                best_deep_decoder_wts = copy.deepcopy(deep_decoder.state_dict())

                # save the best model weights
                # =========================================================== #
                # IMPORTANT:
                # Losing your connection to colab will lead to loss of trained 
                # weights.
                # You should download the trained weights to your local machine. 
                # Later, you can load these weights directly without needing to 
                # train the neural networks again.
                # =========================================================== #
                if save_dir:
                    parent_dir = os.path.join(save_dir, model_name)
                    if os.path.exists(parent_dir) is False:
                        os.mkdir(parent_dir)
                    torch.save(best_keyframe_encoder_wts, os.path.join(parent_dir, 'keyframe_encoder.pth'))
                    torch.save(best_non_keyframe_encoder_wts, os.path.join(parent_dir , 'non_keyframe_encoder.pth'))
                    torch.save(best_keyframe_init_decoder_wts, os.path.join(parent_dir , 'keyframe_init_decoder.pth'))
                    torch.save(best_non_keyframe_init_decoder_wts, os.path.join(parent_dir , 'non_keyframe_init_decoder.pth'))
                    torch.save(best_deep_decoder_wts, os.path.join(parent_dir , 'deep_decoder.pth'))
                    

            # record the train/val accuracies


            if phase == 'val':
                val_psnr_history.append(epoch_psnr)
            else:
                tr_psnr_history.append(epoch_psnr)

            if phase == "train":
                scheduler.step()        
        
    print('Best val PSNR: {:4f}'.format(best_psnr))

    return tr_psnr_history, val_psnr_history


if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")


    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help = "path to parameter file")
    args = parser.parse_args()
    json_path = args.json_path
    
    with open(json_path, 'r') as fp:
        param_dict = json.load(fp)

    train_dataset = Frame_Dataset(
                                param_dict["TRAIN_CSV_PATH"], 
                                param_dict["BLOCK_SIZE"], 
                                param_dict["NUM_FRAMES"], 
                                param_dict["DATA_PERCENT"])


    val_dataset = Frame_Dataset(param_dict["VAL_CSV_PATH"], 
                                param_dict["BLOCK_SIZE"], 
                                param_dict["NUM_FRAMES"], 
                                param_dict["DATA_PERCENT"])

    dataloaders = {"train": train_dataset.get_dataloader(param_dict["BATCH_SIZE"]),
                "val": val_dataset.get_dataloader(param_dict["BATCH_SIZE"])}

    keyframe_encoder = VCSNet_Encoder(image_channels=param_dict["IMG_CHANNELS"],
                                block_size = param_dict["BLOCK_SIZE"], 
                                subsampling_ratio = param_dict["KEYFRAME_SAMPLING_RATIO"])

    non_keyframe_encoder = VCSNet_Encoder(image_channels=param_dict["IMG_CHANNELS"],
                                block_size = param_dict["BLOCK_SIZE"], 
                                subsampling_ratio = param_dict["NON_KEYFRAME_SAMPLING_RATIO"])

    keyframe_initial_decoder = VCSNet_Initial_Decoder(image_channels=param_dict["IMG_CHANNELS"],
                                block_size = param_dict["BLOCK_SIZE"], 
                                subsampling_ratio = param_dict["KEYFRAME_SAMPLING_RATIO"])

    non_keyframe_initial_decoder = VCSNet_Initial_Decoder(image_channels= param_dict["IMG_CHANNELS"],
                                    block_size = param_dict["BLOCK_SIZE"], 
                                    subsampling_ratio = param_dict["NON_KEYFRAME_SAMPLING_RATIO"])
                                    
    deep_decoder = VCSNet_Keyframe_Decoder(image_channels= param_dict["IMG_CHANNELS"],
                                        conv_channels= param_dict["FILTERS_DEEP_CONV"],
                                        kernel_size = param_dict["KERNEL_SIZE_DEEP_CONV"],
                                        stride = param_dict["STRIDE_DEEP_CONV"],
                                        padding = param_dict["PADDING_DEEP_CONV"],
                                        total_convs = param_dict["TOTAL_DEEP_CONVS"],
                                        total_frames = param_dict["NUM_FRAMES"],
                                        keyframe_idxs= param_dict["KEYFRAME_IDXS"],
                                        batch_norm = param_dict["INCLUDE_BATCH_NORM"])

    model_list = [keyframe_encoder, non_keyframe_encoder, keyframe_initial_decoder, non_keyframe_initial_decoder, deep_decoder]
    total_parameters = list(keyframe_encoder.parameters()) + list(non_keyframe_encoder.parameters()) + list(keyframe_initial_decoder.parameters()) + list(non_keyframe_initial_decoder.parameters()) + list(deep_decoder.parameters()) 
    criterion = L1_L2_Loss(param_dict["L1_WEIGHT"])

    if param_dict["OPTIMIZER"] == "SGD":
        optimizer = optim.SGD(params = total_parameters,
                            lr = param_dict["INIT_LR"],
                            weight_decay = param_dict["WEIGHT_DECAY"])
        
    elif param_dict["OPTIMIZER"] == "ADAM":
        optimizer = optim.Adam(params = total_parameters,
                            lr = param_dict["INIT_LR"],
                            weight_decay = param_dict["WEIGHT_DECAY"])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer = optimizer,
                                            milestones = param_dict["MILESTONES"],
                                            gamma = param_dict["LR_DECAY"])

    weights_dir = param_dict["WEIGHTS_DIR"]

    if os.path.exists(weights_dir) is False:
        os.mkdir(weights_dir)

    save_params_file = "train_params.json"

    tr_psnr_history, val_psnr_history = train_model(keyframe_encoder = keyframe_encoder,
                                                    non_keyframe_encoder = non_keyframe_encoder,
                                                    keyframe_initial_decoder = keyframe_initial_decoder,
                                                    non_keyframe_initial_decoder = non_keyframe_initial_decoder,
                                                    deep_decoder = deep_decoder,
                                                    dataloaders = dataloaders,
                                                    optimizer = optimizer,
                                                    scheduler = scheduler,
                                                    criterion = criterion,
                                                    save_dir = weights_dir,
                                                    model_name = param_dict["MODEL_NAME"],
                                                    num_epochs = param_dict["NUM_EPOCHS"],
                                                    device = device
                                                    )
    
    with open(os.path.join(weights_dir, save_params_file), 'w') as fp:
        json.dump(param_dict, fp)