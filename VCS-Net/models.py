import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import *
import argparse, json
import numpy as np

    
class VCSNet_Encoder(nn.Module):

    def __init__(self, image_channels, block_size, subsampling_ratio):
        super().__init__()

        self.total_measurements = int(image_channels*subsampling_ratio*block_size**2)
        self.encoding_conv = nn.Conv2d(in_channels = image_channels,
                                       out_channels = self.total_measurements,
                                       kernel_size = block_size,
                                       stride = block_size)
    
    def forward(self, frames):

        assert len(frames.shape)==5
        batch_size, num_frames, _, _, _ = frames.shape
        result = self.encoding_conv(torch.flatten(frames, end_dim=1))

        return nn.Unflatten(dim=0, unflattened_size = [batch_size, num_frames])(result)
        
        
class VCSNet_Initial_Decoder(nn.Module):

    def __init__(self, image_channels, block_size, subsampling_ratio):
        super().__init__()
        
       
        self.total_filters = image_channels*block_size**2
        self.measurements = int(image_channels*subsampling_ratio*block_size**2)

        self.decoding_conv = nn.Conv2d(in_channels = self.measurements,
                                       out_channels = self.total_filters,
                                       kernel_size = 1,
                                       stride = 1)
        
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=block_size)
    
    def forward(self, measurements):

        assert len(measurements.shape)==5
        batch_size, num_frames, _, _, _ = measurements.shape
        result = self.shuffle_layer(self.decoding_conv(torch.flatten(measurements, end_dim=1)))

        return nn.Unflatten(dim=0, unflattened_size = [batch_size, num_frames])(result)
        

class VCSNet_Keyframe_Decoder(nn.Module):

    def __init__(self, image_channels, conv_channels, kernel_size, stride, padding, total_convs, total_frames, batch_norm = True, keyframe_idxs = [0]):
        super().__init__()
        assert total_convs > 2

        channel_list = [image_channels] + [conv_channels]*(total_convs-2) + [image_channels]

        self.keyframe_convs = nn.ModuleList()
        self.non_keyframe_convs = nn.ModuleList()
        self.ref_convs = nn.ModuleList()
        self.keyframe_idxs = keyframe_idxs
        self.non_keyframe_idxs = [i for i in range(total_frames) if i not in keyframe_idxs]

        for i in range(total_convs-1):

            self.keyframe_convs.append(nn.Sequential(
                nn.Conv2d(in_channels = channel_list[i],
                          out_channels = channel_list[i+1],
                          kernel_size = kernel_size,
                          stride = stride, 
                          padding = padding),
                nn.BatchNorm2d(num_features = channel_list[i+1]) if batch_norm is True else nn.Identity(),
                nn.ReLU()
            ))

            self.non_keyframe_convs.append(nn.Sequential(
                nn.Conv2d(in_channels = channel_list[i],
                          out_channels = channel_list[i+1],
                          kernel_size = kernel_size,
                          stride = stride, 
                          padding = padding),
                nn.BatchNorm2d(num_features = channel_list[i+1]) if batch_norm is True else nn.Identity()
            ))

            self.ref_convs.append(nn.Sequential(
                nn.Conv2d(in_channels = channel_list[i+1],
                          out_channels = channel_list[i+1],
                          kernel_size = kernel_size,
                          stride = stride, 
                          padding = padding),
                nn.BatchNorm2d(num_features = channel_list[i+1]) if batch_norm is True else nn.Identity()
            ))
                
    def forward(self, keyframes, non_keyframes):
    
        keyframe_outputs = [keyframes[:,i] for i in range(len(self.keyframe_idxs))]
        non_keyframe_outputs = [non_keyframes[:,i] for i in range(len(self.non_keyframe_idxs))]


        for keyframe_layer, non_keyframe_layer, ref_layer in zip(self.keyframe_convs, self.non_keyframe_convs, self.ref_convs):

            keyframe_outputs = [keyframe_layer(frame) for frame in keyframe_outputs]
            ref_output = sum([ref_layer(frame) for frame in keyframe_outputs])
            non_keyframe_outputs = [F.relu(non_keyframe_layer(frame) + ref_output) for frame in non_keyframe_outputs]

        keyframe_outputs_tensor = torch.stack(keyframe_outputs, dim=1)
        non_keyframe_outputs_tensor = torch.stack(non_keyframe_outputs, dim=1)

        return keyframe_outputs_tensor, non_keyframe_outputs_tensor


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help = "path to parameter file")
    args = parser.parse_args()
    json_path = args.json_path
    
    with open(json_path, 'r') as fp:
        param_dict = json.load(fp)
        
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
                                        batch_norm = False)
                                        
                                        
    keyframe_idxs = param_dict["KEYFRAME_IDXS"]
    non_keyframe_idxs = [i for i in range(param_dict["NUM_FRAMES"]) if i not in keyframe_idxs]
    
    ip = torch.randn((param_dict["BATCH_SIZE"], param_dict["NUM_FRAMES"], param_dict["IMG_CHANNELS"], 256, 256))
    keyf = ip[:,keyframe_idxs]
    non_keyf = ip[:,non_keyframe_idxs]
    
    keyf_meas = keyframe_encoder(keyf)
    non_keyf_meas = non_keyframe_encoder(non_keyf)
    
    keyf_init_recon = keyframe_initial_decoder(keyf_meas)
    non_keyf_init_recon = non_keyframe_initial_decoder(non_keyf_meas)
    
    keyf_final_recon, non_keyf_final_recon = deep_decoder(keyf_init_recon, non_keyf_init_recon)
    
    final_recon = torch.zeros_like(ip)
    final_recon[:,keyframe_idxs] = keyf_final_recon
    final_recon[:,non_keyframe_idxs] = non_keyf_final_recon
    
    print(final_recon.shape)



class L1_L2_Loss(nn.Module):

    def __init__(self, alpha):
        super().__init__()

        self.alpha = alpha
    
    def forward(self, y_true, y_pred):

        loss = (1 - self.alpha)*nn.MSELoss()(y_true, y_pred) + self.alpha*nn.L1Loss()(y_true, y_pred)
        return loss

if __name__=="__main__":
    main()
