import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Frame_Wise_Convolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_frames):
        super().__init__()

        self.frame_wise_convs = nn.ModuleList()
        self.total_frames = num_frames

        for _ in range(self.total_frames):
            self.frame_wise_convs.append(nn.Conv2d(in_channels = in_channels,
                                                 out_channels = out_channels,
                                                 kernel_size = kernel_size,
                                                 stride = stride,
                                                 bias = False))

    def forward(self, frames):
        
        assert frames.shape[-1] == self.total_frames
        outputs = []

        for i, layer in enumerate(self.frame_wise_convs):
            outputs.append(layer(frames[...,i]))
        
        return torch.stack(outputs, dim=-1)
        

class VCSNet_Encoder(nn.Module):

    def __init__(self, image_channels, block_size, subsampling_ratio, num_frames):
        super().__init__()

        self.total_measurements = int(image_channels*subsampling_ratio*block_size**2)
        self.encoding_conv = Frame_Wise_Convolution(in_channels = image_channels,
                                                    out_channels = self.total_measurements,
                                                    kernel_size = block_size,
                                                    stride = block_size,
                                                    num_frames = num_frames)
    
    def forward(self, frames):
        return self.encoding_conv(frames)
        
        
class VCSNet_Initial_Decoder(nn.Module):

    def __init__(self, image_channels, block_size, subsampling_ratio, num_frames):
        super().__init__()

        self.total_filters = image_channels*block_size**2
        self.measurements = int(image_channels*subsampling_ratio*block_size**2)

        self.decoding_conv = Frame_Wise_Convolution(in_channels = self.measurements,
                                                    out_channels = self.total_filters,
                                                    kernel_size = 1,
                                                    stride = 1,
                                                    num_frames = num_frames)
    
    def forward(self, measurements):
        return self.decoding_conv(measurements)
        

class VCSNet_Keyframe_Decoder(nn.Module):

    def __init__(self, image_channels, conv_channels, kernel_size, stride, padding, total_convs, num_frames, batch_norm = True):
        super().__init__()
        assert total_convs > 2

        channel_list = [image_channels] + [conv_channels]*(total_convs-2) + [image_channels]

        self.keyframe_convs = nn.ModuleList()
        self.non_keyframe_convs = nn.ModuleList()
        self.ref_convs = nn.ModuleList()

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
        
        batch_size, channel, height, width, total_keyframes = keyframes.shape
        _, _, _, _, total_non_keyframes = non_keyframes.shape
        
        keyframes_output = [keyframes[...,i] for i in range(total_keyframes)]
        ref_outputs = []
        non_keyframes_output = [non_keyframes[...,i] for i in range(total_non_keyframes)]

        for keyframe_layer, non_keyframe_layer, ref_layer in zip(self.keyframe_convs, self.non_keyframe_convs, self.ref_convs):

            keyframes_output = [keyframe_layer(frame) for frame in keyframes_output]
            ref_output = sum([ref_layer(frame) for frame in keyframes_output])
            non_keyframes_output = [F.relu(non_keyframe_layer(frame) + ref_output) for frame in non_keyframes_output]

        keyframes_output_tensor = torch.stack(keyframes_output, dim=-1)
        non_keyframes_output_tensor = torch.stack(non_keyframes_output, dim=-1)

        return keyframes_output_tensor, non_keyframes_output_tensor
        

def main():

    IMG_CHANNELS = 3
    NUM_FRAMES = 10
    SAMPLING_RATIO = 2**(-4)
    BLOCK_SIZE = 32
    
    x = torch.randn((2, IMG_CHANNELS, 256, 256, NUM_FRAMES))
    
    encoder = VCSNet_Encoder(image_channels=IMG_CHANNELS,
                             block_size = BLOCK_SIZE, 
                             subsampling_ratio = SAMPLING_RATIO,
                             num_frames = NUM_FRAMES)
                             
    decoder = VCSNet_Initial_Decoder(image_channels=IMG_CHANNELS,
                             block_size = BLOCK_SIZE, 
                             subsampling_ratio = SAMPLING_RATIO,
                             num_frames = NUM_FRAMES)
                             
    key_dec = VCSNet_Keyframe_Decoder(image_channels=IMG_CHANNELS,
                                      conv_channels= 2,
                                      kernel_size = 3,
                                      stride = 1,
                                      padding = 1,
                                      total_convs = 5,
                                      num_frames = NUM_FRAMES)
                                      
    enc = encoder(x)
    dec = decoder(enc)
    out = torch.reshape(dec, x.shape)
    
    KEY_FRAME_IDXS = [0, 5]
    NON_KEYFRAME_IDXS = [i for i in range(NUM_FRAMES) if i not in KEY_FRAME_IDXS]
    
    keyframe_slices = x[...,KEY_FRAME_IDXS]
    non_keyframe_slices = x[...,NON_KEYFRAME_IDXS]
    
    keyframe_outs, non_keyframe_outs = key_dec(keyframe_slices, non_keyframe_slices)
    
    
if __name__=="__main__":
    main()
