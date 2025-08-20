# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import sys
import torch

# from .base_skip import NoChange
from .base_skip import NoChange
from collections import OrderedDict
from ccbdl.network.base import BaseNetwork
from ccbdl.utils import DEVICE

# %%

def magnitude_to_complex(magnitude):
    '''
    Convert the Amplitude and freqeuncy into complex values.

    Parameters
    ----------
    magnitude : torch.tensor
        (real, positive).

    Returns
    -------
    torch.tensor
        real + imaginary stacked.

    '''
    B, T, F = magnitude.shape
    # Random phase in [0, 2π]
    frequency = 2 * torch.pi * torch.arange(0, 512, device=magnitude.device)
    real = magnitude * torch.cos(frequency)
    imag = magnitude * torch.sin(frequency)
    return torch.stack([real, imag], dim=1)  # [B, 2, T, F]


def complex_to_magnitude(complex_tensor):
    '''
    Convert a complex tensor (real + imaginary stacked) back to its magnitude.

    Parameters
    ----------
    complex_tensor : torch.tensor
        A tensor of shape [B, 2, T, F], where the first channel (index 0) is the real part
        and the second channel (index 1) is the imaginary part.

    Returns
    -------
    torch.tensor
        The magnitude tensor of shape [B, T, F].
        This tensor retains its computational graph for gradient propagation.
    '''
    # complex_tensor has shape [B, 2, T, F]
    real = complex_tensor[:, 0, :, :]  # Extract the real part: [B, T, F]
    imag = complex_tensor[:, 1, :, :]  # Extract the imaginary part: [B, T, F]
    magnitude = torch.hypot(real, imag)

    return magnitude



class ModReLU(torch.nn.Module):
    '''
    ModReLU activation function for complex-valued inputs.

    This module performs a modified ReLU operation on the magnitude of complex numbers
    and scales the real and imaginary parts accordingly.

    Parameters
    ----------
    num_channels : int
        Number of channels for which to apply the learnable bias.

    Attributes
    ----------
    bias : torch.nn.Parameter
        A learnable bias term applied to the magnitude before ReLU.

    Returns
    -------
    torch.Tensor
        Complex tensor with the same shape as input, activated via modReLU.
    '''
    def __init__(self, num_channels):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_channels))
        torch.nn.init.constant_(self.bias, 0.1)

    def forward(self, z):
        '''
        Apply modReLU activation to complex input.
    
        The activation is computed as follows:
        scale = ReLU(|z| + bias) / (|z| + ε),
        where |z| is the magnitude of the complex number.
        The real and imaginary parts are scaled by this factor.
    
        Parameters
        ----------
        z : torch.Tensor
            Complex-valued input tensor of shape [B, 2, C, H, W] or [B, 2, H, W],
            where channel 0 is the real part and channel 1 is the imaginary part.
    
        Returns
        -------
        torch.Tensor
            Activated complex-valued tensor of the same shape as input.
        '''
        real, imag = z[:, 0], z[:, 1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        scale = torch.nn.functional.relu(mag + self.bias.view(1, -1, 1, 1)) / (mag + 1e-8)
        return torch.stack([real * scale, imag * scale], dim=1)

class ComplexConv2d(torch.nn.Module):
    '''
    Complex-valued 2D convolutional layer.

    Applies two separate real-valued convolutions to the real and imaginary parts
    of a complex input, combining them to preserve complex structure.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default is 0.

    Returns
    -------
    torch.Tensor
        Complex output tensor of shape [B, 2, C_out, H_out, W_out].
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, ins):
        '''
        Apply complex-valued 2D convolution.
    
        The input tensor is split into real and imaginary parts, and each part
        is convolved separately using real-valued convolution. The outputs are
        recombined to form a complex-valued output.
    
        Parameters
        ----------
        ins : torch.Tensor
            Complex-valued input tensor of shape [B, 2, H, W],
            where channel 0 is real and channel 1 is imaginary.
    
        Returns
        -------
        torch.Tensor
            Complex-valued output tensor of shape [B, 2, C_out, H_out, W_out],
            where channel 0 is real and channel 1 is imaginary.
        '''
        real, imag = ins[:, 0], ins[:, 1]

        real_real = self.real_conv(real)
        imag_imag = self.imag_conv(imag)
        real_part = real_real - imag_imag

        real_imag = self.real_conv(imag)
        imag_real = self.imag_conv(real)
        imag_part = real_imag + imag_real

        return torch.stack([real_part, imag_part], dim=1)  # [B, 2, H', W']


class ComplexConvTranspose2d(torch.nn.Module):
    '''
    Complex-valued 2D transposed convolutional (deconvolution) layer.

    Applies two separate real-valued transposed convolutions to the real and
    imaginary parts of a complex input, combining them to preserve complex structure.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the transposed convolution kernel.
    stride : int or tuple, optional
        Stride of the transposed convolution. Default is 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default is 0.
    output_padding : int or tuple, optional
        Additional size added to one side of the output shape. Default is 0.

    Returns
    -------
    torch.Tensor
        Complex output tensor of shape [B, 2, C_out, H_out, W_out].
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.real = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.imag = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, input):
        '''
        Apply complex-valued 2D transposed convolution (deconvolution).
    
        The real and imaginary parts of the input tensor are separately passed
        through real-valued transposed convolutions and then combined to preserve
        complex structure.
    
        Parameters
        ----------
        input : torch.Tensor
            Complex-valued input tensor of shape [B, 2, H, W],
            where channel 0 is real and channel 1 is imaginary.
    
        Returns
        -------
        torch.Tensor
            Complex-valued output tensor of shape [B, 2, C_out, H_out, W_out],
            where channel 0 is real and channel 1 is imaginary.
        '''
        real, imag = input[:, 0], input[:, 1]
        real_out = self.real(real) - self.imag(imag)
        imag_out = self.real(imag) + self.imag(real)
        return torch.stack([real_out, imag_out], dim=1)
    



def generate_sequence(net_setup, layer = 'Linear', layer_parameters = {},
                      activation_parameters = {}, sequence: torch.nn.Module = None) -> torch.nn.Module:
    '''
    Generate an sequence based on the information in the setup_dict.

    Parameters
    ----------
    setup : dict
        The setup of the model saved in the dict. Needs to contain all information about the model.
    sequence : torch.nn.Module, optional
        Add to defined model to the sequence. If not included, the function will generate a new empty sequence.

    Returns
    -------
    sequence : torch.nn.Module
        The new generated sequence.

    '''
    network_setup = net_setup.split('-')
    sequence = torch.nn.Sequential() if sequence == None else sequence
    layer = getattr(sys.modules[__name__], layer)
    activation = ModReLU
    
    for idx in range(len(network_setup) - 1):
        sequence.add_module(str(len(sequence))+'_'+'Calc', layer(int(network_setup[idx]), 
                            int(network_setup[idx+1]), **layer_parameters))
        sequence.add_module(str(len(sequence))+'_Act', activation(**{'num_channels': int(network_setup[idx+1])}))
    
    del sequence[-1]     
            
    return sequence


class ComplexAutoencoder(BaseNetwork):
    '''
    Create a predefined Autoencoder Neural Network.

    Parameters
    ----------
    setup : dict
        The predefined training scenario.
    preprocess : torch.nn.Module, optional
        DESCRIPTION. The default is None.
    compute : torch.nn.Module, optional
        DESCRIPTION. The default is None.
    postprocess : torch.nn.Module, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    def __init__(self, setup: dict, name: str = '', debug: bool= False,
                 preprocess: torch.nn.Module = None,
                 compute: torch.nn.Module = None, 
                 postprocess: torch.nn.Module = None):
        super().__init__(name, debug)
        
        
        decode_setup = setup.copy()
        decode_setup['net_setup'] = '-'.join(list(reversed(setup['net_setup'].split('-'))))
        decode_setup['layer'] = decode_setup['layer'].replace('Conv2d', 'ComplexConvTranspose2d')
        
        encode_setup = setup.copy()
        encode_setup['layer'] = encode_setup['layer'].replace('Conv2d', 'ComplexConv2d')
        preprocess = torch.nn.Sequential(OrderedDict(
            {'0_skip': NoChange()})) if preprocess == None else preprocess
        encode = generate_sequence(sequence=torch.nn.Sequential(), **encode_setup)
        compute = torch.nn.Sequential(OrderedDict({'0_skip': NoChange()})) if compute == None else compute
        decode = generate_sequence(sequence=torch.nn.Sequential(), **decode_setup)
        postprocess = torch.nn.Sequential(OrderedDict(
            {'1_skip': NoChange()})) if postprocess == None else postprocess
        
        self.network = torch.nn.Sequential()
        self.network.add_module('preprocess', preprocess)
        self.network.add_module('encoder', encode)
        self.network.add_module('compute', compute)
        self.network.add_module('decoder', decode)
        self.network.add_module('postprocess', postprocess)
        
        self.network.to(DEVICE)

    def forward(self, ins: torch.Tensor) -> tuple:
        '''
        Compute the output of the autoencoder with ins

        Parameters
        ----------
        ins : torch.Tensor
            The input of the neural network.

        Returns
        -------
        preprocess, representation, compute, reconstruction, postprocess : torch.Tensor
            The different steps of the autoencoder.

        '''
        preprocess = self.network[0](ins)
        representation = self.network[1](preprocess.unsqueeze(2))
        compute = self.network[2](representation)
        reconstruction = self.network[3](compute)
        postprocess = self.network[4](representation.flatten(start_dim = 1))
        return preprocess, representation, compute, reconstruction, postprocess

def phase_consistency_loss(output, target):
    '''
    Loss to enforce phase reconstruction consistency.

    Parameters
    ----------
    output, target : torch.tensor
        input and reconstruction of the complex autoencoder model.

    Returns
    -------
    torch.tensor
        The mse betweeb the reconstructed phase and the real phase.

    '''
    real1, imag1 = output[:, 0], output[:, 1]
    real2, imag2 = target[:, 0], target[:, 1]

    phase1 = torch.atan2(imag1, real1)
    phase2 = torch.atan2(imag2, real2)
    return torch.nn.functional.mse_loss(phase1, phase2)

# %% test

if __name__ == '__main__':
    
    config = {
                "net_setup": "1-16-32-64-64",
                "layer": "ComplexConv2d",
                "layer_parameters": {
                    "kernel_size": 5,
                    "stride": 2,
                    "padding": 5
                },
                
                "activation_parameters": {
                    "inplace": False
                },
            }
    
    model = ComplexAutoencoder(config)
    image = torch.randn(8, 300, 512).cuda() 
    x_complex = magnitude_to_complex(image)
    _, _, _, reconstruction, _ = model(x_complex)
    reconstruction = reconstruction.squeeze(2)
    inputs = x_complex[:, :, :reconstruction.shape[2], :reconstruction.shape[3]]
    loss = torch.nn.MSELoss()(reconstruction, inputs)
    phase_loss = phase_consistency_loss(reconstruction, inputs)