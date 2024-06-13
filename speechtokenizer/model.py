# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .quantization  import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np
import logging
import GPUtil
from speechtokenizer.discriminator.discriminator import MultiPeriodDiscriminator
from speechtokenizer.discriminator.discriminator import MultiScaleDiscriminator
from speechtokenizer.discriminator.discriminator import MultiScaleSTFTDiscriminator
logger = logging.getLogger("model.py")

class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        '''
        
        Parameters
        ----------
        config : json
            Model Config.

        '''
        super().__init__()
        self.encoder = SEANetEncoder(n_filters=config['model_params']['n_filters'], 
                                     dimension=config['model_params']['dimension'], 
                                     ratios=config['model_params']['strides'],
                                     lstm=config['model_params']['lstm_layers'],
                                     bidirectional=config['model_params']['bidirectional'],
                                     dilation_base=config['model_params']['dilation_base'],
                                     residual_kernel_size=config['model_params']['residual_kernel_size'],
                                     n_residual_layers=config['model_params']['n_residual_layers'],
                                     activation=config['model_params']['activation'])
        self.sample_rate = config['model_params']['sample_rate']
        self.n_q = config['model_params']['n_q']
        self.downsample_rate = np.prod(config['model_params']['strides'])
        if config['model_params']['dimension'] != config['model_params']['semantic_dimension']:
            self.transform = nn.Linear(config['model_params']['dimension'], config['model_params']['semantic_dimension'])
        else:
            self.transform = nn.Identity()
        self.quantizer = ResidualVectorQuantizer(dimension=config['model_params']['dimension'], n_q=config['model_params']['n_q'], bins=config['model_params']['codebook_size'])
        self.decoder = SEANetDecoder(n_filters=config['model_params']['n_filters'], 
                                     dimension=config['model_params']['dimension'], 
                                     ratios=config['model_params']['strides'],
                                     lstm=config['model_params']['lstm_layers'],
                                     bidirectional=False,
                                     dilation_base=config['model_params']['dilation_base'],
                                     residual_kernel_size=config['model_params']['residual_kernel_size'],
                                     n_residual_layers=config['model_params']['n_residual_layers'],
                                     activation=config['model_params']['activation'])
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mstftd = MultiScaleSTFTDiscriminator(32)
        
    @classmethod
    def load_from_checkpoint(cls, 
                             config_path: str, 
                             ckpt_path: str):
        '''

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        '''
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(params)
        return model
    
    def print_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"GPU ID {gpu.id}: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB ({gpu.memoryUtil * 100}%)")

    def forward(self, 
                x: torch.tensor, 
                n_q: int=None, 
                layers: list=[0]):
        '''
        
        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        '''
        n_q = n_q if n_q else self.n_q

        e = self.encoder(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, n_q=n_q, layers=layers)
        feature = rearrange(quantized_list[0], 'b d t -> b t d')
        feature = self.transform(feature)    
        o = self.decoder(quantized)
        
        # MPD
        x_df_r, x_df_g, fmap_f_r, fmap_f_g  = self.mpd(x, o.detach())

        # MSD
        x_ds_r, x_ds_g, fmap_s_r, fmap_s_g = self.msd(x, o.detach())

        #MSTFT
        x_stft_r, fmap_stftd_r  = self.mstftd(x)
        x_stft_gen, fmap_stftd_g = self.mstftd(o.detach())

        fmap_discriminator = fmap_f_g, fmap_f_r, fmap_s_g, fmap_s_r, fmap_stftd_g, fmap_stftd_r
        x_discriminator = x_df_r,x_df_g,x_ds_r,x_ds_g,x_stft_r,x_stft_gen

        return o, commit_loss, feature,x_discriminator, fmap_discriminator
    
    def forward_feature(self, 
                        x: torch.tensor, 
                        layers: list=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is all layers.

        Returns
        -------
        quantized_list : list[torch.tensor]
            Quantized of required layers.

        '''
        e = self.encoder(x)
        layers = layers if layers else list(range(self.n_q))
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)
        return quantized_list
    
    def encode(self, 
               x: torch.tensor, 
               n_q: int=None, 
               st: int=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        '''
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)
        return codes
    
    def decode(self, 
               codes: torch.tensor, 
               st: int=0):
        '''

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        '''
        quantized = self.quantizer.decode(codes, st=st)
        o = self.decoder(quantized)
        return o