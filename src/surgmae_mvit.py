import torch
import numpy as np
import pytorch_lightning as pl
from einops import repeat,rearrange
from torch import nn
from torch.utils.data import DataLoader
from utils import create_multiscale_vision_transformers,create_conv_patch_embed,PatchEmbed,get_sinusoid_encoding_table, create_multiscale_vision_transformers_no_patch_position
from pytorchvideo.layers import SpatioTemporalClsPositionalEncoding

#from pytorchvideo.models.vision_transformers import (
#    create_multiscale_vision_transformers,create_conv_patch_embed
#)

from temporal_hog import temporal_hog,retrieve_spatiotemporal_geysers
from mask_generator import RepresentativeMaskGenerator,TubeMaskingGenerator,CubeMaskGenerator
from videoswintransformer import SwinTransformer3D
from misc.base_params import parse_arguments
from data_utils.protobuf_dataset import process_data_directory_surgery


class SurgMAE_MViT(nn.Module):
    def __init__(self,n_class=7,img_size=224,num_frames=128,feature_dim=108,in_channels=3,patch_embed_dim=96,conv_patch_embed_kernel=(3,7,7),conv_patch_embed_stride=(2,4,4),conv_patch_embed_padding=(1,3,3),embed_dim_mul=[[1,2.0],[3,2.0],[14,2.0]],atten_head_mul=[[1,2.0],[3,2.0],[14,2.0]],pool_q_stride_size=[[1,1,2,2],[3,1,2,2],[14,1,2,2]],pool_kv_stride_adaptive=[1,8,8],pool_kvq_kernel=[3,3,3],head=None,pretrain=False,learnable_posEmbed=False,**kwargs):
        super().__init__()
        self.img_size = img_size
        self.temporal_size = num_frames
        self.stride = conv_patch_embed_stride
        self.patch_embed_dim = patch_embed_dim
        self.downsample_rate = 2**len(pool_q_stride_size)

        # Get Patches
        self.patch_embed = create_conv_patch_embed(
            in_channels=in_channels,
            out_channels=patch_embed_dim,
            conv_kernel_size=conv_patch_embed_kernel,
            conv_stride=conv_patch_embed_stride,
            conv_padding=conv_patch_embed_padding,
            conv=nn.Conv3d
        )
        input_dims = [self.temporal_size,self.img_size,self.img_size]
        input_stride = conv_patch_embed_stride
        patch_embed_shape = [input_dims[i]//input_stride[i] for i in range(len(input_dims))]
        self.cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
            embed_dim=patch_embed_dim,
            patch_embed_shape=patch_embed_shape,
            sep_pos_embed=True,
            has_cls=True
        )

        # MViT Base
        self.mvit = create_multiscale_vision_transformers(
            spatial_size=img_size,
            temporal_size=num_frames//8,
            embed_dim_mul=embed_dim_mul,
            atten_head_mul=atten_head_mul,
            pool_q_stride_size=pool_q_stride_size,
            pool_kv_stride_adaptive=pool_kv_stride_adaptive,
            pool_kvq_kernel=pool_kvq_kernel,
            head=head
        )

        # Conversion to HOG feature dimensions
        in_features = self.mvit.norm_embed.normalized_shape[0]
        out_features = feature_dim
        self.to_latent = nn.Identity()
        self.decoder_pred = nn.Linear(in_features//self.downsample_rate,out_features,bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1,1,patch_embed_dim))

        # Initialize weights
        w = self.patch_embed.patch_model.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0],-1]))
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias,0)
        nn.init.trunc_normal_(self.mask_token,std=0.02)
            
        # Pretrain MViT Base weights
        if not pretrain:
            self.apply(self.__init__weights)
        else:
            # Load checkpoint
            ckpt_path = '/home/paulpak/.cache/torch/hub/checkpoints/MVIT_B_16x4.pyth'
            ckpt = torch.load(ckpt_path)['model_state']
            model_state_dict = self.mvit.state_dict()

            # Add in hidden keys missing from checkpoint
            missing_keys = set(model_state_dict)-set(ckpt)
            for key in missing_keys:
                split = key.split('.')
                original_key = split[0]+'.'+split[1]+'.'+split[2]+'.'+split[4]+'_'+split[3].split('_')[3]+'.'+split[5]
                ckpt[key] = ckpt[original_key]
            msg = self.mvit.load_state_dict(ckpt,strict=False)
            print(msg)

    def __init__weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_context_frames(self,x,num_context=16):
        # Extract HOG and Representative Frames
        hog_features = list()
        indices = list()
        for index,imgs in enumerate(x):
            _,_,hog_features_ = temporal_hog(imgs,temporal_scale=True)
            indices_avg = retrieve_spatiotemporal_geysers(hog_features_,num_representatives=num_context,iterative=False)
            hog_features.append(hog_features_.unsqueeze(0))
            indices.append(indices_avg.unsqueeze(0))
        hog_features = torch.cat(hog_features,dim=0)
        indices = torch.cat(indices,dim=0)
        return hog_features,indices

    def forward_features(self,x,mask,indices):
        # Patch embed entire input sequence
        x = self.patch_embed(x.transpose(1,2))

        # Extract context cubes
        '''
        context_cubes = rearrange(x,'b (t d) c -> b t (d) c',t=self.temporal_size)
        context_x = torch.cat(([torch.cat(([context_cubes[j,i,:].unsqueeze(0) for i in indices[j]]),dim=0).unsqueeze(0) for j  in range(len(indices))]),dim=0)
        context_x = context_x.flatten(1,2)
        print('Context x',context_x.shape)
        '''
        B,L,C = x.shape
        mask_token = self.mask_token.expand(B,L,-1)
        mask = mask.unsqueeze(-1)
        x = x*(1-mask)+mask_token*(mask)
        print('x',x.shape)

        # Get visible tokens from mask
        '''
        X = list()
        for i in range(x.shape[1]//16):
            x_ = self.patch_embed(x[:,(16*i):(16*(i+1)),:,:,:].transpose(1,2))
            print('cube ',i, ' ', x_.shape)
            X.append(x_.unsqueeze(0))
        X = torch.cat(X,dim=0)
        print(X.shape)
        '''
        '''
        if mask is not None:
            # Create Masked Tokens
            mask_token = self.mask_token.expand(B,L,-1)
            dense_mask = repeat(
                mask,
                'b t h w -> b t (h dh) (w dw)',
                dh=int(self.downsample_rate/2),
                dw=int(self.downsample_rate/2)
            )
            w = dense_mask.flatten(1).unsqueeze(-1)
            print('mask',w.shape,mask_token.shape)
            # x = x*(1-w) + mask_token*w
        '''

        # Feed forward transformer backbone (In: Patch Embeddings)
        # x = self.mvit(x.float())
        x = self.mvit(x)
        return x

    def forward_loss(self,x, target_x, mask=None):
        # Compute loss on mask regions
        loss = (x-target_x)**2
        loss = loss.mean(dim=-1)
        # loss = (loss*mask).sum()/(mask.sum()+1e-5)
        return loss

    def forward(self,x,target,indices,mask=None):
        context_x = torch.cat(([torch.cat(([x[j,i,:,:,:].unsqueeze(0) for i in indices[j]]),dim=0).unsqueeze(0) for j  in range(len(indices))]),dim=0)
        print(context_x.shape)
        x = self.forward_features(context_x,mask,indices)

        # x = self.decoder_pred(self.to_latent(x))

        # Remove ClS Token
        x = x[:, 1:, :]

        x = rearrange(
            x,
            'b (t0 h0 w0) (c dt dh dw) -> b (t0 dt) (h0 dh) (w0 dw) c',
            dt=self.stride[0],
            dh=self.stride[1]//2,
            dw=self.stride[2]//2,
            h0=self.img_size//(self.stride[1]//2*self.downsample_rate)//2,
            w0=self.img_size//(self.stride[2]//2*self.downsample_rate)//2
        )
        x = self.decoder_pred(self.to_latent(x))

        # Reshape to the original x
        '''
        x = rearrange(
            x,
            "b (t h w) (dt dc)->b (t dt) h w dc",
            dt=self.stride[0],
            t=self.num_frames // 8 // self.stride[0],
            h=self.img_size // (self.stride[1] * self.downsample_rate//2),
            w=self.img_size // (self.stride[2] * self.downsample_rate//2),
        )
        '''
        print('x/target',x.shape,target.shape)
        loss = self.forward_loss(x,target)

        '''
        if mask is not None:
            x = self.decoder_pred(self.to_latent(x))
            x = x[:,1:,:]
            print('Without cls token',x.shape)
            # Reshape to the original x
            x = rearrange(
                x,
                "b (t h w) (dt dc)->b (t dt) h w dc",
                dt=self.stride[0],
                t=self.num_frames // self.stride[0],
                h=self.img_size // (self.stride[1] * self.downsample_rate),
                w=self.img_size // (self.stride[2] * self.downsample_rate),
            )
            print('reshaped',x.shape)
            loss = forward_loss(x,target,mask)
        else:
            x = self.class_head(self.to_latent(x))
            # TODO: Loss
        return x, loss
        '''
        return x,loss
