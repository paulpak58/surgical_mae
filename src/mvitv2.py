import os
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from fvcore.common.config import CfgNode

import utils
from utils import round_width,calc_mvit_feature_geometry,PatchEmbed,get_3d_sincos_pos_embed
from attention import MultiScaleBlock

################################################################
# MViTv2 implemented from
# https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/video_model_builder.py
################################################################
class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.
    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = utils.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        #### PYTHON 3.7 ####
        # num_patches = math.prod(self.patch_dims)
        from functools import reduce
        import operator
        num_patches = reduce(operator.mul,self.patch_dims,1)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim
            )

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if cfg.MVIT.DIM_MUL_IN_ATT:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=cfg.MVIT.SEPARATE_QKV,
                )

                if cfg.MODEL.ACT_CHECKPOINT:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = utils.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            '''
            self.head = utils.TransformerBasicHead(
                2 * embed_dim
                if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
                else embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
            '''
            self.class_head = utils.TransformerBasicHead(
                2 * embed_dim
                if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
                else embed_dim,
                num_classes=7,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # self.head.projection.weight.data.mul_(head_init_scale)
        # self.head.projection.bias.data.mul_(head_init_scale)
        self.class_head.projection.weight.data.mul_(head_init_scale)
        self.class_head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False, scalars=None):
        x = x[0]
        x, bcthw = self.patch_embed(x)

        # Extract out non-adjacent frames if scalars is not None
        if scalars is not None:
            from einops import rearrange
            x = rearrange(x,'b (t h w) c -> b c t h w',b=x.shape[0],t=bcthw[-3],w=bcthw[-2],c=x.shape[-1])
            x = x[:,:,scalars.long(),:,:]
            bcthw = x.shape
            x = rearrange(x,'b c t h w -> b (t h w) c')
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape


        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw, scalars)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.cls_embed_on:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                # x = self.head([x], bboxes)
                x = self.class_head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.cls_embed_on:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.cls_embed_on:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                # x = self.head(x)
                x = self.class_head(x)

        return x

def build_cfg_mvitv2(size='small'):
    cfg = CfgNode()
    cfg.DATA = CfgNode()
    cfg.MVIT = CfgNode()
    cfg.MODEL = CfgNode()
    cfg.DETECTION = CfgNode()
    cfg.MVIT.REV = CfgNode()
    cfg.CONTRASTIVE = CfgNode()
    
    # DEFAULT MViT
    cfg.MVIT.MODE = "conv"
    cfg.MVIT.POOL_FIRST = False
    cfg.MVIT.PATCH_2D = False
    cfg.MVIT.REV.ENABLE = False
    cfg.MVIT.PATCH_KERNEL = (3,7,7)
    cfg.MVIT.PATCH_STRIDE = (2,4,4)
    cfg.MVIT.PATCH_PADDING = (1,3,3)
    cfg.MVIT.MLP_RATIO = 4.0
    cfg.MVIT.QKV_BIAS = True
    cfg.MVIT.DROPOUT_RATE = 0.0
    cfg.MVIT.LAYER_SCALE_INIT_VALUE = 0.0
    cfg.MVIT.HEAD_INIT_SCALE = 1.0
    cfg.MVIT.CLS_EMBED_ON = True
    cfg.MVIT.USE_MEAN_POOLING = False
    cfg.MVIT.POOL_KVQ_KERNEL = (3,3,3)
    cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1,8,8]
    cfg.MVIT.NORM_STEM = False
    cfg.MVIT.DIM_MUL_IN_ATT = True
    cfg.MVIT.RESIDUAL_POOLING = True
    cfg.MVIT.SEPARATE_QKV = False
    cfg.MVIT.REL_POS_ZERO_INIT = False
    cfg.MVIT.ZERO_DECAY_POS_CLS = False
    cfg.MVIT.REV.ENABLE = False
    cfg.MVIT.REV.RESPATH_FUSE = "concat"

    # DEFAULT Model
    cfg.MODEL.HEAD_ACT = "softmax"
    cfg.MODEL.NUM_CLASSES = 400     # Kinetics
    cfg.MODEL.DROPOUT_RATE = 0.5
    cfg.MODEL.DETACH_FINAL_FC = False
    cfg.DETECTION.ENABLE = False
    cfg.CONTRASTIVE.NUM_MLP_LAYERS = 1
    
    # DEFAULT Data
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3]

    # Positional Embeddings
    cfg.MVIT.USE_ABS_POS = False
    cfg.MVIT.USE_FIXED_SINCOS_POS = False
    cfg.MVIT.SEP_POS_EMBED = False
    cfg.MVIT.REL_POS_SPATIAL = True
    cfg.MVIT.REL_POS_TEMPORAL = True
    cfg.MVIT.NORM = "layernorm"

    if size=='small':
        cfg.DATA.NUM_FRAMES = 16
        cfg.MVIT.EMBED_DIM = 96
        cfg.MVIT.NUM_HEADS = 1
        cfg.MVIT.DEPTH = 16
        cfg.MVIT.DROPPATH_RATE = 0.2
        cfg.MODEL.ACT_CHECKPOINT = False
        cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
        cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
        cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]]
    elif size=='base':
        cfg.DATA.NUM_FRAMES = 32
        cfg.MVIT.EMBED_DIM = 96
        cfg.MVIT.NUM_HEADS = 1
        cfg.MVIT.DEPTH = 24
        cfg.MVIT.DROPPATH_RATE = 0.3
        cfg.MODEL.ACT_CHECKPOINT = False
        cfg.MVIT.DIM_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
        cfg.MVIT.HEAD_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
        cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1], [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1], [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]]
    else:
        cfg.DATA.NUM_FRAMES = 40
        cfg.MVIT.EMBED_DIM = 144
        cfg.MVIT.NUM_HEADS = 2
        cfg.MVIT.DEPTH = 48
        cfg.MVIT.DROPPATH_RATE = 0.75
        cfg.MODEL.ACT_CHECKPOINT = True
        cfg.MVIT.DIM_MUL = [[2, 2.0], [8, 2.0], [44, 2.0]]
        cfg.MVIT.HEAD_MUL = [[2, 2.0], [8, 2.0], [44, 2.0]]
        cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 2, 2], [9, 1, 1, 1], [10, 1, 1, 1],
                  [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1], [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1],
                    [21, 1, 1, 1], [22, 1, 1, 1], [23, 1, 1, 1], [24, 1, 1, 1], [25, 1, 1, 1], [26, 1, 1, 1], [27, 1, 1, 1], [28, 1, 1, 1], [29, 1, 1, 1], [30, 1, 1, 1],
                      [31, 1, 1, 1], [32, 1, 1, 1], [33, 1, 1, 1], [34, 1, 1, 1], [35, 1, 1, 1], [36, 1, 1, 1], [37, 1, 1, 1], [38, 1, 1, 1], [39, 1, 1, 1], [40, 1, 1, 1],
                        [41, 1, 1, 1], [42, 1, 1, 1], [43, 1, 1, 1], [44, 1, 2, 2], [45, 1, 1, 1], [46, 1, 1, 1], [47, 1, 1, 1] ]
    return cfg

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__=='__main__':
    # Small MViTv2 16x4
    cfg = build_cfg_mvitv2(size='small')
    model = MViT(cfg)

    ckpt_path = '/home/paulpak/.cache/torch/hub/checkpoints/MVIT_v2_S_16x4'
    if not os.path.exists(ckpt_path):
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth')
    msg = model.load_state_dict(ckpt['model_state'],strict=False)
    # print(msg)

    x = torch.randn(1,8,3,16,224,224)
    # embed_dim = list(model.children())[-1].normalized_shape[0]
    x = model(x)
    print(x.shape)

    '''
    # Base MViTv2 32x3
    cfg = build_cfg_mvitv2(size='base')
    model = MViT(cfg)
    ckpt_path = '/home/paulpak/.cache/torch/hub/checkpoints/MVIT_v2_B_32x3'
    if not os.path.exists(ckpt_path):
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth')
    msg = model.load_state_dict(ckpt['model_state'])
    print(msg)

    # Base MViTv1 16x4
    ckpt_path = '/home/paulpak/.cache/torch/hub/checkpoints/MVIT_B_16x4'
    if not os.path.exists(ckpt_path):
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth')
    txt = open("cfg.yaml","w")
    txt.write(ckpt['cfg'])
    txt.close()
    Cfg = CfgNode()
    cfg = AttributeDict(dict(Cfg.load_yaml_with_base("cfg.yaml")))
    '''
