# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from functools import partial
from itertools import repeat
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

# from TransID --------------------------------------------------------------------------------------------------------
import torch.nn.functional as F

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, int_classes
else:
    import collections.abc as container_abcs

    int_classes = int


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=(224, 224), patch_size=16, stride_size=20, in_chans=3, embed_dim=768, stem_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size_tuple
        self.num_patches = num_patches
        self.stem_conv = stem_conv
        if self.stem_conv:
            hidden_dim = 64
            stem_stride = 2
            stride_size = patch_size = patch_size[0] // stem_stride
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_chans = hidden_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


# from TransID --------------------------------------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., mask_ratio=0.75, norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_transID_structure=False):
        super().__init__()

        # TODO: --------------------------------------------------------------------------
        self.use_transID_structure = use_transID_structure
        if self.use_transID_structure:
            assert isinstance(img_size, tuple), "使用可重叠patch需要tuple指定图片大小"
            self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size,
                                                  in_chans=in_chans,
                                                  embed_dim=embed_dim)
        else:
            # MAE encoder specifics
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.image_size = img_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # 添加可学习的噪声向量
        self.noise = None
        # 添加遮挡向量嵌入层
        self.mask_embed = nn.Linear(num_patches, self.embed_dim, bias=True)  # [L, 768]
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        if not self.use_transID_structure:
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                        int(self.patch_embed.num_patches ** .5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize mask_embed
        w1 = self.mask_embed.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def patchify_overlap_imgs(self, imgs):
        """
        将给定的张量按照指定的补丁尺寸和步长分割成补丁，并将每个补丁展平。

        参数:
        imgs -- 输入张量，形状为 [batch_size, channels, height, width]

        返回:
        patches -- 补丁张量，形状为 [batch_size, num_patches, patch_size**2 * channels]
        """
        B, C, H, W = imgs.shape
        PH, PW = self.patch_embed.patch_size
        SH, SW = self.patch_embed.stride_size

        # 计算沿高度和宽度方向的补丁数量
        num_patches_h = (H - PH) // SH + 1
        num_patches_w = (W - PW) // SW + 1
        num_patches = num_patches_h * num_patches_w

        # 初始化补丁张量
        patches = torch.zeros((B, num_patches, PH * PW * C), device=imgs.device)

        # 提取并展平补丁
        patch_idx = 0
        for i in range(0, H - PH + 1, SH):
            for j in range(0, W - PW + 1, SW):
                patch = imgs[:, :, i:i + PH, j:j + PW]
                patches[:, patch_idx] = patch.reshape(B, -1)  # 展平补丁
                patch_idx += 1

        return patches

    def unpatchify_overlap_imgs(self, patches):
        """
        将补丁张量重组为原始图像。

        参数:
        patches -- 补丁张量，形状为 [batch_size, num_patches, channels, patch_height, patch_width]
        img_size -- 原始图像的尺寸，形式为 (height, width)

        返回:
        imgs -- 重组后的图像张量，形状为 [batch_size, channels, height, width]
        """
        B, num_patches, C, PH, PW = patches.shape
        H, W = self.image_size
        SH, SW = self.patch_embed.stride_size

        # 初始化图像张量
        imgs = torch.zeros((B, C, H, W), device=patches.device)

        # 重组图像
        patch_idx = 0
        for i in range(0, H - PH + 1, SH):
            for j in range(0, W - PW + 1, SW):
                imgs[:, :, i:i + PH, j:j + PW] = patches[:, patch_idx]
                patch_idx += 1

        return imgs

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def learnable_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if self.noise is None:  # 如果noise是none,就开始初始化
            self.noise = nn.Parameter(torch.rand(N, L, device=x.device))  # [N,L]

        # sort noise for each sample
        ids_shuffle = torch.argsort(self.noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        _N, _L, _D = x_masked.shape
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, self.noise

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        # noise : [N,L]
        x, mask, ids_restore, noise = self.learnable_masking(x, mask_ratio)
        # 将mask embedding嵌入剩下的token中
        # TODO： [N,dim] 广播之后每一个保留的token都是带有相同的mask embed的痕迹,是否需要再增加一个前馈层将其投影到_L的长度呢？
        noise = self.mask_embed(noise)
        x = x + noise[:, None, :]  # [N, _L, dim] + [N, 1, dim]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.use_transID_structure:
            target = self.patchify_overlap_imgs(imgs)
        else:
            target = self.patchify(imgs)  # 补丁化但是不经过投影或者卷积层。

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cmae_vit_base_patch16_transreid(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=(256, 128), stride_size=16, use_transID_structure=True, patch_size=16, embed_dim=768, depth=12,
        num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
#
# size
# mismatch
# for patch_embed.proj.weight: copying
# a
# param
# with shape torch.Size([768, 64, 8, 8]) from checkpoint, the shape in current model is torch.Size([768, 3, 16, 16]).
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 3, 256, 128).to(device)
    # model = mae_vit_base_patch16(img_size=(256, 128), stride_size=16, use_transID_structure=True).to(device)
    model = cmae_vit_base_patch16_transreid().to(device)
    res = model(x,mask_ratio=0.45)
#     #
# import torch
# import torch.nn as nn
# from timm.models.vision_transformer import PatchEmbed, Block
#
# random_tensor = torch.rand(2, 3, 224, 224)
# patch_num = 196
# mask_ratio = 0.75
# # 计算向量中1的数量
# num_ones = int(patch_num * mask_ratio)
#
# # 创建噪声向量，
# noise = nn.Parameter(torch.normal(N,L))
# embed_dim = 768
# decoder_dim = 512
#
# mask_emb = nn.Linear(patch_num, embed_dim, bias=True)
# mask_pred = nn.Linear(embed_dim, patch_num, bias=True)
# embeded_mask = mask_emb(mask_vec) # [768]
# # embeded_mask.shape
# # type(embeded_mask)
# patch_embed = PatchEmbed(224, 16, 3, embed_dim)
# x = patch_embed(random_tensor)
# # add mask emb
# embeded_mask = embeded_mask.unsqueeze(0).unsqueeze(0)
# x = x + embeded_mask
# # 经过encoder-decoder编码以后，和decoder_pred一起通过mask_pred进行重建。
# def conditional_masking(x, mask_ratio, mask_vector=None):
#     N, L, D = x.shape  # batch, length, dim
#     len_keep = int(L * (1 - mask_ratio))
#     if not mask_vector:
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove 生成用来将序列从小到大排序所需要的索引
#     else:
#
#     # sort noise for each sample
#
#     ids_restore = torch.argsort(ids_shuffle, dim=1) # 逆索引
#
#     ids_keep = ids_shuffle[:, :len_keep]
#     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([N, L], device=x.device)
#     mask[:, :len_keep] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)
#
#     return x_masked, mask, ids_restore
#
#
