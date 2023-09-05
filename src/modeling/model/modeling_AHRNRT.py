
from __future__ import absolute_import, division, print_function

import os


import cv2
from torch.nn import functional as F

import torch
import numpy as np
from torch import nn
from .transformer1 import build_transformer_4de
from .position_encoding import build_position_encoding
from src.modeling.model.poolTransformer import poolattnformer_s12_xin
from src.modeling.model.poolTHR import PoolAttnFormer_hr, BasicBlock, net_2d
from src.modeling.model.poolTHR import HR_stream, GroupNorm


class Merge1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(196, 49, 1)
        self.conv_2 = nn.Conv1d(512, 1024, 1)

    def forward(self, x):
        x0 = x[0]  # (1,196,512)
        x1 = x[1]  # (1,49,1024)
        x_new = self.conv_1(x0).permute(0, 2, 1)
        x_new = self.conv_2(x_new).permute(0, 2, 1)
        x_new1 = x_new + x1

        return x_new1


class PatchMerge(nn.Module):

    def __init__(self, stride=2,
                 in_chans=64, embed_dim=128, norm_layer=None):
        super().__init__()
        self.stride = stride
        # self.upsample = nn.subsample(scale_factor=stride, mode='nearest')
        self.poolm = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3,
                               stride=1, padding=1)
        self.proj2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3,
                               stride=1, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj1(x)  # 1,128,56,56
        x = self.proj2(x)  # 1,512,56,56
        x = self.poolm(x)  # 1,512,28,28
        x = self.poolm(x)  # 1,512,14,14
        x = self.poolm(x)

        x = self.norm(x)

        return x


class FastMETRO_Hand_Network(nn.Module):

    def __init__(self, args, mesh_sampler, num_joints=21, num_vertices=195,
                 layers=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[4, 4, 4, 4], num_classes=1000,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 drop_rate=0.1, drop_path_rate=0.1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        # the number of transformer layers
        if 'FastMETRO-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'FastMETRO-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'FastMETRO-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"

        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout,
                                     "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers,
                                     "num_dec_layers": num_dec_layers,
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout,
                                     "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers,
                                     "num_dec_layers": num_dec_layers,
                                     "pos_type": args.pos_type}

        self.transformer_1 = build_transformer_4de(self.transformer_config_1)
        self.transformer_2 = build_transformer_4de(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(self.transformer_config_1["model_dim"],
                                            self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"],
                                            self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'],
                                                           hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'],
                                                           hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)

        # heatmap and feature
        self.inplanes = 64
        self.pooltrans = poolattnformer_s12_xin(pretrained=True)
        self.stage1 = HR_stream(embed_dims, layers, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value)
        self.norm0 = norm_layer(embed_dims[0])

        self.heatmap_layer = self._make_resnet_layer(BasicBlock, self.num_vertices, 2)

        self.heatmap_layer1 = self._make_resnet_layer(BasicBlock, self.num_joints, 2)

        self.feature_extract_layer = self._make_resnet_layer(BasicBlock, 512, 2)
        self.cam_layer1 = nn.Linear(49, 1)

        #
        zeros_1 = torch.tensor(np.zeros((self.num_vertices, self.num_joints + 1)).astype(bool))
        zeros_2 = torch.tensor(
            np.zeros((self.num_joints + 1, (1 + self.num_joints + self.num_vertices))).astype(bool))
        adjacency_indices = torch.load(
            './src/modeling/data/mano_195_adjmat_indices.pt')
        adjacency_matrix_value = torch.load(
            './src/modeling/data/mano_195_adjmat_values.pt')
        adjacency_matrix_size = torch.load(
            './src/modeling/data/mano_195_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value,
                                                   size=adjacency_matrix_size).to_dense()

        self.attention_masks = []
        # attention mask
        update_matrix = (adjacency_matrix.clone() > 0)
        ref_matrix = (adjacency_matrix.clone() > 0)

        temp_mask_1 = (update_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1.clone()], dim=1)
        self.attention_masks.append(torch.cat([zeros_2, temp_mask_2], dim=0))

        for n in range(6):
            arr = torch.arange(adjacency_matrix_size[0])
            for i in range(adjacency_matrix_size[0]):
                idx = arr[update_matrix[i] > 0]
                for j in idx:
                    update_matrix[i] += ref_matrix[j]
            if n in [1, 3, 5]:
                temp_mask_1 = (update_matrix == 0)
                temp_mask_2 = torch.cat([zeros_1, temp_mask_1.clone()], dim=1)
                self.attention_masks.append(torch.cat([zeros_2, temp_mask_2], dim=0))

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        BN_MOMENTUM = 0.1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )  # ,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        layers.append(nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, images):
        device = images.device

        batch, c, h, w = images.shape

        #pooltransformer
        x4, out = self.pooltrans(images)  # x= 1,512,7,7
        img_features = x4.flatten(2).permute(2, 0, 1)   #49,1,512

        # High resolution module
        x_patch, x3 = self.stage1(out)
        x_patch = self.norm0(x_patch)  # x0(1,64,56,56)

        heatmap_ = self.heatmap_layer(x_patch)
        heatmap = heatmap_.clone()
        heatmap = F.softmax(heatmap.flatten(2), dim=-1)  # (1,431,3136)
        #
        heatmap_ = F.interpolate(heatmap_, scale_factor=2)  # 1,195,112,112
        # img_feat =  heatmap_.cpu()
        # gray_x = img_feat.squeeze(0)  # 195,56,56
        # pict = color_heat(gray_x)

        heatmap_ = F.softmax(heatmap_.flatten(2), dim=-1)  # (1,431,12544)

        ###jointheatmap
        heatmap1_ = self.heatmap_layer1(x_patch)
        heatmap1 = heatmap1_.clone()
        heatmap1 = F.softmax(heatmap1.flatten(2), dim=-1)  # (1,21,3136)
        heatmap1_ = F.interpolate(heatmap1_, scale_factor=2)  # 1,21,112,112
        # img_feat =  heatmap1_.cpu()
        # gray_x = img_feat.squeeze(0)  # 21,112,112
        # pict = color_heat(gray_x)

        heatmap1_ = F.softmax(heatmap1_.flatten(2), dim=-1)  # (1,21,12544)


        feature_map = self.feature_extract_layer(x_patch).flatten(2)  # BXCXHW


        cam_p = img_features.permute(1, 0, 2)
        cam_feature = self.cam_layer1(cam_p.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        feature_cat = torch.cat((heatmap, heatmap1), dim=1)  #1,216,3136
        sampled_cat = feature_cat @ feature_map.transpose(1, 2).contiguous()  # BX431XC 1,216,512


        cam_with_jv = torch.cat(
            [cam_feature, sampled_cat], 1)  # 1,217,512
        cam_with_jv = cam_with_jv.permute(1, 0, 2)  # 1,217,512



        _ = 512
        h1 = 7
        w1 = 7

        #
        pos_enc_1 = self.position_encoding_1(batch, h1, w1, device).flatten(2).permute(2, 0, 1)  # 49 X batch_size X 512
        pos_enc_2 = self.position_encoding_2(batch, h1, w1, device).flatten(2).permute(2, 0, 1)
        #

        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(img_features, cam_with_jv,
                                                                               pos_enc_1,
                                                                               attention_mask1=self.attention_masks[
                                                                                   -1].to(device),
                                                                               attention_mask2=self.attention_masks[
                                                                                   -2].to(device))

        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1)  # 1 X batch_size X 128
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1)  # 49 X batch_size X 128
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1)  # (num_joints + num_vertices) X batch_size X 128

        reduced_cam_with_jv = torch.cat([reduced_cam_features_1, reduced_jv_features_1], dim=0)

        cam_features_2, _, jv_features_2 = self.transformer_2(reduced_enc_img_features_1, reduced_cam_with_jv,
                                                              pos_enc_2,
                                                              attention_mask1=self.attention_masks[-3].to(device),
                                                              attention_mask2=self.attention_masks[-4].to(device))

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch, 3)  # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(
            jv_features_2.transpose(0, 1))  # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, :self.num_joints, :]  # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.num_joints:, :]  # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_fine = self.mesh_sampler.upsample(
            pred_3d_vertices_coarse)  # batch_size X num_vertices(fine) X 3

        out = {}
        out['pred_cam'] = pred_cam
        out['pred_3d_joints'] = pred_3d_joints
        out['pred_3d_vertices_coarse'] = pred_3d_vertices_coarse
        out['pred_3d_vertices_fine'] = pred_3d_vertices_fine
        out['heatmap_vert'] = heatmap_
        out['heatmap_joint'] = heatmap1_

        return out
