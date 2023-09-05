
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
from src.modeling.model.poolTransformer import poolattnformer_s12

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, dim=256, norm_layer=GroupNorm):
        super().__init__()
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None))

        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))

        # self.act = act_layer()
        self.norm = norm_layer(dim)
        # self.proj = nn.Conv2d(dim,dim,1)
        self.proj0 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_patch_attn1 = self.patch_pool1(x)
        x_patch_attn2 = self.patch_pool2(x)
        x_patch_attn = x_patch_attn1 @ x_patch_attn2
        x_patch_attn = self.proj0(x_patch_attn)

        x1 = x.contiguous().view(B, C, H * W).transpose(1, 2).contiguous().view(B, H * W, 32, -1)
        x_embdim_attn1 = self.embdim_pool1(x1)
        x_embdim_attn2 = self.embdim_pool2(x1)
        x_embdim_attn = x_embdim_attn1 @ x_embdim_attn2

        x_embdim_attn = x_embdim_attn.contiguous().view(B, H * W, C).transpose(1, 2).contiguous().view(B, C, H, W)
        x_embdim_attn = self.proj1(x_embdim_attn)

        x_out = self.norm(x_patch_attn + x_embdim_attn)
        x_out = self.proj2(x_out)
        return x_out

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class PatchSplit(nn.Module):

    def __init__(self, stride=2,
                 in_chans=256, embed_dim=128, norm_layer=None):
        super().__init__()
        self.stride = stride
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3,
                              stride=1, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        x = self.proj(x)
        x = self.norm(x)

        return x

class HR_stream(nn.Module):

    def __init__(self, dim, layers, mlp_ratio,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        layers = [2, 2, 2, 4]
        mlp_ratio = [2, 2, 2, 4]

        depth1 = layers[1]
        mlp_ratio1 = mlp_ratio[1]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, depth1)]
        self.patch_emb1 = PatchSplit(stride=2,
                                     in_chans=dim[1], embed_dim=dim[0])
        self.Block1 = nn.Sequential(*[PoolFormerBlock(dim[0], mlp_ratio=mlp_ratio1,
                                                      act_layer=act_layer, norm_layer=norm_layer,
                                                      drop=drop_rate, drop_path=dpr1[i],
                                                      use_layer_scale=use_layer_scale,
                                                      layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth1)])

        depth2 = layers[2]
        mlp_ratio2 = mlp_ratio[2]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, depth2)]
        self.patch_emb2 = nn.Sequential(
            PatchSplit(stride=2,
                       in_chans=dim[2], embed_dim=dim[1], norm_layer=norm_layer, ),
            PatchSplit(stride=2,
                       in_chans=dim[1], embed_dim=dim[0]),
        )
        self.Block2 = nn.Sequential(*[PoolFormerBlock(dim[0], mlp_ratio=mlp_ratio2,
                                                      act_layer=act_layer, norm_layer=norm_layer,
                                                      drop=drop_rate, drop_path=dpr2[i],
                                                      use_layer_scale=use_layer_scale,
                                                      layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth2)])

        depth3 = layers[3]
        mlp_ratio3 = mlp_ratio[3]
        dpr3 = [x.item() for x in torch.linspace(0, drop_path_rate, depth3)]
        self.patch_emb3 = nn.Sequential(
            PatchSplit(stride=2,
                       in_chans=dim[3], embed_dim=dim[2], norm_layer=norm_layer, ),
            PatchSplit(stride=2,
                       in_chans=dim[2], embed_dim=dim[1], norm_layer=norm_layer, ),
            PatchSplit(stride=2,
                       in_chans=dim[1], embed_dim=dim[0]),
        )
        self.Block3 = nn.Sequential(*[PoolFormerBlock(dim[0], mlp_ratio=mlp_ratio3,
                                                      act_layer=act_layer, norm_layer=norm_layer,
                                                      drop=drop_rate, drop_path=dpr3[i],
                                                      use_layer_scale=use_layer_scale,
                                                      layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth3)])

    def forward(self, x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]

        x1_new = self.patch_emb1(x1) + x0
        x1_new = self.Block1(x1_new)

        x2_new = self.patch_emb2(x2) + x1_new
        x2_new = self.Block2(x2_new)

        x3_new = self.patch_emb3(x3) + x2_new
        x3_new = self.Block3(x3_new)

        return x3_new, x3


class PoolAttnFormer_hr(nn.Module):
    """
    """

    def __init__(self, img_size=224, layers = [2, 2, 6, 2], embed_dims = [64, 128, 320, 512],
                 mlp_ratios = [4, 4, 4, 4], num_classes=1000,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 drop_rate=0.1, drop_path_rate=0.1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        self.num_classes = num_classes

        self.poolattn_cls = poolattnformer_s12(pretrained=True)

        self.stage1 = HR_stream(embed_dims, layers, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value)

        self.norm0 = norm_layer(embed_dims[0])
        self.norm3 = norm_layer(embed_dims[3])

        #img_size = [img_size[1], img_size[0]]

        self.apply(self.init_weights)

        if pretrained is not None:
            pt_checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.poolattn_cls = load_pretrained_weights(self.poolattn_cls, pt_checkpoint)
            # self.poolattn_cls.load_state_dict(pt_checkpoint, False)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=.02)


    def forward(self, x):
        # through backbone
        x = self.poolattn_cls(x)

        x0, x3 = self.stage1(x)
        x0 = self.norm0(x0)
        x3 = self.norm3(x3)

        return x0, x3


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

BN_MOMENTUM = 0.1
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x




