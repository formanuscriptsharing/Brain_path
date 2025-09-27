import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3dunet.unet3d.buildingblocks import ResNetBlock, create_encoders, create_decoders
from pytorch3dunet.unet3d.utils import number_of_features_per_level


class UNetEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_level=4, resolution=120):
        super(UNetEncoder3D, self).__init__()

        # 定义特征映射的每一层通道数
        self.f_maps = number_of_features_per_level(base_channels, num_level)

        # 创建编码器
        self.encoders = create_encoders(
            in_channels, self.f_maps, ResNetBlock, conv_kernel_size=3,
            conv_padding=1, conv_upscale=2, dropout_prob=0.0,
            layer_order='gcr', num_groups=8, pool_kernel_size=2, is3d=True
        )

        # 输出特征分割层
        self.feature_split = nn.Conv3d(self.f_maps[-1], 128, kernel_size=1)  # 将最后的特征图通道数调整为 256

        # 创建 age map 的卷积层

        # self.age_conv = nn.Conv3d(64, 1, kernel_size=1)  # 用后 128 个通道预测 age map
        self.encoded_feature = nn.Conv3d(64, 1, kernel_size=1)  # 用后 128 个通道预测 age map
        self.age_prediction= nn.Linear(64*15*15*15, 256)
        self.age_projection_head=nn.Linear(256,1)
    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)

        # 分割最后一层的特征
        split_features = F.leaky_relu(self.feature_split(x) ) # 输出形状为 [B, 256, 16, 16, 16]
        decoder_features, age_features = torch.split(split_features, [64, 64], dim=1)
        # features[-1]=decoder_features

        age=self.age_projection_head(F.relu(self.age_prediction(age_features.flatten(start_dim=1))))

        return features, decoder_features, age
    def reparameterize(self,mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        mu, logvar: shape [B,1,D,H,W] (或其他形状)
          logvar = log(sigma^2)
        返回: z = mu + sigma * eps
        """
        # 1) 采样 eps ~ N(0,1), 与 mu 同形状
        eps = torch.randn_like(mu)
        # 2) 计算 sigma = exp(0.5 * logvar)
        sigma = torch.exp(0.5 * logvar)
        # 3) 重参数化输出
        z = mu + sigma * eps
        return z

class UNetDecoder3D(nn.Module):
    def __init__(self, out_channels=1, base_channels=32, num_level=4, resolution=120):
        super(UNetDecoder3D, self).__init__()

        # 定义特征映射的每一层通道数
        self.f_maps = number_of_features_per_level(base_channels, num_level)

        # 创建解码器
        self.decoders = create_decoders(
            self.f_maps, ResNetBlock, conv_kernel_size=3,
            conv_padding=1, layer_order='gcr', num_groups=8,
            upsample='default', dropout_prob=0.0, is3d=True
        )
        self.feature_split = nn.Conv3d( 128,self.f_maps[-1], kernel_size=1)  # 将最后的特征图通道数调整为 256
        # 最终的卷积层生成输出
        self.final_conv = nn.Conv3d(self.f_maps[0], out_channels, kernel_size=3, stride=1, padding=1)

        self.age_conv = nn.Conv3d(1,64, kernel_size=1)  # 用后 128 个通道预测 age map
        self.age_projection_head=nn.Linear(1,15*15*15)
    def forward(self, features, decoder_features,age_features):
        # 初始化解码器特征
        h =  F.leaky_relu(self.age_conv(F.relu(self.age_projection_head(age_features.unsqueeze(1)).view((len(age_features),15,15,15))).unsqueeze(1)))
        features = features[::-1]  # 反转特征顺序
        h=torch.cat((decoder_features,h),dim=1)
        h= F.leaky_relu(self.feature_split(h))
        # self.decoders=self.decoders[::-1]
        for i, decoder in enumerate(self.decoders):
            h = decoder(features[i+1], h)
        h = self.final_conv(h)
        return torch.sigmoid(h)


class UNetModel3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_level=4, resolution=120):
        super(UNetModel3D, self).__init__()
        # 初始化编码器和解码器
        self.encoder = UNetEncoder3D(in_channels, base_channels, num_level, resolution=resolution)
        self.decoder = UNetDecoder3D(out_channels, base_channels, num_level, resolution=resolution)

    def forward(self, x,age_features=None):
        # 编码器处理输入
        features, decoder_features, age_map = self.encoder(x)
        # 解码器处理特征
        if age_features is not None:
            reconstructed = self.decoder( features, decoder_features,age_features)
        else:
            reconstructed = self.decoder(features, decoder_features, age_map[:,0,::])
        return reconstructed, age_map  # 返回重建图像和 age map