import torch
import torch.nn as nn
import numpy as np
import math
import copy
from resnet import resnet18
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        # src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None):
        # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DepthAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()  # 生成深度注意力权重
        )
        self.rgb_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, rgb_feat, depth_feat):
        # 深度引导的注意力机制
        depth_att = self.depth_conv(depth_feat)
        #print(depth_att.shape)
        #print(rgb_feat.shape)
        weighted_rgb = rgb_feat * depth_att

        # 多尺度特征融合
        fused = torch.cat([weighted_rgb, depth_feat], dim=1)
        #print(fused.shape)
        return self.rgb_conv(fused)


class HeadPoseEstimator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_features * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 预测pitch, yaw, roll
        )

    def forward(self, rgb_feat, depth_feat):
        combined = torch.cat([rgb_feat, depth_feat], dim=1)
        return self.regressor(combined)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6
        self.lm=0.1

        self.base_model = resnet18(pretrained=True, maps=maps)
        self.base_model_d = resnet18(pretrained=True, maps=maps)


        # d_model: dim of Q, K, V
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
            maps,
            nhead,
            dim_feedforward,
            dropout)

        encoder_norm = nn.LayerNorm(maps)
        # num_encoder_layer: deeps of layers

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature + 1, maps)

        self.feed = nn.Linear(maps, 2)
        self.feed1 = nn.Linear(maps, 2)

        self.loss_op = nn.L1Loss()
        self.fusion = DepthAwareFusion(maps)


    def forward(self, x_in):

        feature = self.base_model(x_in["face"])
        feature_d = self.base_model_d(x_in["deep"])

        #res_d = torch.mean(feature, dim=[2, 3])
        #print(feature.shape)
        #print(feature_d.shape)
        res_f=self.fusion(feature, feature_d)
        res_f=torch.mean(res_f, dim=[2, 3])
        #print(res_f.shape)

        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)

        cls = self.cls_token.repeat((1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)

        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        #print(feature.shape,pos_feature.shape)
        feature = self.encoder(feature, pos_feature)


        feature = feature.permute(1, 2, 0)

        feature = feature[:, :, 0]

        #print(feature.shape,"###")
        feature1=feature+res_f



        gaze = self.feed(feature1)

        return gaze

    def GazeTo3d(self,head_pose):
        yaw ,pitch,= head_pose[:, 0], head_pose[:, 1]
        head_pose_vector = torch.stack([
            -torch.cos(pitch) * torch.sin(yaw),
            -torch.sin(pitch),
            -torch.cos(pitch) * torch.cos(yaw),

        ], dim=-1)
        return head_pose_vector

    def GazeTo2d1(gaze):
        yaw = np.arctan2(-gaze[0], -gaze[2])
        pitch = np.arcsin(-gaze[1])
        return np.array([yaw, pitch])

    def GazeTo3d1(gaze):
        x = -np.cos(gaze[1]) * np.sin(gaze[0])
        y = -np.sin(gaze[1])
        z = -np.cos(gaze[1]) * np.cos(gaze[0])
        return np.array([x, y, z])

    def loss(self, x_in, label):
        pred_gaze = self.forward(x_in)

        pred_gaze1=self.GazeTo3d(pred_gaze)
        label1 = self.GazeTo3d(label)


        cos_sim = F.cosine_similarity(pred_gaze1, label1)
        angle_loss = torch.acos(cos_sim.clamp(-1 + 1e-6, 1 - 1e-6))
        loss1 = self.loss_op(pred_gaze, label)

        return angle_loss.mean() + loss1

if __name__ == "__main__":
    model = Model()
    dummy_input = {
        "face": torch.randn(10, 3, 224, 224),
        "deep": torch.randn(10, 3, 224, 224),

    }
    output = model(dummy_input)
    #print(output)  # 应输出 torch.Size([2, 2])