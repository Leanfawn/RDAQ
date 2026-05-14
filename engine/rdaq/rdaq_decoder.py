"""
RDAQ: DETR meets Refinement-Driven Adaptive Querying for Dense Aerial Imagery
Copyright (c) 2025 The RDAQ Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from ..core import register

from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ['RDAQ_Decoder']


class SqueezeFusionUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, high_res_feat, low_res_feat):
        low_res_upsampled = self.upsample(low_res_feat)
        fused_feat = torch.cat([high_res_feat, low_res_upsampled], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        spatial_gate = self.spatial_gate(fused_feat)
        refined_high_res = high_res_feat * spatial_gate
        return refined_high_res, spatial_gate


class HSF(nn.Module):
    def __init__(self, d_model, num_levels):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.sfu_units = nn.ModuleList(
            [SqueezeFusionUnit(d_model) for _ in range(self.num_levels - 1)]
        )

    def forward(self, proj_feats: List[torch.Tensor]):
        assert len(proj_feats) == self.num_levels
        refined_feats = list(proj_feats)
        p3_spatial_gate = None

        for i in range(self.num_levels - 2, -1, -1):
            high_res_feat = refined_feats[i]
            low_res_feat = refined_feats[i + 1]
            sfu_unit = self.sfu_units[i]
            refined_feat_i, spatial_gate_i = sfu_unit(high_res_feat, low_res_feat)
            refined_feats[i] = refined_feat_i

            if i == 0:
                p3_spatial_gate = spatial_gate_i

        return refined_feats, p3_spatial_gate


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MultiScaleFeature(nn.Module):
    def __init__(self, channels=256, scale=3):
        super(MultiScaleFeature, self).__init__()
        self.conv = nn.ModuleList(
            [Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1) for _ in range(scale - 1)]
        )

    def forward(self, x):
        x_out = [x]
        for layer in self.conv:
            x_out.append(layer(x_out[-1]))
        return x_out


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x,
                    2,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvNormLayer_fuse(2, 1, kernel_size, stride=1)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale


import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityAwareDynamicFilter(nn.Module):
    def __init__(self, channels, density_channels, kernel_size=5):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.channels = channels

        self.weight_generator = nn.Sequential(
            nn.Conv2d(density_channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * self.k * self.k, 1)
        )

        self.norm = nn.GroupNorm(32, channels)
        self.act = nn.ReLU()

    def forward(self, x, density_feat):
        B, C, H, W = x.shape
        density_feat_resized = F.interpolate(
            density_feat, size=(H, W), mode='bilinear', align_corners=False
        )
        global_density = F.adaptive_avg_pool2d(density_feat_resized, 1)
        dynamic_weights = self.weight_generator(global_density)
        x_reshaped = x.reshape(1, B * C, H, W)
        weights_reshaped = dynamic_weights.view(B * C, 1, self.k, self.k)
        out = F.conv2d(x_reshaped, weights_reshaped, padding=self.pad, groups=B * C)
        out = out.view(B, C, H, W)
        return self.act(self.norm(out + x))


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(
            in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate
        )
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


class DensityGuidedRoutingEstimator(nn.Module):
    def __init__(self, dim, cls_num=4):
        super(DensityGuidedRoutingEstimator, self).__init__()
        self.offset_generator = nn.Conv2d(1, 3 * 3 * 3, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_generator.weight, 0.)
        nn.init.constant_(self.offset_generator.bias, 0.)

        self.deform_weight = nn.Parameter(torch.Tensor(dim, dim, 3, 3))
        nn.init.kaiming_uniform_(self.deform_weight, a=math.sqrt(5))
        self.dim = dim


        self.density_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=3, padding=1, bias=False)
        )

        self.offset_generator = nn.Conv2d(1, 3 * 3 * 3, kernel_size=3, padding=1)
        self.deform_weight = nn.Parameter(torch.Tensor(dim, dim, 3, 3))
        nn.init.kaiming_uniform_(self.deform_weight, a=math.sqrt(5))
        
        self.scatter_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.SiLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.SiLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim // 2, cls_num)
        )

    def forward(self, features, spatial_shapes, p3_spatial_gate):
        features = features.transpose(1, 2)
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]

        v_feat = features[:, :, 0:h * w].view(bs, self.dim, h, w)
  
        x = v_feat * p3_spatial_gate

        density_map = self.density_attn(x)
        attn_weights = F.softmax(density_map.view(bs, 1, -1), dim=-1).view(bs, 1, h, w)

        offset_mask = self.offset_generator(density_map)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1) # 坐标偏移量
        mask = torch.sigmoid(mask)          # 采样点权重

        dense_feat = ops.deform_conv2d(x, offset, self.deform_weight, mask=mask, padding=1)

        scatter_feat = self.scatter_conv(x)

        combined_feat = torch.cat([dense_feat, scatter_feat], dim=1)
        refined_feat = self.fusion(combined_feat)

        global_stat = (refined_feat * attn_weights).sum(dim=(2, 3))

        out_cls = self.classifier(global_stat)
        final_feat = x + refined_feat

        return out_cls, final_feat

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            'num_points_scale',
            torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]
        ).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int]
    ):
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(
                dtype=query.dtype
            ).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = (
                reference_points[:, :, None, :, :2] + offset
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list
        )

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.,
        activation='relu',
        n_levels=4,
        n_points=4,
        cross_attn_method='default',
        layer_scale=None
    ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None
    ):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes
        )

        target = self.gateway(target, self.dropout2(target2))

        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(
            pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1
        )
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat(
            [prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1
        )
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act='relu'
    ):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        dq_num_queris=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed
            )

            if i == 0:
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), reg_scale
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores
        )


@register()
class HSFDFINETransformerplus(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        reg_max=32,
        reg_scale=4.,
        layer_scale=1,
        mlp_act='relu',
        dynamic_query_list=None,
        dgre_cls_num=4,
        using_dynamic_query=True,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic')
        assert cross_attn_method in ('default', 'discrete')
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(
                    hidden_dim,
                    hidden_dim,
                    4 * (self.reg_max + 1),
                    3,
                    act=mlp_act
                ) for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(
                    scaled_dim,
                    scaled_dim,
                    4 * (self.reg_max + 1),
                    3,
                    act=mlp_act
                ) for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(self.reg_max)

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters(feat_channels)

        self.using_dynamic_query = using_dynamic_query
        self.dgre_cls_num = dgre_cls_num
        self.dynamic_query_list = dynamic_query_list
        self.DGRE = CategoricalCounting(self.hidden_dim, cls_num=self.dgre_cls_num)

        self.dddf_module = DensityAwareDynamicFilter(
            channels=self.hidden_dim,
            density_channels=self.hidden_dim,
            kernel_size=5
        )
        self.multiscale = MultiScaleFeature(self.hidden_dim, scale=self.num_levels)
        self.hsf_module = HSF(d_model=self.hidden_dim, num_levels=self.num_levels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))
                    ]))
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))
                    ]))
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        refined_proj_feats, p3_gate = self.hsf_module(proj_feats)

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(refined_proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        feat_flatten = torch.concat(feat_flatten, 1)

        return feat_flatten, spatial_shapes, proj_feats, p3_gate

    def _generate_anchors(
        self,
        spatial_shapes=None,
        grid_size=0.05,
        dtype=torch.float32,
        device='cpu'
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing='ij'
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor(
                [w, h], dtype=dtype
            )
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat(
                [grid_xy, wh], dim=-1
            ).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(
                output_memory, enc_outputs_logits, anchors, self.num_queries
            )

        enc_topk_bbox_unact: torch.Tensor = \
            self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile(
                [memory.shape[0], 1, 1]
            )
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return (
            content,
            enc_topk_bbox_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            enc_outputs_logits
        )

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int
    ):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(
                outputs_logits.max(-1).values, topk, dim=-1
            )
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(
                outputs_logits.flatten(1), topk, dim=-1
            )
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(
                outputs_logits.squeeze(-1), topk, dim=-1
            )

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1, 1, outputs_anchors_unact.shape[-1]
            )
        )
        topk_logits = outputs_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1, 1, outputs_logits.shape[-1]
            )
        ) if self.training else None
        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1, 1, memory.shape[-1]
            )
        )

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        memory, spatial_shapes, proj_feats_raw, p3_gate = self._get_encoder_input(feats)

        counting_output, dgre_feature = self.DGRE(memory, spatial_shapes, p3_gate)
        bs, _, c = memory.shape
        memory_spatial_splits = []
        start_idx = 0

        for (h, w) in spatial_shapes:
            end_idx = start_idx + h * w
            feat_spatial = memory[:, start_idx:end_idx, :].transpose(1, 2).view(
                bs, c, h, w
            )
            memory_spatial_splits.append(feat_spatial)
            start_idx = end_idx

        refined_spatial_splits = []
        for feat in memory_spatial_splits:
            refined_feat = self.dddf_module(feat, dgre_feature)
            refined_spatial_splits.append(refined_feat)

        refined_memory_flatten = []
        for feat in refined_spatial_splits:
            refined_memory_flatten.append(
                feat.flatten(2).transpose(1, 2)
            )

        refined_memory = torch.cat(refined_memory_flatten, dim=1)

        num_queries_list = None
        if self.using_dynamic_query:
            _, predicted = torch.max(counting_output.data, 1)
            num_queries_list = [
                self.dynamic_query_list[i] for i in predicted.tolist()
            ]
            num_select = self.dynamic_query_list[max(predicted.tolist())]
            self.num_queries = num_select

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            if memory.size(0) > 1 and self.using_dynamic_query:
                attn_mask = attn_mask.unsqueeze(0).repeat(
                    memory.size(0) * self.nhead, 1, 1
                )
                for qn in num_queries_list:
                    attn_mask[
                        :,
                        dn_meta['dn_num_split'][0] + qn:,
                        :dn_meta['dn_num_split'][0] + qn
                    ] = True
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                None, None, None, None
            if memory.size(0) > 1 and self.using_dynamic_query:
                attn_mask = torch.full(
                    [self.num_queries, self.num_queries],
                    False,
                    dtype=torch.bool,
                    device=memory.device
                )
                for qn in num_queries_list:
                    attn_mask[qn:, :qn] = True

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            enc_outputs_logits
        ) = self._get_decoder_input(
            refined_memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact
        )

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = \
            self.decoder(
                init_ref_contents,
                init_ref_points_unact,
                refined_memory,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                dn_meta=dn_meta
            )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta['dn_num_split'], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta['dn_num_split'], dim=1
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta['dn_num_split'], dim=2
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta['dn_num_split'], dim=2
            )
            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta['dn_num_split'], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta['dn_num_split'], dim=2
            )

        if self.training:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
                'pred_corners': out_corners[-1],
                'ref_points': out_refs[-1],
                'up': self.up,
                'reg_scale': self.reg_scale,
                'pred_bbox_number': counting_output,
                'num_queries_list': num_queries_list
            }
        else:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
                'num_queries_list': num_queries_list
            }

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1]
            )
            out['enc_aux_outputs'] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list
            )
            out['pre_outputs'] = {
                'pred_logits': pre_logits,
                'pred_boxes': pre_bboxes
            }
            out['enc_meta'] = {
                'class_agnostic': self.query_select_method == 'agnostic'
            }

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1]
                )
                out['dn_pre_outputs'] = {
                    'pred_logits': dn_pre_logits,
                    'pred_boxes': dn_pre_bboxes
                }
                out['dn_meta'] = dn_meta

            out['enc_outputs_logits'] = enc_outputs_logits

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None
    ):
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_corners': c,
                'ref_points': d,
                'teacher_corners': teacher_corners,
                'teacher_logits': teacher_logits
            }
            for a, b, c, d in zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref
            )
        ]
