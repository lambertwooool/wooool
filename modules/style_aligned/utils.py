import torch
import torch.nn as nn
from typing import Union

T = torch.Tensor

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    """
    Expand the first element so it has the same shape as the rest of the batch.
    """
    b = feat.shape[0]
    # feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    feat_style = torch.stack((feat[0], feat[0])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    """
    concat the the feature and the style feature expanded above
    """
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def get_norm_layers(
    layer: nn.Module,
    norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]",
    share_layer_norm: bool,
    share_group_norm: bool,
):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(
                child_layer, norm_layers_, share_layer_norm, share_group_norm
            )