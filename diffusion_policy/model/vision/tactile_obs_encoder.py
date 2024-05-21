import copy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# import torchvision
# from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

# from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from .attentive_pooler import AttentivePooler


class TactileObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        checkpoint_encoder: Optional[str] = None,
        train_encoder: bool = False,
        # shape_meta: dict,
        # rgb_model: Union[nn.Module, Dict[str,nn.Module]],
        # resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        # crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        # random_crop: bool=True,
        # # replace BatchNorm with GroupNorm
        # use_group_norm: bool=False,
        # # use single rgb model for all rgb inputs
        # share_rgb_model: bool=False,
        # # renormalize rgb input with imagenet normalization
        # # assuming input in [0,1]
        # imagenet_norm: bool=False
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder: nn.Module = encoder
        self.train_encoder: bool = train_encoder

        self.rgb_keys = ["digit_thumb", "digit_index"]
        self.low_dim_keys = ["robot_joint", "allegro_joint"]

        if checkpoint_encoder is not None:
            self.load_tactile_encoder(checkpoint_encoder)
            print(f"Tactile encoder loaded successfully from {checkpoint_encoder}")
        else:
            print("No checkpoint provided. Training tactile encoder from scratch.")

        # freeze encoder
        if not self.train_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        # attentive pooling
        self.pooling = AttentivePooler(
            embed_dim=embed_dim,
        )

    # ========= load tactile encoder  ============
    def load_tactile_encoder(self, checkpoint_encoder):
        checkpoint = torch.load(checkpoint_encoder)
        self.encoder.load_state_dict(checkpoint, strict=False)

    # ========= forward pass  ============
    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input

        # pass all rgb obs to rgb model
        imgs = list()
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            imgs.append(img)
        # (N*B,C,H,W)
        imgs = torch.cat(imgs, dim=0)
        # (N*B,D)
        feature = self.encoder(imgs)  # include attentive pooling
        feature = self.pooling(feature).squeeze(1)
        # (N,B,D)
        feature = feature.reshape(-1, batch_size, *feature.shape[1:])
        # (B,N,D)
        feature = torch.moveaxis(feature, 0, 1)
        # (B,N*D)
        feature = feature.reshape(batch_size, -1)
        features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        out_dim = (self.embed_dim * 2) + 7 + 16
        return (out_dim,)
