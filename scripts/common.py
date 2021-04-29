"""Script utilities"""
import sys
from pathlib import Path
from typing import Union, Optional

from torch import nn

from inpainting import backbones as bkb
from inpainting.inpainters.fullconv import FullyConvolutionalInpainter
from inpainting.inpainters.linear_heads import LinearHeadsInpainter

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig
from inpainting.custom_layers import ConVar, ConVarNaive, PartialConvWrapper


def dmfa_from_args(args) -> Union[FullyConvolutionalInpainter, LinearHeadsInpainter]:
    """Load DMFA from parsed script arguments"""
    img_channels = 1 if "mnist" in args.dataset else 3

    backbone_modules = bkb.down_up_backbone_v2(
        (img_channels * 2, args.img_size, args.img_size),
        depth=args.bkb_depth,
        first_channels=args.bkb_fc,
        last_channels=args.bkb_lc,
        kernel_size=5,
        latent_size=-1,
        block_length=args.bkb_block_length,
    )

    if args.architecture == "fullconv":
        inpainter = FullyConvolutionalInpainter(
            a_width=args.num_factors,
            a_amplitude=args.a_amplitude,
            c_h_w=(img_channels, args.img_size, args.img_size),
            last_channels=args.bkb_lc,
            extractor=nn.Sequential(*backbone_modules),
            n_mixes=args.num_mixes,
        )
    elif args.architecture == "linear_heads":
        inpainter = LinearHeadsInpainter(
            c_h_w=(img_channels, args.img_size, args.img_size),
            last_channels=args.bkb_lc,
            a_width=args.num_factors,
            a_amplitude=args.a_amplitude,
            n_mixes=args.num_mixes,
        )
    else:
        raise ValueError("can't initialize inpainter")

    return inpainter


def mfa_from_path(mfa_path: Union[Path, str]):
    # https://github.com/mprzewie/gmm_missing
    # TODO make it less hacky
    mfa_path = Path(mfa_path)
    sys.path.append("../../gmm_missing")
    from mfa_wrapper import MFAWrapper

    return MFAWrapper.from_path(mfa_path)


def acflow_from_path(path: Union[Path, str], batch_size: Optional[int] = None):
    path = Path(path)
    sys.path.append("../../ACFlow/")
    from utils.acflow_wrapper import ACFlowWrapper

    return ACFlowWrapper.from_path(path, batch_size=batch_size)


def mask_configs_from_args(args):
    mask_configs_train = [
        RandomRectangleMaskConfig(
            UNKNOWN_LOSS, args.mask_hidden_h, args.mask_hidden_w, deterministic=False
        )
    ]

    mask_configs_val = [
        RandomRectangleMaskConfig(
            UNKNOWN_LOSS, args.mask_hidden_h, args.mask_hidden_w, deterministic=True
        )
    ]

    if args.mask_unknown_size > 0:
        mask_configs_train.append(
            RandomRectangleMaskConfig(
                UNKNOWN_NO_LOSS,
                args.mask_unknown_size,
                args.mask_unknown_size,
                deterministic=True,
            )
        )
    return mask_configs_train, mask_configs_val


def convar_from_args(args):
    convar_in_channels = 1 if "mnist" in args.dataset else 3

    conv = nn.Conv2d(
        convar_in_channels * 2 if args.convar_append_mask else convar_in_channels,
        args.convar_channels,
        kernel_size=3,
        padding=1,
    )
    convar = (
        ConVar(conv, args.convar_append_mask)
        if args.convar_type == "full"
        else ConVarNaive(conv, args.convar_append_mask)
    )

    if args.convar_type == "partial":
        sys.path.append("../../partialconv/models")
        from partialconv2d import PartialConv2d

        convar = PartialConvWrapper(
            PartialConv2d(
                convar_in_channels,
                args.convar_channels,
                kernel_size=3,
                padding=1,
                multi_channel=True,
            )
        )

    return convar
