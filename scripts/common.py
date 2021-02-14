"""Script utilities"""
import sys
from pathlib import Path
from typing import Union

from torch import nn

from inpainting import backbones as bkb
from inpainting.inpainters.fullconv import FullyConvolutionalInpainter
from inpainting.inpainters.linear_heads import LinearHeadsInpainter


def dmfa_from_args(args) -> Union[FullyConvolutionalInpainter, LinearHeadsInpainter]:
    """Load DMFA from parsed script arguments"""
    img_channels = 1 if "mnist" in args.dataset else 3

    backbone_modules = bkb.down_up_backbone(
        (img_channels * 2, args.img_size, args.img_size),
        depth=args.bkb_depth,
        first_channels=args.bkb_fc,
        last_channels=args.bkb_lc,
        kernel_size=5,
        latent=args.bkb_latent,
        block_length=args.bkb_block_length,
    )

    if args.architecture == "fullconv":
        inpainter = FullyConvolutionalInpainter(
            a_width=args.num_factors,
            a_amplitude=args.a_amplitude,
            c_h_w=(img_channels, args.img_size, args.img_size),
            last_channels=args.bkb_lc,
            extractor=nn.Sequential(*backbone_modules),
            n_mixes=1,
        )
    elif args.architecture == "linear_heads":
        inpainter = LinearHeadsInpainter(
            c_h_w=(img_channels, args.img_size, args.img_size),
            last_channels=args.bkb_lc,
            a_width=args.num_factors,
            a_amplitude=args.a_amplitude,
            n_mixes=1,
        )
    else:
        raise ValueError("can't initialize inpainter")

    return inpainter


def mfa_from_path(mfa_path: Path):
    # https://github.com/mprzewie/gmm_missing
    # TODO make it less hacky
    sys.path.append("../../gmm_missing")
    from mfa_wrapper import MFAWrapper

    return MFAWrapper.from_path(mfa_path)
