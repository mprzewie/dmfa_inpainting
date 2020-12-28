"""Script utilities"""
from typing import Union

from inpainters.fullconv import FullyConvolutionalInpainter
from inpainters.linear_heads import LinearHeadsInpainter
from inpainting import backbones as bkb


def dmfa_from_args(args) -> Union[FullyConvolutionalInpainter, LinearHeadsInpainter]:
    """Load DMFA from parsed script arguments"""
    img_channels = 1 if "mnist" in args.dataset else 3

    if args.architecture == "fullconv":
        inpainter = FullyConvolutionalInpainter(
            a_width=args.num_factors,
            a_amplitude=args.a_amplitude,
            c_h_w=(img_channels, args.img_size, args.simg_size),
            last_channels=args.bkb_lc,
            extractor=bkb.down_up_backbone(
                (img_channels * 2, args.img_size, args.img_size),
                depth=args.bkb_depth,
                first_channels=args.bkb_fc,
                last_channels=args.bkb_lc,
                kernel_size=5,
                latent=args.bkb_latent,
                block_length=args.bkb_block_length,
            ),
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
