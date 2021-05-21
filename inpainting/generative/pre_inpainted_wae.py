"""
Code inspired by
https://github.com/sedelmeyer/wasserstein-auto-encoder/blob/master/Wasserstein-auto-encoder_tutorial.ipynb
"""
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Protocol
from inpainting.datasets import mask_coding as mc
from inpainting.custom_layers import ConVar
from inpainting.utils import freeze_params, free_params, printable_history
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from inpainting.evaluation import fid
from inpainting.generative.wae import (
    get_discriminator,
    WAEMetricFn,
    discriminator_train_loss,
    discriminator_fool_loss,
    wae_reconstruction_loss,
    psnr,
    ssim,
    batches_to_dl,
)


class PreInpaintedWAE(nn.Module):
    def __init__(
        self,
        convar_layer: ConVar,
        encoder: nn.Module,
        decoder: nn.Module,
        discriminator: nn.Module,
        keep_inpainting_gradient: bool = False,
        sigma: float = 1,
    ):
        super().__init__()
        self.convar = convar_layer
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.keep_inpainting_gradient = keep_inpainting_gradient
        self.sigma = sigma

    def forward(
        self,
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
    ):
        b, c, h, w = X.shape
        b, n, l, chw = A.shape

        P_r = P
        M_r = M.reshape(b, n, c, h, w)
        A_r = A.reshape(b, n, l, c, h, w)
        D_r = D.reshape(b, n, c, h, w)

        if not self.keep_inpainting_gradient:
            P_r, M_r, A_r, D_r = [t.detach() for t in [P_r, M_r, A_r, D_r]]
        convar_out = self.convar(X, J, P_r, M_r, A_r, D_r)

        encoder_out = self.encoder(convar_out)
        discriminator_true_encoding_out = self.discriminator(encoder_out)

        fake_encoding = torch.randn_like(encoder_out) * self.sigma
        discriminator_fake_encoding_out = self.discriminator(fake_encoding)

        decoder_out = self.decoder(encoder_out)
        decoder_fake_out = self.decoder(fake_encoding)
        return (
            (encoder_out, decoder_out),
            (fake_encoding, decoder_fake_out),
            (discriminator_true_encoding_out, discriminator_fake_encoding_out),
            ((P, M, A, D), convar_out),
        )


def train_pre_inpainted_wae(
    wae: PreInpaintedWAE,
    data_loader_train: DataLoader,
    data_loaders_val: Dict[str, DataLoader],
    optimizer: Optimizer,
    n_epochs: int,
    device: torch.device,
    max_benchmark_batches: int,
    discriminator_loss_fn: BCELoss,
    reconstruction_loss_fn: BCELoss,
    fid_model: Optional[nn.Module] = None,
) -> List[dict]:
    history = []
    epoch = 0
    metric_fns = dict(
        d_train_loss=discriminator_train_loss(discriminator_loss_fn),
        d_fool_loss=discriminator_fool_loss(discriminator_loss_fn),
        recon_loss=wae_reconstruction_loss(reconstruction_loss_fn),
        psnr=psnr,
        ssim=ssim,
    )
    history.append(
        eval_pre_inpainted_wae(
            wae,
            epoch=epoch,
            data_loaders={
                k: tqdm(v, f"Epoch {epoch}, test_{k}")
                for (k, v) in data_loaders_val.items()
            },
            device=device,
            metric_fns=metric_fns,
            max_benchmark_batches=max_benchmark_batches,
            fid_model=fid_model,
        )
    )
    print(printable_history(history)[-1])
    for e in tqdm(range(epoch + 1, n_epochs + 1), desc="Epoch"):
        epoch_result = train_epoch(
            wae,
            data_loader_train,
            data_loaders_val,
            optimizer,
            device,
            e,
            discriminator_loss_fn=discriminator_loss_fn,
            reconstruction_loss_fn=reconstruction_loss_fn,
            metric_fns=metric_fns,
            max_benchmark_batches=max_benchmark_batches,
            fid_model=fid_model,
        )
        history.append(epoch_result)
        print(printable_history(history)[-1])
    return history


def train_epoch(
    wae: PreInpaintedWAE,
    data_loader_train: DataLoader,
    data_loaders_val: Dict[str, DataLoader],
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    discriminator_loss_fn: BCELoss,
    reconstruction_loss_fn: BCELoss,
    metric_fns: Dict[str, WAEMetricFn],
    max_benchmark_batches: int,
    fid_model: nn.Module,
) -> dict:
    wae.train()

    for (X, J, P, M, A, D, _) in tqdm(data_loader_train, f"Epoch {epoch}, train"):
        X, J, P, M, A, D = [t.to(device) for t in [X, J, P, M, A, D]]

        # train discriminator
        wae.zero_grad()
        freeze_params(wae.convar)
        freeze_params(wae.encoder)
        freeze_params(wae.decoder)
        free_params(wae.discriminator)

        (enc_out, dec_out), (enc_fake, dec_fake_out), (d_true_out, d_fake_out), _ = wae(
            X, J, P, M, A, D
        )

        d_loss = discriminator_train_loss(discriminator_loss_fn)(
            X, J, enc_out, dec_out, enc_fake, dec_fake_out, d_true_out, d_fake_out
        )
        d_loss.backward()
        optimizer.step()

        # train encoder + decoder
        wae.zero_grad()
        free_params(wae.convar)
        free_params(wae.encoder)
        free_params(wae.decoder)
        freeze_params(wae.discriminator)

        (enc_out, dec_out), (enc_fake, dec_fake_out), (d_true_out, d_fake_out), _ = wae(
            X, J, P, M, A, D
        )

        recon_loss = wae_reconstruction_loss(reconstruction_loss_fn)(
            X, J, enc_out, dec_out, enc_fake, dec_fake_out, d_true_out, d_fake_out
        )
        d_loss = discriminator_fool_loss(discriminator_loss_fn)(
            X, J, enc_out, dec_out, enc_fake, dec_fake_out, d_true_out, d_fake_out
        )
        loss = recon_loss + d_loss
        loss.backward()
        optimizer.step()

    return eval_pre_inpainted_wae(
        wae,
        epoch=epoch,
        data_loaders={
            k: tqdm(v, f"Epoch {epoch}, test_{k}")
            for (k, v) in data_loaders_val.items()
        },
        device=device,
        metric_fns=metric_fns,
        max_benchmark_batches=max_benchmark_batches,
        fid_model=fid_model,
    )


def eval_pre_inpainted_wae(
    wae: PreInpaintedWAE,
    epoch: int,
    data_loaders: Dict[str, DataLoader],
    device: torch.device,
    metric_fns: Dict[str, WAEMetricFn],
    max_benchmark_batches: int,
    fid_model: Optional[nn.Module],
) -> dict:
    wae.eval()
    fold_metrics = dict()
    example_predictions = dict()

    for fold, dl in data_loaders.items():
        metrics = []
        batches = []
        for i, (X, J, P, M, A, D, Y) in enumerate(dl):
            if i > max_benchmark_batches and max_benchmark_batches > 0:
                break
            X, J, P, M, A, D = [t.to(device) for t in [X, J, P, M, A, D]]
            (
                (enc_out, dec_out),
                (enc_fake, dec_fake_out),
                (d_true_out, d_fake_out),
                (PMAD_out, convar_out),
            ) = wae(X, J, P, M, A, D)

            metrics.append(
                {
                    m_name: metric_fn(
                        X,
                        J,
                        enc_out,
                        dec_out,
                        enc_fake,
                        dec_fake_out,
                        d_true_out,
                        d_fake_out,
                    ).item()
                    for (m_name, metric_fn) in metric_fns.items()
                }
            )
            batch = {"X": X, "J": J, "dec_out": dec_out, "dec_fake_out": dec_fake_out}

            batches.append({k: v.detach().cpu().numpy() for (k, v) in batch.items()})

            if i == 0:
                (
                    P,
                    M,
                    A,
                    D,
                ) = PMAD_out
                preds = dict(
                    X=X,
                    J=J,
                    P=P,
                    M=M,
                    A=A,
                    D=D,
                    Y=Y,
                    convar_out=convar_out,
                    encoder_out=enc_out,
                    decoder_out=dec_out,
                    discriminator_out=d_true_out,
                    encoder_fake=enc_fake,
                    decoder_fake_out=dec_fake_out,
                )
                example_predictions[fold] = {
                    k: v.cpu().detach().numpy() for (k, v) in preds.items()
                }

        frechet_dist_decoded = 0
        frechet_dist_sampled = 0
        if fid_model is not None:

            frechet_dist_decoded = fid.frechet_distance(
                batches_to_dl(batches, "X", device),
                batches_to_dl(batches, "dec_out", device),
                fid_model,
                il1_key=fold,
                il2_key=f"{epoch}_{fold}_decoding",
            ).item()

            frechet_dist_sampled = fid.frechet_distance(
                batches_to_dl(batches, "X", device),
                batches_to_dl(batches, "dec_fake_out", device),
                fid_model,
                il1_key=fold,
                il2_key=f"{epoch}_{fold}_sampling",
            ).item()

        for m in metrics:
            m["fid_decoding"] = frechet_dist_decoded
            m["fid_sampling"] = frechet_dist_sampled

        fold_metrics[fold] = metrics

    return dict(
        epoch=epoch,
        metrics={
            m_name: {
                fold: np.mean([m[m_name] for m in f_metrics])
                for fold, f_metrics in fold_metrics.items()
            }
            for m_name in list(metric_fns.keys()) + ["fid_decoding", "fid_sampling"]
        },
        sample_results=example_predictions,
    )


def pre_inpainted_wae_iterator_for_fid(
    data_loader: DataLoader,
    wae: PreInpaintedWAE,
    return_mode: str,
    device: torch.device,
):
    for (X, J, P, M, A, D), y in data_loader:
        X, J = [t.to(device) for t in [X, J]]

        (enc_out, dec_out), (enc_fake, dec_fake_out), (d_true_out, d_fake_out), _ = wae(
            X, J
        )

        if return_mode == "decoding":
            yield dec_out

        elif return_mode == "sampling":
            yield dec_fake_out

        else:
            raise TypeError(return_mode)
