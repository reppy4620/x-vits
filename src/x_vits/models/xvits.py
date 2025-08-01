import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from x_vits.losses import kl_loss
from x_vits.modules.handler import DurationHandlerOutput
from x_vits.utils.model import rand_slice_segments, slice_segments, to_log_scale


class XVITS(nn.Module):
    def __init__(
        self,
        text_encoder,
        duration_handler,
        frame_prior_network,
        pitch_predictor,
        flow,
        posterior_encoder,
        vocoder,
        spec_tfm,
        segment_size,
        style_encoder=None,
        style_diffusion=None,
        context_embedder=None,
        ref_segment_size=None,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.duration_handler = duration_handler
        self.frame_prior_network = frame_prior_network
        self.pitch_predictor = pitch_predictor
        self.flow = flow
        self.posterior_encoder = posterior_encoder
        self.vocoder = vocoder
        self.dec = vocoder

        self.style_encoder = style_encoder
        self.style_diffusion = style_diffusion
        self.context_embedder = context_embedder

        self.spec_tfm = spec_tfm
        self.segment_size = segment_size
        self.ref_segment_size = ref_segment_size
        if self.style_encoder:
            self.style_dim = style_encoder.style_dim

    def training_step(
        self,
        x,
        x_lengths,
        spec,
        spec_lengths,
        cf0,
        vuv,
        duration=None,
        raw_texts=None,
    ):
        mel = self.spec_tfm.spec_to_mel(spec)
        if self.style_encoder and self.style_diffusion:
            ref_mel, _ = rand_slice_segments(mel, spec_lengths, self.ref_segment_size)
            cond = self.style_encoder(ref_mel)
        else:
            cond = None

        context, context_lengths = (
            self.context_embedder(raw_texts) if self.context_embedder else (None, None)
        )
        x, x_mask = self.text_encoder(
            x, x_lengths, context=context, context_lengths=context_lengths
        )
        phoneme_level_feature = x.detach().transpose(-1, -2)

        duration_handler_output: DurationHandlerOutput = self.duration_handler(
            x=x,
            x_mask=x_mask,
            x_lengths=x_lengths,
            mel=mel,
            mel_lengths=spec_lengths,
            duration=duration,
            cond=cond,
        )
        x_frame, p_attn, _, y_mask, loss_dict = duration_handler_output
        x_frame, m_p, logs_p = self.frame_prior_network(x_frame, y_mask, cond=cond)
        pitch_pred = self.pitch_predictor(x_frame, y_mask, cond=cond)
        log_cf0_pred, vuv_logit_pred = pitch_pred.chunk(2, dim=1)
        vuv_pred = vuv_logit_pred.sigmoid() * y_mask

        z, m_q, logs_q = self.posterior_encoder(spec, y_mask, cond=cond)
        z_p = self.flow(z, y_mask, cond=cond)

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        f0 = (
            cf0 * vuv
            if random.random() < 0.8
            else (log_cf0_pred.exp() * torch.where(vuv_pred > 0.5, 1.0, 0.0)).detach()
        )
        f0 = slice_segments(f0, ids_slice, self.segment_size)
        o, o_mb = self.vocoder(z_slice, f0, cond=cond)

        with torch.autocast(device_type="cuda", enabled=False):
            log_cf0 = to_log_scale(cf0)
            loss_cf0 = (
                F.mse_loss(log_cf0_pred * vuv, log_cf0 * vuv, reduction="sum") / spec_lengths.sum()
            )
            loss_vuv = F.mse_loss(vuv_pred, vuv, reduction="sum") / spec_lengths.sum()
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
            if self.style_encoder and self.style_diffusion:
                assert cond is not None, (
                    "Conditioning should not be None when style diffusion is used"
                )
                loss_diff = self.style_diffusion(
                    cond.detach().unsqueeze(1),
                    embedding=phoneme_level_feature,
                )
                recon = self.style_diffusion.sampler(
                    torch.randn(x.size(0), 1, self.style_dim, device=x.device),
                    num_steps=random.randint(3, 5),
                    embedding=phoneme_level_feature,
                    embedding_scale=1.0,
                )
                loss_diff_recon = F.l1_loss(cond, recon.squeeze(1))
            else:
                loss_diff = torch.tensor(0.0, device=x.device)
                loss_diff_recon = torch.tensor(0.0, device=x.device)
        loss_dict = loss_dict | dict(
            cf0=loss_cf0, vuv=loss_vuv, kl=loss_kl, diff=loss_diff, diff_recon=loss_diff_recon
        )
        return o, o_mb, ids_slice, p_attn, loss_dict

    def forward(self, x, x_lengths, raw_texts, noise_scale=0.667):
        context, context_lengths = (
            self.context_embedder(raw_texts) if self.context_embedder else (None, None)
        )
        x, x_mask = self.text_encoder(
            x, x_lengths, context=context, context_lengths=context_lengths
        )
        if self.style_diffusion:
            cond = self.style_diffusion.sampler(
                torch.randn(x.size(0), 1, self.style_dim, device=x.device),
                num_steps=5,
                embedding=x.transpose(-1, -2),
                embedding_scale=1.0,
            ).reshape(x.size(0), self.style_dim)
        else:
            cond = None

        x_frame, p_attn, duration, y_mask, _ = self.duration_handler.infer(x, x_mask, cond=cond)

        x_frame, m_p, logs_p = self.frame_prior_network(x_frame, y_mask, cond=cond)
        pitch_pred = self.pitch_predictor(x_frame * y_mask, y_mask, cond=cond)
        log_cf0_pred, vuv_logit_pred = pitch_pred.chunk(2, dim=1)
        cf0 = torch.exp(log_cf0_pred)
        vuv = torch.where(vuv_logit_pred.sigmoid() > 0.5, 1.0, 0.0)
        f0 = cf0 * vuv

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow.reverse(z_p * y_mask, y_mask, cond=cond)
        o, _ = self.vocoder(z * y_mask, f0, cond=cond)
        return o, (p_attn, f0, cf0, vuv, duration)
