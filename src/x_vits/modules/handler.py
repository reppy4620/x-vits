from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from x_vits.modules.alignment import viterbi_decode
from x_vits.utils.model import generate_path, length_to_mask, to_log_scale


class DurationHandlerOutput(NamedTuple):
    x_frame: torch.Tensor
    p_attn: torch.Tensor
    duration: torch.Tensor
    y_mask: torch.Tensor
    loss_dict: Optional[dict[str, torch.Tensor]] = None


class DurationHandler(nn.Module):
    def __init__(self, duration_predictor, alignment_module, length_regulator, duration_loss=None):
        super().__init__()
        self.duration_predictor = duration_predictor
        self.alignment_module = alignment_module
        self.length_regulator = length_regulator
        if duration_loss is None:
            self.duration_loss = nn.MSELoss(reduction="sum")
        else:
            self.duration_loss = duration_loss

    def infer(self, x, x_mask, cond=None):
        log_duration_pred = self.duration_predictor(x, x_mask, cond=cond)
        duration = log_duration_pred.exp().round()
        y_lengths = duration.sum(dim=[1, 2]).clamp_min(1).long()
        y_mask = length_to_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)
        attn_mask = (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)
        attn = generate_path(duration.squeeze(1), attn_mask)
        x_frame = x @ attn
        output = DurationHandlerOutput(
            x_frame=x_frame,
            p_attn=attn,
            duration=duration,
            y_mask=y_mask,
        )
        return output


class UnsupervisedDurationHandler(DurationHandler):
    def forward(self, x, x_mask, x_lengths, mel, mel_mask, mel_lengths, cond=None, **kwargs):
        log_duration_pred = self.duration_predictor(x.detach(), x_mask, cond=cond)

        log_p_attn = self.alignment_module(
            text=x.transpose(1, 2),
            feats=mel.transpose(1, 2),
            text_lengths=x_lengths,
            feats_lengths=mel_lengths,
            x_masks=x_mask.squeeze(1).bool().logical_not(),
        )
        duration, loss_bin = viterbi_decode(log_p_attn, x_lengths, mel_lengths)
        loss_forwardsum = self.forwardsum_loss(log_p_attn, x_lengths, mel_lengths)
        y_mask = length_to_mask(mel_lengths).unsqueeze(1).to(x_mask.dtype)
        x_frame, p_attn = self.length_regulator(
            hs=x.transpose(1, 2),
            ds=duration,
            h_masks=y_mask.squeeze(1).bool(),
            d_masks=x_mask.squeeze(1).bool(),
        )
        x_frame = x_frame.transpose(1, 2)
        p_attn = p_attn.transpose(1, 2)
        log_duration = to_log_scale(duration)
        loss_duration = (
            self.duration_loss(log_duration_pred.squeeze(1), log_duration) / x_lengths.sum()
        )
        output = DurationHandlerOutput(
            x_frame=x_frame,
            p_attn=p_attn,
            duration=duration.unsqueeze(1),
            y_mask=y_mask,
            loss_dict=dict(
                dur=loss_duration,
                bin=loss_bin,
                forwardsum=loss_forwardsum,
            ),
        )
        return output


class SupervisedDurationHandler(DurationHandler):
    def forward(self, x, x_mask, x_lengths, mel_lengths, duration, cond=None, **kwargs):
        log_duration_pred = self.duration_predictor(x.detach(), x_mask, cond=cond)

        y_mask = length_to_mask(mel_lengths).unsqueeze(1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn_mask = (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)
        attn = generate_path(duration.squeeze(1), attn_mask)

        x_frame = x @ attn
        log_duration = to_log_scale(duration)
        loss_duration = self.duration_loss(log_duration_pred, log_duration) / x_lengths.sum()
        output = DurationHandlerOutput(
            x_frame=x_frame,
            p_attn=attn,
            duration=duration,
            y_mask=y_mask,
            loss_dict=dict(dur=loss_duration),
        )
        return output
