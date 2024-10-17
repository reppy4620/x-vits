# adapted from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/jets/alignments.py
# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from scipy.stats import betabinom


class AlignmentModule(nn.Module):
    """Alignment Learning Framework proposed for parallel TTS models in:

    https://arxiv.org/abs/2108.10447

    """

    def __init__(self, adim, odim, cache_prior=True):
        """Initialize AlignmentModule.

        Args:
            adim (int): Dimension of attention.
            odim (int): Dimension of feats.
            cache_prior (bool): Whether to cache beta-binomial prior.

        """
        super().__init__()
        self.cache_prior = cache_prior
        self._cache = {}

        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

        self.f_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, text_lengths, feats_lengths, x_masks=None):
        """Calculate alignment loss.

        Args:
            text (Tensor): Batched text embedding (B, T_text, adim).
            feats (Tensor): Batched acoustic feature (B, T_feats, odim).
            text_lengths (Tensor): Text length tensor (B,).
            feats_lengths (Tensor): Feature length tensor (B,).
            x_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Log probability of attention matrix (B, T_feats, T_text).

        """
        text = text.transpose(1, 2)
        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose(1, 2)

        feats = feats.transpose(1, 2)
        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose(1, 2)

        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = torch.norm(dist, p=2, dim=3)
        score = -dist

        if x_masks is not None:
            x_masks = x_masks.unsqueeze(-2)
            score = score.masked_fill(x_masks, -np.inf)

        log_p_attn = F.log_softmax(score, dim=-1)

        # add beta-binomial prior
        bb_prior = self._generate_prior(
            text_lengths,
            feats_lengths,
        ).to(dtype=log_p_attn.dtype, device=log_p_attn.device)
        log_p_attn = log_p_attn + bb_prior

        return log_p_attn

    def _generate_prior(self, text_lengths, feats_lengths, w=1) -> torch.Tensor:
        """Generate alignment prior formulated as beta-binomial distribution

        Args:
            text_lengths (Tensor): Batch of the lengths of each input (B,).
            feats_lengths (Tensor): Batch of the lengths of each target (B,).
            w (float): Scaling factor; lower -> wider the width.

        Returns:
            Tensor: Batched 2d static prior matrix (B, T_feats, T_text).

        """
        B = len(text_lengths)
        T_text = text_lengths.max()
        T_feats = feats_lengths.max()

        bb_prior = torch.full((B, T_feats, T_text), fill_value=-np.inf)
        for bidx in range(B):
            T = feats_lengths[bidx].item()
            N = text_lengths[bidx].item()

            key = str(T) + "," + str(N)
            if self.cache_prior and key in self._cache:
                prob = self._cache[key]
            else:
                alpha = w * np.arange(1, T + 1, dtype=float)  # (T,)
                beta = w * np.array([T - t + 1 for t in alpha])
                k = np.arange(N)
                batched_k = k[..., None]  # (N,1)
                prob = betabinom.logpmf(batched_k, N, alpha, beta)  # (N,T)

            # store cache
            if self.cache_prior and key not in self._cache:
                self._cache[key] = prob

            prob = torch.from_numpy(prob).transpose(0, 1)  # -> (T,N)
            bb_prior[bidx, :T, :N] = prob

        return bb_prior


@jit(nopython=True)
def _monotonic_alignment_search(log_p_attn):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A


def viterbi_decode(log_p_attn, text_lengths, feats_lengths):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search(cur_log_p_attn.detach().cpu().numpy())
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


# adapted from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/jets/length_regulator.py
# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
class GaussianUpsampling(torch.nn.Module):
    """Gaussian upsampling with fixed temperature as in:

    https://arxiv.org/abs/2010.04301

    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """Upsample hidden states according to durations.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor): Mask tensor (B, T_feats).
            d_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        """
        B = ds.size(0)
        device = ds.device

        if ds.sum() == 0:
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if h_masks is None:
            T_feats = ds.sum().int()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)
        return hs, p_attn


class HardAlignmentUpsampler(nn.Module):
    def forward(self, x, attn):
        """Normal upsampler

        Args:
            x (torch.Tensor): [B, C, P]
            attn (torch.Tensor): [B, P, T]

        Returns:
            torch.Tensor: [B, C, T]
        """
        return x @ attn
