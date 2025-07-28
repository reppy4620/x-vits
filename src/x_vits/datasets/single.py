from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from x_vits.frontend.en import text_to_sequence as text_to_sequence_en
from x_vits.frontend.ja import text_to_sequence as text_to_sequence_ja
from x_vits.utils.const import LANGUAGE
from x_vits.utils.logging import logger


class SingleSpeakerDataset(Dataset):
    def __init__(self, df_file, wav_dir, cf0_dir, vuv_dir, spec_tfm, language="JP"):
        logger.info(f"Loading dataset... : {df_file}")
        self.wav_dir = Path(wav_dir)
        self.cf0_dir = Path(cf0_dir)
        self.vuv_dir = Path(vuv_dir)
        self.spec_tfm = spec_tfm
        if LANGUAGE.from_str(language) == LANGUAGE.JAPANESE:
            self.text_to_sequence = text_to_sequence_ja
        elif LANGUAGE.from_str(language) == LANGUAGE.ENGLISH:
            self.text_to_sequence = text_to_sequence_en
        else:
            raise ValueError(f"Invalid language: {language}")

        df = pd.read_csv(df_file)
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, phoneme_str, duration_str, _, raw_text = self.data[idx]
        bname = Path(fname).stem

        phoneme = self.text_to_sequence(phoneme_str)
        phoneme = torch.tensor(phoneme, dtype=torch.long)

        duration = [int(d) for d in duration_str.split()]
        duration = torch.tensor(duration, dtype=torch.float)

        wav_file = self.wav_dir / f"{bname}.wav"
        cf0_file = self.cf0_dir / f"{bname}.npy"
        vuv_file = self.vuv_dir / f"{bname}.npy"

        wav, _ = torchaudio.load(wav_file)
        spec = self.spec_tfm.to_spec(wav).squeeze(0)

        cf0 = torch.tensor(np.load(cf0_file), dtype=torch.float)
        vuv = torch.tensor(np.load(vuv_file), dtype=torch.float)
        assert cf0.shape[-1] == vuv.shape[-1]
        assert abs(spec.shape[-1] - cf0.shape[-1]) <= 1, (spec.shape[-1], cf0.shape[-1])
        cf0 = cf0[..., : spec.shape[-1]]
        vuv = vuv[..., : spec.shape[-1]]
        assert abs(duration.sum().item() - spec.shape[-1]) <= 1, (
            duration.sum().item(),
            spec.shape[-1],
        )
        return bname, phoneme, duration, spec, cf0, vuv, wav, raw_text

    def num_tokens(self, idx):
        return int(self.data[idx][3])

    def ordered_indices(self):
        lengths = np.array([int(x[3]) for x in self.data])
        indices = np.random.permutation(len(self))
        indices = indices[np.argsort(np.array(lengths)[indices], kind="mergesort")]
        return indices


class SingleSpeakerCollator:
    def __call__(self, batch):
        # I'm not sure whether using `torch.nn.utils.rnn.pad_sequence` is good or if the following method is better
        (bnames, phonemes, durations, specs, cf0s, vuvs, wavs, raw_texts) = zip(*batch)

        B = len(bnames)
        phone_lengths = [x.size(-1) for x in phonemes]
        frame_lengths = [x.size(-1) for x in specs]
        sample_lengths = [x.size(-1) for x in wavs]

        phone_max_length = max(phone_lengths)
        frame_max_length = max(frame_lengths)
        sample_max_length = max(sample_lengths)
        spec_dim = specs[0].size(0)

        phoneme_pad = torch.zeros(size=(B, phone_max_length), dtype=torch.long)
        duration_pad = torch.zeros(size=(B, 1, phone_max_length), dtype=torch.float)
        spec_pad = torch.zeros(size=(B, spec_dim, frame_max_length), dtype=torch.float)
        cf0_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        vuv_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        wav_pad = torch.zeros(size=(B, 1, sample_max_length), dtype=torch.float)
        for i in range(B):
            p_l, f_l, s_l = phone_lengths[i], frame_lengths[i], sample_lengths[i]
            phoneme_pad[i, :p_l] = phonemes[i]
            duration_pad[i, :, :p_l] = durations[i]
            spec_pad[i, :, :f_l] = specs[i]
            cf0_pad[i, :, :f_l] = cf0s[i]
            vuv_pad[i, :, :f_l] = vuvs[i]
            wav_pad[i, :, :s_l] = wavs[i]

        phone_lengths = torch.LongTensor(phone_lengths)
        frame_lengths = torch.LongTensor(frame_lengths)
        sample_lengths = torch.LongTensor(sample_lengths)

        return (
            bnames,
            phoneme_pad,
            duration_pad,
            spec_pad,
            cf0_pad,
            vuv_pad,
            wav_pad,
            phone_lengths,
            frame_lengths,
            sample_lengths,
            raw_texts,
        )
