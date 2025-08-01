import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from x_vits.utils.const import ENGLISH_BERT_MODEL_NAME, JAPANESE_BERT_MODEL_NAME, LANGUAGE

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings


class ContextEmbedder(nn.Module):
    def __init__(self, language=LANGUAGE.JAPANESE):
        super().__init__()
        lang = LANGUAGE.from_str(language)
        if lang == LANGUAGE.JAPANESE:
            model_name = JAPANESE_BERT_MODEL_NAME
        elif lang == LANGUAGE.ENGLISH:
            model_name = ENGLISH_BERT_MODEL_NAME
        else:
            raise ValueError(f"Invalid language: {language}")
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

        self.register_buffer("dummy", torch.tensor(0))

        for p in self.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return self.dummy.device

    @torch.no_grad()
    def forward(self, raw_texts):
        inputs = self.tokenizer(raw_texts, return_tensors="pt", padding=True).to(self.device)
        context_lengths = inputs.attention_mask.sum(dim=1)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, context_lengths
