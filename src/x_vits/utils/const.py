from enum import Enum

LRELU_SLOPE = 0.1

JAPANESE_BERT_MODEL_NAME = "sbintuitions/modernbert-ja-30m"
ENGLISH_BERT_MODEL_NAME = "microsoft/deberta-v3-xsmall"


class LANGUAGE(Enum):
    ENGLISH = "EN"
    JAPANESE = "JP"

    @classmethod
    def from_str(self, s):
        return dict(
            EN=LANGUAGE.ENGLISH,
            JP=LANGUAGE.JAPANESE,
        )[s]


class PreprocessType(Enum):
    JSUT = "JSUT"
    LJSPEECH = "LJSPEECH"

    @classmethod
    def from_str(self, s):
        return dict(
            JSUT=PreprocessType.JSUT,
            LJSPEECH=PreprocessType.LJSPEECH,
        )[s]
