# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass
from fairseq.data import Dictionary

import unicodedata
import sys
import re

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class SentencepiecePreTokenizer(object):

    def __init__(self):
        self.transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”—–-",  u"'''\"\"---") ] )

    def handle_single_quote(self, tokens):
        line = ' '.join(tokens)
        line = re.sub(r"' ([smdSMDtT])\b", r"'\1", line)
        line = re.sub(r"' ll\b", "'ll", line)
        line = re.sub(r"' re\b", "'re", line)
        line = re.sub(r"' ve\b", "'ve", line)
        line = re.sub(r"' LL\b", "'LL ", line)
        line = re.sub(r"' RE\b", "'RE ", line)
        line = re.sub(r"' VE\b", "'VE ", line)
        return line.split()

    def split_on_cont_punc(self, tokens):
        new_tokens = []
        for token in tokens:
            if len(token) > 1:
                last_j = 0
                pre_is_punc = _is_punctuation(token[0])
                for j, ch in enumerate(token):
                    is_punc = _is_punctuation(ch)
                    if is_punc != pre_is_punc:
                        new_tokens.append(token[last_j: j])
                        last_j = j
                    pre_is_punc = is_punc
                if last_j < len(token):
                    new_tokens.append(token[last_j:])
            else:
                new_tokens.append(token)
        return new_tokens

    def split_pre_and_post_punc(self, tokens):
        def pre_punc(token):
            last_j = 0
            for j in range(1, len(token)):
                if not _is_punctuation(token[j]):
                    last_j = j
                    break
            return token[:last_j], token[last_j:]
        def post_punc(token):
            last_j = len(token)
            for j in range(len(token) - 2, -1, -1):
                is_punc = _is_punctuation(token[j])
                if not _is_punctuation(token[j]):
                    last_j = j + 1
                    break
            return token[:last_j], token[last_j:]
        new_tokens = []
        for token in tokens:
            if len(token) > 1 and _is_punctuation(token[0]):
                a, b = pre_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    if _is_punctuation(b[-1]):
                        c, d = post_punc(b)
                        if c:
                            new_tokens.append(c)
                        if d:
                            new_tokens.append(d)
                    else:
                        new_tokens.append(b)
            elif len(token) > 1 and _is_punctuation(token[-1]):
                a, b = post_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    new_tokens.append(b)
            else:
                new_tokens.append(token)
        return new_tokens

    def tokenize(self, line):
        line = line.strip()
        line = line.replace("``", '"').replace("''", '"')
        line = line.translate(self.transl_table)
        tokens = line.split()
        tokens = self.split_pre_and_post_punc(tokens)
        tokens = self.handle_single_quote(tokens)
        return tokens

@dataclass
class SentencepieceConfig(FairseqDataclass):
    sentencepiece_model: str = field(
        default="???", metadata={"help": "path to sentencepiece model"}
    )
    vocab: str = field(
        default="???", metadata={"help": "path to dict.txt"}
    )

@register_bpe("sentencepiece", dataclass=SentencepieceConfig)
class SentencepieceBPE(object):
    def __init__(self, cfg):
        sentencepiece_model = file_utils.cached_path(cfg.sentencepiece_model)
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
            self.pre_tokenizer = SentencepiecePreTokenizer()
            self.dictionary = Dictionary.load(cfg.vocab)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        return ' '.self.tokenize(x)

    def decode(self, x: str) -> str:
        return x.replace(' ', '').replace('\u2581', ' ').strip()
    
    
    def skip_space(self, tokens):
        new_tokens = []
        for i, token in enumerate(tokens):
            skip = False
            # skip single space, to reduce total length
            if token == '\u2581':
                if i == len(tokens) - 1 or _is_punctuation(tokens[i + 1][0]):
                    skip = True
            if not skip:
                new_tokens.append(token)
        return new_tokens

    def tokenize(self, x):
        x = ' '.join(self.pre_tokenizer.tokenize(x))
        tokens = self.sp.EncodeAsPieces(x)
        tokens = self.skip_space(tokens)
        return tokens

    def convert_tokens_to_ids(self, tokens: list):
        ret = []
        for token in tokens:
            ret.append(self.dictionary.index(token))
        return ret

    def convert_tokens_to_string(self, tokens: list):
        return self.decode(" ".join(tokens))

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")
