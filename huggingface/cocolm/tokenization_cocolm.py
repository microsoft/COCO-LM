# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# The script is largely adapted from the huggingface transformers library

import re
import os
import unicodedata

from transformers.tokenization_utils import PreTrainedTokenizer
from cocolm.tokenization_utils import Dictionary


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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


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

COCOLM_VOCAB_FILES_NAMES = {"vocab_file": "sp.model", "dict_file": "dict.txt"}

COCOLM_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cocolm-cased": "https://huggingface.co/microsoft/cocolm-base/resolve/main/sp.model",
    },
    "dict_file": {
        "cocolm-cased": "https://huggingface.co/microsoft/cocolm-base/resolve/main/dict.txt"
    }
}

COCOLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cocolm-cased": 512,
}

class COCOLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = COCOLM_VOCAB_FILES_NAMES
    pretrained_vocab_files_map = COCOLM_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = COCOLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, dict_file, **kwargs):
        super(COCOLMTokenizer, self).__init__(**kwargs)
        if not os.path.exists(vocab_file):
            raise EnvironmentError("file {} not found".format(vocab_file))
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab_file)
            self.pre_tokenizer = SentencepiecePreTokenizer()
            self.dictionary = Dictionary.load(dict_file)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')
        self.dictionary.add_symbol('<mask>')

    @property
    def cls_token(self):
        return self.dictionary.alias_mapper[self.dictionary.bos_word]

    @property
    def sep_token(self):
        return self.dictionary.alias_mapper[self.dictionary.eos_word]

    @property
    def pad_token(self):
        return self.dictionary.alias_mapper[self.dictionary.pad_word]

    @property
    def unk_token(self):
        return self.dictionary.alias_mapper[self.dictionary.unk_word]

    @property
    def cls_token_id(self):
        return self.dictionary.bos_index

    @property
    def sep_token_id(self):
        return self.dictionary.eos_index

    @property
    def pad_token_id(self):
        return self.dictionary.pad_index

    @property
    def mask_token_id(self):
        return self.dictionary.index('<mask>')

    @property
    def unk_token_id(self):
        return self.dictionary.unk_index

    def encode_plus(self, text_a, text_b=None, add_special_tokens=True, max_length=512):
        tokens_a = self.tokenize(text_a)
        if text_b is not None:
            tokens_b = self.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 4)
        else:
            if len(tokens_a) > max_length-2:
                tokens_a = tokens_a[:max_length-2]

        if add_special_tokens:
            tokens = [self.dictionary.bos_word] + tokens_a + [self.dictionary.eos_word]
            if text_b is not None:
                tokens += [self.dictionary.eos_word] + tokens_b + [self.dictionary.eos_word]
        else:
            tokens = tokens_a + tokens_b

        ids = self.convert_tokens_to_ids(tokens)
        return {"input_ids": ids}

    def encode(self, x: str, add_special_tokens=False) -> str:
        tokens = self.tokenize(x)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list) -> str:
        x = "".join([self._convert_id_to_token(token_id) for token_id in ids])
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
        if isinstance(tokens, str):
            return self.dictionary.index(tokens)
        for token in tokens:
            ret.append(self.dictionary.index(token))
        return ret
    
    def _convert_id_to_token(self, index):
        """ Converts a token (str) in an id using the vocab. """
        token = self.dictionary[index]
        return token

    def convert_tokens_to_string(self, tokens: list):
        x = " ".join(tokens)
        return x.replace(' ', '').replace('\u2581', ' ').strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")
