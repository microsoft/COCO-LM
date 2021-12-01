#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentencepiece-model",
        help='path to encoder.json',
    )
    parser.add_argument('--vocab', type=str)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()


    encoder = MultiprocessingEncoder(args)
    with Pool(args.workers, initializer=encoder.initializer) as pool:
        for text in pool.imap(encoder.encode, sys.stdin, chunksize=4096):
            sys.stdout.write(text)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = SentencepieceBPE(self.args)

    def encode(self, line):
        global bpe
        enc_line = ' '.join(bpe.tokenize(line))
        return enc_line + '\n'

if __name__ == "__main__":
    main()
