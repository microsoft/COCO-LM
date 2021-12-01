# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    SelfMultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.in_proj_weight.data)
        # normal_(module.q_proj.weight.data)
        # normal_(module.k_proj.weight.data)
        # normal_(module.v_proj.weight.data)



def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = torch.sign(relative_position)
    num_buckets //= 2
    n = torch.abs(relative_position)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact
    max_bucket_val = num_buckets - 1 - max_exact
    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + torch.ceil(
        torch.log(n.float() / max_exact) / math.log((max_distance - 1) / max_exact) * (max_bucket_val)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret = torch.where(is_small, n, val_if_large) * sign
    return ret


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        share_embed_tokens: object = None,
        share_embed_positions: object = None,
        share_emb_layer_norm: object = None,
        shared_embedding_dim: int = 768,
        rel_pos: int = 0,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        checkpoint_activations: bool = False,
        offload_activations: bool = False
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.shared_embedding_dim = shared_embedding_dim
        self.num_attention_heads = num_attention_heads

        self.checkpoint_activations = checkpoint_activations
        self.offload_activations = offload_activations

        self.embed_linear = None
        if share_embed_tokens is None:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.embedding_dim, self.padding_idx
            )
        else:
            self.embed_tokens = share_embed_tokens
            if self.shared_embedding_dim != self.embedding_dim:
                self.embed_linear = nn.Linear(self.shared_embedding_dim, self.embedding_dim)
        
        if share_embed_positions is None:
            self.embed_positions = nn.Embedding(self.max_seq_len, self.embedding_dim)
        else:
            self.embed_positions = share_embed_positions
            
        # self.embed_tokens = self.build_embedding(
        #     self.vocab_size, self.embedding_dim, self.padding_idx
        # )
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        if share_emb_layer_norm is None:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = share_emb_layer_norm

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    encoder_normalize_before=encoder_normalize_before,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        self.rel_pos = True if rel_pos == 1 else False

        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.rel_pos_bins, self.num_attention_heads)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.rel_pos_bins,
                max_distance=self.max_rel_pos
            )
            self.rp_bucket -= self.rp_bucket.min()

    def get_rel_pos_bias(self, x):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != x.device:
            self.rp_bucket = self.rp_bucket.to(x.device)
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        encoder_normalize_before,
    ):
        layer = TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            encoder_normalize_before=encoder_normalize_before,
        )
        if self.checkpoint_activations:
            layer = checkpoint_wrapper(layer, offload_to_cpu=self.offload_activations)
        return layer

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_ext_padding_mask: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_tpu = tokens.device.type == "xla"

        # compute padding mask. This is needed for multi-head attention
        if is_tpu or self.traceable:
            padding_mask = tokens.eq(self.padding_idx)
        elif use_ext_padding_mask:
            padding_mask = padding_mask
        else:
            padding_mask = tokens.eq(self.padding_idx)
            if not padding_mask.any():
                padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.embed_linear is not None:
            x = self.embed_linear(x)

        if self.embed_scale is not None:
            x *= self.embed_scale

        seq_len = tokens.size(1)
        if self.embed_positions is not None:
            weight = self.embed_positions.weight[:seq_len, :]
            x += weight

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        rel_pos_bias = self.get_rel_pos_bias(tokens).repeat(tokens.size(0), 1, 1) if self.rel_pos else None
        if attn_mask is None:
            attn_mask = rel_pos_bias
        else:
            attn_mask += rel_pos_bias

        if attn_mask is not None and padding_mask is not None:
            # merge key_padding_mask and attn_mask
            attn_mask = attn_mask.view(tokens.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
            
        for layer in self.layers:   
            x, _ = layer(x, self_attn_padding_mask=padding_mask, attn_bias=attn_mask)
            if not last_state_only:
                inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
