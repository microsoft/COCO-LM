# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# The script is largely adapted from the huggingface transformers library


from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.nn import Parameter

from transformers.modeling_utils import PreTrainedModel, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
from transformers.models.bert.modeling_bert import ACT2FN
from transformers.file_utils import WEIGHTS_NAME
from transformers.activations import get_activation

from cocolm.configuration_cocolm import COCOLMConfig
from cocolm.convert_state_dict import get_checkpoint_from_transformer_cache

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as COCOLMLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm as COCOLMLayerNorm


COCOLM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'microsoft/cocolm-base': "https://huggingface.co/microsoft/cocolm-base/resolve/main/pytorch_model.bin",
    'microsoft/cocolm-large': "https://huggingface.co/microsoft/cocolm-large/resolve/main/pytorch_model.bin",
}

class COCOLMPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization 
        and a simple interface for dowloading and loading pretrained models.
    """
    config_class = COCOLMConfig
    supported_convert_pretrained_model_archive_map = {
        "cocolm": COCOLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    }
    base_model_prefix = "cocolm"
    pretrained_model_archive_map = {
        **COCOLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, COCOLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=True,
            drop_parameters=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'cocolm')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[
                    model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(
                        pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(
                        pretrained_model_name_or_path, WEIGHTS_NAME), map_location='cpu')
                    kwargs["state_dict"] = state_dict
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError(
                        "No pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("s2s-ft does't support the model !")
            raise NotImplementedError()

        state_dict = kwargs["state_dict"]
        _k = 'cocolm.embeddings.position_embeddings.weight'
        if _k in state_dict and "config" in kwargs:
            config = kwargs["config"]
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(
                    data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(
                    mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift +
                                               delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " %
                                (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(
                    data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(
                    mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(
                    state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        if drop_parameters is not None:
            if not isinstance(drop_parameters, list):
                raise RuntimeError()
            not_drop_state_dict = {}
            for key in state_dict:
                drop_flag = False
                for prefix in drop_parameters:
                    if key.startswith(prefix):
                        drop_flag = True
                        break

                if drop_flag:
                    logger.info("Drop %s" % key)
                else:
                    not_drop_state_dict[key] = state_dict[key]

            kwargs["state_dict"] = not_drop_state_dict
            del state_dict
        if pretrained_model_name_or_path in pretrained_model_archive_map:
            pretrained_model_name_or_path = pretrained_model_archive_map[pretrained_model_name_or_path]
        elif not os.path.isfile(pretrained_model_name_or_path):
            pretrained_model_name_or_path = 'microsoft/cocolm-large'  
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class COCOLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(COCOLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        fix_word_embedding = getattr(config, "fix_word_embedding", None)
        if fix_word_embedding:
            self.word_embeddings.weight.requires_grad = False
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.layer_norm_type = config.layer_norm_type
        if self.layer_norm_type in ('post',):
            self.LayerNorm = COCOLMLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        if self.token_type_embeddings:
            embeddings = embeddings + \
                self.token_type_embeddings(token_type_ids)

        if self.layer_norm_type in ('post',):
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids


class SelfMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.qk_head_dim = self.head_dim
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask,
        need_weights,
        attn_mask,
        attn_bias,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.qk_head_dim == self.head_dim:
            # , self.k_proj(query), self.v_proj(query)
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:
            q, k = self.qk_proj(query).chunk(2, dim=-1)
            v = self.v_proj(query)

        q = (
            q.contiguous().view(tgt_len, bsz * self.num_heads, self.qk_head_dim)
            .transpose(0, 1) * self.scaling
        )
        if k is not None:
            k = (
                k.contiguous().view(-1, bsz * self.num_heads, self.qk_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous().view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_bias is not None:
            attn_weights += attn_bias

        if before_softmax:
            return attn_weights, v

        attn_probs = nn.Softmax(dim=-1)(attn_weights)
        attn_probs = self.dropout(attn_probs)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        ret_attn_weights = None
        if need_weights:
            ret_attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                ret_attn_weights = attn_weights.mean(dim=0)

        return attn, ret_attn_weights


class COCOLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = SelfMultiheadAttention(config.hidden_size, config.num_attention_heads,
                                                dropout=config.attention_probs_dropout_prob)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask, attn_bias):
        x = x.transpose(0, 1)
        residual = x
        x, attn = self.self_attn(query=x, key_padding_mask=attention_mask, need_weights=True if self.config.output_attentions else False, attn_mask=None, attn_bias=attn_bias)
        x[x != x] = 0
        x = self.dropout(x)
        x = residual + x
        x = self.LayerNorm(x)
        if attn is not None:
            return (x.transpose(0, 1), ) + attn
        return (x.transpose(0, 1), )


class COCOLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class COCOLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm_type = config.layer_norm_type
        if self.layer_norm_type in ('post', 'hybrid'):
            self.LayerNorm = COCOLMLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.layer_norm_type == 'pre':
            return hidden_states + input_tensor
        elif self.layer_norm_type == 'hybrid':
            return hidden_states + self.LayerNorm(input_tensor) + input_tensor
        else:
            return self.LayerNorm(hidden_states + input_tensor)


class COCOLMLayer(nn.Module):
    def __init__(self, config):
        super(COCOLMLayer, self).__init__()
        self.attention = COCOLMAttention(config)
        if hasattr(config, 'num_ffn_layers') and config.num_ffn_layers > 1:
            self.num_ffn_layers = config.num_ffn_layers
            self.intermediate = nn.ModuleList(
                [COCOLMIntermediate(config) for _ in range(config.num_ffn_layers)])
            self.output = nn.ModuleList(
                [COCOLMOutput(config) for _ in range(config.num_ffn_layers)])
        else:
            self.num_ffn_layers = 0
            self.intermediate = COCOLMIntermediate(config)
            self.output = COCOLMOutput(config)

        self.layer_norm_type = config.layer_norm_type
        if self.layer_norm_type in ('pre', 'hybrid'):
            if self.num_ffn_layers > 0:
                self.LayerNorm = nn.ModuleList([COCOLMLayerNorm(
                    config.hidden_size, eps=1e-5) for i in range(self.num_ffn_layers)])
            else:
                self.LayerNorm = COCOLMLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, rel_pos)
        attention_output = self_attention_outputs[0]

        if isinstance(self.intermediate, nn.ModuleList):
            layer_output = attention_output
            for i, (intermediate_layer, output_layer) in enumerate(zip(self.intermediate, self.output)):
                if self.layer_norm_type in ('pre', 'hybrid'):
                    _attention_output = self.LayerNorm[i](layer_output)
                else:
                    _attention_output = layer_output
                intermediate_output = intermediate_layer(_attention_output)
                layer_output = output_layer(intermediate_output, layer_output)
        else:
            if self.layer_norm_type in ('pre', 'hybrid'):
                _attention_output = self.LayerNorm(attention_output)
            else:
                _attention_output = attention_output
            intermediate_output = self.intermediate(_attention_output)
            layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


class COCOLMEncoder(nn.Module):
    def __init__(self, config):
        super(COCOLMEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([COCOLMLayer(config)
                                    for _ in range(config.num_hidden_layers)])
        self.layer_norm_type = config.layer_norm_type
        if self.layer_norm_type in ('pre', 'hybrid'):
            self.LayerNorm = COCOLMLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                split_lengths=split_lengths, rel_pos=rel_pos)
            hidden_states = layer_outputs[0]

            if (self.layer_norm_type in ('pre', 'hybrid')) and (i == len(self.layer) - 1):
                # pre-layernorm: apply layernorm for the topmost hidden states
                hidden_states = self.LayerNorm(hidden_states)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class COCOLMBinaryPredictions(nn.Module):
    """Binary prediction module for the main model."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, hidden_states):
        logits = self.out_proj(hidden_states).squeeze(-1)
        return logits


class COCOLMCLMHead(nn.Module):
    def __init__(self, config):
        super(COCOLMCLMHead, self).__init__()
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        x = self.decoder(hidden_states)
        return x


class COCOLMSCLHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation("gelu")
        self.LayerNorm = COCOLMLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.LayerNorm(pooled_output)
        return pooled_output


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


class COCOLMModel(COCOLMPreTrainedModel):

    def __init__(self, config):
        super(COCOLMModel, self).__init__(config)
        self.config = config

        self.embeddings = COCOLMEmbeddings(config)
        self.encoder = COCOLMEncoder(config)
        
        if hasattr(config, 'need_pooler') and getattr(config, 'need_pooler'):
            self.scl_head = COCOLMSCLHead(config)
        else:
            self.scl_head = None

        if self.config.rel_pos_bins > 0:
            assert self.config.rel_pos_bins % 2 == 0
            self.relative_attention_bias = nn.Embedding(self.config.rel_pos_bins, self.config.num_attention_heads)
            context_position = torch.arange(self.config.max_position_embeddings, dtype=torch.long)[:, None]
            memory_position = torch.arange(self.config.max_position_embeddings, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.config.rel_pos_bins,
                max_distance=self.config.max_rel_pos
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, split_lengths=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = attention_mask == 0

        embedding_output, position_ids = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        embedding_output = embedding_output * (attention_mask.unsqueeze(-1).type_as(embedding_output))

        rel_pos_bias = self.get_rel_pos_bias(input_ids).repeat(input_ids.size(0), 1, 1) if self.config.rel_pos_bins > 0 else None
        seq_len = input_ids.size(1)

        if rel_pos_bias is not None and extended_attention_mask is not None:
            # merge key_padding_mask and attn_mask
            rel_pos_bias = rel_pos_bias.view(input_ids.size(0), -1, seq_len, seq_len)
            rel_pos_bias.masked_fill_(
                extended_attention_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )
            rel_pos_bias = rel_pos_bias.view(-1, seq_len, seq_len)
            extended_attention_mask = None

        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask,
            split_lengths=split_lengths, rel_pos=rel_pos_bias)
        sequence_output = encoder_outputs[0]

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, ) + encoder_outputs[1:]
        if self.scl_head is None:
            # sequence_output, pooled_output, (hidden_states), (attentions)
            return outputs
        else:
            pooled_output = self.scl_head(sequence_output)
            return sequence_output, pooled_output


class COCOLMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.cls_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class COCOLMForSequenceClassification(COCOLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.cocolm = COCOLMModel(config)
        self.classifier = COCOLMClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.cocolm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        logits = self.classifier(outputs[0])

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class PoolerLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.dense.bias.data.zero_()

    def forward(
        self, hidden_states, p_mask = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            x.masked_fill_(p_mask, float('-inf'))
        return x


class SQuADHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.start_logits = PoolerLogits(hidden_size)
        self.end_logits = PoolerLogits(hidden_size)

    def forward(
        self,
        hidden_states,
        start_positions=None,
        end_positions=None,
        p_mask = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the first token for the labeled span.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the last token for the labeled span.
            is_impossible (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        end_logits = self.end_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            def loss_fct(logits, targets):
                return F.nll_loss(
                    F.log_softmax(
                        logits.view(-1, logits.size(-1)),
                        dim=-1,
                        dtype=torch.float32,
                    ),
                    targets.view(-1),
                    reduction='sum',
                )
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) * 0.5
            return total_loss
        else:
            return start_logits, end_logits


class COCOLMForQuestionAnswering(COCOLMPreTrainedModel):

    def __init__(self, config):
        super(COCOLMForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.cocolm = COCOLMModel(config)
        self.qa_outputs = SQuADHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.cocolm(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        squad_outputs = self.qa_outputs(sequence_output, start_positions, end_positions)
        outputs = (squad_outputs,) + outputs[2:]

        return outputs