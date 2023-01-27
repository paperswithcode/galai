from parallelformers.policies.base import Layer, Policy
from parallelformers.utils.dist_utils import AllReduceLinear

from transformers.models.opt.modeling_opt import OPTDecoderLayer


__all__ = ["OPTDecoderLayerPolicyNoBias"]


class OPTDecoderLayerPolicyNoBias(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            "self_attn.embed_dim": config.hidden_size // world_size,
            "self_attn.num_heads": config.num_attention_heads // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="self_attn.q_proj.weight",
            ),
            Layer(
                weight="self_attn.k_proj.weight",
            ),
            Layer(
                weight="self_attn.v_proj.weight",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="self_attn.out_proj.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="fc1.weight",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="fc2.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return OPTDecoderLayer
