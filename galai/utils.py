import re
from typing import List
import math
import html

from dataclasses import dataclass


__all__ = [
    "escape_custom_split_sequence", "ModelInfo",
]


# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


REFERENCE_RE = re.compile(r"\[START_REF\](.*?)\[END_REF\]", flags=re.DOTALL)


def extract_references_from_text(text: str) -> List[str]:
    return [cit.strip() for cit in REFERENCE_RE.findall(text)]


@dataclass
class ModelInfo:
    name: str
    num_layers: int
    num_heads: int
    head_size: int = 128
    vocab_size: int = 50000
    max_positions: int = 2048

    @property
    def hidden_dimension(self) -> int:
        return self.head_size * self.num_heads

    @property
    def parameters(self) -> int:
        layer_norm_elementwise_affine = True
        enable_bias = True
        h_dim = self.hidden_dimension
        bias = h_dim if enable_bias else 0
        embed_tokens_size = self.vocab_size * h_dim
        embed_positions_size = (self.max_positions + 2) * h_dim
        layer_norm_size = 2 * h_dim if layer_norm_elementwise_affine else 0
        self_attn_size = 4 * (h_dim * h_dim + bias)  # 4 = k_proj+v_proj+q_proj+out_proj
        ffn_dim = 4 * h_dim
        fc_size = 2 * h_dim * ffn_dim + 5 * bias  # 2 = fc1 + fc2
        decoder_layer_size = self_attn_size + fc_size + 2 * layer_norm_size
        decoder_size = self.num_layers * decoder_layer_size + layer_norm_size + embed_tokens_size + embed_positions_size

        return decoder_size

    @property
    def disk_size(self) -> int:
        """Approximate dist size in bytes of checkpoints files"""
        return self.parameters * 2

    def weights_size(self, dtype="float16") -> int:
        """Approximate total size of model weights in memory"""
        element_size = 2 if dtype == "float16" else 4
        return self.parameters * element_size

    def memory_per_token(self, dtype="float16") -> int:
        """Approximate memory size required to store intermediate activations and cached outputs"""
        element_size = 2 if dtype == "float16" else 4
        return 2 * self.num_layers * self.num_heads * self.head_size * element_size

    @staticmethod
    def by_name(name: str) -> "ModelInfo":
        return _MODEL_INFO_BY_NAME[name]

    @staticmethod
    def all() -> List["ModelInfo"]:
        return _MODEL_INFO


def _humanize(parameters):
    scale = min(int(math.log10(parameters)) // 3, 4)
    suffix = " KMBT"[scale]

    return f"{parameters / math.pow(10, 3 * scale):.1f} {suffix}".rstrip()


class ModelInfoList(list):
    def _repr_html_(self):
        if not self:
            return ""
        columns = {
            "Name": lambda m: f"<strong>{html.escape(m.name)}</strong>",
            "Parameters": lambda m: _humanize(m.parameters),
            "Layers": lambda m: str(m.num_layers),
            "Heads": lambda m: str(m.num_heads),
            "Head Size": lambda m: str(m.head_size),
            "Vocabulary Size": lambda m: str(m.vocab_size),
            "Context Size": lambda m: str(m.max_positions),
        }
        output = ["<table><thead><tr>"]
        for col in columns:
            output.append(f"<th>{col}</th>")
        output.append("</tr></thead><tbody>")
        for mi in self:
            output.append("<tr>")
            for extractor in columns.values():
                output.append(f"<td>{extractor(mi)}</td>")
            output.append("</tr>")
        output.append("</tbody></table>")
        return "".join(output)


_MODEL_INFO = ModelInfoList([
    ModelInfo("mini",      num_layers=12, num_heads=12, head_size=64),
    ModelInfo("base",      num_layers=24, num_heads=32, head_size=64),
    ModelInfo("standard",  num_layers=32, num_heads=32, head_size=128),
    ModelInfo("large",     num_layers=48, num_heads=56, head_size=128),
    ModelInfo("huge",      num_layers=96, num_heads=80, head_size=128),
])

_MODEL_INFO_BY_NAME = {model.name: model for model in _MODEL_INFO}
