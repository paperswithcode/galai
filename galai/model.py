import os
import torch

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from parallelformers import parallelize
from tokenizers import Tokenizer

from galai.architecture import OPTForCausalLM, OPTConfig
from galai.utils import escape_custom_split_sequence


class Model(object):
    """
    Model class holding the GALACTICA models. We configure a class to encapsulate the HuggingFace model,
    the tokenizer, and the specific tokenization logic for GALACTICA. For low-level access, we recommend
    using the standard HuggingFace API.
    """

    def __init__(self, name: str, dtype: str, num_gpus: int):
        """
        Initializes a new model

        Parameters
        ----------
        name : str
            Model name, e.g. `standard`.
        """
        self.name = name
        self.num_gpus = num_gpus
        self.dtype = dtype
        self.is_loaded = False

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Loads the checkpoint for the model

        Parameters
        ----------
        checkpoint_path : str
            Path for the checkpoint (str)
        """
        self.config = OPTConfig.from_pretrained(checkpoint_path)
        
        with init_empty_weights():
            self.model = OPTForCausalLM(self.config)
        
        self.model.tie_weights()

        device_map = {
            'decoder.embed_tokens': 0, 
            'decoder.embed_positions': 0, 
            'decoder.layer_norm': 0,
        }
        
        n_layers = self.config.num_hidden_layers
        
        for i in range(n_layers):
            device_map[f"decoder.layers.{i}"] = i * self.num_gpus // n_layers

        if 'mini' in checkpoint_path or 'base' in checkpoint_path:
            checkpoint_path = checkpoint_path + '/pytorch_model.bin'

        load_checkpoint_and_dispatch(
            self.model.model, 
            checkpoint_path, 
            device_map=device_map, 
            offload_folder=None, 
            dtype=self.dtype, 
            offload_state_dict=True
        )

        self.model.tie_weights()
        self.model.eval()

    def _set_tokenizer(self, tokenizer_path: str):
        """
        Configures the tokenizer for the model

        Parameters
        ----------
        tokenizer_path : str
            Path for the tokenizer (str)
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(direction="left", pad_id=1, pad_type_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=2020, direction="left")

    def generate(self, input_text: str, max_length=60, new_doc=False, top_p=None) -> str:
        """
        Generates text using the model

        Parameters
        ----------
        input_text : str
            Input context for the model to use for its generation, 
            e.g. "Attention Is All You Need [START_REF]"

        max_length: int
            Maximum length of the generated text

        new_doc : bool
            If True, treats generation a new document, otherwise assumes generation could be
            anywhere within document. Use new_doc=True if you are generating documents, e.g.
            # Schwarzschild Radius, # Transformer (machine learning), 
            Title: Transformers, A Survey. For general prompting, turn off. Default is False.

        top_p : float or None
            If None, uses greedy decoding. If a number, e.g. 0.7, performs top p sampling.
            Default is None.

        Returns
        ----------
        str - generated text from the model
        """
        texts = [escape_custom_split_sequence(input_text)]

        if new_doc:
            pad_id = self.tokenizer.padding["pad_id"]
            pad_token = self.tokenizer.id_to_token(pad_id)
            texts = [pad_token + t for t in texts]

        list_encoded = self.tokenizer.encode_batch(texts)
        context_tokens = [encoded.ids for encoded in list_encoded]
        input_v = torch.LongTensor(context_tokens).to(self.model.device)

        if top_p is not None:
            out = self.model.generate(
                input_v, 
                max_length=max_length, 
                return_dict_in_generate=True, 
                output_hidden_states=True,
                top_p=top_p,
                do_sample=True
            )
        else:
            out = self.model.generate(
                input_v, 
                max_length=max_length, 
                return_dict_in_generate=True, 
                output_hidden_states=True
            )
                
        return self.tokenizer.decode_batch(
            out['sequences'].tolist(), 
            skip_special_tokens=False)[0].lstrip('<pad>')
