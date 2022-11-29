from galai.model import Model
import torch

HF_MAPPING = {
    "mini": ("facebook/galactica-125m", torch.float32),
    "base": ("facebook/galactica-1.3b", torch.float32),
    "standard": ("facebook/galactica-6.7b", torch.float32),
    "large": ("facebook/galactica-30b", torch.float32),
    "huge": ("facebook/galactica-120b", torch.float16)
}


def load_model(name: str, dtype: str=None):
    """
    Utility function for loading the model

    Parameters
    ----------
    name : str
        Name of the model

    dtype: str
        Optional dtype; default float32 for all models but 'huge'

    Returns
    ----------
    Model - model object
    """

    if name not in HF_MAPPING:
        raise ValueError("Invalid model name. Must be one of 'mini', 'base', 'standard', 'large', 'huge'.")

    hf_model, default_dtype = HF_MAPPING[name]

    model = Model(name=name, dtype=default_dtype if dtype is None else dtype)
    model._set_tokenizer(hf_model)
    model._load_checkpoint(checkpoint_path=hf_model)

    return model
