from galai.model import Model
import torch

HF_MAPPING = {
    "mini": ("facebook/galactica-125m", torch.float32),
    "base": ("facebook/galactica-1.3b", torch.float32),
    "standard": ("facebook/galactica-6.7b", torch.float32),
    "large": ("facebook/galactica-30b", torch.float32),
    "huge": ("facebook/galactica-120b", torch.float16)
}

def load_model(name: str):
    """
    Utility function for loading the model

    Parameters
    ----------
    name : str
        Name of the model

    dtype: str
        Optional dtype; default float32 for smaller models

    num_gpus: int
        Number of GPUs to use, default 8 GPUs

    Returns
    ----------
    Model - model object
    """

    if name not in HF_MAPPING:
        raise ValueError("Invalid model name. Must be one of 'mini', 'base', 'standard', 'large', 'huge'.")

    # TODO: consider the dtypes
    
    hf_model, dtype = HF_MAPPING[name]

    model = Model(name=name, dtype=dtype)
    model._set_tokenizer(hf_model)
    model._load_checkpoint(checkpoint_path=hf_model)

    return model
