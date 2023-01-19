from typing import Union

from galai.model import Model
from galai.utils import ModelInfo
import torch
import warnings
from pathlib import Path

HF_MAPPING = {
    "mini": ("facebook/galactica-125m", torch.float32),
    "base": ("facebook/galactica-1.3b", torch.float32),
    "standard": ("facebook/galactica-6.7b", torch.float32),
    "large": ("facebook/galactica-30b", torch.float32),
    "huge": ("facebook/galactica-120b", torch.float16)
}


def load_model(
    name: str,
    dtype: Union[str, torch.dtype] = None,
    num_gpus: int = None,
    parallelize: bool = False
):
    """
    Utility function for loading the model

    Parameters
    ----------
    name: str
        Name of the model

    dtype: str
        Optional dtype; default float32 for all models but 'huge'

    num_gpus : int (optional)
        Number of GPUs to use for the inference. If None, all available GPUs are used. If 0 (or if
        None and there are no GPUs) only a CPU is used. If a positive number n, then the first n CUDA
        devices are used.

    parallelize : bool; default False
        Specify if to use model tensor parallelizm. Ignored in CPU or single GPU inference.

        By the default (when parallelize is False) the multi-GPU inference is run using accelerate's
        pipeline parallelizm in which each GPU is responsible for evaluating a given subset of
        model's layers. In this mode evaluations are run sequentially. This mode is well suited for
        developing in model's internals as it is more robust in terms of recovering from exceptions
        due to not using additional processes. However, because of the sequential nature of
        pipeline parallelizm, at any given time only a single GPU is working.

        If parallelize is True, parallelformers' model tensor parallelizm is used instead.

    Returns
    ----------
    Model - model object
    """

    if name in HF_MAPPING:
        hf_model, default_dtype = HF_MAPPING[name]
        galai_model = True
    elif Path(name).exists():
        hf_model = name
        default_dtype = torch.float32
        galai_model = False
    else:
        raise ValueError(
            "Invalid model name. Must be one of 'mini', 'base', 'standard', 'large', 'huge', " +
            "a path to a local checkpoint dir, or a model name available on HuggingFace hub."
        )

    if dtype is None:
        dtype = default_dtype

    if isinstance(dtype, str):
        dtype = getattr(torch, "float16", None)
    if dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype: {dtype}"
        )

    if dtype == torch.bfloat16 and parallelize:
        raise ValueError(
            "Model tensor parallel does not support bfloat16 dtype. Use either dtype='float16' " +
            "or dtype='float32', or disable tenros parallelizm with parallelize=False."
        )

    if num_gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 0
    elif num_gpus > 0:
        # make sure CUDA is available
        if not torch.cuda.is_available():
            warnings.warn(
                "No CUDA support detected, falling back to CPU inference. If you want to run " +
                "inference on GPU make sure CUDA is configured correctly and pytorch is " +
                "installed with CUDA support. Set num_gpus=None to avoid this warning.",
                UserWarning
            )
            num_gpus = 0
        elif num_gpus > torch.cuda.device_count():
            available = torch.cuda.device_count()
            warnings.warn(
                f"num_gpus={num_gpus} is higher than the number of available CUDA devices. " +
                f"Setting it to {available}.",
                UserWarning
            )
            num_gpus = available
    if num_gpus > 1 and parallelize and galai_model:
        mi = ModelInfo.by_name(name)
        if mi.num_heads % num_gpus != 0:
            raise ValueError(
                f"With parallelize=True the number of model heads ({mi.num_heads} for '{name}' " +
                "model) must be divisible by the num_gpus. Adapt the number of GPUs, try a " +
                "different model or set parallelize=False"
            )
    if num_gpus <= 1 and parallelize:
        warnings.warn(
            "parallelize=True requires at least two GPUs. Setting it back to False.",
            UserWarning
        )
        parallelize = False

    model = Model(
        name=name,
        dtype=dtype,
        num_gpus=num_gpus,
        tensor_parallel=parallelize,
    )
    model._set_tokenizer(hf_model)
    model._load_checkpoint(checkpoint_path=hf_model)

    return model
