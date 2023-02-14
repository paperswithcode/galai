import warnings
from typing import Union, List

import torch

from transformers import AutoTokenizer, OPTForCausalLM, StoppingCriteriaList, StoppingCriteria
from parallelformers import parallelize
import psutil

from galai.utils import escape_custom_split_sequence


__all__ = ["Model"]


class FinishedReferenceCriteria(StoppingCriteria):
    """
    A custom criteria to stop generation as soon as all the sequences in the batch have at least
    one [END_REF] marker after the prompt.
    """
    def __init__(self, prompt_length: int, end_ref_id: int):
        """
        Create a new criteria instance for a given generation run.

        Parameters
        ----------
        prompt_length : int
            The length of the prompt in tokens used to distinguish [END_REF] tokens in the prompt
            from the generated [END_REF] tokens. For a batch of multiple prompts of different
            lengths this should be the length of the longest prompt and other prompts should be
            padded.
        end_ref_id : int
            The [END_REF] token id.
        """
        self.prompt_length = prompt_length
        self.end_ref_id = end_ref_id

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        is_end_ref = (input_ids[:, self.prompt_length:] == self.end_ref_id)
        has_end_ref = is_end_ref.any(dim=-1)
        return has_end_ref.all()


class Model(object):
    """
    Model class holding the GALACTICA models. We configure a class to encapsulate the HuggingFace model,
    the tokenizer, and the specific tokenization logic for GALACTICA. For low-level access, we recommend
    using the standard HuggingFace API.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        num_gpus: int,
        tensor_parallel: bool = False,
    ):
        """
        Initializes a new model

        Parameters
        ----------
        name : str
            Model name, e.g. `standard`.

        dtype: torch.dtype
            Model weights type.

        num_gpus : int
            Number of GPUs to use for the inference. If 0 only a CPU is used. If a positive number
            n, then the first n CUDA devices are used.

        tensor_parallel : bool
            Specify if to use model tensor parallelizm. Ignored in CPU or single GPU inference.
        """

        self.name = name
        self.dtype = dtype
        self.is_loaded = False
        self.num_gpus = num_gpus
        self.tensor_parallel = tensor_parallel
        self.max_input_length = 2020
        self._master_port = None

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Loads the checkpoint for the model

        Parameters
        ----------
        checkpoint_path : str
            Path for the checkpoint (str)
        """

        # query available memory size of the GPUs we want to use. If tensor_parallel is True,
        # we just load the model's weights to RAM, as it needs to be sliced by parallelformers
        # before loading to VRAM.
        device_map = None
        max_memory = {}
        if self.num_gpus > 0 and not self.tensor_parallel:
            # based on https://github.com/huggingface/accelerate/blob/5315290b55ea9babd95a281a27c51d87b89d7c85/src/accelerate/utils/modeling.py#L274
            for i in range(self.num_gpus):
                _ = torch.tensor([0], device=i)
            for i in range(self.num_gpus):
                max_memory[i] = torch.cuda.mem_get_info(i)[0]
            device_map = "auto"
        max_memory["cpu"] = psutil.virtual_memory().available

        self.model = OPTForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.model.eval()

        if self.tensor_parallel:
            self._parallelize()

    def _parallelize(self) -> None:
        """
        Parallelize the model for a tensor-parallel multi-GPU inference.
        """

        if self.num_gpus < 2:
            warnings.warn("At least two GPUs are required to parallelize the model.", UserWarning)
            return

        self._master_port = 13000 + (id(self.model) % 32749)

        custom_policies = None
        if self.model.config.model_type == "opt" and not self.model.config.enable_bias:
            from galai.parallel_policy import OPTDecoderLayerPolicyNoBias
            custom_policies = [OPTDecoderLayerPolicyNoBias]

        parallelize(
            self.model, num_gpus=self.num_gpus, fp16=self.dtype == torch.float16,
            master_port=self._master_port,
            custom_policies=custom_policies,
        )

    def _set_tokenizer(self, tokenizer_path: str):
        """
        Configures the tokenizer for the model

        Parameters
        ----------
        tokenizer_path : str
            Path for the tokenizer (str)
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # setup padding
        tokenizer.pad_token_id = 1
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "left"

        # setup truncation
        tokenizer.truncation_side = "left"

        # setup special tokens
        tokenizer.bos_token_id = 0
        tokenizer.bos_token = "<s>"

        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"

        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = 3

        self.tokenizer = tokenizer

    def _tokenize(self, input_text: List[str], new_doc: bool) -> torch.LongTensor:
        """
        Apply custom preprocessing to input texts and tokenize them.

        Returns
        -------
            input_text : list[str]
                Texts to be tokenized
            new_doc : bool
                If True, prepends the end-of-document (</s>) token to each sequence and fixes
                padding.
        """
        texts = []
        for text in input_text:
            text = escape_custom_split_sequence(text)
            if not text:
                warnings.warn(
                    "Found an empty input text. Changing to end-of-document token instead.",
                    UserWarning
                )
                text = self.tokenizer.eos_token
            texts.append(text)

        if new_doc:
            pad_token = self.tokenizer.pad_token
            texts = [pad_token + t for t in texts]

        encoded = self.tokenizer(
            texts,
            padding="longest",
            max_length=self.max_input_length,
            truncation=True
        )
        context_tokens = encoded["input_ids"]
        input_v = torch.LongTensor(context_tokens).to(self.model.device)

        if new_doc:
            input_v[input_v[:, 0] == self.tokenizer.pad_token_id, 0] = self.tokenizer.eos_token_id
        return input_v

    @torch.inference_mode()
    def generate(
        self,
        input_text: Union[str, List[str]],
        max_length=None,
        max_new_tokens=None,
        new_doc=False,
        top_p=None,
        top_k=None,
        penalty_alpha=None,
        num_beams=1,
        num_return_sequences=1,
        return_full_text=True,
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generates text using the model

        Parameters
        ----------
        input_text : str or list[str]
            Input context for the model to use for its generation,
            e.g. "Attention Is All You Need [START_REF]"

        max_length : int (optional)
            Maximum length in tokens of the generated text (including prompt). Only one of
            max_length and max_new_tokens should be specified. If neither is set, then
            max_new_tokens is set to 60.

        max_new_tokens : int (optional)
            Maximum length in tokens of the generated text (excluding prompt). Only one of
            max_length and max_new_tokens should be specified. If neither is set, then
            max_new_tokens is set to 60.

        new_doc : bool
            If True, treats generation a new document, otherwise assumes generation could be
            anywhere within document. Use new_doc=True if you are generating documents, e.g.
            # Schwarzschild Radius, # Transformer (machine learning), 
            Title: Transformers, A Survey. For general prompting, turn off. Default is False.

        top_p : float or None
            If a number, e.g. 0.7, performs top p sampling. Default is None.

        top_k : int or None
            If a number, performs top k sampling (if penalty_alpha is None) or contrastive search
            decoding (if penalty_alpha > 0). Default is None.

        penalty_alpha : float or None
            If a positive number and top_k is set, performs contrastive search decoding with top_k
            candidates reranking. Default is None.

        num_beams : int, default 1
            Number of beams to use in beam search.

        num_return_sequences : int, default 1
            Number of generations to return for each prompt.

        Returns
        ----------
        str, list[str] or list[list[str]] - generated texts from the model. If input_text is a
            singe string, then the output is str if num_return_sequences == 1 or a list of
            strings if num_return_sequences > 1. If input_text is an iterable of strings, then the
            output is either a list of strings if num_return_sequences == 1 or a list of lists of
            strings, in which each inner list contains the generations for a given input prompt.
        """
        texts = [input_text] if isinstance(input_text, str) else input_text
        input_v = self._tokenize(texts, new_doc)
        options = {}
        if penalty_alpha is not None:
            options["penalty_alpha"] = penalty_alpha
            options["top_k"] = top_k
        else:
            if top_p is not None:
                options["do_sample"] = True
                options["top_p"] = top_p
            if top_k is not None:
                options["do_sample"] = True
                options["top_k"] = top_k

        if max_new_tokens is None and max_length is None:
            max_new_tokens = 60
        out = self.model.generate(
            input_v,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **options
        )

        out_tokens = out['sequences']
        if not return_full_text:
            out_tokens = out_tokens[:, input_v.shape[1]:]
        # we keep special tokens such as [START_REF] or <work>
        decoded = self.tokenizer.batch_decode(
            out_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # so we manually remove </s> and <pad>
        decoded = [
            text.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
            for text in decoded
        ]

        if num_return_sequences == 1:
            return decoded[0] if isinstance(input_text, str) else decoded
        if isinstance(input_text, str):
            return decoded
        else:
            return [
                decoded[num_return_sequences * i:num_return_sequences * (i+1)]
                for i in range(len(texts))
            ]

    @torch.inference_mode()
    def generate_reference(
        self,
        input_text: Union[str, List[str]],
        max_length=None,
        max_new_tokens=None,
        new_doc=False,
        top_p=None,
        suggestions=1,
        diversity_penalty=0.0,
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generates reference.

        Parameters
        ----------
        input_text : str or list[str]
            Input context for the model to use for its generation,
            e.g. "Attention Is All You Need [START_REF]"

        max_length : int (optional)
            Maximum length in tokens of the generated text (including prompt). Only one of
            max_length and max_new_tokens should be specified.

        max_new_tokens : int (optional)
            Maximum length in tokens of the generated text (excluding prompt). Only one of
            max_length and max_new_tokens should be specified. If neither is set, then
            max_new_tokens is set to 60.

        new_doc : bool
            If True, treats generation a new document, otherwise assumes generation could be
            anywhere within document. Use new_doc=True if you are generating documents, e.g.
            # Schwarzschild Radius, # Transformer (machine learning),
            Title: Transformers, A Survey. For general prompting, turn off. Default is False.

        top_p : float or None
            If None, uses greedy decoding. If a number, e.g. 0.7, performs top p sampling.
            Default is None.

        suggestions : int, default 1
            Number of suggestions to return for each input prompt. Uses beam search to return more
            suggestions. Ignored when sampling.

        diversity_penalty : float, default 0.0, ignored if sampling or suggestions == 1

        Returns
        ----------
        str, list[str] or list[list[str]] - generated reference suggestions from the model. If
            input_text is a singe string, then the output is str if suggestions == 1 or a list of
            strings if suggestions > 1. If input_text is an iterable of strings, then the output is
            either a list of strings if suggestions == 1 or a list of lists of strings, in which
            each inner list contains the suggestions for a given input prompt.
        """
        texts = [input_text] if isinstance(input_text, str) else input_text
        # append [START_REF] token if missing
        fixed_texts = []
        for text in texts:
            start_ref_pos = text.rfind("[START_REF]")
            if start_ref_pos == -1:
                fixed_texts.append(text + "[START_REF]")
            else:
                end_ref_pos = text.find("[END_REF]", start_ref_pos)
                if end_ref_pos != -1:
                    # the last [START_REF] is closed with [END_REF], let's add another one
                    fixed_texts.append(text + "[START_REF]")
                else:
                    # avoid spaces after [START_REF] token for better results
                    fixed_texts.append(text.rstrip())

        input_v = self._tokenize(fixed_texts, new_doc)

        prompt_length = input_v.shape[1]
        finished_reference_criteria = FinishedReferenceCriteria(
            prompt_length=prompt_length,
            end_ref_id=self.tokenizer.convert_tokens_to_ids("[END_REF]"),
        )

        if max_new_tokens is None and max_length is None:
            max_new_tokens = 60

        stopping_criteria = StoppingCriteriaList([finished_reference_criteria])
        if top_p is not None:
            out = self.model.generate(
                input_v,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=False,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=suggestions,
                stopping_criteria=stopping_criteria,
            )
        else:
            out = self.model.generate(
                input_v,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=suggestions,
                num_return_sequences=suggestions,
                num_beam_groups=suggestions if diversity_penalty > 0.0 else 1,
                diversity_penalty=diversity_penalty,
                return_dict_in_generate=True,
                output_hidden_states=False,
                stopping_criteria=stopping_criteria,
            )
        # cut-off the prompts
        generated_tokens = out["sequences"][:, prompt_length:]
        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        references = []
        unfinished_generation = False
        for text in decoded:
            end_ref_pos = text.find("[END_REF]")
            if end_ref_pos == -1:
                unfinished_generation = True
                references.append(text.strip())
            else:
                references.append(text[:end_ref_pos].strip())
        if unfinished_generation:
            warnings.warn(
                "At least one of the generated references may be incomplete. Consider increasing max_length or max_new_tokens.",
                UserWarning
            )

        if suggestions == 1:
            return references[0] if isinstance(input_text, str) else references
        if isinstance(input_text, str):
            return references
        else:
            return [
                references[suggestions * i:suggestions * (i+1)]
                for i in range(len(texts))
            ]
