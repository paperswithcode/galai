<p align="center">
    <br>
    <img src="https://github.com/paperswithcode/galai/raw/main/docs/source/img/logo.png#gh-dark-mode-only" width="400"/>
    <img src="https://github.com/paperswithcode/galai/raw/main/docs/source/img/logo_black.png#gh-light-mode-only" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/paperswithcode/galai/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/paperswithcode/galai.svg">
    </a>
    <a href="https://github.com/paperswithcode/galai/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/paperswithcode/galai.svg">
    </a>
</p>

**GALACTICA** is a general-purpose scientific language model. It is trained on a large corpus of scientific text and data. It can perform scientific NLP tasks at a high level, as well as tasks such as citation prediction, mathematical reasoning, molecular property prediction and protein annotation. More information is available at [galactica.org](https://galactica.org).

## Install

From pip:
    
```bash
pip install galai
```

From repository:
    
```bash
pip install git+https://github.com/paperswithcode/galai
```

## Models

There are five GALACTICA models available which we detail below:

|  Size       | Parameters  |
|:-----------:|:-----------:|
| `mini`      |    125 M    |
| `base`      |    1.3 B    |
| `standard`  |    6.7 B    |
| `large`     |     30 B    |
| `huge`      |    120 B    |

## Quickstart

```python
import galai as gal

model = gal.load_model("standard")
model.generate("Scaled dot product attention:\n\n\\[")
# Scaled dot product attention:\n\n\\[ \\displaystyle\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^{T}}{\\sqrt{d_{k}}}%\n)V \\]
```

## Hugging Face Integration

You can find all the model weights with their model cards and inference widget in the [Hugging Face Hub](https://huggingface.co/models?other=galactica). All the models can be used out of the box with the `transformers` library.

```bash
pip install transformers accelerate
```

You can run inference using the high-level `pipeline` API

```python
from transformers import pipeline

model = pipeline("text-generation", model="facebook/galactica-6.7b")
input_text = "The Transformer architecture [START_REF]"
model(input_text)
``` 

Or for more control you can use the lower level `OPTForCausalLM` class. See the model cards of the respective repo to learn how to use the model in CPU, GPU, and different precisions.

```python
from transformers import AutoTokenizer, OPTForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")

input_text = "The Transformer architecture [START_REF]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## Capabilities

GALACTICA is a stand-alone LM which is not instruction tuned. Because of this you need to use the correct prompts to get good results. In this note, we go over some of the special tokens, and prompt styles you will need to use to get good results.

We demonstrate some examples using the standard (6.7B) model below.

📚 **Predict Citations**:

You need to use `[START_REF]`:

```python
model.generate("The Transformer architecture [START_REF]")
# The Transformer architecture [START_REF] Attention is All you Need, Vaswani[END_REF] is a sequence-to-sequence model that uses self-attention to capture long-range dependencies between input and output tokens. The Transformer has been shown to achieve state-of-the-art results on a wide range of natural
```

🔢 **Predict LaTeX**:

```python
model.generate("The Schwarzschild radius is defined as: \\[")
# The Schwarzschild radius is defined as: \\[r_{s}=\\frac{2GM}{c^{2}}\\]\n\nwhere \\(G\\) is the gravitational constant, \\(M\\) is the mass of the black hole, and
```

🤔 **Reasoning**:

Reasoning uses the special `<work>` token:

```python
model.generate("A force of 0.6N is applied to an object, which accelerates at 3m/s. What is its mass? <work>")
# What force should be applied to accelerate an object of mass 3kg to 10m/s? <work>\nWe can use Newton's second law: F = ma. We can substitute variables to get:\n\n\\[ F = \\left(66kg
```

⚛️ **Generate Molecules**:

```python
model.generate("[START_I_SMILES]", max_length=200)
# [START_I_SMILES]CCC1=CC=C(C=C1)C(=O)NC2=CC=CC(=C2)C(=O)NC3=CC=C(C=C3)S(=O)(=O)N[END_I_SMILES]\n\n### Molecular Formula\n\nC22H21N3O4S\n\n## Chemical and Physical Properties\n\nThe following are chemical properties for 3-[[3-(4-ethylphenyl)-3-oxo-propanoyl]amino]-N-(4-sulfamoylphenyl)benzamide.\n\n### Computed Properties\n\n| Property Name | Property Value\n| --- | ----------- |\n| Molecular Weight | 423.5\n| XLogP3-AA Log P | 3.2\n| Hydrogen Bond Donor Count | 3\n| Hydrogen Bond Acceptor Count 
```

🧑‍🔬 **Predict Protein Annotations**:

```python
model.generate("[START_AMINO]GHMQSITAGQKVISKHKNGRFYQCEVVRLTTETFYEVNFDDGSFSDNLYPEDIVSQDCLQFGPPAEGEVVQVRWTDGQVYGAKFVASHPIQMYQVEFEDGSQLVVKRDDVYTLDEELP[END_AMINO] ## Keywords", max_length=200)
# '[START_AMINO]GHMQSITAGQKVISKHKNGRFYQCEVVRLTTETFYEVNFDDGSFSDNLYPEDIVSQDCLQFGPPAEGEVVQVRWTDGQVYGAKFVASHPIQMYQVEFEDGSQLVVKRDDVYTLDEELP[END_AMINO] ## Keywords\n\nCytoplasm, Methyltransferase, rRNA processing, S-adenosyl-L-methionine, Transferase\n\n## References\n\nQuestion: What are some articles for Ribosomal RNA small subunit methyltransferase H?\n\nAnswer: \n\n[START_REF] Comparative Genomics of 28 Salmonella enterica Isolates: Evidence for CRISPR-Mediated Adaptive Sublineage Evolution, Fricke[END_REF]\n\n</s>'
```

🖱️ **Free-Form Generation**

If you want autocomplete based functionality, it is often good to experiment with turning off `new_doc=True`. This makes it more likely for the model to think it is in the middle of a document, as opposed to the beginning.

```python
model.generate("The reason why Transformers replaced RNNs was because", new_doc=False)
```

❓ **Question Answering**
  
In the paper we prefix questions with "Q:" or "Question:". A typical format is "Question: question.\n\nAnswer:", for example:

```python
model.generate("Question: What is the notch signaling pathway?\n\nAnswer:")
```

📄 **Documents**
  
When starting a document, you must use the start document token for good results. To do this, set `new_doc=True` in generate:

For some article types, like Wikipedia style articles and GitHub repositories, use `#` to begin, e.g:
  
```python
model.generate("# Multi-Head Attention\n\n", new_doc=True)
```
  
For paper documents, use Title, e.g:

```python
model.generate("Title: Self-Supervised Learning, A Survey\n\n", new_doc=True)
```

You can also try alternative sampling techniques for less repetitions, e.g.

```python
model.generate("Lecture 1: The Ising Model\n\n", new_doc=True, top_p=0.7, do_sample=True, max_length=200)
# 'Lecture 1: The Ising Model\n\n# 13 Introduction\n\nWe will now look at a simple model for magnetism, the Ising model, which is\na lattice model in which we consider only two spin values, up or down, and\nwe want to understand how these spins interact with each other and how\nthey get arranged in a particular state.\n\nWe will first consider the one-dimensional case, and then move on to\nthe case of two-dimensional lattices, and then to higher dimensions.\n\n# 14 The One-Dimensional Ising Model\n\n# 14.1 The Model\n\nThe one-dimensional Ising model is the simplest case of the model, in\nwhich the lattice is a line of \\(N\\) spins, each with two possible spin\nvalues, up or down. In other words, we consider a line of \\(N\\) spins\nwhere each spin can point up or down'
```

## Citation

```bibtex
@inproceedings{GALACTICA,
    title={GALACTICA: A Large Language Model for Science},
    author={Ross Taylor and Marcin Kardas and Guillem Cucurull and Thomas Scialom and Anthony Hartshorn and Elvis Saravia and Andrew Poulton and Viktor Kerkez and Robert Stojnic},
    year={2022}
}
```
