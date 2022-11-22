# PromptBOOK

**GALACTICA** is a stand-alone LM which is not instruction tuned. Because of this you need to use the correct prompts to get good results. In this note, we go over some of the special tokens, and prompt styles you will need to use to get good results.

## Special Tokens

### Citations

To cite, you need to use `[START_REF]`.

```python
model.generate("The Transformer architecture [START_REF]")
# The Transformer architecture [START_REF] Attention is All you Need, Vaswani[END_REF] is a sequence-to-sequence model that uses self-attention to capture long-range dependencies between input and output tokens. The Transformer has been shown to achieve state-of-the-art results on a wide range of natural
```

### Reasoning

To try step-by-step reasoning, use `<work>`:

```python
model.generate("A force of 0.6N is applied to an object, which accelerates at 3m/s. What is its mass? <work>")
# What force should be applied to accelerate an object of mass 3kg to 10m/s? <work>\nWe can use Newton's second law: F = ma. We can substitute variables to get:\n\n\\[ F = \\left(66kg
```
  
### SMILES

For standard SMILES use `[START_SMILES]`

```python
model.generate("[START_SMILES]", top_p=0.6, max_length=200)
```

For Isomeric SMILES use `[START_I_SMILES]`:

```python
model.generate("[START_I_SMILES]", top_p=0.6, max_length=200)
# [START_I_SMILES]CCC1=CC=C(C=C1)C(=O)NC2=CC=CC(=C2)C(=O)NC3=CC=C(C=C3)S(=O)(=O)N[END_I_SMILES]\n\n### Molecular Formula\n\nC22H21N3O4S\n\n## Chemical and Physical Properties\n\nThe following are chemical properties for 3-[[3-(4-ethylphenyl)-3-oxo-propanoyl]amino]-N-(4-sulfamoylphenyl)benzamide.\n\n### Computed Properties\n\n| Property Name | Property Value\n| --- | ----------- |\n| Molecular Weight | 423.5\n| XLogP3-AA Log P | 3.2\n| Hydrogen Bond Donor Count | 3\n| Hydrogen Bond Acceptor Count 
```

### Protein Sequences
  
For protein sequences, use `[START_AMINO]`:

```python
model.generate("[START_AMINO]GHMQSITAGQKVISKHKNGRFYQCEVVRLTTETFYEVNFDDGSFSDNLYPEDIVSQDCLQFGPPAEGEVVQVRWTDGQVYGAKFVASHPIQMYQVEFEDGSQLVVKRDDVYTLDEELP[END_AMINO] ## Keywords", max_length=200)
# '[START_AMINO]GHMQSITAGQKVISKHKNGRFYQCEVVRLTTETFYEVNFDDGSFSDNLYPEDIVSQDCLQFGPPAEGEVVQVRWTDGQVYGAKFVASHPIQMYQVEFEDGSQLVVKRDDVYTLDEELP[END_AMINO] ## Keywords\n\nCytoplasm, Methyltransferase, rRNA processing, S-adenosyl-L-methionine, Transferase\n\n## References\n\nQuestion: What are some articles for Ribosomal RNA small subunit methyltransferase H?\n\nAnswer: \n\n[START_REF] Comparative Genomics of 28 Salmonella enterica Isolates: Evidence for CRISPR-Mediated Adaptive Sublineage Evolution, Fricke[END_REF]\n\n</s>'
```
  
## Documents
  
When starting a document, you must use the start document token for good results. To do this, set `new_doc=True` in generate:

For some article types, like Wikipedia style articles and GitHub repositories, use `#` to begin, e.g:
  
```python
model.generate("# Multi-Head Attention", new_doc=True)
```
  
For paper documents, use Title, e.g:

```python
model.generate("Title: Self-Supervised Learning, A Survey", new_doc=True)
```

## Free-Form Generation

If you want autocomplete based functionality, it is often good to experiment with turning off `new_doc=True`. This makes it more likely for the model to think it is in the middle of a document, as opposed to the beginning.

```python
model.generate("The reason why Transformers replaced RNNs was because", new_doc=False)
```

## Questions
  
In the paper we prefix questions with "Q:" or "Question:". A typical format is "Question: question.\n\nAnswer:", for example:

```python
model.generate("Question: What is the notch signaling pathway?\n\nAnswer:")
```
 
