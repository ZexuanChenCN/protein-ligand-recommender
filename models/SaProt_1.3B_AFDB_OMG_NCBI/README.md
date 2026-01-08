---
license: mit
---
We further trained a 1.3 billion parameter version of the SaProt model, setting the context length to 1536 during training, and used a combined dataset of AFDB, OMG_prot50, and NCBI (70% identity filtering), totaling 383 million sequences. The training strategy is similar to that of [SaProt-O](https://github.com/westlake-repl/Denovo-Pinal/wiki/Tutorial), employing multimodal input integration (sequence and structural data) to ensure better alignment with real-world research applications. 

Specifically, the training data is a mixture of UniRef50 (40%), OMG (30%), and NCBI (30%). 

For sequences from OMG and NCBI lacking corresponding structural information, we employ mask language modeling where the model predicts the masked amino acid tokens. 

For the UniRef50 dataset, which includes structural data, we applied four distinct training strategies, each sampled with equal probability (25%):

- Predicting all amino acid tokens given partial masked structural tokens.
- Predicting all amino acid tokens given complete structural tokens.
- Predicting partial amino acid tokens given their amino acid token context and partial masked structural tokens.
- Predicting partial amino acid tokens given their amino acid token context and complete structural tokens.

SaProt_1.3B_AFDB_OMG_NCBI is  also a model very useful for protein editing. For instance, if you wish to modify certain regions of your protein—whether natural proteins or de novo designed—you can easily mask these amino acids by inputting partial or complete structures. Remarkably, the model functions effectively even if only sequence data is provided. If you have text data and would like to incorporate it, please refer to [SaProt-T/O](http://113.45.254.183:9527/). The relevant link can be found in the interface of [Pinal](http://www.denovo-pinal.com/).

### Loading model from huggingface 

> SaProt_1.3B_AFDB_OMG_NCBI, unlike  [SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2), **does not support** loading from the esm library

The following code shows how to load the model.

```
from transformers import EsmTokenizer, EsmForMaskedLM

model_path = "/your/path/to/SaProt_1.3B_AFDB_OMG_NCBI"
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)

#################### Example ####################
device = "cuda"
model.to(device)

seq = "M#EvVpQpL#VyQdYaKv" # Here "#" represents lower plDDT regions (plddt < 70)
tokens = tokenizer.tokenize(seq)
print(tokens)

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)
print(outputs.logits.shape)

"""
['M#', 'Ev', 'Vp', 'Qp', 'L#', 'Vy', 'Qd', 'Ya', 'Kv']
torch.Size([1, 11, 446])
"""
```