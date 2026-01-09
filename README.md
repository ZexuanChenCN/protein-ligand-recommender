# Protein-Ligand Recommender
A contrastive learning-based recommendation system for protein-ligand pairs, supporting bidirectional retrieval (protein → ligand / ligand → protein) with efficient Maximum Inner Product Search (MIPS).

- Embedding Backbones:
  Protein embedding: Saprot (specialized for protein sequence representation)
  Ligand embedding: Chemberta (optimized for SMILES string encoding)
- Dataset: Built on BALM/BALM-benchmark`https://huggingface.co/datasets/BALM/BALM-benchmark`
- Traing Framework: CLIP-style contrastive learning to align protein/ligand embeddings
- Efficient Retrieval: MIPS-based inference for fast top-k recommendation
- Web Interface: A web for protein/ligand recommendation, just supprots the intranet of Westlake University. Web Site:`http://127.0.0.1:5000`(If no response, please send a email to zexuanchencn@163.com, I will run the web.py as fast as I can.)


- training: model.train.py
- Inference: inference.mips_retrieval.py
- Web: web.app.py
  
