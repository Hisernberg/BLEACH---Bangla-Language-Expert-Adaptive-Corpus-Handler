# BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler
BLEACH  Bangla Language Expert Adaptive Corpus Handler A Low-Resource Dialect-Aware Sparse Mixture-of-Experts Language Model

Overview
BLEACH is a research-grade, low-resource Mixture-of-Experts (MoE) language modeling framework designed for Bangla dialectal NLP.
It introduces dialect-aligned expert specialization within a 50–60M parameter sparse transformer, enabling efficient, interpretable, and scalable modeling under free-tier compute constraints (Google Colab / Kaggle).

Each expert is explicitly aligned with a major Bangla dialect—Chittagong, Sylhet, Barishal, Noakhali, and Mymensingh—allowing BLEACH to learn dialect-specific linguistic patterns while maintaining a shared backbone for generalization.
This project targets ACL / EMNLP / NAACL-level research, with a strong emphasis on analysis, efficiency, and social impact.

Key Contributions
Dialect-Aligned Sparse MoE Architecture
Each expert specializes in a distinct Bangla dialect while sharing a common attention backbone.

Low-Resource MoE at 50–60M Parameters
Demonstrates that sparse MoE models can be trained effectively without large-scale infrastructure.
Efficient Training on Free Compute
Designed to run on a single T4 / P100 GPU using:
FP16 (AMP)
Gradient checkpointing
Sparse routing
Parameter-efficient tuning (LoRA / QLoRA)
Interpretable Routing & Analysis
Includes expert utilization analysis, routing heatmaps, and load-balancing diagnostics.

Dialect-Inclusive NLP
Advances NLP support for under-represented Bangla dialect communities.

Architecture Overview
Backbone: Transformer with RoPE positional embeddings
Experts: 5 FFN experts (~8–10M params each)

Routing:
Switch Transformer–style Top-1 routing
GShard-inspired capacity factor
Load-balancing auxiliary loss
Activations: SwiGLU / GLU
Parameter Efficiency: LoRA / QLoRA (experts only)

 Dataset
The model is trained on a cleaned and merged Bangla dialect corpus with explicit dialect labels.
Dialects covered:
Chittagong (ctg)
Sylhet (syl)
Barishal
Noakhali
Mymensingh

Data splits:
cleaned_bangla_train.csv
cleaned_bangla_val.csv
cleaned_bangla_test.csv
Each sample follows:

{
  "text": "string",
  "dialect": "Chittagong | Sylhet | Barisal | Noakhali | Mymensingh"
}

Evaluation & Analysis
BLEACH evaluates not only performance but system behavior:
Language Modeling: Perplexity
Generation: BLEU
Classification: Accuracy (dialect probing)

MoE Analysis:
Expert utilization
Routing entropy
Load-balancing efficiency
Sparse vs dense comparison

Generalization:
Cross-dialect evaluation
Low-resource robustness

Training Features:
Mixed-precision FP16 training (torch.cuda.amp)
Gradient checkpointing for memory efficiency
R-Drop regularization
Label smoothing
Expert dropout

Optional 4-bit activation simulation

Single-GPU only (no DeepSpeed / FSDP)

