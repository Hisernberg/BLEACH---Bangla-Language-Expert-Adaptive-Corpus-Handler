BLEACH

Bangla Language Expert Adaptive Corpus Handler
A low-resource, dialect-aware sparse Mixture-of-Experts language model for Bangla










Overview

BLEACH is a language modeling framework designed to address a core challenge in Bangla NLP: capturing dialectal variation under severe resource constraints. Rather than scaling model size, BLEACH adopts a sparse Mixture-of-Experts (MoE) architecture that enables dialect-specific specialization while keeping both training and inference costs extremely low.

Despite its compact footprint, BLEACH matches or outperforms much larger general-purpose and domain-specific models in language modeling quality, while uniquely offering explicit dialect awareness.

This repository provides the full experimental pipeline needed to reproduce the reported results, including preprocessing, training, evaluation, and interpretability analyses.

Key Features
State-of-the-Art Language Modeling

BLEACH achieves highly competitive perplexity on Bangla text, outperforming large general-purpose models and strong Bangla-specific baselines while using orders of magnitude fewer parameters.

Explicit Dialect Awareness

The model learns to route inputs to specialized experts that align with major Bangla dialect groups. This capability is absent from most existing Bangla language models and enables dialect-sensitive modeling and analysis.

Low-Cost and Efficient

Training and inference run on modest hardware, with minimal memory requirements and very low monetary cost, making BLEACH practical for researchers and developers without access to large compute clusters.

Interpretable Expert Routing

Routing patterns can be visualized and analyzed, offering insight into how the model organizes linguistic variation across dialects rather than treating Bangla as a single homogeneous language.

Results Summary
Language Modeling Performance
Model	Parameters	Perplexity ↓
BLEACH (ours)	~50–60M (sparse)	8.23
DeepSeek-V3	671B	9.20
BanglaLLaMA-7B	7B	12.40
BanglaRoBERTa	125M	19.30

BLEACH achieves the best perplexity despite being 2–3 orders of magnitude smaller than competing models.

Dialect Classification Performance
Model	Dialect-Aware	Macro F1 ↑
BLEACH (ours)	Yes	0.937
BanglaLLaMA-7B	No	–
DeepSeek-V3	No	–

BLEACH is the only evaluated model with explicit multi-dialect modeling capability.

Repository Structure
.
├── BLEACH-01(model_architecture-setup-and_vizualization).py
│   Core MoE architecture, expert routing, and visualization code
├── preprocessing.ipynb
│   Data cleaning, normalization, tokenization, and dialect tagging
├── evaluation_results.json
│   Stored evaluation outputs for reproducibility
├── *.png
│   Routing heatmaps, entropy plots, expert utilization figures

Getting Started
Requirements

Python 3.7 or later

PyTorch

HuggingFace Transformers

Clone the repository and install dependencies:

git clone https://github.com/Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler.git
cd BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler
pip install -r requirements.txt

Data Preparation

Run preprocessing.ipynb to clean and tokenize the Bangla corpus.
This step handles dialect labels, normalization, and dataset splitting.

Training

Training uses a sparse Mixture-of-Experts setup optimized for limited compute.
Hyperparameters can be adjusted in the main model script to match available hardware and dataset size.

Evaluation

Evaluation scripts compute:

Perplexity

Macro-averaged precision, recall, and F1

Routing entropy and expert utilization statistics

Results are saved in JSON format for reproducibility.

Visualizations

The repository includes visual analyses of:

Expert routing distribution across dialects

Layer-wise expert engagement

Model uncertainty and routing entropy

These figures help explain how BLEACH adapts linguistically, not just how well it performs.

Citation

If you use BLEACH in your research, please cite the associated paper and this repository.
You may reference its improvements in perplexity, efficiency, and dialect awareness relative to much larger models.

Contributions

Contributions are welcome. This includes:

Improved preprocessing or evaluation

Downstream tasks such as classification or generation

Extension to other low-resource or dialect-rich languages

Please open an issue or submit a pull request.

License

This project is released for research and educational use.
See the license file for details and attribution guidelines.
