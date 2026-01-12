BLEACH

Bangla Language Expert Adaptive Corpus Handler
A low-resource, dialect-aware sparse mixture-of-experts language model for Bangla.

Overview

BLEACH is a language modeling framework built to solve a practical problem in natural language processing: how to train powerful, dialect-aware Bangla models without the enormous compute budgets most modern models require. Instead of relying on brute-force scale, BLEACH uses a carefully designed sparse mixture-of-experts architecture that lets it match or beat much larger models while staying lean in both training cost and inference resource use.

This repository contains the code, data preprocessing scripts, evaluation tools, and visualizations needed to reproduce the key results.

Key Features

State-of-the-art language modeling
BLEACH achieves highly competitive perplexity on Bangla text, outperforming both large general-purpose models and specialized baselines, despite using far fewer resources.

Dialect awareness
The model naturally splits linguistic patterns into expert clusters that align with major Bangla dialect groups. This ability isn’t present in most off-the-shelf models and opens doors for dialect-focused research and applications.

Low-cost and efficient
Training and inference are feasible on modest hardware, with very low monetary cost and memory footprint, making BLEACH accessible to researchers and developers without access to massive compute.

Interpretable encoding
Expert routing patterns can be visualized and interpreted, providing insight into how the model differentiates between dialectal and linguistic structures.

Repository Structure

BLEACH-01(model_architecture-setup-and_vizualization).py: Core model architecture and routing visualization code.

preprocessing.ipynb: Dataset cleaning and preparation pipeline.

evaluation_results.json: Evaluation outputs used for benchmarking.

Visualization files (*.png): Dialect expert heatmaps, routing entropy, layer-wise expert use, etc.

Getting Started
Requirements

Clone the repository and install dependencies:

git clone https://github.com/Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler.git
cd BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler
pip install -r requirements.txt


Make sure you have Python 3.7+ and appropriate ML libraries (PyTorch, Transformers) installed.

Data Preparation

Before training or evaluation, run the preprocessing notebook to clean and tokenize your Bangla corpus. This step handles dialect tags and text normalization.

Training

Training scripts use a sparse mixture-of-experts setup. Check and modify hyperparameters in the model script for your hardware and dataset sizes.

Evaluation

Once trained, use the evaluation scripts to compute perplexity and F1 scores across dialect subsets.

Visualizations

The repository includes several graphs that illustrate:

Expert routing distribution across dialects.

Confidence and uncertainty in predictions.

Layer-wise expert engagement patterns.

These help you understand where and how BLEACH adapts linguistically.

Citation

If you use BLEACH in your work, please cite the associated paper and repository. You can mention the key improvements in perplexity and dialect performance that BLEACH achieves compared to much larger models.

Contribution

Contributions are welcome. Whether you want to improve preprocessing, add downstream tasks (classification, generation), or extend BLEACH to other languages, open an issue or a pull request.

License

The repository is open for research use. Please see the license file for details and attribution guidelines.
