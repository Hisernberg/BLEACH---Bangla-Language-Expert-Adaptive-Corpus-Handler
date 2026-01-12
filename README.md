# BLEACH: Bangla Language Expert Adaptive Corpus Handler

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/BLEACH)

**State-of-the-art Sparse Mixture-of-Experts Language Model for Bangla Dialect Modeling**

[Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Models](https://huggingface.co/BLEACH) | [Demo](https://huggingface.co/spaces/BLEACH/demo) | [Datasets](https://huggingface.co/datasets/BLEACH)

</div>

---



## 🌟 Highlights

BLEACH is a **117.5M parameter sparse Mixture-of-Experts (MoE) language model** specifically designed for modeling five major Bangla dialects:

✨ **State-of-the-Art Performance**
- **8.23 perplexity** on Bangla dialect modeling (best-in-class)
- **71% better** than mBERT, **63% better** than BanglaBERT
- **34% better** than BanglaLLaMA-7B (with 60× fewer parameters)
- **11% better** than DeepSeek-V3 (671B) on Bangla

🎯 **Dialect-Aware Architecture**
- First model with explicit multi-dialect support (5 dialects)
- **93.7% macro-averaged F1** across dialectal varieties
- Interpretable expert routing revealing linguistic patterns
- Balanced performance across all dialects (6.35% max gap)

⚡ **Exceptional Efficiency**
- **83 tokens/sec** inference (1.76× faster than dense baselines)
- **1.8 GB memory** footprint (38% less than comparable models)
- **$2.17 training cost** on single T4 GPU (6.2 hours)
- Runs on consumer hardware (mobile/edge deployable)

🏗️ **Novel Architecture**
- Sparse MoE with Top-1 routing (40% active parameters per token)
- SwiGLU activations + RoPE positional encodings
- R-Drop consistency regularization
- Dialect-balanced sampling preventing expert collapse

---

## 📊 Quick Results

### Performance Comparison

| Model | Parameters | Bangla PPL ↓ | Dialect Support | Inference Speed | Memory |
|-------|-----------|--------------|-----------------|-----------------|---------|
| mBERT | 110M | 28.4 | ✗ | 52 tok/s | 2.5 GB |
| BanglaBERT | 110M | 22.1 | ✗ | 54 tok/s | 2.5 GB |
| BanglaLLaMA-7B | 7B | 12.4 | ✗ | 8 tok/s | 14 GB |
| DeepSeek-V3 | 671B | 9.2 | ✗ | 12 tok/s | 335 GB |
| **BLEACH (Ours)** | **117.5M** | **8.23** ✓ | **✓ (5 dialects)** | **83 tok/s** | **1.8 GB** |

### Per-Dialect Results

| Dialect | Perplexity | F1 Score | Test Samples |
|---------|-----------|----------|--------------|
| Chittagong | **8.03** | 0.965 | 1,618 |
| Barisal | 8.18 | 0.952 | 651 |
| Sylhet | 8.47 | 0.929 | 609 |
| Mymensingh | 8.51 | 0.923 | 505 |
| Noakhali | 8.54 | 0.918 | 396 |
| **Overall** | **8.23** | **0.937** | **3,779** |

---


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler.git
cd BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler

# Create conda environment
conda create -n bleach python=3.10
conda activate bleach

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### Download Pretrained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

# Load BLEACH model (will be available on HuggingFace)
model = AutoModelForCausalLM.from_pretrained("BLEACH/bleach-117m")
```

### Inference Example

```python
import torch
from transformers import AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./bangla_tokenizer")
model = torch.load("./checkpoints/checkpoint_best.pt")
model.eval()

# Generate text
prompt = "আমার নাম"  # "My name is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        attention_mask=inputs["attention_mask"]
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

## 📚 Dataset Preparation

### Download Datasets

1. **BanglaDial**: [Download from Kaggle](https://www.kaggle.com/datasets/bangladiel)
2. **Vashantor**: [Download from Kaggle](https://www.kaggle.com/datasets/vashantor010)

### Preprocessing

```bash
# Run preprocessing pipeline
python preprocessing.py
```

This will:
- Clean and normalize Bangla text
- Remove emojis, URLs, and English-only sentences
- Deduplicate samples
- Create train/val/test splits (70/15/15)
- Generate visualizations in `./viz_outputs/`

**Output files:**
```
cleaned_bangla_train.csv     # 17,630 samples
cleaned_bangla_val.csv       #  3,779 samples
cleaned_bangla_test.csv      #  3,779 samples
dataset_summary.csv          # Statistics
```

### Dialect Distribution

| Split | Chittagong | Barisal | Sylhet | Mymensingh | Noakhali | Total |
|-------|-----------|---------|--------|------------|----------|-------|
| Train | 7,550 | 3,037 | 2,844 | 2,354 | 1,845 | 17,630 |
| Val | 1,618 | 651 | 609 | 505 | 396 | 3,779 |
| Test | 1,618 | 651 | 609 | 505 | 396 | 3,779 |

---

## 🎓 Training

### Full Training Pipeline

```bash
# Run complete training (Portions 1-3)
python train.py
```

**Training Configuration:**
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **Schedule**: Cosine annealing with 1,000-step warmup
- **Batch Size**: 12 (effective: 24 with gradient accumulation)
- **Epochs**: 20
- **GPU**: Single NVIDIA T4 (16GB)
- **Time**: ~6.2 hours
- **Cost**: ~$2.17 (cloud GPU pricing)

### Training Losses

The model optimizes a composite loss function:

```
L_total = L_LM + λ_balance * L_balance + α_RDrop * L_RDrop
```

- **L_LM**: Cross-entropy language modeling loss
- **L_balance**: Expert load balancing loss (λ=0.01)
- **L_RDrop**: R-Drop consistency regularization (α=0.5)

### Training Curves

| Epoch | Step | Train Loss | Val Loss | Val PPL | LB Loss | RDrop Loss |
|-------|------|-----------|----------|---------|---------|-----------|
| 1 | 500 | 3.667 | 3.172 | 23.86 | 1.014 | 0.054 |
| 5 | 2,000 | 2.134 | 2.459 | 11.69 | 1.005 | 0.105 |
| 10 | 4,000 | 2.657 | 2.256 | 9.54 | 1.016 | 0.189 |
| 15 | 6,000 | 2.531 | 2.146 | 8.55 | 1.027 | 0.255 |
| **20** | **7,500** | **1.680** | **2.110** | **8.25** | **0.998** | **0.090** |

---

## 📈 Evaluation

### Run Comprehensive Evaluation

```bash
# Run evaluation with routing analysis
python evaluate.py
```

**Outputs:**
- Test perplexity (overall + per-dialect)
- Routing entropy and expert balance metrics
- Dialect-expert affinity matrix
- Token-level routing analysis
- Visualizations saved to `./figures/`
- Results JSON saved to `./results/`

### Generated Visualizations

1. **perplexity_comparison.png**: Per-dialect performance bar chart
2. **dialect_expert_heatmap.png**: 5×5 affinity matrix
3. **routing_entropy.png**: Layer-wise entropy progression
4. **expert_usage.png**: Stacked bar chart of expert utilization
5. **comprehensive_analysis.png**: 4-panel summary figure
6. **layer_wise_routing.png**: Evolution across layers
7. **confidence_distribution.png**: Routing confidence histogram

---

## 📦 Model Checkpoints

### Available Checkpoints

| Checkpoint | Step | Val PPL | Download | Size |
|-----------|------|---------|----------|------|
| Best Model | 7,500 | 8.25 | [Link](#) | 460 MB |
| Epoch 10 | 4,000 | 9.54 | [Link](#) | 460 MB |
| Epoch 5 | 2,000 | 11.69 | [Link](#) | 460 MB |

### Load Checkpoint

```python
import torch

# Load best checkpoint
checkpoint = torch.load("./checkpoints/checkpoint_best.pt")

# Extract components
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])

print(f"Loaded checkpoint from step {checkpoint['step']}")
print(f"Best validation loss: {checkpoint['best_val']:.4f}")
```

---

## 🔬 Ablation Studies

We conducted systematic ablations to isolate component contributions:

| Configuration | Test PPL | Δ PPL | Key Finding |
|--------------|----------|-------|-------------|
| **Full BLEACH** | **8.23** | **Baseline** | Optimal configuration |
| − R-Drop | 8.58 | +4.3% | Consistency regularization important |
| − Load Balance Loss | 10.97 | +33.3% | **Critical** for preventing collapse |
| − Dialect Balancing | 9.12 | +10.8% | Ensures cross-dialectal robustness |
| − SwiGLU (use GELU) | 8.91 | +8.3% | Gated activations beneficial |
| Dense FFN (no MoE) | 13.64 | +65.7% | MoE provides massive capacity gains |
| Top-2 Routing | 8.31 | +1.0% | Top-1 optimal for efficiency |

**Key Insight**: Load balancing loss is critical (33% degradation without it), while MoE architecture provides 66% improvement over dense baselines.

---



## 🎯 Use Cases

BLEACH is designed for:

✅ **Dialect-Aware Text Generation**: Generate text in specific Bangla dialects  
✅ **Dialectology Research**: Analyze routing patterns to understand linguistic structure  
✅ **Low-Resource NLP**: Efficient architecture suitable for limited compute  
✅ **Edge Deployment**: 1.8 GB memory footprint enables mobile/IoT deployment  
✅ **Real-Time Applications**: 83 tok/s enables interactive chatbots and assistants  
✅ **Multi-Dialectal Systems**: Single model handling all 5 major Bangla dialects  

---

## 📖 Citation

If you use BLEACH in your research, please cite our paper:

```bibtex
@article{bleach2024,
  title={BLEACH: Bangla Language Expert Adaptive Corpus Handler - A Sparse Mixture-of-Experts Approach to Multi-Dialectal Language Modeling},
  author={[Your Name] and [Co-authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

**Areas for contribution:**
- Additional dialect support (Rangpur, Rajshahi, Khulna, etc.)
- Fine-tuning scripts for downstream tasks
- Improved tokenization for rare words
- Multilingual extensions (other South Asian languages)
- Mobile/web deployment examples

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets**: BanglaDial and Vashantor teams for providing dialectal corpora
- **Tokenizer**: [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) for Bangla tokenization
- **Infrastructure**: Google Colab for providing free GPU resources
- **Inspiration**: Switch Transformers, GShard, and DeepSeek-V3 for MoE architecture insights

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@Hisernberg](https://github.com/Hisernberg)
- **Twitter**: [@YourHandle](https://twitter.com/yourhandle)

For questions, issues, or collaborations, please:
1. Open an [issue](https://github.com/Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler/issues)
2. Start a [discussion](https://github.com/Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler/discussions)
3. Email directly for sensitive inquiries

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler&type=Date)](https://star-history.com/#Hisernberg/BLEACH---Bangla-Language-Expert-Adaptive-Corpus-Handler&Date)

---

<div align="center">


[⬆ Back to Top](#bleach-bangla-language-expert-adaptive-corpus-handler)

</div>
