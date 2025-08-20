# Medical Assistant

A comprehensive framework for training medical AI assistants using both decoder-only and encoder-decoder models. This project provides pipelines for fine-tuning language models on medical Q&A datasets with advanced evaluation metrics.

## The Approach

The data presented is akin to a *Sequence-to-Sequence* learning task. The data is too small to induce any knowledge, so the learning goal is mostly for the model to learn the question-answer format within the medical data domain. Considering the maximum size of the answers (~4000 tokens), it implies a *decoder-only* architecture. I chose a QWen-0.6B-base model as a starting point for the fine-tuning task on the provided data. For the sake of reducing the training time in a more manageable time-window, I ended up to reduce the context window to 256 tokens. As such, it becomes a "mock" training that doesn't capture the full scale of the actual training job that would need to happen in a real setting.

By reducing the sequence scale to maximum 256 tokens, using an *Encoder-Decoder* architecture becomes potentially more adapted. I used a T5-FLAN-small model (77M parameters) as a pretrained model for the fine-tuning job.

Both models are trained by computing the loss function only on the completion and the decoder-only model is trained with a LoRA adapter. I use BertScore as a validation metric measured on the generated answers compared to reference answers. Bertscore is good to validate that the model starts to use more of a medical jargon, but it will also provide a sense of domain knowledge in the responses provided. Considering the simple question-answer format following learning task, perplexity is also meaningful metric to consider. As expected, the 2025 500M parameter QWen model outperformed the 2021 77M parameter T5-FLAN model in both those scores!

Considering the time-constraint, no cross-validation, nor a full exploration of the training techniques could be performed beyond the ones provided

## The BERTScore

BERTScore computes semantic similarity using contextual embeddings with the following formulas:

**Step 1: Token Embeddings**
- Generate contextualized embeddings for each token using a pre-trained model (DeBERTa-XLarge-MNLI)
- Generated text: $\mathbf{x} = \{x_1, x_2, \ldots, x_k\}$ 
- Reference text: $\mathbf{y} = \{y_1, y_2, \ldots, y_l\}$

**Step 2: Similarity Matrix**
- Compute cosine similarity between all token pairs:

$$\text{sim}(x_i, y_j) = \frac{x_i \cdot y_j}{||x_i|| \cdot ||y_j||}$$

**Step 3: Precision, Recall, F1**

Precision (how well generated tokens match reference):
$$P = \frac{1}{k} \sum_{i=1}^{k} \max_{j} \text{sim}(x_i, y_j)$$

Recall (how well reference tokens are covered):
$$R = \frac{1}{l} \sum_{j=1}^{l} \max_{i} \text{sim}(x_i, y_j)$$

F1 Score (harmonic mean):
$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

**Why BERTScore for Medical Q&A:**
- Captures semantic similarity beyond exact word matches
- Better handles medical terminology and paraphrasing
- DeBERTa-XLarge-MNLI provides strong domain understanding
- Correlates better with human judgment than traditional metrics like BLEU

**Performance Interpretation:**
- 0.86+ (QWen): Excellent semantic similarity
- 0.56 (T5-FLAN): Moderate similarity, room for improvement
- Baseline: Random text ~0.3-0.4

## Features

- **Dual Training Modes**: Support for both decoder-only (QWen) and encoder-decoder (T5-FLAN) architectures
- **Parameter-Efficient Training**: LoRA adapters for decoder-only models to reduce computational requirements
- **Advanced Evaluation**: BERTScore F1 metrics for semantic similarity assessment
- **Data Processing**: Automated data loading, filtering, and preprocessing pipelines
- **Hugging Face Integration**: Seamless model and dataset management with Hub integration
- **Flexible Configuration**: Command-line interface for easy model selection

## Architecture

### Decoder-Only Training (QWen)
- Uses LoRA adapters (rank=64) for parameter-efficient fine-tuning
- Supervised fine-tuning with custom generation-based evaluation
- BERTScore evaluation using DeBERTa-XLarge-MNLI
- Completion-only loss computation

### Encoder-Decoder Training (T5-FLAN)
- Full seq2seq training with proper input/target tokenization
- DataCollator for sequence-to-sequence tasks
- Generation-based evaluation with BERTScore metrics
- Predict-with-generate mode for evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/damienbenveniste/Medical-assistant-model.git
cd Medical_Assistant

# Install dependencies using uv (recommended)
uv sync

# Or generate requirements.txt and use pip
uv export --format requirements-txt > requirements.txt
pip install -r requirements.txt
```

## Dependencies

- `datasets>=4.0.0` - Dataset loading and processing
- `transformers[torch]>=4.55.2` - Model architectures and training
- `torch>=2.8.0` - Deep learning framework
- `trl>=0.21.0` - Transformer Reinforcement Learning
- `peft>=0.17.0` - Parameter-Efficient Fine-Tuning
- `evaluate>=0.4.5` - Evaluation metrics
- `bert-score>=0.3.13` - Semantic similarity evaluation
- `huggingface-hub[cli]>=0.34.4` - Model hub integration
- `python-dotenv>=1.1.1` - Environment variable management

## Usage

### Basic Training

```bash
# Train decoder-only model (default)
python src/main.py

# Explicitly specify decoder-only training
python src/main.py --model-type decoder

# Train encoder-decoder model
python src/main.py --model-type encoder-decoder
```

### Configuration

Set environment variables in `.env` file:

```bash
HF_TOKEN=your_huggingface_token_here
```

### Data Format

The system expects CSV files with `question` and `answer` columns:

```csv
question,answer
"What are the symptoms of diabetes?","Common symptoms include increased thirst, frequent urination, and unexplained weight loss."
"How is hypertension treated?","Treatment typically involves lifestyle changes and medications like ACE inhibitors."
```

## Project Structure

```
src/
├── main.py              # Entry point with CLI interface
├── data.py              # Data loading and splitting utilities  
├── data_processing.py   # Tokenization and preprocessing
├── model.py             # Model loading and configuration
└─── training.py          # Training classes and evaluation

data/
└── medical_data.csv     # Training dataset

pyproject.toml           # Project configuration and dependencies
```

## Training Results

### QWen Model (Decoder-Only)

**Training Configuration:**
- Model: QWen with LoRA adapters (rank=64)
- Epochs: 3
- Batch Size: 8
- Max Length: 256 tokens

**Performance Metrics:**

| Epoch | Train Loss | Eval Loss | BERTScore F1 | Token Accuracy |
|-------|------------|-----------|--------------|----------------|
| 1     | 1.456      | 1.413     | 0.864        | 66.7%          |
| 2     | 1.338      | 1.384     | 0.865        | 67.3%          |
| 3     | 1.308      | 1.367     | 0.864        | 67.5%          |

**Key Observations:**
- Consistent improvement in training loss across epochs
- Stable BERTScore F1 around 0.86, indicating high semantic quality
- Token accuracy improved from 66.7% to 67.5%
- Model shows good convergence without overfitting

### T5-FLAN Model (Encoder-Decoder)

**Training Configuration:**
- Model: T5-FLAN (encoder-decoder)
- Epochs: 5
- Batch Size: 8
- Max Length: 256 tokens

**Performance Metrics:**

| Epoch | Train Loss | Eval Loss | BERTScore F1 |
|-------|------------|-----------|--------------|
| 1     | 2.210      | 1.551     | 0.541        |
| 2     | 1.626      | 1.476     | 0.560        |
| 3     | 1.569      | 1.477     | 0.557        |
| 4     | 1.539      | 1.383     | 0.553        |
| 5     | 1.525      | 1.406     | 0.561        |

**Key Observations:**
- Significant loss reduction from epoch 1 to 2 (2.21 → 1.63)
- Lower BERTScore compared to QWen (~0.56 vs 0.86)
- Stable convergence after epoch 2
- May require more epochs or different hyperparameters for optimal performance

## Model Comparison

| Aspect | QWen (Decoder-Only) | T5-FLAN (Encoder-Decoder) |
|--------|---------------------|---------------------------|
| **BERTScore F1** | 0.864 | 0.561 |
| **Training Efficiency** | High (LoRA) | Standard |
| **Memory Usage** | Lower | Higher |
| **Convergence** | 3 epochs | 5+ epochs needed |
| **Best Use Case** | General medical Q&A | Structured transformations |

## Trained Models

The following fine-tuned models are available on Hugging Face Hub:

### QWen Model (Decoder-Only)
- **Model ID**: `damienbenveniste/medical_assistant`
- **Hub URL**: https://huggingface.co/damienbenveniste/medical_assistant
- **Architecture**: QWen with LoRA adapters (rank=64)
- **Best Performance**: BERTScore F1 = 0.864

### T5-FLAN Model (Encoder-Decoder)  
- **Model ID**: `damienbenveniste/medical_assistant_seq2seq`
- **Hub URL**: https://huggingface.co/damienbenveniste/medical_assistant_seq2seq
- **Architecture**: T5-FLAN base
- **Best Performance**: BERTScore F1 = 0.561

### Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load QWen model
tokenizer = AutoTokenizer.from_pretrained("damienbenveniste/medical_assistant")
model = AutoModelForCausalLM.from_pretrained("damienbenveniste/medical_assistant")

# Load T5-FLAN model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("damienbenveniste/medical_assistant_seq2seq") 
model = AutoModelForSeq2SeqLM.from_pretrained("damienbenveniste/medical_assistant_seq2seq")
```

## API Reference

### DataConnector
Handles dataset loading and preprocessing:
- `load()`: Load and filter medical Q&A data
- `split()`: Create train/validation/test splits

### DataProcessor  
Tokenizes data for encoder-decoder models:
- `transform()`: Filter and tokenize dataset for seq2seq training

### SupervisedTrainer
Trains decoder-only models with LoRA:
- `add_adapter()`: Add LoRA adapter to model
- `train()`: Execute training pipeline
- `compute_metrics()`: Calculate BERTScore evaluation

### EncDecTrainer
Trains encoder-decoder models:
- `train()`: Execute seq2seq training pipeline
- `compute_metrics()`: Calculate BERTScore evaluation


