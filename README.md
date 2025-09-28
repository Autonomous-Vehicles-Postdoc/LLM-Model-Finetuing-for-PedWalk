
# LLM Fine-tuning with LoRA: LLaMA 2 Educational Implementation

**By: Dr. Basharat Hussain**  
**Topic: Large Language Model Fine-tuning using Parameter-Efficient Methods**

## 📋 Table of Contents
- [Overview](#overview)
- [Technical Approach](#technical-approach)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Configuration Details](#configuration-details)
- [Results and Evaluation](#results-and-evaluation)
- [Common Issues & Solutions](#common-issues--solutions)
- [Educational Objectives](#educational-objectives)
- [References](#references)

## 🎯 Overview

This project demonstrates **Parameter-Efficient Fine-tuning** of Large Language Models using **LoRA (Low-Rank Adaptation)** technique. We fine-tune Meta's LLaMA 2 7B model on a custom instruction-following dataset to showcase how modern LLMs can be adapted for specific tasks without requiring massive computational resources.

### Key Features
- **Model**: LLaMA 2 7B Chat model
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Dataset**: 104 custom instruction-response pairs
- **Hardware**: Single GPU (Tesla T4/V100/A100)
- **Memory Efficient**: 4-bit quantization + LoRA adapters

## 🔬 Technical Approach

### Parameter-Efficient Fine-tuning (PEFT)
Instead of fine-tuning all 7 billion parameters, we use **LoRA** which:
- Freezes the original model weights
- Adds small trainable adapter modules
- Reduces trainable parameters from 7B to ~5.3M (0.29%)
- Maintains model performance while being memory efficient

### QLoRA (Quantized LoRA)
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Double Quantization**: Further compression for efficiency
- **NF4 Data Type**: Optimized for neural network weights
- **BFloat16 Compute**: Maintains numerical stability

### Architecture Details
```
Base Model: LLaMA 2 7B (1.8B trainable params after quantization)
LoRA Rank (r): 8
LoRA Alpha: 32
Target Modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
Dropout: 0.05
```

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: CUDA-compatible with 15+ GB VRAM (Tesla T4 minimum)
- **RAM**: 16+ GB system memory
- **Storage**: 15+ GB free space

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.0+
- Google Colab (recommended) or local Jupyter environment

## 🚀 Installation

### Option 1: Google Colab (Recommended)
```bash
# Install required packages
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb
```

### Option 2: Local Installation
```bash
# Create virtual environment
python -m venv llm_finetune
source llm_finetune/bin/activate  # Linux/Mac
# llm_finetune\Scripts\activate  # Windows

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft trl datasets bitsandbytes einops wandb
```

### Hugging Face Authentication
```python
from huggingface_hub import login
login("your_huggingface_token_here")  # Get token from hf.co/settings/tokens
```

## 📊 Dataset

Our training dataset consists of **104 instruction-response pairs** covering diverse topics:

### Dataset Structure
```
### HUMAN:
[Question or instruction]

### RESPONSE:
[Detailed, helpful response]
```

### Topic Coverage
- **Science & Technology**: AI, quantum physics, renewable energy
- **Health & Wellness**: Exercise, nutrition, mental health
- **Business & Finance**: Investment, economics, leadership
- **Environment**: Climate change, sustainability, ecosystems
- **Education**: Learning strategies, critical thinking

### Sample Entry
```
### HUMAN:
What are the benefits of renewable energy?

### RESPONSE:
Renewable energy offers numerous benefits including reduced greenhouse 
gas emissions, energy independence, job creation in green sectors, and 
long-term cost savings. It helps combat climate change while providing 
sustainable power sources like solar, wind, and hydroelectric energy.
```

## 💻 Usage

### 1. Basic Fine-tuning
```python
# Load the notebook and run cells sequentially
# Key steps:
python
# 1. Model and tokenizer loading
base_model, tokenizer = load_model_and_tokenizer()

# 2. LoRA configuration
peft_config = setup_lora_config()
model = get_peft_model(base_model, peft_config)

# 3. Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

### 2. Testing the Model
```python
# Test with custom prompts
test_prompt = "### HUMAN:\nExplain machine learning in simple terms.\n\n### RESPONSE:\n"
result = test_base_model(model, tokenizer, test_prompt)
print(result)
```

### 3. Model Merging (Optional)
```python
# Merge LoRA adapters with base model
merged_model, tokenizer = merge_and_save_model()
```

## ⚙️ Configuration Details

### Training Parameters
```python
training_args = TrainingArguments(
    output_dir="llama_finetuned_output",
    per_device_train_batch_size=1,      # Adjust based on GPU memory
    gradient_accumulation_steps=4,       # Effective batch size = 4
    num_train_epochs=3,                  # 3 epochs for demonstration
    learning_rate=2e-4,                  # Conservative learning rate
    bf16=True,                          # Mixed precision training
    logging_steps=1,                     # Log every step
    save_strategy="epoch",               # Save after each epoch
    max_grad_norm=0.3,                  # Gradient clipping
    warmup_ratio=0.03,                  # Learning rate warmup
    lr_scheduler_type="constant",        # Constant LR after warmup
)
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=8,                                # Rank: balance between efficiency and performance
    lora_alpha=32,                      # Scaling factor (typically 2-4x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,                  # Regularization
    bias="none",                        # Don't adapt bias terms
    task_type="CAUSAL_LM"              # Causal language modeling
)
```

### Quantization Setup
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit quantization
    bnb_4bit_use_double_quant=True,     # Double quantization
    bnb_4bit_quant_type="nf4",         # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16  # Computation dtype
)
```

## 📈 Results and Evaluation

### Training Metrics
- **Total Training Steps**: 78 (26 steps × 3 epochs)
- **Final Training Loss**: ~1.91-2.25 range
- **Trainable Parameters**: 5,324,800 (0.29% of total)
- **Training Time**: ~7 minutes on Tesla T4

### Model Performance
The model demonstrates improved instruction-following capabilities:

**Before Fine-tuning** (Base LLaMA 2):
- Generic, verbose responses
- Less structured output
- May not follow instruction format

**After Fine-tuning**:
- Follows ### HUMAN/RESPONSE format
- More focused, relevant answers
- Better instruction adherence

### Memory Usage
- **Base Model**: ~2.4 GB (quantized)
- **With LoRA Adapters**: ~2.4 GB (minimal overhead)
- **Peak Training Memory**: ~6-8 GB

## 🔧 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution**:
```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=2

# Or use smaller model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Instead of 13b
```

### Issue 2: BFloat16 Error
**Problem**: `RuntimeError: expected scalar type Float but found BFloat16`

**Solution**:
```python
# Ensure consistent dtype in pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,  # Add this
    device_map="auto"
)
```

### Issue 3: Slow Training
**Solutions**:
- Enable mixed precision: `bf16=True`
- Use gradient checkpointing: automatically enabled
- Optimize batch size and accumulation steps

### Issue 4: Model Access Issues
**Solution**:
```python
# Ensure proper HuggingFace authentication
from huggingface_hub import login
login("your_token_here")

# Or request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

## 🎓 Educational Objectives

This implementation serves several pedagogical purposes:

### 1. **Understanding PEFT Methods**
- Compare full fine-tuning vs. LoRA
- Analyze parameter efficiency trade-offs
- Explore quantization benefits

### 2. **Practical Implementation Skills**
- Hands-on experience with modern fine-tuning
- Working with large-scale models
- Memory optimization techniques

### 3. **Research Applications**
- Foundation for custom domain adaptation
- Baseline for comparing PEFT methods
- Template for research projects

### 4. **Industry Relevance**
- Cost-effective model customization
- Production-ready techniques
- Scalable training approaches

## 📝 Key Takeaways

1. **LoRA enables efficient fine-tuning** with minimal parameter overhead
2. **Quantization dramatically reduces memory requirements** without significant performance loss
3. **Small datasets can be effective** for instruction-following adaptation
4. **Modern libraries** (transformers, peft, trl) simplify complex implementations
5. **GPU resources are accessible** through platforms like Google Colab

## 🔗 References

### Academic Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

### Technical Resources
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)

### Model Resources
- [LLaMA 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/)

---

## 🤝 Contributing

This educational project welcomes contributions:
- Additional training examples
- Improved evaluation metrics
- Documentation enhancements
- Bug fixes and optimizations

## 📄 License

This project is for educational purposes. Please respect:
- LLaMA 2 Custom Commercial License
- Hugging Face Terms of Service
- Academic Use Guidelines

---

**Date**: August 2025  
**Instructor**: Basharat Hussain  

*This implementation serves as a practical introduction to modern LLM fine-tuning techniques for students and researchers.*
