# Ollama Finetuning for Equity Research - Complete Guide

A comprehensive system for finetuning Ollama models specifically for institutional-quality equity research and financial analysis tasks.

## 🎯 Overview

This finetuning framework transforms general-purpose Ollama models into specialized equity research assistants using:

- **LoRA (Low-Rank Adaptation)** for parameter-efficient training
- **Domain-specific training data** with real financial examples
- **Comprehensive evaluation metrics** for financial analysis quality
- **Automated workflow orchestration** with error handling and monitoring

## 📋 Prerequisites

### System Requirements
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 20GB+ free space
- **OS**: macOS, Linux, or Windows with WSL
- **Python**: 3.8+

### Software Dependencies
```bash
# Core requirements
pip install pandas numpy yfinance requests

# Training frameworks (choose based on platform)
pip install mlx-lm          # For Apple Silicon (M1/M2/M3)
pip install unsloth         # For CUDA/CPU systems
pip install transformers    # Alternative framework
```

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull base model
ollama pull llama3.1:8b
```

## 🚀 Quick Start

### Option 1: Complete Automated Workflow
```bash
# Run the entire finetuning pipeline
python run_complete_finetuning.py

# This will automatically:
# 1. Generate training data
# 2. Execute LoRA finetuning
# 3. Create finetuned model
# 4. Run comprehensive evaluation
# 5. Generate performance report
```

### Option 2: Step-by-Step Process
```bash
# Step 1: Generate training dataset
python create_training_data.py

# Step 2: Execute LoRA finetuning
python finetune_ollama.py

# Step 3: Create Ollama model
python finetune_ollama.py --create-model

# Step 4: Evaluate performance
python evaluate_model.py equity_research_llama

# Step 5: Benchmark against base model
python evaluate_model.py equity_research_llama --benchmark
```

## 📊 Training Strategy

### LoRA Configuration
```json
{
  "lora_config": {
    "rank": 16,              // Balance between efficiency and quality
    "alpha": 32,             // Learning rate scaling factor
    "dropout": 0.1,          // Prevent overfitting
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  }
}
```

### Hyperparameters
- **Learning Rate**: 2e-5 (conservative for financial domain)
- **Batch Size**: 4 (memory-efficient)
- **Epochs**: 3 (prevent overfitting)
- **Warmup Steps**: 100 (stable training start)
- **Gradient Accumulation**: 8 steps

### Training Data Composition
- **Executive Summaries**: Investment recommendations and highlights
- **Financial Analysis**: Profitability, efficiency, and quality metrics
- **Valuation Models**: DCF analysis and comparable company methods
- **Investment Thesis**: Bull/bear cases with specific catalysts
- **Risk Assessment**: Business, financial, and regulatory risks
- **Industry Analysis**: Competitive landscape and market dynamics

## 📈 Evaluation Metrics

### Quantitative Metrics
1. **Financial Accuracy** (25% weight)
   - Correct usage of financial terminology
   - Presence of quantitative calculations
   - Accuracy of financial ratios and metrics

2. **Investment Reasoning** (30% weight)
   - Clear buy/hold/sell recommendations
   - Specific price targets with upside/downside
   - Logical catalyst identification
   - Comprehensive risk assessment

3. **Technical Depth** (25% weight)
   - Advanced financial concepts (DCF, WACC, etc.)
   - Industry-specific knowledge
   - Comparative analysis and benchmarking
   - Methodological sophistication

4. **Structure Quality** (20% weight)
   - Professional formatting and organization
   - Clear section headers and bullet points
   - Appropriate response length
   - Institutional report structure

### Performance Benchmarks
- **Excellent**: 80-100 (Production ready)
- **Good**: 65-79 (Minor improvements needed)
- **Acceptable**: 50-64 (Additional training recommended)
- **Poor**: <50 (Significant retraining required)

## 🔧 Configuration

### Model Selection
```json
{
  "model_selection": {
    "base_model": "llama3.1:8b",      // Recommended balance
    "alternatives": {
      "fast": "llama2:7b",            // Faster inference
      "quality": "llama3.1:70b",      // Higher quality
      "efficient": "mistral:7b"       // Memory efficient
    }
  }
}
```

### Training Data Customization
```python
# Add custom training examples
custom_examples = [
    {
        "instruction": "Analyze XYZ Corp's competitive position...",
        "response": "## Competitive Analysis - XYZ Corp..."
    }
]

# Modify data generator
generator = EquityResearchTrainingDataGenerator()
generator.add_custom_examples(custom_examples)
```

### Evaluation Customization
```python
# Custom evaluation prompts
custom_prompts = [
    "Generate a DCF model for Apple with 10% revenue growth",
    "Assess Tesla's competitive risks in the EV market",
    "Compare Microsoft and Google's cloud strategies"
]

evaluator = EquityResearchEvaluator("equity_research_llama")
results = evaluator.comprehensive_evaluation(custom_prompts)
```

## 📁 Project Structure

```
ERT/
├── finetuning_workflow/          # Workflow state and reports
├── training_data/                # Generated training datasets
├── finetuned_models/             # LoRA adapters and models
├── evaluation_results/           # Performance evaluations
├── finetune_config.json         # Configuration settings
├── create_training_data.py      # Training data generator
├── finetune_ollama.py           # LoRA finetuning engine
├── evaluate_model.py            # Evaluation system
└── run_complete_finetuning.py   # Complete workflow orchestrator
```

## 🎛️ Advanced Usage

### Custom Model Creation
```bash
# Finetune with custom configuration
python finetune_ollama.py --config custom_config.json

# Create model with specific name
python finetune_ollama.py --create-model --name my_equity_model

# Benchmark multiple models
python evaluate_model.py model1 model2 model3 --compare
```

### Training Data Expansion
```python
# Generate larger dataset
generator = EquityResearchTrainingDataGenerator()
generator.generate_dynamic_training_examples(num_examples=500)

# Add sector-specific examples
generator.add_sector_examples("technology", count=100)
generator.add_sector_examples("healthcare", count=100)
```

### Performance Optimization
```bash
# Training with higher quality model
python run_complete_finetuning.py --config high_quality_config.json

# Distributed training (if supported)
python finetune_ollama.py --distributed --gpus 2

# Memory optimization for large models
python finetune_ollama.py --use-qlora --4bit
```

## 🔍 Monitoring and Debugging

### Training Progress
```bash
# Monitor training logs
tail -f logs/ollama_equity_research.log

# Check GPU utilization (if applicable)
nvidia-smi

# Monitor system resources
htop
```

### Common Issues & Solutions

#### 1. Out of Memory Errors
```bash
# Reduce batch size
# Increase gradient accumulation steps
# Use QLoRA instead of LoRA
# Switch to smaller base model
```

#### 2. Slow Training
```bash
# Check system resources
# Reduce context length
# Use smaller rank for LoRA
# Enable gradient checkpointing
```

#### 3. Poor Performance
```bash
# Increase training data size
# Adjust learning rate
# Train for more epochs
# Use higher-quality base model
```

#### 4. Model Creation Failures
```bash
# Verify adapter files exist
# Check Ollama server status
# Validate Modelfile syntax
# Ensure sufficient disk space
```

## 📊 Performance Optimization

### Training Speed
- **Apple Silicon**: Use MLX-LM framework for optimal performance
- **NVIDIA GPUs**: Use Unsloth with CUDA acceleration
- **CPU Only**: Reduce model size and batch size

### Memory Efficiency
- **4-bit Quantization**: Use QLoRA for reduced memory usage
- **Gradient Checkpointing**: Trade compute for memory
- **Batch Size Tuning**: Find optimal size for your hardware

### Quality Improvement
- **Data Quality**: Focus on high-quality, diverse training examples
- **Hyperparameter Tuning**: Experiment with learning rates and schedules
- **Evaluation-Driven**: Use evaluation metrics to guide improvements

## 🚀 Production Deployment

### Model Integration
```python
# Update your equity research generator
generator = EnhancedEquityResearchGenerator()
generator.ollama_engine.default_model = "equity_research_llama"

# Generate reports with finetuned model
report = generator.generate_comprehensive_report("AAPL")
```

### Performance Monitoring
```python
# Continuous evaluation in production
evaluator = EquityResearchEvaluator("equity_research_llama")
daily_score = evaluator.evaluate_recent_outputs()

if daily_score < threshold:
    trigger_retraining_alert()
```

### Model Updates
```bash
# Incremental training with new data
python finetune_ollama.py --continue-training --new-data recent_examples.jsonl

# Version management
ollama create equity_research_llama:v2 -f updated_Modelfile
```

## 📚 Best Practices

### Data Quality
- Ensure diverse representation across sectors and market conditions
- Include both bullish and bearish examples
- Validate financial accuracy of training examples
- Regular data quality audits and updates

### Training Strategy
- Start with smaller models for rapid iteration
- Use validation loss for early stopping
- Monitor for catastrophic forgetting
- Regular checkpointing during training

### Evaluation Rigor
- Test on held-out data not seen during training
- Include both quantitative and qualitative metrics
- Compare against base model performance
- Regular performance monitoring in production

### Ethical Considerations
- Ensure training data doesn't contain insider information
- Avoid biased recommendations or market manipulation
- Include appropriate disclaimers in model outputs
- Regular compliance and ethics reviews

## 🆘 Troubleshooting

### General Issues
```bash
# Check system status
python run_complete_finetuning.py --validate-only

# View detailed logs
cat logs/ollama_equity_research.log

# Clean up corrupted files
rm -rf training_data finetuned_models
python create_training_data.py
```

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Logs**: Review detailed logs in `logs/` directory
- **Configuration**: Validate configuration files
- **Community**: Ollama and LoRA community forums

## 📈 Roadmap

### Planned Improvements
- [ ] Multi-model ensemble training
- [ ] Real-time market data integration
- [ ] Advanced risk metrics evaluation
- [ ] Automated hyperparameter optimization
- [ ] Distributed training support
- [ ] Model versioning and A/B testing

### Contributing
Contributions welcome! Focus areas:
- Additional evaluation metrics
- Training data quality improvements
- Performance optimizations
- Documentation enhancements

---

**Note**: This is a sophisticated finetuning system designed for professional equity research. Ensure proper validation and testing before production deployment. Always include appropriate disclaimers in generated research reports.

For support or questions, review the troubleshooting section and check the generated logs for detailed error information.