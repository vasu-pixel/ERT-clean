#!/bin/bash

# Ollama LoRA Finetuning Script for Equity Research
# Generated on 2025-09-20T19:57:36.249121

set -ex

echo "🚀 Starting Ollama LoRA Finetuning for Equity Research"
echo "=================================================="

# Configuration
BASE_MODEL="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="equity_research_llama31"
TRAIN_FILE="training_data/equity_research_train.jsonl"
VAL_FILE="training_data/equity_research_validation.jsonl"
OUTPUT_DIR="finetuned_models"

# Training parameters (with fallbacks to handle different config structures)
LEARNING_RATE=2e-05
BATCH_SIZE=4
EPOCHS=3
LORA_RANK=16
LORA_ALPHA=32

echo "📋 Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Output Model: $MODEL_NAME"
echo "  Training Data: $TRAIN_FILE"
echo "  Validation Data: $VAL_FILE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA Rank: $LORA_RANK"
echo ""

# Check if Ollama is running
echo "🔍 Checking Ollama server..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama server not running. Please start with: ollama serve"
    exit 1
fi
echo "✅ Ollama server is running"

# Check if base model exists (for HuggingFace models, we'll let MLX-LM handle downloading)
echo "🔍 Checking base model availability..."
if [[ "$BASE_MODEL" == *"/"* ]]; then
    echo "📥 Using HuggingFace model: $BASE_MODEL (will be downloaded by MLX-LM if needed)"
else
    # For Ollama models
    if ! ollama list | grep -q "$BASE_MODEL"; then
        echo "📥 Pulling base model: $BASE_MODEL"
        ollama pull $BASE_MODEL
    fi
fi
echo "✅ Base model configuration ready: $BASE_MODEL"

# Create output directory
mkdir -p $OUTPUT_DIR

# Convert training data format for MLX-LM and create proper directory structure
echo "🔄 Converting training data format for MLX-LM..."
mkdir -p $OUTPUT_DIR/data
/opt/anaconda3/bin/python -c "
import json

def convert_to_mlx_format(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # Convert to MLX-LM format with instruction-response structure
            text = f"[INST] {data['instruction']} [/INST] {data['response']}"
            mlx_data = {'text': text}
            f_out.write(json.dumps(mlx_data) + '\n')

# Create properly named files for MLX-LM
convert_to_mlx_format('$TRAIN_FILE', '$OUTPUT_DIR/data/train.jsonl')
convert_to_mlx_format('$VAL_FILE', '$OUTPUT_DIR/data/valid.jsonl')
print('✅ Data conversion completed')
"

# Start LoRA training using MLX-LM (if available) or alternative method
echo "🎯 Starting LoRA finetuning..."

# Method 1: Using MLX-LM (for Apple Silicon)
if command -v /opt/anaconda3/bin/python &> /dev/null && /opt/anaconda3/bin/python -c "import mlx_lm" 2>/dev/null; then
    echo "Using MLX-LM for training..."

    /opt/anaconda3/bin/python -m mlx_lm lora         --model $BASE_MODEL         --train         --data $OUTPUT_DIR/data         --num-layers 8         --batch-size 1         --iters 100         --val-batches 1         --learning-rate $LEARNING_RATE         --steps-per-report 10         --steps-per-eval 25         --adapter-path $OUTPUT_DIR/adapters         --save-every 25         --max-seq-length 512

else
    echo "⚠️  No compatible training framework found."
    echo "Please install MLX-LM (Apple Silicon) or Unsloth for LoRA training."
    echo ""
    echo "Installation commands:"
    echo "  pip install mlx-lm  # For Apple Silicon"
    exit 1
fi

echo ""
echo "🎉 LoRA finetuning completed!"
echo "📁 Adapter saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Create Ollama model: /opt/anaconda3/bin/python finetune_ollama.py --create-model"
echo "2. Test the model: ollama run $MODEL_NAME"
