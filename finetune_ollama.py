#!/usr/bin/env python3
"""
Ollama LoRA Finetuning System for Equity Research

This script implements parameter-efficient finetuning using LoRA (Low-Rank Adaptation)
for Ollama models, specifically optimized for equity research tasks.
"""

import os
import json
import logging
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLoRAFineTuner:
    """LoRA finetuning system for Ollama models"""

    def __init__(self, config_path: str = "finetune_config.json"):
        self.config = self._load_config(config_path)
        self.output_dir = Path("finetuned_models")
        self.output_dir.mkdir(exist_ok=True)

        # Training state tracking
        self.training_metrics = []
        self.best_validation_loss = float('inf')
        self.early_stopping_patience = 3
        self.current_patience = 0

    def _load_config(self, config_path: str) -> Dict:
        """Load finetuning configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default finetuning configuration"""
        return {
            "finetuning_strategy": {
                "model_selection": {
                    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "target_name": "equity_research_llama31"
                },
                "hyperparameters": {
                    "learning_rate": 2e-5,
                    "max_epochs": 3,
                    "batch_size": 4,
                    "lora_rank": 16,
                    "lora_alpha": 32
                }
            },
            "data_config": {
                "train_file": "training_data/equity_research_train.jsonl",
                "validation_file": "training_data/equity_research_validation.jsonl"
            },
            "model_config": {
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "model_name": "equity_research_llama31",
                "context_length": 4096
            },
            "lora_config": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "training_config": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "scheduler": "cosine_with_warmup",
                "save_every_n_steps": 500,
                "eval_every_n_steps": 250,
                "logging_steps": 50
            }
        }

    def prepare_modelfile(self, adapter_path: str) -> str:
        """Create Ollama Modelfile for the finetuned model"""

        modelfile_content = f"""
FROM {self.config.get('model_config', {}).get('base_model') or self.config.get('finetuning_strategy', {}).get('model_selection', {}).get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')}

# Set custom model parameters for equity research
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Load LoRA adapter
ADAPTER {adapter_path}

# System message for equity research
SYSTEM '''You are a senior equity research analyst at a top-tier investment bank.
You specialize in generating institutional-quality research reports with:

- Comprehensive financial analysis using standard metrics (P/E, EV/EBITDA, ROE, FCF, etc.)
- DCF valuation models with detailed assumptions and sensitivity analysis
- Investment thesis with specific bull/bear cases and catalysts
- Risk assessment covering business, financial, and regulatory factors
- Industry analysis with competitive positioning and market dynamics
- Executive summaries with clear buy/hold/sell recommendations and price targets

Always provide specific, actionable insights backed by quantitative analysis.
Use professional financial terminology and maintain institutional research standards.
Format outputs with clear sections and bullet points for readability.'''

# Template for equity research responses
TEMPLATE '''{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>'''
"""

        model_name = self.config.get('model_config', {}).get('model_name', 'equity_research_llama31')
        modelfile_path = self.output_dir / f"{model_name}_Modelfile"

        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        logger.info(f"Modelfile created at {modelfile_path}")
        return str(modelfile_path)

    def validate_training_data(self) -> bool:
        """Validate training data format and quality"""

        train_file = self.config.get('data_config', {}).get('train_file', 'training_data/equity_research_train.jsonl')
        val_file = self.config.get('data_config', {}).get('validation_file', 'training_data/equity_research_validation.jsonl')

        for file_path in [train_file, val_file]:
            if not Path(file_path).exists():
                logger.error(f"Training file not found: {file_path}")
                return False

            try:
                with open(file_path, 'r') as f:
                    line_count = 0
                    for line in f:
                        data = json.loads(line)

                        # Validate required fields
                        if 'instruction' not in data or 'response' not in data:
                            logger.error(f"Invalid format in {file_path}: missing instruction or response")
                            return False

                        # Check content quality
                        if len(data['instruction']) < 10 or len(data['response']) < 50:
                            logger.warning(f"Short content detected in {file_path}")

                        line_count += 1

                    logger.info(f"Validated {line_count} examples in {file_path}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {e}")
                return False

        return True

    def create_training_script(self) -> str:
        """Create training script for LoRA finetuning"""
        logger.info("=== ENTERING create_training_script method ===")
        logger.info(f"Target directory: {self.output_dir}")
        logger.info(f"Script path will be: {self.output_dir / 'train_lora.sh'}")

        try:
            # Log before directory creation
            logger.info("Creating output directory...")
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Directory created/verified: {self.output_dir}")
            logger.info(f"Directory exists: {self.output_dir.exists()}")

            # Check directory permissions
            self._check_directory_permissions()

            # Log before script generation
            logger.info("Generating training script content...")
            script_content = f"""#!/bin/bash

# Ollama LoRA Finetuning Script for Equity Research
# Generated on {datetime.now().isoformat()}

set -ex

echo "🚀 Starting Ollama LoRA Finetuning for Equity Research"
echo "=================================================="

# Configuration
BASE_MODEL="{self.config.get('model_config', {}).get('base_model') or self.config.get('finetuning_strategy', {}).get('model_selection', {}).get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')}"
MODEL_NAME="{self.config.get('model_config', {}).get('model_name', 'equity_research_llama31')}"
TRAIN_FILE="{self.config.get('data_config', {}).get('train_file', 'training_data/equity_research_train.jsonl')}"
VAL_FILE="{self.config.get('data_config', {}).get('validation_file', 'training_data/equity_research_validation.jsonl')}"
OUTPUT_DIR="{self.output_dir}"

# Training parameters (with fallbacks to handle different config structures)
LEARNING_RATE={self.config.get('training_config', {}).get('learning_rate') or self.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('learning_rate', 2e-5)}
BATCH_SIZE={self.config.get('training_config', {}).get('batch_size') or self.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('batch_size', 4)}
EPOCHS={self.config.get('training_config', {}).get('num_epochs') or self.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('max_epochs', 3)}
LORA_RANK={self.config.get('lora_config', {}).get('rank') or self.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('lora_rank', 16)}
LORA_ALPHA={self.config.get('lora_config', {}).get('alpha') or self.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('lora_alpha', 32)}

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
{sys.executable} -c "
import json

def convert_to_mlx_format(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # Convert to MLX-LM format with instruction-response structure
            text = f\"[INST] {{data['instruction']}} [/INST] {{data['response']}}\"
            mlx_data = {{'text': text}}
            f_out.write(json.dumps(mlx_data) + '\\n')

# Create properly named files for MLX-LM
convert_to_mlx_format('$TRAIN_FILE', '$OUTPUT_DIR/data/train.jsonl')
convert_to_mlx_format('$VAL_FILE', '$OUTPUT_DIR/data/valid.jsonl')
print('✅ Data conversion completed')
"

# Start LoRA training using MLX-LM (if available) or alternative method
echo "🎯 Starting LoRA finetuning..."

# Method 1: Using MLX-LM (for Apple Silicon)
if command -v {sys.executable} &> /dev/null && {sys.executable} -c "import mlx_lm" 2>/dev/null; then
    echo "Using MLX-LM for training..."

    {sys.executable} -m mlx_lm lora \
        --model $BASE_MODEL \
        --train \
        --data $OUTPUT_DIR/data \
        --num-layers 8 \
        --batch-size 1 \
        --iters 100 \
        --val-batches 1 \
        --learning-rate $LEARNING_RATE \
        --steps-per-report 10 \
        --steps-per-eval 25 \
        --adapter-path $OUTPUT_DIR/adapters \
        --save-every 25 \
        --max-seq-length 512

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
echo "1. Create Ollama model: {sys.executable} finetune_ollama.py --create-model"
echo "2. Test the model: ollama run $MODEL_NAME"
"""

            script_path = self.output_dir / "train_lora.sh"
            logger.info(f"Script content generated, length: {len(script_content)} characters")

            # Log before file write
            logger.info(f"Attempting to write script to: {script_path}")
            logger.info(f"Using absolute path: {script_path.absolute()}")

            with open(script_path, 'w') as f:
                f.write(script_content)

            # Verify file was created
            if script_path.exists():
                file_size = script_path.stat().st_size
                logger.info(f"SUCCESS: Script file created at {script_path}")
                logger.info(f"File size: {file_size} bytes")
                logger.info(f"File permissions: {oct(script_path.stat().st_mode)[-3:]}")
            else:
                logger.error(f"CRITICAL: Script file NOT found after write operation")
                raise FileNotFoundError(f"Script file was not created at {script_path}")

            # Make script executable
            logger.info("Setting script permissions to executable...")
            os.chmod(script_path, 0o755)
            logger.info(f"Script permissions set to executable: {oct(script_path.stat().st_mode)[-3:]}")

            # Final verification
            final_check = script_path.exists() and script_path.stat().st_size > 0
            logger.info(f"Final verification - Script exists and has content: {final_check}")

            if not final_check:
                raise RuntimeError(f"Script creation verification failed at {script_path}")

            logger.info(f"Training script created successfully at {script_path}")
            return str(script_path)

        except Exception as e:
            logger.error(f"EXCEPTION in create_training_script: {type(e).__name__}: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise

    def _check_directory_permissions(self):
        """Check if we can write to the output directory"""
        logger.info(f"Checking permissions for {self.output_dir}")

        if not self.output_dir.exists():
            logger.info("Directory doesn't exist yet")
            return

        # Check if we can write to the directory
        test_file = self.output_dir / '.test_write'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()  # Remove test file
            logger.info("Directory write permissions: OK")
        except Exception as e:
            logger.error(f"Directory write permissions: FAILED - {e}")
            raise

    def execute_training(self) -> bool:
        """Execute the LoRA finetuning process"""
        logger.info("=== STARTING execute_training method ===")

        try:
            # 1. Validate training data
            logger.info("Validating training data...")
            if not self.validate_training_data():
                logger.error("Training data validation failed")
                return False

            # 2. Create training script with enhanced verification
            logger.info("About to call create_training_script...")
            script_path = self.create_training_script()
            logger.info(f"create_training_script returned: {script_path}")

            # Convert to Path object for consistency
            script_path_obj = Path(script_path)

            # Double-check file existence with detailed logging
            logger.info(f"Verifying script existence at: {script_path}")
            logger.info(f"Script path exists: {script_path_obj.exists()}")
            logger.info(f"Script path is file: {script_path_obj.is_file()}")

            if script_path_obj.exists():
                logger.info(f"Script file size: {script_path_obj.stat().st_size} bytes")
                logger.info(f"Script file permissions: {oct(script_path_obj.stat().st_mode)}")

            if not script_path_obj.exists():
                logger.error(f"CRITICAL: Training script was not created at {script_path}")
                logger.error(f"Directory contents: {list(self.output_dir.iterdir())}")
                raise FileNotFoundError(f"Training script was not created at {script_path}")

            if script_path_obj.stat().st_size == 0:
                logger.error(f"CRITICAL: Training script is empty at {script_path}")
                raise RuntimeError(f"Training script is empty at {script_path}")

            # 3. Execute training
            logger.info("Starting LoRA finetuning...")
            start_time = time.time()

            process = subprocess.Popen([script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            stderr = process.communicate()[1]

            if process.returncode != 0:
                logger.error(f"❌ Training failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr}")
                print(stderr)
                return False

            end_time = time.time()
            training_duration = end_time - start_time
            logger.info(f"✅ Training completed successfully in {training_duration:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"Training execution error: {type(e).__name__}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def create_ollama_model(self, adapter_path: str) -> bool:
        """Create Ollama model with LoRA adapter"""

        try:
            # 1. Prepare Modelfile
            logger.info("Creating Ollama Modelfile...")
            modelfile_path = self.prepare_modelfile(adapter_path)

            # 2. Create Ollama model
            model_name = self.config.get('model_config', {}).get('model_name', 'equity_research_llama31')
            logger.info(f"Creating Ollama model: {model_name}")

            create_cmd = f"ollama create {model_name} -f {modelfile_path}"
            process = subprocess.run(create_cmd.split(), capture_output=True, text=True)

            if process.returncode == 0:
                logger.info(f"✅ Ollama model created successfully: {model_name}")

                # 3. Test the model
                logger.info("Testing the finetuned model...")
                test_prompt = "Provide a brief investment analysis for Apple Inc. (AAPL)."

                test_cmd = f"ollama run {model_name} '{test_prompt}'"
                test_process = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)

                if test_process.returncode == 0:
                    logger.info("✅ Model test successful")
                    logger.info(f"Sample output: {test_process.stdout[:200]}...")
                else:
                    logger.warning("⚠️  Model test failed, but model was created")

                return True
            else:
                logger.error(f"❌ Failed to create Ollama model: {process.stderr}")
                return False

        except Exception as e:
            logger.error(f"Model creation error: {e}")
            return False

    def benchmark_model(self, model_name: str) -> Dict:
        """Benchmark the finetuned model against base model"""

        test_prompts = [
            "Generate an executive summary for Microsoft (MSFT) with a BUY recommendation.",
            "Analyze the financial performance of Tesla focusing on profitability metrics.",
            "Conduct a DCF valuation for Amazon with 20% AWS growth assumptions.",
            "Assess the investment risks for NVIDIA in the AI semiconductor market."
        ]

        base_model = self.config.get('model_config', {}).get('base_model') or self.config.get('finetuning_strategy', {}).get('model_selection', {}).get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')
        results = {
            'base_model_responses': [],
            'finetuned_responses': [],
            'comparison_metrics': {}
        }

        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt[:50]}...")

            # Test base model
            base_cmd = f"ollama run {base_model} '{prompt}'"
            base_process = subprocess.run(base_cmd, shell=True, capture_output=True, text=True)

            # Test finetuned model
            ft_cmd = f"ollama run {model_name} '{prompt}'"
            ft_process = subprocess.run(ft_cmd, shell=True, capture_output=True, text=True)

            results['base_model_responses'].append({
                'prompt': prompt,
                'response': base_process.stdout,
                'length': len(base_process.stdout.split())
            })

            results['finetuned_responses'].append({
                'prompt': prompt,
                'response': ft_process.stdout,
                'length': len(ft_process.stdout.split())
            })

        # Calculate basic metrics
        base_avg_length = sum(r['length'] for r in results['base_model_responses']) / len(test_prompts)
        ft_avg_length = sum(r['length'] for r in results['finetuned_responses']) / len(test_prompts)

        results['comparison_metrics'] = {
            'base_model_avg_length': base_avg_length,
            'finetuned_avg_length': ft_avg_length,
            'length_improvement': (ft_avg_length - base_avg_length) / base_avg_length * 100
        }

        # Save benchmark results
        benchmark_path = self.output_dir / f"{model_name}_benchmark.json"
        with open(benchmark_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to {benchmark_path}")
        return results

def main():
    """Main finetuning workflow"""

    print("🎯 Ollama LoRA Finetuning for Equity Research")
    print("=" * 50)

    finetuner = OllamaLoRAFineTuner()

    # Check if training data exists
    if not Path("training_data").exists():
        print("❌ Training data not found. Please run: python create_training_data.py")
        return

    print("📋 Finetuning Configuration:")
    base_model = finetuner.config.get('finetuning_strategy', {}).get('model_selection', {}).get('base_model', 'llama3.1:8b')
    target_model = finetuner.config.get('finetuning_strategy', {}).get('model_selection', {}).get('target_name', 'equity_research_llama')
    lora_rank = finetuner.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('lora_rank', 16)
    learning_rate = finetuner.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('learning_rate', 2e-5)
    epochs = finetuner.config.get('finetuning_strategy', {}).get('hyperparameters', {}).get('max_epochs', 3)

    print(f"  Base Model: {base_model}")
    print(f"  Target Model: {target_model}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print()

    # Execute training workflow
    success = finetuner.execute_training()

    if success:
        print("🎉 Finetuning completed successfully!")

        # Create Ollama model
        adapter_path = finetuner.output_dir / "lora_model"
        if adapter_path.exists():
            model_created = finetuner.create_ollama_model(str(adapter_path))

            if model_created:
                model_name = finetuner.config.get('model_config', {}).get('model_name', 'equity_research_llama')
                print(f"✅ Finetuned model ready: {model_name}")
                print(f"🧪 Run benchmark: {sys.executable} finetune_ollama.py --benchmark {model_name}")
                print(f"🚀 Test model: ollama run {model_name}")
            else:
                print("❌ Failed to create Ollama model")
        else:
            print("❌ Adapter not found. Training may have failed.")
    else:
        print("❌ Finetuning failed. Check logs for details.")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ollama LoRA Finetuning for Equity Research")
    parser.add_argument('--benchmark', type=str, help='Benchmark a finetuned model')
    parser.add_argument('--create-model', action='store_true', help='Create Ollama model from adapter')

    args = parser.parse_args()

    if args.benchmark:
        finetuner = OllamaLoRAFineTuner()
        results = finetuner.benchmark_model(args.benchmark)
        print(f"📊 Benchmark completed. Results saved.")
    elif args.create_model:
        finetuner = OllamaLoRAFineTuner()
        adapter_path = finetuner.output_dir / "lora_model"
        finetuner.create_ollama_model(str(adapter_path))
    else:
        main()