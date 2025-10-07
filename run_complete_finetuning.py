#!/usr/bin/env python3
"""
Complete Ollama Finetuning Workflow for Equity Research

This script orchestrates the entire finetuning process:
1. Data generation and validation
2. LoRA finetuning execution
3. Model evaluation and benchmarking
4. Performance monitoring and reporting
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EquityResearchFineTuningWorkflow:
    """Complete finetuning workflow orchestrator"""

    def __init__(self, config_path: str = "finetune_config.json"):
        self.config_path = config_path
        self.workflow_dir = Path("finetuning_workflow")
        self.workflow_dir.mkdir(exist_ok=True)

        # Initialize workflow state
        self.workflow_state = {
            'stage': 'initialization',
            'start_time': datetime.now().isoformat(),
            'completed_stages': [],
            'current_stage_start': None,
            'errors': [],
            'metrics': {}
        }

    def log_stage(self, stage_name: str, status: str, details: Optional[Dict] = None):
        """Log workflow stage progress"""

        self.workflow_state['stage'] = stage_name

        if status == 'start':
            self.workflow_state['current_stage_start'] = datetime.now().isoformat()
            logger.info(f"🚀 Starting stage: {stage_name}")
        elif status == 'complete':
            if stage_name not in self.workflow_state['completed_stages']:
                self.workflow_state['completed_stages'].append(stage_name)
            duration = self._calculate_stage_duration()
            logger.info(f"✅ Completed stage: {stage_name} (Duration: {duration:.2f}s)")
            if details:
                self.workflow_state['metrics'][stage_name] = details
        elif status == 'error':
            error_info = {
                'stage': stage_name,
                'timestamp': datetime.now().isoformat(),
                'details': details or {}
            }
            self.workflow_state['errors'].append(error_info)
            logger.error(f"❌ Error in stage: {stage_name}")

        # Save workflow state
        self._save_workflow_state()

    def _calculate_stage_duration(self) -> float:
        """Calculate duration of current stage"""
        if self.workflow_state['current_stage_start']:
            start = datetime.fromisoformat(self.workflow_state['current_stage_start'])
            return (datetime.now() - start).total_seconds()
        return 0.0

    def _save_workflow_state(self):
        """Save current workflow state to file"""
        state_file = self.workflow_dir / "workflow_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.workflow_state, f, indent=2)

    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites for finetuning"""

        self.log_stage('prerequisites_validation', 'start')

        try:
            # Check Ollama installation
            if not self._check_ollama_available():
                raise Exception("Ollama not available")

            # Check Python dependencies
            required_packages = ['pandas', 'numpy', 'yfinance', 'requests']
            missing_packages = []

            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                raise Exception(f"Missing packages: {missing_packages}")

            # Check disk space (minimum 10GB)
            free_space = self._get_free_disk_space()
            if free_space < 10:
                raise Exception(f"Insufficient disk space: {free_space:.1f}GB available, 10GB required")

            # Check base model availability
            base_model = self._get_base_model_name()
            if not self._check_model_available(base_model):
                logger.info(f"Pulling base model: {base_model}")
                self._pull_ollama_model(base_model)

            details = {
                'ollama_available': True,
                'packages_installed': len(required_packages) - len(missing_packages),
                'free_disk_space_gb': free_space,
                'base_model_ready': True
            }

            self.log_stage('prerequisites_validation', 'complete', details)
            return True

        except Exception as e:
            self.log_stage('prerequisites_validation', 'error', {'error': str(e)})
            return False

    def generate_training_data(self) -> bool:
        """Generate comprehensive training dataset"""

        self.log_stage('data_generation', 'start')

        try:
            # Run training data generation script
            cmd = [sys.executable, 'create_training_data.py']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise Exception(f"Data generation failed: {result.stderr}")

            # Validate generated data
            train_file = Path("training_data/equity_research_train.jsonl")
            val_file = Path("training_data/equity_research_validation.jsonl")

            if not train_file.exists() or not val_file.exists():
                raise Exception("Training data files not generated")

            # Count examples
            train_count = self._count_jsonl_lines(train_file)
            val_count = self._count_jsonl_lines(val_file)

            details = {
                'training_examples': train_count,
                'validation_examples': val_count,
                'total_examples': train_count + val_count,
                'data_files_created': 2
            }

            self.log_stage('data_generation', 'complete', details)
            return True

        except Exception as e:
            self.log_stage('data_generation', 'error', {'error': str(e)})
            return False

    def execute_lora_finetuning(self) -> bool:
        """Execute LoRA finetuning process"""

        self.log_stage('lora_finetuning', 'start')

        try:
            # Check for training framework availability
            framework = self._detect_training_framework()

            if not framework:
                # Install appropriate framework
                self._install_training_framework()
                framework = self._detect_training_framework()

                if not framework:
                    raise Exception("No compatible training framework available")

            # Execute finetuning
            cmd = [sys.executable, 'finetune_ollama.py']

            # Start training process with timeout
            logger.info("Starting LoRA finetuning process...")
            start_time = time.time()

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Monitor training progress
            stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout

            if process.returncode != 0:
                raise Exception(f"Finetuning failed: {stderr}")

            duration = time.time() - start_time

            # Verify adapter creation
            adapter_path = Path("finetuned_models/lora_model")
            if not adapter_path.exists():
                raise Exception("LoRA adapter not created")

            details = {
                'training_framework': framework,
                'training_duration_seconds': duration,
                'adapter_created': True,
                'output_directory': str(adapter_path)
            }

            self.log_stage('lora_finetuning', 'complete', details)
            return True

        except subprocess.TimeoutExpired:
            process.kill()
            self.log_stage('lora_finetuning', 'error', {'error': 'Training timeout after 1 hour'})
            return False
        except Exception as e:
            self.log_stage('lora_finetuning', 'error', {'error': str(e)})
            return False

    def create_finetuned_model(self) -> bool:
        """Create Ollama model with LoRA adapter"""

        self.log_stage('model_creation', 'start')

        try:
            # Create Ollama model from adapter
            cmd = [sys.executable, 'finetune_ollama.py', '--create-model']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise Exception(f"Model creation failed: {result.stderr}")

            # Verify model creation
            model_name = self._get_finetuned_model_name()
            if not self._check_model_available(model_name):
                raise Exception(f"Finetuned model not available: {model_name}")

            # Test basic functionality
            test_response = self._test_model_basic(model_name)
            if not test_response:
                raise Exception("Model basic test failed")

            details = {
                'model_name': model_name,
                'model_available': True,
                'basic_test_passed': True,
                'test_response_length': len(test_response.split())
            }

            self.log_stage('model_creation', 'complete', details)
            return True

        except Exception as e:
            self.log_stage('model_creation', 'error', {'error': str(e)})
            return False

    def evaluate_finetuned_model(self) -> bool:
        """Comprehensive evaluation of finetuned model"""

        self.log_stage('model_evaluation', 'start')

        try:
            model_name = self._get_finetuned_model_name()

            # Run comprehensive evaluation
            cmd = [sys.executable, 'evaluate_model.py', model_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

            if result.returncode != 0:
                raise Exception(f"Evaluation failed: {result.stderr}")

            # Load evaluation results
            evaluation_file = self._find_latest_evaluation_file(model_name)
            if not evaluation_file:
                raise Exception("Evaluation results not found")

            with open(evaluation_file, 'r') as f:
                evaluation_results = json.load(f)

            overall_score = evaluation_results.get('overall_score', 0)

            details = {
                'model_name': model_name,
                'overall_score': overall_score,
                'evaluation_file': str(evaluation_file),
                'metric_count': len(evaluation_results.get('metric_scores', {})),
                'test_prompts_evaluated': evaluation_results.get('successful_responses', 0)
            }

            self.log_stage('model_evaluation', 'complete', details)
            return True

        except Exception as e:
            self.log_stage('model_evaluation', 'error', {'error': str(e)})
            return False

    def generate_final_report(self) -> bool:
        """Generate final finetuning report"""

        self.log_stage('final_report', 'start')

        try:
            # Compile workflow summary
            total_duration = (datetime.now() - datetime.fromisoformat(self.workflow_state['start_time'])).total_seconds()

            # Create comprehensive report
            report = {
                'workflow_summary': {
                    'total_duration_seconds': total_duration,
                    'total_duration_hours': total_duration / 3600,
                    'completed_stages': len(self.workflow_state['completed_stages']),
                    'total_stages': 6,
                    'success_rate': len(self.workflow_state['completed_stages']) / 6 * 100,
                    'errors_encountered': len(self.workflow_state['errors'])
                },
                'stage_metrics': self.workflow_state['metrics'],
                'model_info': {
                    'base_model': self._get_base_model_name(),
                    'finetuned_model': self._get_finetuned_model_name(),
                    'training_method': 'LoRA',
                    'dataset_size': self.workflow_state['metrics'].get('data_generation', {}).get('total_examples', 0)
                },
                'performance_summary': self._get_performance_summary(),
                'recommendations': self._generate_recommendations(),
                'timestamp': datetime.now().isoformat()
            }

            # Save final report
            report_file = self.workflow_dir / f"finetuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            # Create markdown summary
            markdown_summary = self._create_markdown_summary(report)
            summary_file = self.workflow_dir / "finetuning_summary.md"
            with open(summary_file, 'w') as f:
                f.write(markdown_summary)

            details = {
                'report_file': str(report_file),
                'summary_file': str(summary_file),
                'total_duration_hours': total_duration / 3600,
                'overall_success': len(self.workflow_state['errors']) == 0
            }

            self.log_stage('final_report', 'complete', details)
            return True

        except Exception as e:
            self.log_stage('final_report', 'error', {'error': str(e)})
            return False

    def run_complete_workflow(self) -> bool:
        """Execute the complete finetuning workflow"""

        logger.info("🎯 Starting Complete Ollama Finetuning Workflow for Equity Research")
        logger.info("=" * 80)

        workflow_steps = [
            ('Prerequisites Validation', self.validate_prerequisites),
            ('Training Data Generation', self.generate_training_data),
            ('LoRA Finetuning Execution', self.execute_lora_finetuning),
            ('Finetuned Model Creation', self.create_finetuned_model),
            ('Model Evaluation', self.evaluate_finetuned_model),
            ('Final Report Generation', self.generate_final_report)
        ]

        overall_success = True

        for step_name, step_function in workflow_steps:
            logger.info(f"\n📋 {step_name}")
            logger.info("-" * 40)

            success = step_function()

            if success:
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
                overall_success = False

                # Ask user if they want to continue
                if not self._should_continue_after_failure(step_name):
                    break

        # Final summary
        logger.info("\n" + "=" * 80)
        if overall_success:
            logger.info("🎉 FINETUNING WORKFLOW COMPLETED SUCCESSFULLY!")
            self._print_success_summary()
        else:
            logger.info("⚠️  FINETUNING WORKFLOW COMPLETED WITH ERRORS")
            self._print_error_summary()

        return overall_success

    # Helper methods
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def _get_free_disk_space(self) -> float:
        """Get free disk space in GB"""
        try:
            import shutil
            free_bytes = shutil.disk_usage('.').free
            return free_bytes / (1024**3)
        except:
            return 0.0

    def _get_base_model_name(self) -> str:
        """Get base model name from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('finetuning_strategy', {}).get('model_selection', {}).get('base_model', 'llama3.1:8b')
        except:
            return 'llama3.1:8b'

    def _get_finetuned_model_name(self) -> str:
        """Get finetuned model name from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('finetuning_strategy', {}).get('model_selection', {}).get('target_name', 'equity_research_llama')
        except:
            return 'equity_research_llama'

    def _check_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            return model_name in result.stdout
        except:
            return False

    def _pull_ollama_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            result = subprocess.run(['ollama', 'pull', model_name], timeout=1800)
            return result.returncode == 0
        except:
            return False

    def _count_jsonl_lines(self, file_path: Path) -> int:
        """Count lines in JSONL file"""
        try:
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
        except:
            return 0

    def _detect_training_framework(self) -> Optional[str]:
        """Detect available training framework"""
        frameworks = ['mlx_lm', 'unsloth', 'transformers']

        for framework in frameworks:
            try:
                __import__(framework)
                return framework
            except ImportError:
                continue

        return None

    def _install_training_framework(self):
        """Install appropriate training framework"""
        # Detect platform and install appropriate framework
        import platform

        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Apple Silicon - install MLX
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlx-lm'])
        else:
            # Other platforms - install Unsloth
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'unsloth'])

    def _test_model_basic(self, model_name: str) -> str:
        """Basic test of model functionality"""
        try:
            cmd = f"ollama run {model_name} 'What is 2+2?'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout.strip() if result.returncode == 0 else ""
        except:
            return ""

    def _find_latest_evaluation_file(self, model_name: str) -> Optional[Path]:
        """Find latest evaluation file for model"""
        eval_dir = Path("evaluation_results")
        if not eval_dir.exists():
            return None

        eval_files = list(eval_dir.glob(f"{model_name}_evaluation_*.json"))
        return max(eval_files, key=lambda x: x.stat().st_mtime) if eval_files else None

    def _get_performance_summary(self) -> Dict:
        """Get performance summary from evaluation"""
        model_name = self._get_finetuned_model_name()
        eval_file = self._find_latest_evaluation_file(model_name)

        if not eval_file:
            return {'status': 'evaluation_not_found'}

        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)

            return {
                'overall_score': results.get('overall_score', 0),
                'best_metric': max(results.get('metric_scores', {}).items(), key=lambda x: x[1].get('mean', 0)),
                'worst_metric': min(results.get('metric_scores', {}).items(), key=lambda x: x[1].get('mean', 0)),
                'response_success_rate': results.get('successful_responses', 0) / results.get('test_prompts_count', 1) * 100
            }
        except:
            return {'status': 'evaluation_parse_error'}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on workflow results"""
        recommendations = []

        # Check if there were errors
        if self.workflow_state['errors']:
            recommendations.append("Review error logs and address failed components before production use")

        # Check performance
        performance = self._get_performance_summary()
        overall_score = performance.get('overall_score', 0)

        if overall_score >= 80:
            recommendations.append("Model demonstrates excellent performance - ready for production use")
        elif overall_score >= 65:
            recommendations.append("Model shows good performance - consider additional training on specific weak areas")
        elif overall_score >= 50:
            recommendations.append("Model requires additional training before production deployment")
        else:
            recommendations.append("Model performance is below acceptable threshold - significant retraining required")

        # Check training data size
        data_metrics = self.workflow_state['metrics'].get('data_generation', {})
        total_examples = data_metrics.get('total_examples', 0)

        if total_examples < 1000:
            recommendations.append("Consider expanding training dataset for improved performance")

        return recommendations

    def _create_markdown_summary(self, report: Dict) -> str:
        """Create markdown summary of finetuning results"""

        summary = f"""# Ollama Finetuning Summary - Equity Research

## Workflow Overview
- **Duration**: {report['workflow_summary']['total_duration_hours']:.2f} hours
- **Success Rate**: {report['workflow_summary']['success_rate']:.1f}%
- **Completed Stages**: {report['workflow_summary']['completed_stages']}/{report['workflow_summary']['total_stages']}

## Model Information
- **Base Model**: {report['model_info']['base_model']}
- **Finetuned Model**: {report['model_info']['finetuned_model']}
- **Training Method**: {report['model_info']['training_method']}
- **Dataset Size**: {report['model_info']['dataset_size']} examples

## Performance Results
"""

        performance = report.get('performance_summary', {})
        if 'overall_score' in performance:
            summary += f"- **Overall Score**: {performance['overall_score']:.1f}/100\n"
            summary += f"- **Response Success Rate**: {performance.get('response_success_rate', 0):.1f}%\n"

        summary += "\n## Recommendations\n"
        for rec in report['recommendations']:
            summary += f"- {rec}\n"

        summary += f"\n## Generated On\n{report['timestamp']}\n"

        return summary

    def _should_continue_after_failure(self, failed_step: str) -> bool:
        """Ask user if workflow should continue after failure"""
        # For now, continue automatically for non-critical failures
        non_critical_steps = ['model_evaluation', 'final_report']
        return failed_step.lower().replace(' ', '_') in non_critical_steps

    def _print_success_summary(self):
        """Print success summary"""
        model_name = self._get_finetuned_model_name()
        logger.info(f"🎯 Finetuned model ready: {model_name}")
        logger.info("🧪 Next steps:")
        logger.info(f"   - Test model: ollama run {model_name}")
        logger.info(f"   - Evaluate: python evaluate_model.py {model_name}")
        logger.info("   - Update your report generator to use the finetuned model")

    def _print_error_summary(self):
        """Print error summary"""
        logger.info("💡 Common troubleshooting steps:")
        logger.info("   - Ensure Ollama is running: ollama serve")
        logger.info("   - Check available disk space (10GB+ required)")
        logger.info("   - Verify Python dependencies are installed")
        logger.info("   - Review workflow logs in finetuning_workflow/")

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description="Complete Ollama Finetuning Workflow for Equity Research")
    parser.add_argument('--config', type=str, default='finetune_config.json', help='Configuration file path')
    parser.add_argument('--skip-prerequisites', action='store_true', help='Skip prerequisites validation')
    parser.add_argument('--resume-from', type=str, help='Resume workflow from specific stage')

    args = parser.parse_args()

    # Initialize workflow
    workflow = EquityResearchFineTuningWorkflow(args.config)

    # Run complete workflow
    success = workflow.run_complete_workflow()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()