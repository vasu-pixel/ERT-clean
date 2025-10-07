# Enhanced Equity Research Tool (ERT) - Ollama Edition

A professional equity research report generator powered by **Ollama** (local LLM) instead of OpenAI API. This tool generates institutional-quality 35-50 page research reports using locally-hosted AI models.

## 🚀 Key Features

- **Local AI Processing**: Uses Ollama for private, offline AI analysis
- **No API Costs**: Run unlimited reports without per-token charges
- **Institutional Quality**: Generates comprehensive 6-section research reports
- **Privacy First**: All data processing happens locally
- **Multiple Models**: Support for various Ollama models (Llama, Mistral, etc.)
- **Professional Output**: Markdown reports with financial metrics tables

## 📋 Prerequisites

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 🛠️ Setup Instructions

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Pull Required AI Model
```bash
# Recommended model (8B parameters, good balance of speed/quality)
ollama pull llama3.1:8b

# Alternative models:
# ollama pull llama2:7b        # Faster, smaller
# ollama pull llama3.1:70b     # Higher quality, slower
# ollama pull mistral:7b       # Alternative model
```

### 3. Validate Setup
```bash
python run_generate_report_ollama.py --validate-only
```

## 🎯 Usage

### Interactive Mode (Recommended for beginners)
```bash
python run_generate_report_ollama.py --interactive
```

### Single Company Report
```bash
python run_generate_report_ollama.py --ticker AAPL
```

### Batch Processing
```bash
python run_generate_report_ollama.py --batch AAPL MSFT GOOGL TSLA
```

### Custom Model
```bash
python run_generate_report_ollama.py --ticker AAPL --model llama3.1:70b
```

### Demo Run
```bash
python run_generate_report_ollama.py --demo
```

## 📁 Project Structure

```
ERT/
├── src/
│   ├── utils/
│   │   ├── ollama_engine.py          # Ollama API interface
│   │   └── gpt_engine.py             # Legacy OpenAI interface
│   ├── stock_report_generator_ollama.py  # Main Ollama-based generator
│   ├── stock_report_generator.py    # Legacy OpenAI generator
│   └── ...
├── reports/                          # Generated reports
├── run_generate_report_ollama.py     # Main Ollama runner
├── run_generate_report.py            # Legacy OpenAI runner
├── config.json                       # Configuration
├── .env                             # Environment variables
└── requirements.txt                  # Python dependencies
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# AI Engine Configuration
AI_ENGINE=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# OpenAI Configuration (backup/alternative)
# OPENAI_API_KEY=your_key_here
```

### Config File (config.json)
```json
{
  "model": "llama3.1:8b",
  "max_tokens": 4000,
  "temperature": 0.3,
  "ai_engine": "ollama",
  "ollama": {
    "model": "llama3.1:8b",
    "base_url": "http://localhost:11434"
  }
}
```

## 📊 Report Sections

Each generated report includes:

1. **Executive Summary** - Investment recommendation, price target, key highlights
2. **Market Research** - Industry analysis, competitive landscape, ESG factors
3. **Financial Analysis** - Revenue, profitability, balance sheet, cash flow
4. **Valuation Analysis** - DCF modeling, peer comparison, target price methodology
5. **Investment Thesis** - Bull/bear cases, catalysts, risks
6. **Risk Analysis** - Business, financial, and regulatory risk assessment

## 🎛️ Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama2:7b` | 7B | Fast | Good | Quick analysis, testing |
| `llama3.1:8b` | 8B | Medium | Very Good | **Recommended balance** |
| `llama3.1:70b` | 70B | Slow | Excellent | Highest quality reports |
| `mistral:7b` | 7B | Fast | Good | Alternative model |

## 🚀 Performance Tips

### Speed Optimization
- Use smaller models (7B-8B) for faster generation
- Run on machines with more RAM for better performance
- Consider GPU acceleration if available

### Quality Optimization
- Use larger models (70B) for highest quality analysis
- Adjust temperature settings in config (lower = more consistent)
- Review and iterate on prompts for better outputs

## 🔧 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve

# Check available models
ollama list
```

### Model Not Found
```bash
# Pull missing model
ollama pull llama3.1:8b

# List available models
ollama list
```

### Memory Issues
- Use smaller models (7B instead of 70B)
- Close other applications to free RAM
- Check system requirements for your chosen model

### Generation Quality Issues
- Try different models
- Adjust temperature settings
- Ensure model is fully downloaded

## 🔄 Migration from OpenAI

If migrating from the OpenAI version:

1. **Keep both versions**: The original OpenAI-based files are preserved
2. **Use new files**: Run `run_generate_report_ollama.py` instead of `run_generate_report.py`
3. **Environment**: Update `.env` file to set `AI_ENGINE=ollama`
4. **Compare outputs**: Generate reports with both systems to compare quality

## 📈 Expected Performance

### Generation Times (approximate)
- **7B models**: 2-4 minutes per report
- **8B models**: 3-5 minutes per report
- **70B models**: 10-20 minutes per report

### System Requirements
- **Minimum**: 8GB RAM, 7B models
- **Recommended**: 16GB RAM, 8B models
- **High-end**: 32GB+ RAM, 70B models

## 🆘 Support

### Common Issues
1. **"Ollama server not accessible"** → Run `ollama serve`
2. **"Model not found"** → Run `ollama pull <model_name>`
3. **Slow generation** → Use smaller model or check system resources
4. **Poor quality** → Try larger model or adjust prompts

### Getting Help
- Check Ollama documentation: https://ollama.ai/docs
- Review configuration files and logs
- Test with `--validate-only` flag

## 🎯 Best Practices

### For Production Use
1. Use stable models (llama3.1:8b recommended)
2. Monitor system resources
3. Implement proper error handling
4. Regular model updates
5. Backup important reports

### For Development
1. Start with validation: `--validate-only`
2. Test with demo: `--demo`
3. Use interactive mode for experimentation
4. Compare different models for your use case

## 🔐 Privacy & Security

- **Data Privacy**: All processing happens locally
- **No API Calls**: No data sent to external services
- **Offline Capable**: Works without internet connection (after model download)
- **Secure**: No API keys or external dependencies required

## 📝 License

This tool is designed for professional equity research and analysis. Please ensure compliance with relevant financial regulations and use responsibly.

---

**Note**: This is the Ollama-powered version of the Enhanced Equity Research Tool. The original OpenAI-based version remains available in the legacy files.