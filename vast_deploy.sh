#!/bin/bash
# Vast.ai deployment script for ERT with Ollama

echo "=== ERT Vast.ai Deployment Script ==="

# Update system
apt-get update && apt-get upgrade -y

# Install Python and pip
apt-get install -y python3 python3-pip git curl

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl start ollama
systemctl enable ollama

# Clone your repository
cd /workspace
git clone https://github.com/yourusername/ERT.git || echo "Update the git URL"
cd ERT

# Install Python dependencies
pip3 install -r requirements-compatible.txt

# Pull AI model (adjust model name as needed)
ollama pull llama3.1:8b

# Create start script
cat > start_ert.sh << 'EOF'
#!/bin/bash
cd /workspace/ERT
export PYTHONPATH=/workspace/ERT/src
export FLASK_ENV=production
python3 production_server.py --host 0.0.0.0 --port 5000
EOF

chmod +x start_ert.sh

echo "=== Setup Complete ==="
echo "Run: ./start_ert.sh to start ERT"
echo "Access via: http://[instance-ip]:5000"