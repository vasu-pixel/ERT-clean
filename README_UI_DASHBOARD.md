# ERT Status Dashboard - User Interface Documentation

A comprehensive web-based dashboard for monitoring and managing Enhanced Equity Research Tool (ERT) report generation in real-time.

## 🌟 Features

### **Real-time Monitoring**
- **Live Progress Tracking** - Watch reports generate in real-time with progress bars
- **WebSocket Updates** - Instant status updates without page refresh
- **System Health Monitoring** - Check Ollama connection, data access, and disk space
- **Queue Management** - Monitor active reports and queue size

### **Report Generation**
- **One-Click Generation** - Enter ticker symbol and generate reports instantly
- **Quick Buttons** - Pre-configured buttons for popular stocks (AAPL, MSFT, etc.)
- **Batch Processing** - Queue multiple reports for sequential generation
- **Progress Visualization** - Real-time progress bars with section-by-section updates

### **Report Management**
- **Report History** - View all completed and failed reports
- **Download Reports** - Direct download links for generated reports
- **Report Details** - Detailed information about each report's generation process
- **Status Filtering** - Filter reports by status (completed, failed, active)

### **System Status**
- **Connection Monitoring** - Real-time Ollama and data service status
- **Performance Metrics** - Active reports, queue size, completion statistics
- **Health Checks** - Automatic system validation and status reporting
- **Resource Monitoring** - Disk space and system resource usage

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
# Install UI-specific dependencies
pip install -r requirements_ui.txt

# Or install individual packages
pip install flask flask-socketio requests eventlet
```

### **2. Launch Dashboard**
```bash
# Simple launch
python launch_dashboard.py

# Development mode with debug
python launch_dashboard.py --dev

# Custom port
python launch_dashboard.py --port 8080

# Check system status only
python launch_dashboard.py --check-only
```

### **3. Access Dashboard**
- **URL**: http://localhost:5000
- **API**: http://localhost:5000/api/status
- **Reports**: http://localhost:5000/reports/

## 📊 Dashboard Interface

### **Main Dashboard Layout**

```
┌─────────────────────────────────────────────────────────────┐
│                    ERT Status Dashboard                     │
│                 [Ollama] [Data] [Storage]                   │
├─────────────────────────────────────────────────────────────┤
│  Generate New Report                                        │
│  [Ticker Input] [Generate Report]                          │
│  [AAPL] [MSFT] [GOOGL] [TSLA] [AMZN]                      │
├─────────────────────┬───────────────────────────────────────┤
│    System Status    │         Active Reports               │
│   Active: 2         │  ┌─────────────────────────────────┐ │
│   Queue: 1          │  │ AAPL [████████░░] 80%           │ │
│   Completed: 15     │  │ Generating investment thesis... │ │
│                     │  └─────────────────────────────────┘ │
└─────────────────────┴───────────────────────────────────────┘
│                    Recent Reports                           │
│  ✅ MSFT - Completed (2,547 words) [Download] [Details]    │
│  ❌ INVALID - Failed                [Details]               │
│  ✅ GOOGL - Completed (3,102 words) [Download] [Details]   │
└─────────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Header Section**
- **Status Badges**: Real-time system status indicators
  - 🤖 Ollama: Connection status to local LLM
  - 📊 Data: Market data access status
  - 💾 Storage: Available disk space

#### **2. Report Generation Panel**
- **Ticker Input**: Enter stock symbols (auto-uppercase)
- **Generate Button**: Start report generation
- **Quick Buttons**: One-click generation for popular stocks
- **Keyboard Support**: Press Enter to generate

#### **3. System Status Panel**
- **Active Reports**: Currently generating reports count
- **Queue Size**: Number of reports waiting to be processed
- **Completed Today**: Reports finished today

#### **4. Active Reports Panel**
- **Real-time Progress**: Live progress bars (0-100%)
- **Current Section**: What section is being generated
- **Time Estimates**: Estimated completion time
- **Status Updates**: Current processing status

#### **5. Recent Reports Panel**
- **Report History**: Last 20 completed reports
- **Status Indicators**: Success/failure icons
- **Download Links**: Direct access to generated reports
- **Report Details**: Metadata and generation info

## 🔧 Technical Architecture

### **Backend (Flask + SocketIO)**

```python
# Core Components
├── status_server.py          # Main Flask application
├── ReportStatusManager       # Status tracking and queue management
├── ReportProgress           # Individual report progress tracking
└── Background Worker        # Report generation worker thread
```

**Key Features:**
- **WebSocket Communication**: Real-time bidirectional updates
- **Queue Management**: Thread-safe report queue processing
- **Progress Tracking**: Detailed progress monitoring per report
- **API Endpoints**: RESTful API for status and control
- **Background Processing**: Non-blocking report generation

### **Frontend (HTML + JavaScript)**

```javascript
// Core Components
├── WebSocket Client         # Real-time server communication
├── Progress Visualization   # Dynamic progress bars and charts
├── Report Management       # Report listing and actions
└── System Monitoring       # Health status and metrics
```

**Key Features:**
- **Real-time Updates**: Instant UI updates via WebSocket
- **Responsive Design**: Mobile-friendly responsive layout
- **Interactive Elements**: Hover effects, animations, modals
- **Toast Notifications**: User feedback for actions
- **Keyboard Shortcuts**: Enhanced user experience

### **Data Flow**

```
User Input → Flask API → Queue → Background Worker → Ollama → Progress Updates → WebSocket → UI Update
```

1. **User Action**: Enter ticker and click generate
2. **API Call**: POST to `/api/generate` endpoint
3. **Queue Addition**: Report added to processing queue
4. **Background Processing**: Worker picks up report from queue
5. **Ollama Integration**: AI generates report sections
6. **Progress Updates**: Real-time progress sent via WebSocket
7. **UI Updates**: Dashboard updates automatically
8. **Completion**: Report saved and available for download

## 📱 User Interface Guide

### **Generating Reports**

1. **Single Report**:
   - Enter ticker symbol in input field
   - Click "Generate Report" or press Enter
   - Monitor progress in Active Reports panel

2. **Quick Generation**:
   - Click any quick button (AAPL, MSFT, etc.)
   - Report automatically added to queue

3. **Batch Processing**:
   - Generate multiple reports by entering tickers one by one
   - Each report processes sequentially

### **Monitoring Progress**

- **Progress Bar**: Visual representation (0-100%)
- **Current Section**: Real-time status updates
- **Time Estimates**: Estimated completion time
- **Error Handling**: Clear error messages if generation fails

### **Managing Reports**

- **Download**: Click download button for completed reports
- **Details**: View detailed generation information
- **History**: Browse recent report history
- **Status**: Filter by completion status

### **System Health**

- **Status Badges**: Quick system health overview
- **Connection Status**: WebSocket connection indicator
- **Resource Monitoring**: Disk space and system resources
- **Error Detection**: Automatic problem detection and alerts

## 🎨 Customization

### **Visual Styling**

The dashboard supports extensive customization:

- **Color Themes**: Modify CSS variables for custom colors
- **Dark Mode**: Automatic dark mode detection
- **Responsive Design**: Mobile and tablet optimized
- **Animations**: Smooth transitions and hover effects

### **Configuration**

```python
# Server Configuration
PORT = 5000                    # Dashboard port
DEBUG_MODE = False            # Development mode
WEBSOCKET_CORS = "*"          # CORS settings
MAX_REPORTS = 100             # Maximum concurrent reports
```

### **API Endpoints**

- `GET /` - Main dashboard page
- `GET /api/status` - Current system status
- `POST /api/generate` - Start report generation
- `GET /api/reports` - List recent reports
- `GET /api/reports/<id>` - Specific report details
- `GET /reports/<filename>` - Download report files

## 🔒 Security Features

- **Input Validation**: Ticker symbol validation
- **Rate Limiting**: Prevent spam requests
- **CORS Configuration**: Controlled cross-origin access
- **File Access**: Secure report file serving
- **Error Handling**: Graceful error recovery

## 🚨 Troubleshooting

### **Common Issues**

1. **Dashboard Won't Start**
   ```bash
   # Check dependencies
   python launch_dashboard.py --check-only

   # Install missing packages
   pip install -r requirements_ui.txt
   ```

2. **Ollama Connection Failed**
   ```bash
   # Start Ollama server
   ollama serve

   # Check available models
   ollama list
   ```

3. **Reports Not Generating**
   - Verify Ollama is running
   - Check internet connection for market data
   - Ensure sufficient disk space
   - Validate ticker symbols

4. **WebSocket Disconnection**
   - Refresh browser page
   - Check network connectivity
   - Restart dashboard server

### **Debug Mode**

```bash
# Launch in development mode for detailed logs
python launch_dashboard.py --dev

# Check system status
python launch_dashboard.py --check-only

# View server logs
tail -f logs/ollama_equity_research.log
```

## 📈 Performance Optimization

### **Server Performance**
- **Background Processing**: Non-blocking report generation
- **Queue Management**: Efficient task scheduling
- **Memory Usage**: Optimized data structures
- **Connection Pooling**: Efficient WebSocket handling

### **Client Performance**
- **Lazy Loading**: Progressive content loading
- **Efficient Updates**: Minimal DOM manipulation
- **Caching**: Browser caching for static assets
- **Compression**: Gzipped responses

## 🔄 Updates and Maintenance

### **Automatic Updates**
- **Status Refresh**: Every 30 seconds
- **Progress Updates**: Real-time via WebSocket
- **Health Checks**: Continuous monitoring
- **Error Recovery**: Automatic reconnection

### **Manual Maintenance**
- **Log Rotation**: Automatic log file management
- **Report Cleanup**: Old report file management
- **Cache Clearing**: Periodic cache cleanup
- **Performance Monitoring**: Resource usage tracking

## 📞 Support

### **Getting Help**
- **System Status**: Use `--check-only` flag
- **Debug Mode**: Enable with `--dev` flag
- **Log Files**: Check `logs/` directory
- **API Testing**: Use `/api/status` endpoint

### **Common Solutions**
- **Port Conflicts**: Use `--port` to specify different port
- **Permission Issues**: Check file system permissions
- **Network Issues**: Verify firewall settings
- **Resource Limits**: Monitor system resources

---

**Note**: This dashboard is designed to work seamlessly with the Enhanced Equity Research Tool's Ollama-based report generation system. For the best experience, ensure Ollama is properly configured and running before starting the dashboard.