# 🚀 Intelligent Real-Time Financial Market Analyzer

A comprehensive AI-powered real-time financial market analytics dashboard that combines machine learning, real-time data streaming, and modern web technologies to provide predictive market insights.

## 🎯 Features

### Real-Time Data Ingestion
- **AsyncIO + WebSockets**: High-performance real-time data streaming
- **DuckDB Integration**: Fast analytical database for market data storage
- **Multi-Asset Support**: Stocks, cryptocurrencies, and forex pairs

### AI-Powered Analytics
- **PyTorch LSTM Models**: Deep learning for price prediction
- **Hugging Face Transformers**: Advanced sentiment analysis of market news
- **Scikit-learn Preprocessing**: Feature engineering and data preprocessing
- **Joblib Parallel Processing**: Optimized model inference

### REST API & WebSocket Endpoints
- **FastAPI Backend**: High-performance async API framework
- **Real-time WebSocket Streams**: Live market data and signal feeds
- **Pydantic Validation**: Type-safe request/response handling

### Modern Web Interface
- **Responsive Dashboard**: Real-time visualization of market data
- **Interactive Charts**: Live price feeds and prediction displays
- **Signal Monitoring**: AI-generated buy/sell/hold recommendations

### Production-Ready Architecture
- **Docker Containerization**: Complete microservice setup
- **Nginx Load Balancing**: Production-grade web server configuration
- **Redis Caching**: High-performance data caching layer
- **Structured Logging**: Comprehensive logging with Loguru

## 🛠 Tech Stack

- **Backend**: Python 3.11, FastAPI, AsyncIO
- **AI/ML**: PyTorch, Hugging Face Transformers, Scikit-learn
- **Database**: DuckDB, Redis
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **DevOps**: Docker, Docker Compose, Nginx
- **Data Sources**: yfinance, Alpha Vantage API

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Intelligent-RealTime-Financial-Market-Analyzer
```

2. **Create and activate virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys (optional for basic functionality)
```

5. **Run the application**
```bash
python main.py
```

6. **Access the dashboard**
- API Documentation: http://localhost:8000/docs
- Web Dashboard: Open `frontend/index.html` in your browser
- Health Check: http://localhost:8000/api/v1/health

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Access services**
- Web Dashboard: http://localhost
- API: http://localhost:8000
- Redis: localhost:6379

## 📊 API Endpoints

### Market Data
- `GET /api/v1/market-data/{symbol}` - Get historical market data
- `WebSocket /api/v1/ws/market-data` - Real-time market data stream

### AI Predictions
- `POST /api/v1/predict` - Generate price predictions
- `GET /api/v1/signals/{symbol}` - Get AI-generated market signals
- `WebSocket /api/v1/ws/signals` - Real-time signal stream

### Sentiment Analysis
- `POST /api/v1/sentiment` - Analyze text sentiment

### System
- `GET /api/v1/health` - Health check endpoint

## 🔧 Configuration

### Environment Variables
```bash
# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Database
DATABASE_URL=duckdb:///./data/market_data.db

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# AI Models
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
MODEL_PATH=./models/

# WebSocket Settings
WS_MAX_CONNECTIONS=100
WS_HEARTBEAT_INTERVAL=30
```

## 🧠 AI Models

### Price Prediction Model
- **Architecture**: LSTM Neural Network
- **Features**: Technical indicators (RSI, MACD, SMA), price history, volume
- **Output**: Price prediction with confidence score

### Sentiment Analysis
- **Model**: RoBERTa-based transformer from Hugging Face
- **Input**: Market news, social media text, analyst reports
- **Output**: Sentiment classification (positive/negative/neutral) with confidence

### Signal Generation
- **Approach**: Ensemble of technical analysis and ML predictions
- **Indicators**: RSI, MACD, Moving averages, volatility
- **Output**: Buy/Sell/Hold signals with strength scores

## 📈 Usage Examples

### Get Market Data
```python
import requests

response = requests.get("http://localhost:8000/api/v1/market-data/AAPL")
data = response.json()
```

### Generate Prediction
```python
prediction_request = {
    "symbol": "AAPL",
    "timeframe": "1h"
}
response = requests.post("http://localhost:8000/api/v1/predict", json=prediction_request)
prediction = response.json()
```

### Analyze Sentiment
```python
sentiment_request = {
    "text": "Apple stock shows strong momentum with excellent quarterly results!"
}
response = requests.post("http://localhost:8000/api/v1/sentiment", json=sentiment_request)
sentiment = response.json()
```

## 🔄 Real-Time Features

### WebSocket Connections
```javascript
// Market data stream
const marketWs = new WebSocket('ws://localhost:8000/api/v1/ws/market-data');
marketWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Live market data:', data);
};

// Signal stream
const signalWs = new WebSocket('ws://localhost:8000/api/v1/ws/signals');
signalWs.onmessage = (event) => {
    const signal = JSON.parse(event.data);
    console.log('New signal:', signal);
};
```

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Nginx         │    │   FastAPI       │
│   (HTML/JS)     │◄──►│   (Reverse      │◄──►│   (Python)      │
│                 │    │    Proxy)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐             │
                       │   Redis         │◄────────────┤
                       │   (Cache)       │             │
                       └─────────────────┘             │
                                                       │
                       ┌─────────────────┐             │
                       │   DuckDB        │◄────────────┘
                       │   (Analytics)   │
                       └─────────────────┘
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Lint code
flake8 app/
black app/
```

## 📝 Development

### Project Structure
```
├── app/
│   ├── api/           # FastAPI endpoints
│   ├── core/          # Configuration and logging
│   ├── models/        # Pydantic schemas
│   ├── services/      # Business logic
│   ├── data/          # Data processing
│   └── utils/         # Utilities
├── frontend/          # Web interface
├── tests/            # Test suite
├── data/             # Database files
├── logs/             # Application logs
├── models/           # Trained ML models
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### Adding New Features
1. Create service in `app/services/`
2. Add endpoints in `app/api/endpoints.py`
3. Update schemas in `app/models/schemas.py`
4. Add tests in `tests/`

## 🚨 Troubleshooting

### Common Issues

**Port already in use**
```bash
lsof -ti:8000 | xargs kill -9
```

**Database connection issues**
- Ensure `data/` directory exists and is writable
- Check DuckDB file permissions

**Model loading errors**
- First run may take time to download Hugging Face models
- Ensure sufficient disk space for model cache

**WebSocket connection failures**
- Check firewall settings
- Verify CORS configuration for frontend

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation at `/docs` endpoint
- Review the API documentation at `/redoc` endpoint

---

**Built with ❤️ using modern AI and web technologies** AI-based signal detection and live streaming data pipelines.
