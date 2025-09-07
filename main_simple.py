"""
Simplified FastAPI application for testing without heavy ML dependencies
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
from typing import List
import json

from app.core.config import settings
from app.core.logging import setup_logging
from app.models.schemas import HealthCheck, MarketDataPoint, AssetType

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Intelligent Real-Time Financial Market Analyzer",
    description="AI-powered real-time financial market analytics with predictive signals",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint - redirect to frontend"""
    return {"message": "Intelligent Real-Time Financial Market Analyzer API", "docs": "/docs"}


@app.get("/api/v1/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        services={
            "api": "healthy",
            "database": "healthy",
            "logging": "healthy"
        }
    )


@app.get("/api/v1/market-data/{symbol}", response_model=List[MarketDataPoint])
async def get_market_data(symbol: str, limit: int = 100):
    """Get mock market data for testing"""
    try:
        # Generate mock data for testing
        mock_data = []
        base_price = 150.0
        
        for i in range(min(limit, 10)):
            price = base_price + (i * 0.5) + ((-1) ** i * 2.0)
            mock_data.append(MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=price,
                volume=1000000 + (i * 50000),
                asset_type=AssetType.STOCK
            ))
        
        logger.info(f"Generated {len(mock_data)} mock data points for {symbol}")
        return mock_data
        
    except Exception as e:
        logger.error(f"Error generating mock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict")
async def predict_price(request: dict):
    """Mock prediction endpoint"""
    try:
        symbol = request.get("symbol", "UNKNOWN")
        current_price = 150.0
        predicted_price = current_price * 1.02  # 2% increase
        
        return {
            "symbol": symbol,
            "prediction": predicted_price,
            "confidence": 0.75,
            "timestamp": datetime.utcnow().isoformat(),
            "features_used": ["price", "volume", "sma_5", "rsi"]
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sentiment")
async def analyze_sentiment(request: dict):
    """Mock sentiment analysis endpoint"""
    try:
        text = request.get("text", "")
        
        # Simple mock sentiment based on keywords
        positive_words = ["good", "great", "excellent", "positive", "bullish", "up", "gain"]
        negative_words = ["bad", "terrible", "negative", "bearish", "down", "loss", "crash"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = 0.8
        elif negative_count > positive_count:
            sentiment = "negative" 
            confidence = 0.8
        else:
            sentiment = "neutral"
            confidence = 0.6
            
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                "positive": confidence if sentiment == "positive" else 0.3,
                "neutral": confidence if sentiment == "neutral" else 0.4,
                "negative": confidence if sentiment == "negative" else 0.3
            }
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/signals/{symbol}")
async def get_market_signals(symbol: str, limit: int = 10):
    """Mock market signals endpoint"""
    try:
        # Generate mock signal
        signal = {
            "symbol": symbol,
            "signal_type": "buy",
            "strength": 0.75,
            "timestamp": datetime.utcnow().isoformat(),
            "indicators": {
                "rsi": 45.2,
                "macd": 1.2,
                "sma_5": 148.5,
                "sma_20": 145.8,
                "current_price": 150.0,
                "predicted_price": 153.0
            },
            "sentiment_score": 0.7
        }
        
        return [signal]
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Intelligent Real-Time Financial Market Analyzer (Simplified)")
    logger.info(f"Server running on {settings.host}:{settings.port}")
    logger.info("Frontend available at: /static/index.html")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Financial Market Analyzer")


if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
