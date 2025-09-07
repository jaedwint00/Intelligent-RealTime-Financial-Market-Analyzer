"""
Pydantic models for request/response validation
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"


class MarketDataPoint(BaseModel):
    """Individual market data point"""
    symbol: str = Field(..., description="Asset symbol (e.g., AAPL, BTC-USD)")
    timestamp: datetime = Field(..., description="Data timestamp")
    price: float = Field(..., description="Current price")
    volume: Optional[float] = Field(None, description="Trading volume")
    asset_type: AssetType = Field(..., description="Type of asset")


class PredictionRequest(BaseModel):
    """Request for price prediction"""
    symbol: str = Field(..., description="Asset symbol")
    timeframe: str = Field("1h", description="Prediction timeframe")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features")


class PredictionResponse(BaseModel):
    """Response with prediction results"""
    symbol: str
    prediction: float = Field(..., description="Predicted price")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    timestamp: datetime
    features_used: List[str]


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    text: str = Field(..., description="Text to analyze")
    symbol: Optional[str] = Field(None, description="Related asset symbol")


class SentimentAnalysisResponse(BaseModel):
    """Response with sentiment analysis results"""
    text: str
    sentiment: str = Field(..., description="Sentiment label")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scores: Dict[str, float] = Field(..., description="Individual sentiment scores")


class MarketSignal(BaseModel):
    """Market signal detection result"""
    symbol: str
    signal_type: str = Field(..., description="Type of signal (buy/sell/hold)")
    strength: float = Field(..., description="Signal strength (0-1)")
    timestamp: datetime
    indicators: Dict[str, float] = Field(..., description="Technical indicators")
    sentiment_score: Optional[float] = Field(None, description="Sentiment component")


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
