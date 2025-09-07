"""
FastAPI endpoints for the Financial Market Analyzer
"""

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.schemas import (
    MarketDataPoint,
    PredictionRequest,
    PredictionResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    MarketSignal,
    HealthCheck,
)
from ..services.data_ingestion import DataIngestionService
from ..services.prediction import PredictionService
from ..services.sentiment import SentimentService
from ..core.logging import setup_logging

logger = setup_logging()
router = APIRouter()

# Initialize services
data_service = DataIngestionService()
prediction_service = PredictionService()
sentiment_service = SentimentService()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        services={
            "data_ingestion": "healthy",
            "prediction": "healthy",
            "sentiment": "healthy",
        }
    )


@router.get("/market-data/{symbol}", response_model=List[MarketDataPoint])
async def get_market_data(symbol: str, limit: int = 100):
    """Get historical market data for a symbol"""
    try:
        data = await data_service.get_historical_data(symbol, limit)
        return data
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Generate price prediction for a symbol"""
    try:
        prediction = await prediction_service.predict(
            request.symbol, request.timeframe, request.features
        )
        return prediction
    except Exception as e:
        logger.error(f"Error predicting price for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of text"""
    try:
        result = await sentiment_service.analyze(request.text, request.symbol)
        return result
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{symbol}", response_model=List[MarketSignal])
async def get_market_signals(symbol: str, limit: int = 10):
    """Get market signals for a symbol"""
    try:
        signals = await prediction_service.get_signals(symbol, limit)
        return signals
    except Exception as e:
        logger.error(f"Error fetching signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Wait for client message (symbol subscription)
            message = await websocket.receive_text()
            logger.info(f"Received subscription request: {message}")

            # Start streaming data for the requested symbol
            await data_service.stream_data(websocket, message)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time market signals"""
    await websocket.accept()
    logger.info("Signal WebSocket connection established")

    try:
        while True:
            # Stream real-time signals
            await prediction_service.stream_signals(websocket)
            await asyncio.sleep(5)  # Send signals every 5 seconds

    except WebSocketDisconnect:
        logger.info("Signal WebSocket connection closed")
    except Exception as e:
        logger.error(f"Signal WebSocket error: {e}")
        await websocket.close()
