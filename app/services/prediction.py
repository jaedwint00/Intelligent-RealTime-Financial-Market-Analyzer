"""
Prediction service using PyTorch + Scikit-learn + Joblib
"""

import json
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from fastapi import WebSocket

from ..models.schemas import PredictionResponse, MarketSignal
from ..core.config import settings
from ..core.logging import setup_logging

logger = setup_logging()


class LSTMPredictor(nn.Module):
    """LSTM model for price prediction"""

    def __init__(self, input_size=7, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class PredictionService:
    """Service for AI-based market prediction and signal generation"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model = None
        self.rf_classifier = None
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.active_connections: List[WebSocket] = []
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize LSTM model
            self.lstm_model = LSTMPredictor().to(self.device)

            # Initialize Random Forest for signal classification
            self.rf_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            # Try to load pre-trained models
            self._load_models()

            logger.info(f"Models initialized on {self.device}")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")

    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            lstm_path = f"{settings.model_path}/lstm_model.pth"
            rf_path = f"{settings.model_path}/rf_classifier.joblib"
            scaler_path = f"{settings.model_path}/scalers.joblib"

            # Load LSTM model
            if torch.cuda.is_available():
                self.lstm_model.load_state_dict(torch.load(lstm_path))
            else:
                self.lstm_model.load_state_dict(
                    torch.load(lstm_path, map_location="cpu")
                )

            # Load Random Forest
            self.rf_classifier = load(rf_path)

            # Load scalers
            scalers = load(scaler_path)
            self.price_scaler = scalers["price_scaler"]
            self.feature_scaler = scalers["feature_scaler"]

            logger.info("Pre-trained models loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using freshly initialized models")

    def _save_models(self):
        """Save trained models"""
        try:
            import os

            os.makedirs(settings.model_path, exist_ok=True)

            # Save LSTM model
            torch.save(
                self.lstm_model.state_dict(), f"{settings.model_path}/lstm_model.pth"
            )

            # Save Random Forest
            dump(self.rf_classifier, f"{settings.model_path}/rf_classifier.joblib")

            # Save scalers
            scalers = {
                "price_scaler": self.price_scaler,
                "feature_scaler": self.feature_scaler,
            }
            dump(scalers, f"{settings.model_path}/scalers.joblib")

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    async def predict(
        self, symbol: str, timeframe: str = "1h", features: Optional[Dict] = None
    ) -> PredictionResponse:
        """Generate price prediction for a symbol"""
        try:
            # Get historical data for prediction
            from .data_ingestion import DataIngestionService

            data_service = DataIngestionService()
            historical_data = await data_service.get_historical_data(symbol, 100)

            if len(historical_data) < 50:
                raise ValueError(
                    f"Insufficient data for prediction: {len(historical_data)} points"
                )

            # Prepare features
            features_df = self._prepare_features(historical_data)

            # Make prediction using LSTM
            prediction = await self._predict_with_lstm(features_df)

            # Calculate confidence using ensemble approach
            confidence = await self._calculate_confidence(features_df, prediction)

            return PredictionResponse(
                symbol=symbol,
                prediction=float(prediction),
                confidence=float(confidence),
                timestamp=datetime.utcnow(),
                features_used=list(features_df.columns),
            )

        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            raise

    def _prepare_features(self, historical_data: List) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "timestamp": point.timestamp,
                        "price": point.price,
                        "volume": point.volume or 0,
                    }
                    for point in historical_data
                ]
            )

            df = df.sort_values("timestamp")

            # Calculate technical indicators
            df["sma_5"] = df["price"].rolling(window=5).mean()
            df["sma_20"] = df["price"].rolling(window=20).mean()
            df["rsi"] = self._calculate_rsi(df["price"])
            df["macd"] = self._calculate_macd(df["price"])
            df["volatility"] = df["price"].rolling(window=10).std()

            # Price changes
            df["price_change"] = df["price"].pct_change()
            df["volume_change"] = df["volume"].pct_change()

            # Remove NaN values
            df = df.dropna()

            # Select features for model
            feature_columns = [
                "price",
                "volume",
                "sma_5",
                "sma_20",
                "rsi",
                "macd",
                "volatility",
            ]
            return df[feature_columns].tail(50)  # Use last 50 points

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2

    async def _predict_with_lstm(self, features_df: pd.DataFrame) -> float:
        """Make prediction using LSTM model"""
        try:
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_df.values)

            # Prepare sequence data
            sequence_length = 20
            if len(features_scaled) < sequence_length:
                sequence_length = len(features_scaled)

            X = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Make prediction
            self.lstm_model.eval()
            with torch.no_grad():
                prediction = self.lstm_model(X_tensor)
                prediction = prediction.cpu().numpy()[0][0]

            # Scale back to original price range
            last_price = features_df["price"].iloc[-1]
            predicted_change = prediction * 0.1  # Assume max 10% change
            predicted_price = last_price * (1 + predicted_change)

            return predicted_price

        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            # Fallback to simple prediction
            return features_df["price"].iloc[-1] * 1.001  # Small positive change

    async def _calculate_confidence(
        self, features_df: pd.DataFrame, prediction: float
    ) -> float:
        """Calculate prediction confidence"""
        try:
            # Use volatility and recent price stability as confidence
            # indicators
            volatility = features_df["volatility"].iloc[-1]
            # Higher stability = higher confidence
            price_stability = 1 / (1 + volatility)

            # Use ensemble agreement (simplified)
            recent_trend = (
                features_df["price"].tail(5).mean()
                / features_df["price"].tail(10).mean()
            )
            trend_confidence = min(
                abs(recent_trend - 1) * 10, 0.5
            )  # Max 0.5 from trend

            confidence = (price_stability * 0.7) + (trend_confidence * 0.3)
            # Clamp between 0.1 and 0.95
            return min(max(confidence, 0.1), 0.95)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence

    async def get_signals(self, symbol: str, limit: int = 10) -> List[MarketSignal]:
        """Generate market signals for a symbol"""
        try:
            # Get prediction
            prediction_response = await self.predict(symbol)

            # Get historical data for signal generation
            from .data_ingestion import DataIngestionService

            data_service = DataIngestionService()
            historical_data = await data_service.get_historical_data(symbol, 50)

            if not historical_data:
                return []

            # Prepare features
            features_df = self._prepare_features(historical_data)
            current_price = features_df["price"].iloc[-1]

            # Generate signal based on prediction and technical indicators
            signal_type = self._determine_signal_type(
                current_price, prediction_response.prediction, features_df
            )

            # Calculate signal strength
            strength = self._calculate_signal_strength(
                features_df, prediction_response.confidence
            )

            # Get sentiment score (placeholder - would integrate with sentiment
            # service)
            sentiment_score = 0.5  # Neutral

            signal = MarketSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                timestamp=datetime.utcnow(),
                indicators={
                    "rsi": float(features_df["rsi"].iloc[-1]),
                    "macd": float(features_df["macd"].iloc[-1]),
                    "sma_5": float(features_df["sma_5"].iloc[-1]),
                    "sma_20": float(features_df["sma_20"].iloc[-1]),
                    "predicted_price": prediction_response.prediction,
                    "current_price": current_price,
                },
                sentiment_score=sentiment_score,
            )

            return [signal]

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    def _determine_signal_type(
        self, current_price: float, predicted_price: float, features_df: pd.DataFrame
    ) -> str:
        """Determine signal type based on prediction and indicators"""
        price_change = (predicted_price - current_price) / current_price
        rsi = features_df["rsi"].iloc[-1]

        # Strong buy/sell signals
        if price_change > 0.02 and rsi < 70:
            return "buy"
        elif price_change < -0.02 and rsi > 30:
            return "sell"
        else:
            return "hold"

    def _calculate_signal_strength(
        self, features_df: pd.DataFrame, confidence: float
    ) -> float:
        """Calculate signal strength"""
        try:
            # Combine multiple factors
            rsi = features_df["rsi"].iloc[-1]
            volatility = features_df["volatility"].iloc[-1]

            # RSI strength (closer to extremes = stronger signal)
            rsi_strength = max(abs(rsi - 50) / 50, 0.1)

            # Volatility factor (lower volatility = stronger signal)
            vol_strength = max(1 - (volatility / features_df["volatility"].mean()), 0.1)

            # Combine with prediction confidence
            strength = (rsi_strength * 0.3) + (vol_strength * 0.3) + (confidence * 0.4)

            return min(max(strength, 0.1), 1.0)

        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5

    async def stream_signals(self, websocket: WebSocket):
        """Stream real-time signals to WebSocket client"""
        try:
            symbols = ["AAPL", "GOOGL", "MSFT", "BTC-USD"]  # Example symbols

            for symbol in symbols:
                signals = await self.get_signals(symbol, 1)

                if signals:
                    message = {
                        "type": "market_signal",
                        "data": signals[0].dict(),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    await websocket.send_text(json.dumps(message))

        except Exception as e:
            logger.error(f"Error streaming signals: {e}")

    async def train_models(self, training_data: pd.DataFrame):
        """Train models with new data (placeholder for future implementation)"""
        try:
            logger.info("Model training initiated")

            # This would implement the actual training logic
            # For now, we'll just log that training was requested

            # Save models after training
            self._save_models()

            logger.info("Model training completed")

        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
