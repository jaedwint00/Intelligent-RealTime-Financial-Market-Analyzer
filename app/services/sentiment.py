"""
Sentiment analysis service using Hugging Face Transformers
"""

import asyncio
from typing import Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from joblib import Parallel, delayed

from ..models.schemas import SentimentAnalysisResponse
from ..core.config import settings
from ..core.logging import setup_logging

logger = setup_logging()


class SentimentService:
    """Service for sentiment analysis using Hugging Face models"""

    def __init__(self):
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {settings.sentiment_model}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(settings.sentiment_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.sentiment_model
            )

            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )

            logger.info(f"Sentiment model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            # Fallback to a simpler model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True,
                )
                logger.info("Loaded fallback sentiment model")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise

    async def analyze(
        self, text: str, symbol: Optional[str] = None
    ) -> SentimentAnalysisResponse:
        """Analyze sentiment of given text"""
        try:
            # Run sentiment analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._analyze_text, text)

            # Process results - handle both single result and list format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Multiple results format
                    result = result[0]
                scores = {item["label"].lower(): item["score"] for item in result}
            else:
                # Fallback scores
                scores = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

            # Determine dominant sentiment
            dominant_sentiment = max(scores.items(), key=lambda x: x[1])

            return SentimentAnalysisResponse(
                text=text,
                sentiment=dominant_sentiment[0],
                confidence=dominant_sentiment[1],
                scores=scores,
            )

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Return neutral sentiment as fallback
            return SentimentAnalysisResponse(
                text=text,
                sentiment="neutral",
                confidence=0.5,
                scores={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            )

    def _analyze_text(self, text: str) -> list:
        """Perform sentiment analysis on text"""
        if not self.sentiment_pipeline:
            raise ValueError("Sentiment pipeline not initialized")

        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        return self.sentiment_pipeline(text)

    async def batch_analyze(self, texts: list, symbols: Optional[list] = None) -> list:
        """Analyze sentiment for multiple texts in parallel"""
        try:
            # Use joblib for parallel processing
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: Parallel(n_jobs=-1)(
                    delayed(self._analyze_text)(text) for text in texts
                ),
            )

            # Process results
            responses = []
            for i, result in enumerate(results):
                scores = {item["label"].lower(): item["score"] for item in result}
                dominant_sentiment = max(scores.items(), key=lambda x: x[1])

                responses.append(
                    SentimentAnalysisResponse(
                        text=texts[i],
                        sentiment=dominant_sentiment[0],
                        confidence=dominant_sentiment[1],
                        scores=scores,
                    )
                )

            return responses

        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return []

    async def analyze_market_news(
        self, symbol: str, news_texts: list
    ) -> Dict[str, float]:
        """Analyze sentiment of market news for a specific symbol"""
        try:
            # Filter news related to the symbol
            relevant_texts = [
                text for text in news_texts if symbol.lower() in text.lower()
            ]

            if not relevant_texts:
                return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

            # Analyze sentiment of relevant news
            results = await self.batch_analyze(relevant_texts)

            # Aggregate scores
            total_scores = {"positive": 0, "neutral": 0, "negative": 0}
            count = len(results)

            for result in results:
                for sentiment, score in result.scores.items():
                    if sentiment in total_scores:
                        total_scores[sentiment] += score

            # Average the scores
            avg_scores = {
                sentiment: score / count for sentiment, score in total_scores.items()
            }

            return avg_scores

        except Exception as e:
            logger.error(f"Error analyzing market news sentiment: {e}")
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": settings.sentiment_model,
            "device": self.device,
            "status": "loaded" if self.sentiment_pipeline else "not_loaded",
        }
