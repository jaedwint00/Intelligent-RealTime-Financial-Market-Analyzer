"""
Real-time data ingestion service using AsyncIO + WebSockets + DuckDB
"""

import asyncio
import json
from datetime import datetime
from typing import List

import duckdb
import yfinance as yf
from fastapi import WebSocket

from ..models.schemas import MarketDataPoint, AssetType
from ..core.logging import setup_logging

logger = setup_logging()


class DataIngestionService:
    """Service for ingesting real-time market data"""

    def __init__(self):
        self.db_connection = None
        self.active_connections: List[WebSocket] = []
        self.setup_database()

    def setup_database(self):
        """Initialize DuckDB database and tables"""
        try:
            # Use in-memory database to avoid file locking issues
            self.db_connection = duckdb.connect(":memory:")

            # Create market data table
            self.db_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER,
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    price DOUBLE,
                    volume DOUBLE,
                    asset_type VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index for faster queries
            self.db_connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
                ON market_data(symbol, timestamp)
            """
            )

            logger.info("Database initialized successfully (in-memory)")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def get_historical_data(
        self, symbol: str, limit: int = 100
    ) -> List[MarketDataPoint]:
        """Fetch historical data from database or external API"""
        try:
            # First try to get from database
            query = """
                SELECT symbol, timestamp, price, volume, asset_type
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """

            result = self.db_connection.execute(query, [symbol, limit]).fetchall()

            if result:
                return [
                    MarketDataPoint(
                        symbol=row[0],
                        timestamp=row[1],
                        price=row[2],
                        volume=row[3],
                        asset_type=AssetType(row[4]),
                    )
                    for row in result
                ]

            # If no data in database, fetch from external API
            return await self._fetch_external_data(symbol, limit)

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

    async def _fetch_external_data(
        self, symbol: str, limit: int
    ) -> List[MarketDataPoint]:
        """Fetch data from external APIs (yfinance)"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")

            data_points = []
            for timestamp, row in hist.tail(limit).iterrows():
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=float(row["Close"]),
                    volume=float(row["Volume"]) if "Volume" in row else None,
                    asset_type=self._determine_asset_type(symbol),
                )
                data_points.append(data_point)

                # Store in database
                await self._store_data_point(data_point)

            return data_points

        except Exception as e:
            logger.error(f"Error fetching external data for {symbol}: {e}")
            return []

    def _determine_asset_type(self, symbol: str) -> AssetType:
        """Determine asset type based on symbol"""
        if symbol.endswith("-USD") or symbol.endswith("USDT"):
            return AssetType.CRYPTO
        if "/" in symbol:
            return AssetType.FOREX
        return AssetType.STOCK

    async def _store_data_point(self, data_point: MarketDataPoint):
        """Store data point in database"""
        try:
            # Generate a simple ID based on timestamp and symbol hash
            import hashlib

            id_hash = hashlib.md5(
                f"{data_point.symbol}{data_point.timestamp}".encode()
            ).hexdigest()
            data_id = int(id_hash[:8], 16) % 1000000  # Convert to integer

            query = """
                INSERT INTO market_data
                (id, symbol, timestamp, price, volume, asset_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """

            self.db_connection.execute(
                query,
                [
                    data_id,
                    data_point.symbol,
                    data_point.timestamp,
                    data_point.price,
                    data_point.volume,
                    data_point.asset_type.value,
                ],
            )

        except Exception as e:
            logger.error(f"Error storing data point: {e}")

    async def stream_data(self, websocket: WebSocket, symbol: str):
        """Stream real-time data to WebSocket client"""
        self.active_connections.append(websocket)

        try:
            while True:
                # Fetch latest data point
                data_points = await self.get_historical_data(symbol, 1)

                if data_points:
                    message = {
                        "type": "market_data",
                        "data": data_points[0].dict(),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    await websocket.send_text(json.dumps(message))

                # Wait before next update
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error streaming data: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def start_background_ingestion(self):
        """Start background task for continuous data ingestion"""
        symbols = ["AAPL", "GOOGL", "MSFT", "BTC-USD", "ETH-USD"]  # Example symbols

        while True:
            try:
                tasks = [self._ingest_symbol_data(symbol) for symbol in symbols]
                await asyncio.gather(*tasks)

                # Wait before next batch
                await asyncio.sleep(60)  # Ingest every minute

            except Exception as e:
                logger.error(f"Background ingestion error: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def _ingest_symbol_data(self, symbol: str):
        """Ingest data for a single symbol"""
        try:
            data_points = await self._fetch_external_data(symbol, 1)
            logger.info(f"Ingested {len(data_points)} data points for {symbol}")

        except Exception as e:
            logger.error(f"Error ingesting data for {symbol}: {e}")

    def close(self):
        """Close database connection"""
        if self.db_connection:
            self.db_connection.close()
