import pandas as pd
import numpy as np
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import logging
import requests


class CoinRegistry:
    """Centralized cryptocurrency discovery and management system"""

    def __init__(self, config_manager, data_manager=None):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)

        # Registry storage
        self.coins = {}
        self.coin_mappings = {}
        self.exchange_mappings = {}
        self.metadata_cache = {}
        self._lock = threading.Lock()

        # Registry files
        self.registry_path = Path("data/registry")
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.coins_file = self.registry_path / "coins.json"
        self.mappings_file = self.registry_path / "mappings.json"

        # Update tracking
        self.last_update = None
        self.update_interval_hours = 24
        self.auto_update_active = False

        # Initialize registry
        self._load_registry()
        self._initialize_default_coins()

        # Start auto-update if configured
        if self.config_manager.get("auto_update_registry", True):
            self.start_auto_update()

    def _load_registry(self):
        """Load registry from disk"""
        try:
            # Load coins data
            if self.coins_file.exists():
                with open(self.coins_file, "r") as f:
                    self.coins = json.load(f)
                self.logger.info(f"Loaded {len(self.coins)} coins from registry")

            # Load mappings data
            if self.mappings_file.exists():
                with open(self.mappings_file, "r") as f:
                    mappings_data = json.load(f)
                    self.coin_mappings = mappings_data.get("coin_mappings", {})
                    self.exchange_mappings = mappings_data.get("exchange_mappings", {})
                    self.last_update = mappings_data.get("last_update")
                self.logger.info("Loaded registry mappings")

        except Exception as e:
            self.logger.error(f"Error loading registry: {str(e)}")
            self.coins = {}
            self.coin_mappings = {}
            self.exchange_mappings = {}

    def _save_registry(self):
        """Save registry to disk"""
        try:
            # Save coins data
            with open(self.coins_file, "w") as f:
                json.dump(self.coins, f, indent=2)

            # Save mappings data
            mappings_data = {
                "coin_mappings": self.coin_mappings,
                "exchange_mappings": self.exchange_mappings,
                "last_update": self.last_update,
            }
            with open(self.mappings_file, "w") as f:
                json.dump(mappings_data, f, indent=2)

            self.logger.debug("Registry saved to disk")

        except Exception as e:
            self.logger.error(f"Error saving registry: {str(e)}")

    def _initialize_default_coins(self):
        """Initialize with default cryptocurrency data"""
        if not self.coins:
            default_coins = self._get_default_coin_list()

            for coin in default_coins:
                self.add_coin(
                    coin["symbol"],
                    coin["name"],
                    coin.get("market_cap_rank", 999),
                    coin.get("metadata", {}),
                )

            self.logger.info(f"Initialized registry with {len(default_coins)} default coins")

    def _get_default_coin_list(self) -> List[Dict[str, Any]]:
        """Get default list of major cryptocurrencies"""
        return [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap_rank": 1},
            {"symbol": "ETH", "name": "Ethereum", "market_cap_rank": 2},
            {"symbol": "BNB", "name": "BNB", "market_cap_rank": 3},
            {"symbol": "XRP", "name": "XRP", "market_cap_rank": 4},
            {"symbol": "ADA", "name": "Cardano", "market_cap_rank": 5},
            {"symbol": "AVAX", "name": "Avalanche", "market_cap_rank": 6},
            {"symbol": "DOGE", "name": "Dogecoin", "market_cap_rank": 7},
            {"symbol": "SOL", "name": "Solana", "market_cap_rank": 8},
            {"symbol": "DOT", "name": "Polkadot", "market_cap_rank": 9},
            {"symbol": "MATIC", "name": "Polygon", "market_cap_rank": 10},
            {"symbol": "LTC", "name": "Litecoin", "market_cap_rank": 11},
            {"symbol": "SHIB", "name": "Shiba Inu", "market_cap_rank": 12},
            {"symbol": "TRX", "name": "TRON", "market_cap_rank": 13},
            {"symbol": "UNI", "name": "Uniswap", "market_cap_rank": 14},
            {"symbol": "LINK", "name": "Chainlink", "market_cap_rank": 15},
            {"symbol": "ATOM", "name": "Cosmos", "market_cap_rank": 16},
            {"symbol": "XMR", "name": "Monero", "market_cap_rank": 17},
            {"symbol": "ETC", "name": "Ethereum Classic", "market_cap_rank": 18},
            {"symbol": "BCH", "name": "Bitcoin Cash", "market_cap_rank": 19},
            {"symbol": "XLM", "name": "Stellar", "market_cap_rank": 20},
            {"symbol": "NEAR", "name": "NEAR Protocol", "market_cap_rank": 21},
            {"symbol": "ALGO", "name": "Algorand", "market_cap_rank": 22},
            {"symbol": "VET", "name": "VeChain", "market_cap_rank": 23},
            {"symbol": "FLOW", "name": "Flow", "market_cap_rank": 24},
            {"symbol": "ICP", "name": "Internet Computer", "market_cap_rank": 25},
            {"symbol": "APT", "name": "Aptos", "market_cap_rank": 26},
            {"symbol": "QNT", "name": "Quant", "market_cap_rank": 27},
            {"symbol": "FIL", "name": "Filecoin", "market_cap_rank": 28},
            {"symbol": "HBAR", "name": "Hedera", "market_cap_rank": 29},
            {"symbol": "MANA", "name": "Decentraland", "market_cap_rank": 30},
            {"symbol": "SAND", "name": "The Sandbox", "market_cap_rank": 31},
            {"symbol": "AXS", "name": "Axie Infinity", "market_cap_rank": 32},
            {"symbol": "THETA", "name": "THETA", "market_cap_rank": 33},
            {"symbol": "AAVE", "name": "Aave", "market_cap_rank": 34},
            {"symbol": "GRT", "name": "The Graph", "market_cap_rank": 35},
            {"symbol": "ENJ", "name": "Enjin Coin", "market_cap_rank": 36},
            {"symbol": "CRV", "name": "Curve DAO Token", "market_cap_rank": 37},
            {"symbol": "MKR", "name": "Maker", "market_cap_rank": 38},
            {"symbol": "SNX", "name": "Synthetix", "market_cap_rank": 39},
            {"symbol": "COMP", "name": "Compound", "market_cap_rank": 40},
            {"symbol": "YFI", "name": "yearn.finance", "market_cap_rank": 41},
            {"symbol": "1INCH", "name": "1inch Network", "market_cap_rank": 42},
            {"symbol": "SUSHI", "name": "SushiSwap", "market_cap_rank": 43},
            {"symbol": "BAT", "name": "Basic Attention Token", "market_cap_rank": 44},
            {"symbol": "ZRX", "name": "0x", "market_cap_rank": 45},
            {"symbol": "LRC", "name": "Loopring", "market_cap_rank": 46},
            {"symbol": "CHZ", "name": "Chiliz", "market_cap_rank": 47},
            {"symbol": "HOT", "name": "Holo", "market_cap_rank": 48},
            {"symbol": "IOTA", "name": "IOTA", "market_cap_rank": 49},
            {"symbol": "ZEC", "name": "Zcash", "market_cap_rank": 50},
        ]

    def start_auto_update(self):
        """Start automatic registry updates"""
        if not self.auto_update_active:
            self.auto_update_active = True
            self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
            self.update_thread.start()
            self.logger.info("Auto-update started for coin registry")

    def stop_auto_update(self):
        """Stop automatic registry updates"""
        self.auto_update_active = False
        self.logger.info("Auto-update stopped for coin registry")

    def _auto_update_loop(self):
        """Auto-update loop"""
        while self.auto_update_active:
            try:
                # Check if update is needed
                if self._should_update():
                    self.update_from_external_source()

                # Sleep for 1 hour between checks
                time.sleep(3600)

            except Exception as e:
                self.logger.error(f"Error in auto-update loop: {str(e)}")
                time.sleep(1800)  # Sleep 30 minutes on error

    def _should_update(self) -> bool:
        """Check if registry should be updated"""
        if self.last_update is None:
            return True

        try:
            last_update_dt = datetime.fromisoformat(self.last_update)
            hours_since_update = (datetime.now() - last_update_dt).total_seconds() / 3600
            return hours_since_update >= self.update_interval_hours
        except Exception:
            return True

    def add_coin(
        self, symbol: str, name: str, rank: int = 999, metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a new coin to the registry"""
        try:
            with self._lock:
                coin_id = symbol.upper()

                coin_data = {
                    "symbol": coin_id,
                    "name": name,
                    "market_cap_rank": rank,
                    "added_at": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "active": True,
                    "exchanges": [],
                    "trading_pairs": [],
                }

                self.coins[coin_id] = coin_data

                # Update mappings
                self.coin_mappings[coin_id] = {
                    "primary_symbol": coin_id,
                    "alternative_symbols": [symbol.lower(), symbol.upper()],
                    "name_variations": [name, name.lower(), name.upper()],
                }

                self._save_registry()

            self.logger.info(f"Added coin {coin_id} ({name}) to registry")
            return True

        except Exception as e:
            self.logger.error(f"Error adding coin {symbol}: {str(e)}")
            return False

    def remove_coin(self, symbol: str) -> bool:
        """Remove a coin from the registry"""
        try:
            with self._lock:
                coin_id = symbol.upper()

                if coin_id in self.coins:
                    # Mark as inactive instead of deleting
                    self.coins[coin_id]["active"] = False
                    self.coins[coin_id]["deactivated_at"] = datetime.now().isoformat()

                    self._save_registry()

                    self.logger.info(f"Deactivated coin {coin_id} in registry")
                    return True
                else:
                    self.logger.warning(f"Coin {coin_id} not found in registry")
                    return False

        except Exception as e:
            self.logger.error(f"Error removing coin {symbol}: {str(e)}")
            return False

    def get_coin(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get coin information"""
        with self._lock:
            coin_id = self._resolve_symbol(symbol)
            if coin_id and coin_id in self.coins:
                return self.coins[coin_id].copy()
            return None

    def get_all_coins(self, active_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get all coins in registry"""
        with self._lock:
            if active_only:
                return {k: v for k, v in self.coins.items() if v.get("active", True)}
            else:
                return self.coins.copy()

    def get_coins_by_rank(
        self, max_rank: int = 100, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get coins by market cap rank"""
        coins = self.get_all_coins(active_only)

        # Filter by rank and sort
        ranked_coins = [
            coin for coin in coins.values() if coin.get("market_cap_rank", 999) <= max_rank
        ]

        return sorted(ranked_coins, key=lambda x: x.get("market_cap_rank", 999))

    def search_coins(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search coins by symbol or name"""
        query_lower = query.lower()
        matches = []

        with self._lock:
            for coin in self.coins.values():
                if not coin.get("active", True):
                    continue

                # Check symbol match
                if query_lower in coin["symbol"].lower():
                    matches.append((coin, 3))  # High priority for symbol match
                    continue

                # Check name match
                if query_lower in coin["name"].lower():
                    matches.append((coin, 2))  # Medium priority for name match
                    continue

                # Check alternative symbols
                mappings = self.coin_mappings.get(coin["symbol"], {})
                alt_symbols = mappings.get("alternative_symbols", [])
                if any(query_lower in alt.lower() for alt in alt_symbols):
                    matches.append((coin, 1))  # Low priority for alternative match

        # Sort by priority and return
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:limit]]

    def _resolve_symbol(self, symbol: str) -> Optional[str]:
        """Resolve symbol to primary coin ID"""
        symbol_upper = symbol.upper()

        # Direct match
        if symbol_upper in self.coins:
            return symbol_upper

        # Check mappings
        for coin_id, mappings in self.coin_mappings.items():
            alt_symbols = mappings.get("alternative_symbols", [])
            if (
                symbol in alt_symbols
                or symbol.upper() in alt_symbols
                or symbol.lower() in alt_symbols
            ):
                return coin_id

        return None

    def update_coin_metadata(self, symbol: str, metadata: Dict[str, Any]) -> bool:
        """Update coin metadata"""
        try:
            with self._lock:
                coin_id = self._resolve_symbol(symbol)
                if coin_id and coin_id in self.coins:
                    self.coins[coin_id]["metadata"].update(metadata)
                    self.coins[coin_id]["updated_at"] = datetime.now().isoformat()
                    self._save_registry()
                    return True
                else:
                    return False

        except Exception as e:
            self.logger.error(f"Error updating metadata for {symbol}: {str(e)}")
            return False

    def add_exchange_mapping(self, coin_symbol: str, exchange: str, exchange_symbol: str) -> bool:
        """Add exchange-specific symbol mapping"""
        try:
            with self._lock:
                coin_id = self._resolve_symbol(coin_symbol)
                if not coin_id:
                    return False

                if exchange not in self.exchange_mappings:
                    self.exchange_mappings[exchange] = {}

                self.exchange_mappings[exchange][exchange_symbol] = coin_id

                # Update coin's exchange list
                if coin_id in self.coins:
                    exchanges = self.coins[coin_id].get("exchanges", [])
                    if exchange not in exchanges:
                        exchanges.append(exchange)
                        self.coins[coin_id]["exchanges"] = exchanges

                self._save_registry()
                return True

        except Exception as e:
            self.logger.error(f"Error adding exchange mapping: {str(e)}")
            return False

    def resolve_exchange_symbol(self, exchange: str, exchange_symbol: str) -> Optional[str]:
        """Resolve exchange-specific symbol to standard symbol"""
        with self._lock:
            exchange_mappings = self.exchange_mappings.get(exchange, {})
            return exchange_mappings.get(exchange_symbol)

    def get_exchange_symbols(self, exchange: str) -> Dict[str, str]:
        """Get all symbols for a specific exchange"""
        with self._lock:
            return self.exchange_mappings.get(exchange, {}).copy()

    def update_from_external_source(self, source: str = "coingecko") -> bool:
        """Update registry from external data source"""
        try:
            if source == "coingecko":
                return self._update_from_coingecko()
            elif source == "coinmarketcap":
                return self._update_from_coinmarketcap()
            else:
                self.logger.error(f"Unknown external source: {source}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating from external source: {str(e)}")
            return False

    def _update_from_coingecko(self) -> bool:
        """Update from CoinGecko API"""
        try:
            # Free CoinGecko API endpoint
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 500,  # Get top 500 coins
                "page": 1,
                "sparkline": "false",
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            coins_data = response.json()
            updated_count = 0

            with self._lock:
                for coin_data in coins_data:
                    symbol = coin_data["symbol"].upper()
                    name = coin_data["name"]
                    rank = coin_data["market_cap_rank"] or 999

                    # Prepare metadata
                    metadata = {
                        "coingecko_id": coin_data["id"],
                        "current_price": coin_data["current_price"],
                        "market_cap": coin_data["market_cap"],
                        "total_volume": coin_data["total_volume"],
                        "price_change_24h": coin_data["price_change_percentage_24h"],
                        "updated_from_coingecko": datetime.now().isoformat(),
                    }

                    # Add or update coin
                    if symbol in self.coins:
                        # Update existing coin
                        self.coins[symbol].update(
                            {
                                "name": name,
                                "market_cap_rank": rank,
                                "metadata": {**self.coins[symbol].get("metadata", {}), **metadata},
                                "updated_at": datetime.now().isoformat(),
                            }
                        )
                    else:
                        # Add new coin
                        self.add_coin(symbol, name, rank, metadata)

                    updated_count += 1

                self.last_update = datetime.now().isoformat()
                self._save_registry()

            self.logger.info(f"Updated {updated_count} coins from CoinGecko")
            return True

        except requests.RequestException as e:
            self.logger.error(f"Network error updating from CoinGecko: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating from CoinGecko: {str(e)}")
            return False

    def _update_from_coinmarketcap(self) -> bool:
        """Update from CoinMarketCap API (requires API key)"""
        try:
            api_key = self.config_manager.get("api_keys", {}).get("coinmarketcap", "")
            if not api_key:
                self.logger.warning("CoinMarketCap API key not found")
                return False

            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}
            params = {"start": "1", "limit": "500", "convert": "USD"}

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            coins_data = data["data"]
            updated_count = 0

            with self._lock:
                for coin_data in coins_data:
                    symbol = coin_data["symbol"].upper()
                    name = coin_data["name"]
                    rank = coin_data["cmc_rank"]

                    # Prepare metadata
                    quote = coin_data["quote"]["USD"]
                    metadata = {
                        "coinmarketcap_id": coin_data["id"],
                        "current_price": quote["price"],
                        "market_cap": quote["market_cap"],
                        "total_volume": quote["volume_24h"],
                        "price_change_24h": quote["percent_change_24h"],
                        "updated_from_coinmarketcap": datetime.now().isoformat(),
                    }

                    # Add or update coin
                    if symbol in self.coins:
                        # Update existing coin
                        self.coins[symbol].update(
                            {
                                "name": name,
                                "market_cap_rank": rank,
                                "metadata": {**self.coins[symbol].get("metadata", {}), **metadata},
                                "updated_at": datetime.now().isoformat(),
                            }
                        )
                    else:
                        # Add new coin
                        self.add_coin(symbol, name, rank, metadata)

                    updated_count += 1

                self.last_update = datetime.now().isoformat()
                self._save_registry()

            self.logger.info(f"Updated {updated_count} coins from CoinMarketCap")
            return True

        except requests.RequestException as e:
            self.logger.error(f"Network error updating from CoinMarketCap: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating from CoinMarketCap: {str(e)}")
            return False

    def discover_exchange_coins(self, exchange_manager) -> int:
        """Discover coins from connected exchanges"""
        try:
            discovered_count = 0

            if not exchange_manager:
                self.logger.warning("No exchange manager provided")
                return 0

            available_exchanges = exchange_manager.get_available_exchanges()

            for exchange_name in available_exchanges:
                try:
                    # Get markets from exchange
                    markets = exchange_manager.get_exchange_markets(exchange_name)

                    for symbol, market in markets.items():
                        try:
                            # Extract base currency
                            base_currency = market.get("base", "").upper()
                            if not base_currency:
                                continue

                            # Add exchange mapping
                            self.add_exchange_mapping(base_currency, exchange_name, symbol)

                            # Add coin if not exists
                            if base_currency not in self.coins:
                                # Use market info to create coin entry
                                coin_name = market.get("baseId", base_currency)
                                self.add_coin(base_currency, coin_name)
                                discovered_count += 1

                            # Update trading pairs
                            coin_data = self.coins.get(base_currency, {})
                            trading_pairs = coin_data.get("trading_pairs", [])
                            if symbol not in trading_pairs:
                                trading_pairs.append(symbol)
                                self.coins[base_currency]["trading_pairs"] = trading_pairs

                        except Exception as e:
                            self.logger.debug(f"Error processing market {symbol}: {str(e)}")
                            continue

                except Exception as e:
                    self.logger.error(f"Error discovering coins from {exchange_name}: {str(e)}")
                    continue

            if discovered_count > 0:
                with self._lock:
                    self._save_registry()

            self.logger.info(f"Discovered {discovered_count} new coins from exchanges")
            return discovered_count

        except Exception as e:
            self.logger.error(f"Error in coin discovery: {str(e)}")
            return 0

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            active_coins = [coin for coin in self.coins.values() if coin.get("active", True)]

            # Count by exchanges
            exchange_counts = {}
            for coin in active_coins:
                for exchange in coin.get("exchanges", []):
                    exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1

            # Count by rank tiers
            rank_tiers = {"top_100": 0, "top_500": 0, "others": 0}
            for coin in active_coins:
                rank = coin.get("market_cap_rank", 999)
                if rank <= 100:
                    rank_tiers["top_100"] += 1
                elif rank <= 500:
                    rank_tiers["top_500"] += 1
                else:
                    rank_tiers["others"] += 1

            return {
                "total_coins": len(self.coins),
                "active_coins": len(active_coins),
                "inactive_coins": len(self.coins) - len(active_coins),
                "last_update": self.last_update,
                "exchange_mappings": len(self.exchange_mappings),
                "coin_mappings": len(self.coin_mappings),
                "exchange_counts": exchange_counts,
                "rank_distribution": rank_tiers,
                "auto_update_active": self.auto_update_active,
            }

    def export_registry(self, file_path: str = None) -> bool:
        """Export registry to JSON file"""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = self.registry_path / f"registry_export_{timestamp}.json"

            export_data = {
                "coins": self.coins,
                "coin_mappings": self.coin_mappings,
                "exchange_mappings": self.exchange_mappings,
                "exported_at": datetime.now().isoformat(),
                "stats": self.get_registry_stats(),
            }

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Registry exported to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting registry: {str(e)}")
            return False

    def import_registry(self, file_path: str, merge: bool = True) -> bool:
        """Import registry from JSON file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"Import file {file_path} not found")
                return False

            with open(file_path, "r") as f:
                import_data = json.load(f)

            with self._lock:
                if merge:
                    # Merge with existing data
                    self.coins.update(import_data.get("coins", {}))
                    self.coin_mappings.update(import_data.get("coin_mappings", {}))
                    self.exchange_mappings.update(import_data.get("exchange_mappings", {}))
                else:
                    # Replace existing data
                    self.coins = import_data.get("coins", {})
                    self.coin_mappings = import_data.get("coin_mappings", {})
                    self.exchange_mappings = import_data.get("exchange_mappings", {})

                self.last_update = datetime.now().isoformat()
                self._save_registry()

            self.logger.info(f"Registry imported from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error importing registry: {str(e)}")
            return False

    def validate_registry(self) -> Dict[str, Any]:
        """Validate registry data integrity"""
        validation_results = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

        try:
            with self._lock:
                # Check for required fields
                for coin_id, coin_data in self.coins.items():
                    if not isinstance(coin_data, dict):
                        validation_results["errors"].append(
                            f"Coin {coin_id} data is not a dictionary"
                        )
                        continue

                    required_fields = ["symbol", "name", "market_cap_rank"]
                    for field in required_fields:
                        if field not in coin_data:
                            validation_results["errors"].append(
                                f"Coin {coin_id} missing required field: {field}"
                            )

                    # Check symbol consistency
                    if coin_data.get("symbol", "").upper() != coin_id:
                        validation_results["warnings"].append(
                            f"Coin {coin_id} has inconsistent symbol: {coin_data.get('symbol')}"
                        )

                # Check mappings consistency
                for coin_id, mappings in self.coin_mappings.items():
                    if coin_id not in self.coins:
                        validation_results["warnings"].append(
                            f"Mapping exists for non-existent coin: {coin_id}"
                        )

                # Check exchange mappings
                for exchange, mappings in self.exchange_mappings.items():
                    for exchange_symbol, coin_id in mappings.items():
                        if coin_id not in self.coins:
                            validation_results["warnings"].append(
                                f"Exchange mapping {exchange}:{exchange_symbol} points to non-existent coin: {coin_id}"
                            )

                # Statistics
                validation_results["statistics"] = {
                    "total_coins_checked": len(self.coins),
                    "total_errors": len(validation_results["errors"]),
                    "total_warnings": len(validation_results["warnings"]),
                }

                if validation_results["errors"]:
                    validation_results["valid"] = False

            return validation_results

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
            return validation_results

    def cleanup_registry(
        self, remove_inactive: bool = False, remove_old_days: int = 365
    ) -> Dict[str, int]:
        """Clean up registry by removing old or inactive entries"""
        cleanup_stats = {"coins_removed": 0, "mappings_cleaned": 0, "exchange_mappings_cleaned": 0}

        try:
            cutoff_date = datetime.now() - timedelta(days=remove_old_days)

            with self._lock:
                # Clean up coins
                coins_to_remove = []

                for coin_id, coin_data in self.coins.items():
                    should_remove = False

                    # Remove inactive coins if requested
                    if remove_inactive and not coin_data.get("active", True):
                        should_remove = True

                    # Remove very old coins
                    if coin_data.get("deactivated_at"):
                        try:
                            deactivated_dt = datetime.fromisoformat(coin_data["deactivated_at"])
                            if deactivated_dt < cutoff_date:
                                should_remove = True
                        except Exception:
                            pass

                    if should_remove:
                        coins_to_remove.append(coin_id)

                # Remove coins
                for coin_id in coins_to_remove:
                    del self.coins[coin_id]
                    cleanup_stats["coins_removed"] += 1

                    # Clean up related mappings
                    if coin_id in self.coin_mappings:
                        del self.coin_mappings[coin_id]
                        cleanup_stats["mappings_cleaned"] += 1

                # Clean up exchange mappings
                for exchange, mappings in list(self.exchange_mappings.items()):
                    cleaned_mappings = {}
                    for exchange_symbol, coin_id in mappings.items():
                        if coin_id in self.coins:
                            cleaned_mappings[exchange_symbol] = coin_id
                        else:
                            cleanup_stats["exchange_mappings_cleaned"] += 1

                    self.exchange_mappings[exchange] = cleaned_mappings

                # Save cleaned registry
                self._save_registry()

            self.logger.info(f"Registry cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            self.logger.error(f"Error in registry cleanup: {str(e)}")
            return cleanup_stats


# Global registry instance
_registry_instance = None
_registry_lock = threading.Lock()


def get_coin_registry(config_manager, data_manager=None):
    """Get global coin registry instance"""
    global _registry_instance

    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = CoinRegistry(config_manager, data_manager)
        return _registry_instance
