"""
CryptoSmartTrader V2 - Daily Analysis Scheduler
Coordinates continuous ML analysis and scraping for daily reporting
"""

import asyncio
import logging
import threading
import time
import subprocess
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import schedule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cache_manager import CacheManager
from core.config_manager import ConfigManager
from core.health_monitor import HealthMonitor
from core.openai_enhanced_analyzer import OpenAIEnhancedAnalyzer


class DailyAnalysisScheduler:
    """Scheduler for coordinating continuous analysis and daily reporting"""

    def __init__(
        self,
        config_manager: ConfigManager,
        cache_manager: CacheManager,
        health_monitor: HealthMonitor,
    ):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI Enhanced Analyzer
        self.openai_analyzer = OpenAIEnhancedAnalyzer(config_manager)

        # Service management
        self.running_services = {}
        self.daily_results = {}
        self.is_running = False

        # Paths
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"

        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize daily analysis cache
        self._init_daily_cache()

    def _init_daily_cache(self):
        """Initialize daily analysis cache structure"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        if not self.cache_manager.get(cache_key):
            daily_structure = {
                "date": today,
                "ml_analysis": {
                    "status": "pending",
                    "predictions": [],
                    "accuracy_metrics": {},
                    "last_update": None,
                },
                "social_sentiment": {
                    "status": "pending",
                    "reddit_data": [],
                    "twitter_data": [],
                    "sentiment_scores": {},
                    "last_update": None,
                },
                "technical_analysis": {
                    "status": "pending",
                    "indicators": {},
                    "signals": [],
                    "last_update": None,
                },
                "daily_summary": {
                    "market_trend": None,
                    "top_performers": [],
                    "sentiment_overview": None,
                    "recommendations": [],
                    "generated_at": None,
                },
            }
            self.cache_manager.set(cache_key, daily_structure, ttl_minutes=1440)  # 24 hours

    def start_daily_analysis(self):
        """Start the daily analysis scheduler"""
        if self.is_running:
            self.logger.warning("Daily analysis scheduler already running")
            return

        self.is_running = True
        self.logger.info("Starting daily analysis scheduler")

        # Schedule daily tasks with OpenAI enhancement
        if schedule:
            schedule.every().day.at("06:00").do(self._start_morning_analysis)
            schedule.every().day.at("12:00").do(self._update_midday_analysis)
            schedule.every().day.at("18:00").do(self._update_evening_analysis)
            schedule.every().day.at("23:30").do(self._generate_daily_report)

            # Schedule continuous tasks with AI enhancement
            schedule.every(15).minutes.do(self._update_ml_predictions)
            schedule.every(5).minutes.do(self._update_social_sentiment)
            schedule.every(30).minutes.do(self._update_technical_analysis)

            # Schedule OpenAI enhanced analysis batches
            schedule.every().hour.do(self._run_openai_enhancement_batch)
            schedule.every().day.at("09:00").do(self._generate_ai_market_insights)
            schedule.every().day.at("15:00").do(self._generate_ai_market_insights)
            schedule.every().day.at("21:00").do(self._generate_ai_market_insights)

        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()

        # Start background services immediately
        self._start_background_services()

    def stop_daily_analysis(self):
        """Stop the daily analysis scheduler"""
        self.is_running = False
        self._stop_background_services()
        if schedule:
            schedule.clear()
        self.logger.info("Daily analysis scheduler stopped")

    def _run_scheduler(self):
        """Run the schedule loop"""
        while self.is_running:
            try:
                if schedule:
                    schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)

    def _start_background_services(self):
        """Start ML analysis and social scraping services"""
        try:
            # Start ML analysis service
            self._start_ml_service()

            # Start social scraping service
            self._start_social_service()

            self.logger.info("Background services started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")

    def _start_ml_service(self):
        """Start ML analysis background service"""
        if sys.platform.startswith("win"):
            script_path = self.project_root / "scripts" / "start_ml_analysis.bat"
            if script_path.exists():
                process = subprocess.Popen(
                    str(script_path),
                    shell=True,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.running_services["ml_analysis"] = process
                self.logger.info("ML analysis service started")
        else:
            # For non-Windows systems, start Python service directly
            self._start_ml_service_python()

    def _start_social_service(self):
        """Start social scraping background service"""
        if sys.platform.startswith("win"):
            script_path = self.project_root / "scripts" / "start_social_scraper.bat"
            if script_path.exists():
                process = subprocess.Popen(
                    str(script_path),
                    shell=True,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.running_services["social_scraper"] = process
                self.logger.info("Social scraping service started")
        else:
            # For non-Windows systems, start Python service directly
            self._start_social_service_python()

    def _start_ml_service_python(self):
        """Start ML service using Python directly"""
        try:
            ml_script = self.project_root / "scripts" / "ml_background_service.py"
            if ml_script.exists():
                process = subprocess.Popen(
                    [sys.executable, str(ml_script)],
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.running_services["ml_analysis"] = process
                self.logger.info("ML analysis Python service started")
        except Exception as e:
            self.logger.error(f"Failed to start ML Python service: {e}")

    def _start_social_service_python(self):
        """Start social service using Python directly"""
        try:
            social_script = self.project_root / "scripts" / "social_scraper_service.py"
            if social_script.exists():
                process = subprocess.Popen(
                    [sys.executable, str(social_script)],
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.running_services["social_scraper"] = process
                self.logger.info("Social scraping Python service started")
        except Exception as e:
            self.logger.error(f"Failed to start social Python service: {e}")

    def _stop_background_services(self):
        """Stop all background services"""
        for service_name, process in self.running_services.items():
            try:
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                    self.logger.info(f"Stopped {service_name} service")
            except Exception as e:
                self.logger.error(f"Error stopping {service_name}: {e}")

        self.running_services.clear()

    def _start_morning_analysis(self):
        """Start comprehensive morning analysis"""
        self.logger.info("Starting morning analysis")
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        # Update analysis status
        daily_data = self.cache_manager.get(cache_key) or {}
        daily_data["ml_analysis"]["status"] = "running"
        daily_data["social_sentiment"]["status"] = "running"
        daily_data["technical_analysis"]["status"] = "running"

        self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)

    def _update_midday_analysis(self):
        """Update analysis at midday"""
        self.logger.info("Updating midday analysis")
        self._collect_analysis_results()

    def _update_evening_analysis(self):
        """Update analysis in the evening"""
        self.logger.info("Updating evening analysis")
        self._collect_analysis_results()

    def _generate_daily_report(self):
        """Generate comprehensive daily report"""
        self.logger.info("Generating daily report")
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        daily_data = self.cache_manager.get(cache_key) or {}

        # Generate summary
        summary = self._generate_daily_summary(daily_data)
        daily_data["daily_summary"] = summary
        daily_data["daily_summary"]["generated_at"] = datetime.now().isoformat()

        # Save to cache and file
        self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)
        self._save_daily_report(daily_data)

    def _update_ml_predictions(self):
        """Update ML predictions"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        # Collect ML results from data directory
        ml_results = self._collect_ml_results()

        if ml_results:
            daily_data = self.cache_manager.get(cache_key) or {}
            daily_data["ml_analysis"]["predictions"].extend(ml_results)
            daily_data["ml_analysis"]["last_update"] = datetime.now().isoformat()
            daily_data["ml_analysis"]["status"] = "active"

            self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)

    def _update_social_sentiment(self):
        """Update social sentiment data"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        # Collect social sentiment results
        social_results = self._collect_social_results()

        if social_results:
            daily_data = self.cache_manager.get(cache_key) or {}
            if social_results.get("reddit"):
                daily_data["social_sentiment"]["reddit_data"].extend(social_results["reddit"])
            if social_results.get("twitter"):
                daily_data["social_sentiment"]["twitter_data"].extend(social_results["twitter"])

            daily_data["social_sentiment"]["last_update"] = datetime.now().isoformat()
            daily_data["social_sentiment"]["status"] = "active"

            self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)

    def _update_technical_analysis(self):
        """Update technical analysis data"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        # Collect technical analysis results
        technical_results = self._collect_technical_results()

        if technical_results:
            daily_data = self.cache_manager.get(cache_key) or {}
            daily_data["technical_analysis"]["indicators"].update(
                technical_results.get("indicators", {})
            daily_data["technical_analysis"]["signals"].extend(technical_results.get("signals", []))
            daily_data["technical_analysis"]["last_update"] = datetime.now().isoformat()
            daily_data["technical_analysis"]["status"] = "active"

            self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)

    def _collect_ml_results(self) -> List[Dict]:
        """Collect ML prediction results from data directory"""
        try:
            ml_data_dir = self.data_dir / "ml_predictions"
            if not ml_data_dir.exists():
                return []

            results = []
            today = datetime.now().strftime("%Y-%m-%d")

            # Look for today's prediction files
            for file_path in ml_data_dir.glob(f"predictions_{today}*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    self.logger.error(f"Error reading ML file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error collecting ML results: {e}")
            return []

    def _collect_social_results(self) -> Dict:
        """Collect social sentiment results"""
        try:
            social_data_dir = self.data_dir / "social"
            if not social_data_dir.exists():
                return {}

            results = {"reddit": [], "twitter": []}
            today = datetime.now().strftime("%Y-%m-%d")

            # Collect Reddit data
            reddit_file = social_data_dir / f"reddit_{today}.json"
            if reddit_file.exists():
                with open(reddit_file, "r") as f:
                    results["reddit"] = json.load(f)

            # Collect Twitter data
            twitter_file = social_data_dir / f"twitter_{today}.json"
            if twitter_file.exists():
                with open(twitter_file, "r") as f:
                    results["twitter"] = json.load(f)

            return results
        except Exception as e:
            self.logger.error(f"Error collecting social results: {e}")
            return {}

    def _collect_technical_results(self) -> Dict:
        """Collect technical analysis results"""
        try:
            tech_data_dir = self.data_dir / "technical"
            if not tech_data_dir.exists():
                return {}

            today = datetime.now().strftime("%Y-%m-%d")
            tech_file = tech_data_dir / f"technical_{today}.json"

            if tech_file.exists():
                with open(tech_file, "r") as f:
                    return json.load(f)

            return {}
        except Exception as e:
            self.logger.error(f"Error collecting technical results: {e}")
            return {}

    def _collect_analysis_results(self):
        """Collect all analysis results"""
        self._update_ml_predictions()
        self._update_social_sentiment()
        self._update_technical_analysis()

    def _generate_daily_summary(self, daily_data: Dict) -> Dict:
        """Generate daily analysis summary"""
        summary = {
            "market_trend": "neutral",
            "top_performers": [],
            "sentiment_overview": "neutral",
            "recommendations": [],
            "confidence_score": 0.5,
        }

        try:
            # Analyze ML predictions
            ml_predictions = daily_data.get("ml_analysis", {}).get("predictions", [])
            if ml_predictions:
                # Determine market trend from predictions
                positive_predictions = sum(1 for p in ml_predictions if p.get("change", 0) > 0)
                total_predictions = len(ml_predictions)

                if positive_predictions / total_predictions > 0.6:
                    summary["market_trend"] = "bullish"
                elif positive_predictions / total_predictions < 0.4:
                    summary["market_trend"] = "bearish"

                # Extract top performers
                sorted_predictions = sorted(
                    ml_predictions, key=lambda x: x.get("change", 0), reverse=True
                )
                summary["top_performers"] = sorted_predictions[:5]

            # Analyze sentiment
            reddit_data = daily_data.get("social_sentiment", {}).get("reddit_data", [])
            twitter_data = daily_data.get("social_sentiment", {}).get("twitter_data", [])

            if reddit_data or twitter_data:
                # Calculate overall sentiment
                sentiment_scores = []
                for data in reddit_data + twitter_data:
                    if "sentiment" in data:
                        sentiment_scores.append(data["sentiment"])

                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    if avg_sentiment > 0.6:
                        summary["sentiment_overview"] = "positive"
                    elif avg_sentiment < 0.4:
                        summary["sentiment_overview"] = "negative"

            # Generate recommendations
            recommendations = []

            if summary["market_trend"] == "bullish" and summary["sentiment_overview"] == "positive":
                recommendations.append("Strong buy signals detected across multiple indicators")
                summary["confidence_score"] = 0.8
            elif (
                summary["market_trend"] == "bearish" and summary["sentiment_overview"] == "negative"
            ):
                recommendations.append("Caution advised - bearish signals detected")
                summary["confidence_score"] = 0.7
            else:
                recommendations.append("Mixed signals - monitor market closely")
                summary["confidence_score"] = 0.5

            # Technical analysis recommendations
            technical_data = daily_data.get("technical_analysis", {})
            if technical_data.get("signals"):
                strong_signals = [
                    s for s in technical_data["signals"] if s.get("strength", 0) > 0.7
                ]
                if strong_signals:
                    recommendations.append(
                        f"Strong technical signals detected for {len(strong_signals)} assets"
                    )

            summary["recommendations"] = recommendations

        except Exception as e:
            self.logger.error(f"Error generating daily summary: {e}")

        return summary

    def _save_daily_report(self, daily_data: Dict):
        """Save daily report to file"""
        try:
            reports_dir = self.data_dir / "daily_reports"
            reports_dir.mkdir(exist_ok=True)

            today = datetime.now().strftime("%Y-%m-%d")
            report_file = reports_dir / f"daily_report_{today}.json"

            with open(report_file, "w") as f:
                json.dump(daily_data, f, indent=2, default=str)

            self.logger.info(f"Daily report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Error saving daily report: {e}")

    def get_daily_analysis_status(self) -> Dict:
        """Get current daily analysis status"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        daily_data = self.cache_manager.get(cache_key) or {}

        # Add service status
        status = {
            "date": today,
            "scheduler_running": self.is_running,
            "services": {
                "ml_analysis": "running" if "ml_analysis" in self.running_services else "stopped",
                "social_scraper": "running"
                if "social_scraper" in self.running_services
                else "stopped",
            },
            "analysis_data": daily_data,
        }

        return status

    def get_latest_daily_report(self) -> Optional[Dict]:
        """Get the latest daily report"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"

        return self.cache_manager.get(cache_key)

    def _run_openai_enhancement_batch(self):
        """Run OpenAI enhancement batch processing"""
        try:
            self.logger.info("Running OpenAI enhancement batch")
            today = datetime.now().strftime("%Y-%m-%d")
            cache_key = f"daily_analysis_{today}"

            # Get current analysis data
            daily_data = self.cache_manager.get(cache_key) or {}

            # Prepare batch data for OpenAI enhancement
            batch_data = {}

            if daily_data.get("social_sentiment", {}).get("status") == "active":
                batch_data["sentiment"] = daily_data["social_sentiment"]

            if daily_data.get("technical_analysis", {}).get("status") == "active":
                batch_data["technical"] = daily_data["technical_analysis"]

            if daily_data.get("ml_analysis", {}).get("status") == "active":
                batch_data["ml_predictions"] = daily_data["ml_analysis"]

            if batch_data:
                # Process with OpenAI (run in thread to avoid blocking)
                threading.Thread(
                    target=self._process_openai_batch, args=(batch_data, today), daemon=True
                ).start()

        except Exception as e:
            self.logger.error(f"OpenAI enhancement batch failed: {e}")

    def _process_openai_batch(self, batch_data: Dict, date: str):
        """Process OpenAI batch in background thread"""
        try:
            # Run async OpenAI processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            enhanced_results = loop.run_until_complete(
                self.openai_analyzer.process_analysis_batch(batch_data)

            # Store enhanced results
            cache_key = f"daily_analysis_{date}"
            daily_data = self.cache_manager.get(cache_key) or {}

            daily_data["openai_enhanced"] = {
                "enhanced_results": enhanced_results,
                "enhancement_timestamp": datetime.now().isoformat(),
                "batch_status": "completed",
            }

            self.cache_manager.set(cache_key, daily_data, ttl_minutes=1440)
            self.logger.info("OpenAI enhancement batch completed successfully")

        except Exception as e:
            self.logger.error(f"OpenAI batch processing failed: {e}")
        finally:
            loop.close()

    def _generate_ai_market_insights(self):
        """Generate AI-powered market insights"""
        try:
            self.logger.info("Generating AI market insights")
            today = datetime.now().strftime("%Y-%m-%d")
            cache_key = f"daily_analysis_{today}"

            # Get all available analysis data
            daily_data = self.cache_manager.get(cache_key) or {}

            if daily_data:
                # Generate comprehensive insights using OpenAI
                threading.Thread(
                    target=self._process_market_insights, args=(daily_data, today), daemon=True
                ).start()

        except Exception as e:
            self.logger.error(f"AI market insights generation failed: {e}")

    def _process_market_insights(self, daily_data: Dict, date: str):
        """Process market insights in background thread"""
        try:
            # Run async market insights generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            market_insights = loop.run_until_complete(
                self.openai_analyzer.generate_market_insights(daily_data)

            # Store market insights
            cache_key = f"daily_analysis_{date}"
            current_data = self.cache_manager.get(cache_key) or {}

            current_data["ai_market_insights"] = market_insights

            # Update daily summary with AI insights
            if "daily_summary" not in current_data:
                current_data["daily_summary"] = {}

            ai_insights = market_insights.get("ai_insights", {})
            current_data["daily_summary"].update(
                {
                    "ai_enhanced": True,
                    "ai_market_trend": ai_insights.get("market_trend", "neutral"),
                    "ai_confidence": ai_insights.get("confidence_score", 0.5),
                    "ai_recommendations": ai_insights.get("strategic_recommendations", []),
                    "ai_insights_timestamp": datetime.now().isoformat(),
                }
            )

            self.cache_manager.set(cache_key, current_data, ttl_minutes=1440)
            self.logger.info("AI market insights generated successfully")

        except Exception as e:
            self.logger.error(f"Market insights processing failed: {e}")
        finally:
            loop.close()
