#!/usr/bin/env python3
"""
Daily Health Aggregator
Centralize all health data into single daily directory
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def aggregate_daily_health():
    """Aggregate all health data for today"""

    date_str = datetime.now().strftime("%Y%m%d")
    daily_dir = Path(f"logs/daily/{date_str}")
    daily_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate from various sources
    sources = [
        ("logs/application.log", "system_logs.txt"),
        ("logs/trading/*.json", "trading_data.json"),
        ("logs/ml/*.json", "ml_metrics.json"),
        ("logs/agents/*.json", "agent_status.json"),
    ]

    health_summary = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "sources_aggregated": 0,
        "total_files": 0,
    }

    for source_pattern, target_name in sources:
        source_path = (
            Path(source_pattern.split("*")[0]) if "*" in source_pattern else Path(source_pattern)
        )

        if source_path.exists():
            try:
                if source_path.is_file():
                    shutil.copy2(source_path, daily_dir / target_name)
                    health_summary["sources_aggregated"] += 1
                elif source_path.is_dir():
                    # Aggregate JSON files from directory
                    json_files = list(source_path.glob("*.json"))
                    if json_files:
                        aggregated_data = []
                        for json_file in json_files:
                            with open(json_file, "r") as f:
                                data = json.load(f)
                                aggregated_data.append(data)

                        with open(daily_dir / target_name, "w") as f:
                            json.dump(aggregated_data, f, indent=2)

                        health_summary["sources_aggregated"] += 1
                        health_summary["total_files"] += len(json_files)
            except Exception as e:
                print(f"Error aggregating {source_pattern}: {e}")

    # Save health summary
    with open(daily_dir / "health_summary.json", "w") as f:
        json.dump(health_summary, f, indent=2)

    print(f"Daily health data aggregated to: {daily_dir}")
    return health_summary


if __name__ == "__main__":
    aggregate_daily_health()
