"""
File Storage Adapter - Concrete implementation of StoragePort for file system storage

Provides file-based storage following the StoragePort contract.
"""

import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging
import fnmatch

from ..interfaces.storage_port import (
    StoragePort,
    StorageRequest,
    StorageResponse,
    DataFormat,
    StorageError,
)


class FileStorageAdapter(StoragePort):
    """File system storage implementation"""

    def __init__(self, base_path: str = "data", default_format: DataFormat = DataFormat.JSON):
        """
        Initialize file storage adapter

        Args:
            base_path: Base directory for file storage
            default_format: Default data format for storage
        """
        self.base_path = Path(base_path)
        self.default_format = default_format
        self.logger = logging.getLogger(__name__)

        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Storage statistics
        self._stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "operations_count": 0,
            "last_cleanup": None,
        }

    def store(
        self, key: str, data: Any, ttl: Optional[int] = None, metadata: Optional[Dict] = None
    ) -> StorageResponse:
        """Store data to file system"""

        try:
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare storage object
            storage_obj = {
                "data": data,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl": ttl,
                "key": key,
            }

            # Determine format and save
            format_type = self._determine_format(data)

            if format_type == DataFormat.JSON:
                with open(file_path.with_suffix(".json"), "w") as f:
                    json.dump(storage_obj, f, indent=2, default=self._json_serializer)

            elif format_type == DataFormat.PARQUET and isinstance(data, pd.DataFrame):
                # Store DataFrame as parquet with metadata as separate JSON
                data.to_parquet(file_path.with_suffix(".parquet"))
                metadata_path = file_path.with_suffix(".meta.json")
                with open(metadata_path, "w") as f:
                    json.dump(
                        {
                            "metadata": metadata or {},
                            "timestamp": storage_obj["timestamp"],
                            "ttl": ttl,
                            "key": key,
                            "format": "parquet",
                        },
                        f,
                        indent=2,
                    )

            elif format_type == DataFormat.PICKLE:
                with open(file_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(storage_obj, f)

            else:
                # Default to JSON
                with open(file_path.with_suffix(".json"), "w") as f:
                    json.dump(storage_obj, f, indent=2, default=self._json_serializer)

            self._update_stats()
            self.logger.debug(f"Stored data for key: {key}")

            return StorageResponse(
                success=True, metadata={"file_path": str(file_path), "format": format_type.value}
            )

        except Exception as e:
            self.logger.error(f"Failed to store data for key {key}: {e}")
            return StorageResponse(success=False, error=f"Storage failed: {str(e)}")

    def retrieve(self, key: str) -> StorageResponse:
        """Retrieve data from file system"""

        try:
            file_path = self._get_file_path(key)

            # Try different formats
            formats_to_try = [
                (file_path.with_suffix(".json"), DataFormat.JSON),
                (file_path.with_suffix(".parquet"), DataFormat.PARQUET),
                (file_path.with_suffix(".pkl"), DataFormat.PICKLE),
            ]

            for path, format_type in formats_to_try:
                if path.exists():
                    if format_type == DataFormat.JSON:
                        with open(path, "r") as f:
                            storage_obj = json.load(f)

                        # Check TTL
                        if self._is_expired(storage_obj):
                            self.delete(key)
                            return StorageResponse(success=False, error="Data expired")

                        return StorageResponse(
                            success=True,
                            data=storage_obj["data"],
                            metadata=storage_obj.get("metadata", {}),
                        )

                    elif format_type == DataFormat.PARQUET:
                        # Load DataFrame and metadata
                        df = pd.read_parquet(path)
                        metadata_path = path.with_suffix(".meta.json")

                        metadata = {}
                        if metadata_path.exists():
                            with open(metadata_path, "r") as f:
                                meta_obj = json.load(f)

                            # Check TTL
                            if self._is_expired(meta_obj):
                                self.delete(key)
                                return StorageResponse(success=False, error="Data expired")

                            metadata = meta_obj.get("metadata", {})

                        return StorageResponse(success=True, data=df, metadata=metadata)

                    elif format_type == DataFormat.PICKLE:
                        with open(path, "rb") as f:
                            storage_obj = pickle.load(f)

                        # Check TTL
                        if self._is_expired(storage_obj):
                            self.delete(key)
                            return StorageResponse(success=False, error="Data expired")

                        return StorageResponse(
                            success=True,
                            data=storage_obj["data"],
                            metadata=storage_obj.get("metadata", {}),
                        )

            return StorageResponse(success=False, error=f"Key '{key}' not found")

        except Exception as e:
            self.logger.error(f"Failed to retrieve data for key {key}: {e}")
            return StorageResponse(success=False, error=f"Retrieval failed: {str(e)}")

    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""

        file_path = self._get_file_path(key)

        # Check for any of the possible formats
        return (
            file_path.with_suffix(".json").exists()
            or file_path.with_suffix(".parquet").exists()
            or file_path.with_suffix(".pkl").exists()
        )

    def delete(self, key: str) -> StorageResponse:
        """Delete data by key"""

        try:
            file_path = self._get_file_path(key)
            deleted_files = []

            # Try to delete all possible format files
            for suffix in [".json", ".parquet", ".pkl", ".meta.json"]:
                path = file_path.with_suffix(suffix)
                if path.exists():
                    path.unlink()
                    deleted_files.append(str(path))

            if deleted_files:
                self.logger.debug(f"Deleted files for key {key}: {deleted_files}")
                return StorageResponse(success=True, metadata={"deleted_files": deleted_files})
            else:
                return StorageResponse(success=False, error=f"Key '{key}' not found")

        except Exception as e:
            self.logger.error(f"Failed to delete key {key}: {e}")
            return StorageResponse(success=False, error=f"Deletion failed: {str(e)}")

    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by pattern"""

        try:
            keys = set()

            # Walk through all files in base path
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    # Extract key from file path
                    relative_path = file_path.relative_to(self.base_path)

                    # Remove file extension to get key
                    key_parts = list(relative_path.parts[:-1])  # Directory parts
                    filename = relative_path.stem

                    # Handle metadata files
                    if filename.endswith(".meta"):
                        filename = filename[:-5]  # Remove .meta suffix

                    key_parts.append(filename)
                    key = "/".join(key_parts)

                    # Apply pattern filter if specified
                    if pattern is None or fnmatch.fnmatch(key, pattern):
                        keys.add(key)

            return sorted(list(keys))

        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []

    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear storage, optionally filtered by pattern"""

        try:
            keys_to_delete = self.list_keys(pattern)
            deleted_count = 0

            for key in keys_to_delete:
                result = self.delete(key)
                if result.success:
                    deleted_count += 1

            self.logger.info(f"Cleared {deleted_count} keys with pattern: {pattern}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to clear storage: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""

        try:
            self._update_stats()
            return self._stats.copy()
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def _get_file_path(self, key: str) -> Path:
        """Convert key to file path"""
        # Replace special characters and create nested structure
        safe_key = key.replace(":", "/").replace("*", "_star_").replace("?", "_q_")
        return self.base_path / safe_key

    def _determine_format(self, data: Any) -> DataFormat:
        """Determine best format for data type"""
        if isinstance(data, pd.DataFrame):
            return DataFormat.PARQUET
        elif isinstance(data, (dict, list, str, int, float, bool)) or data is None:
            return DataFormat.JSON
        else:
            return DataFormat.PICKLE

    def _is_expired(self, storage_obj: Dict) -> bool:
        """Check if stored object has expired"""
        ttl = storage_obj.get("ttl")
        if ttl is None:
            return False

        timestamp_str = storage_obj.get("timestamp")
        if not timestamp_str:
            return True

        try:
            stored_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            current_time = datetime.now(timezone.utc)
            age_seconds = (current_time - stored_time).total_seconds()

            return age_seconds > ttl
        except Exception:
            return True

    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif hasattr(obj, "tolist"):  # NumPy arrays
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _update_stats(self):
        """Update storage statistics"""
        try:
            total_files = 0
            total_size = 0

            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size

            self._stats.update(
                {
                    "total_files": total_files,
                    "total_size_bytes": total_size,
                    "operations_count": self._stats.get("operations_count", 0) + 1,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to update stats: {e}")
