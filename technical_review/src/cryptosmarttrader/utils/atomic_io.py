#!/usr/bin/env python3
"""
Atomic File Operations - Safe file writing
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Union
import pandas as pd

def atomic_write_json(data: Any, file_path: Union[str, Path]) -> None:
    """Atomically write JSON data"""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f'{file_path.stem}_tmp_',
        suffix=file_path.suffix,
        delete=False
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_file_path = tmp_file.name

    # Atomic rename
    os.rename(tmp_file_path, file_path)

def atomic_write_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Atomically write CSV data"""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    tmp_file_path = file_path.with_suffix(f'{file_path.suffix}.tmp')
    df.to_csv(tmp_file_path, index=False)

    # Atomic rename
    os.rename(tmp_file_path, file_path)

def atomic_write_text(content: str, file_path: Union[str, Path]) -> None:
    """Atomically write text content"""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f'{file_path.stem}_tmp_',
        suffix=file_path.suffix,
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    # Atomic rename
    os.rename(tmp_file_path, file_path)
