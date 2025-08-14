def validate_timestamps(df, target_col="target_720h"):
    """Validate no look-ahead bias in timestamps"""
    if "timestamp" in df.columns and "label_timestamp" in df.columns:
        assert (df["timestamp"] < df["label_timestamp"]).all(), "Look-ahead bias detected!"

    # Check no future features
    late_cols = [c for c in df.columns if c.startswith("feat_")]
    if late_cols and target_col in df.columns:
        future_mask = df[late_cols].isna().any(axis=1) & df[target_col].notna()
        assert not future_mask.any(), "Future features detected!"

    return True


def normalize_timestamp(dt, tz="UTC"):
    """Normalize timestamp to UTC and floor to hour"""
    import pandas as pd

    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return dt.tz_localize(tz) if dt.tz is None else dt.tz_convert(tz)
