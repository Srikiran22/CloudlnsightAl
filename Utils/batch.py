# Helpers for batch ingestion: turning many uploaded files into one dataset.

import pandas as pd


def merge_frames(frames):
    """Merge (filename, DataFrame) pairs into one dataset.

    A provenance column is inserted first so rows stay traceable to the file
    they came from. It is named `uploaded_file` when `source_file` is data.
    """
    if not frames:
        raise ValueError("No data was extracted from the uploaded files.")

    provenance_column = "source_file"
    if any(provenance_column in df.columns for _, df in frames):
        provenance_column = "uploaded_file"

    parts = []
    for name, df in frames:
        part = df.copy()
        part.insert(0, provenance_column, str(name))
        parts.append(part)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    return merged
