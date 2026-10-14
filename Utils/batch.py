# Helpers for batch ingestion: turning many uploaded files into one dataset.

import pandas as pd


def merge_frames(frames):
    """Merge (filename, DataFrame) pairs into one dataset.

    A provenance column is inserted first so rows stay traceable to the file
    they came from. The name avoids colliding with columns the data already
    uses: `source_file` is preferred, then `uploaded_file`, then numbered
    fallbacks -- data columns are never duplicated or overwritten.
    """
    if not frames:
        raise ValueError("No data was extracted from the uploaded files.")

    existing = set()
    for _, df in frames:
        existing.update(df.columns)

    base_names = ["source_file", "uploaded_file"]
    provenance_column = next(
        (name for name in base_names if name not in existing),
        None,
    )
    if provenance_column is None:
        counter = 2
        while f"source_file_{counter}" in existing:
            counter += 1
        provenance_column = f"source_file_{counter}"

    parts = []
    for name, df in frames:
        part = df.copy()
        part.insert(0, provenance_column, str(name))
        parts.append(part)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    return merged
