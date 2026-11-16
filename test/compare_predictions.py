"""Compare test predictions with the expected ones (probabilities within TOLERANCE)."""

import sys

import numpy as np
import pandas as pd

TOLERANCE = 1e-3

observed, expected = (pd.read_csv(path, sep="\t") for path in sys.argv[1:3])
assert list(observed["locus_id"]) == list(expected["locus_id"]), "the scored loci differ"
assert (observed["status"] == expected["status"]).all(), "the scored windows differ"
difference = np.nanmax(np.abs(observed["probability"] - expected["probability"]))
assert difference <= TOLERANCE, f"probabilities differ by up to {difference:.2g}"
print(f"test passed: {len(observed)} loci, largest probability difference {difference:.2g}")
print(observed[["locus_id", "chrom", "start", "end", "status", "probability", "cdh"]].to_string(index=False))
