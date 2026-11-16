#!/bin/bash
# Download the released CDHscan model from Zenodo into resources/model/ and check it.
# Content: weights of 3 seeds, locus catalogue, 10 kb bin table, panel reference (0.8 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

ZENODO_URL="https://zenodo.org/records/ZENODO_RECORD_ID/files/CDHscan_model_v1.tar.gz"
ARCHIVE_MD5="33f2891f3b0586e2b23ff74aeb200c40"

mkdir -p resources/model
ARCHIVE=resources/model/CDHscan_model_v1.tar.gz
curl -fL --retry 5 -o "$ARCHIVE" "$ZENODO_URL"
echo "${ARCHIVE_MD5}  ${ARCHIVE}" | md5sum -c -
tar -xzf "$ARCHIVE" -C resources/model
(cd resources/model/CDHscan_model_v1 && md5sum -c --quiet MD5SUMS)
rm "$ARCHIVE"
echo "model ready: resources/model/CDHscan_model_v1"
