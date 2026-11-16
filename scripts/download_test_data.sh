#!/bin/bash
# Download the CDHscan test data from Zenodo into test/data and check it.
# Content: one person (HG00097) on a 1.6 Mb region of CHM13 chr20 (about 115 MB).
set -euo pipefail
cd "$(dirname "$0")/.."

ZENODO_URL="https://zenodo.org/records/ZENODO_RECORD_ID/files/CDHscan_test_data_v1.tar.gz"
ARCHIVE_MD5="3b568fe09ba7c0a16b121d97845a0615"

ARCHIVE=test/CDHscan_test_data_v1.tar.gz
curl -fL --retry 5 -o "$ARCHIVE" "$ZENODO_URL"
echo "${ARCHIVE_MD5}  ${ARCHIVE}" | md5sum -c -
tar -xzf "$ARCHIVE" -C test
mv test/CDHscan_test_data_v1 test/data
(cd test/data && md5sum -c --quiet MD5SUMS)
rm "$ARCHIVE"
echo "test data ready: test/data"
