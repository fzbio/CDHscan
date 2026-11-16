#!/bin/bash
# Run CDHscan on the test data and compare the predictions with the expected ones.
#   bash test/run_test.sh            predict target only (a few minutes)
#   bash test/run_test.sh cosigt     also the cosigt target (needs Apptainer or conda; pulls the
#                                    cosigt containers on first use)
#   bash test/run_test.sh reconstruct  cosigt, then the reconstruct target (remapping of test Hi-C reads)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=resources/model/CDHscan_model_v1
[[ -d "$MODEL" ]] || bash scripts/download_model.sh
[[ -d test/data ]] || bash scripts/download_test_data.sh
# The test model folder holds the files of the 9 test loci; the weights are the released ones.
ln -sfn "$(realpath "$MODEL/models")" test/data/model/models

snakemake predict --configfile test/config.yaml --cores 2
python test/compare_predictions.py results_test/HG00097/predictions.tsv test/expected/predictions.tsv

if [[ "${1:-}" == "cosigt" || "${1:-}" == "reconstruct" ]]; then
    snakemake cosigt --configfile test/config.yaml --cores 4
    cat results_test/HG00097/proxy_haplotypes.tsv
fi

if [[ "${1:-}" == "reconstruct" ]]; then
    snakemake reconstruct --configfile test/config.yaml --cores 4
    cat results_test/HG00097/reconstruction/contacts.tsv results_test/HG00097/reconstruction/region.tsv
fi
