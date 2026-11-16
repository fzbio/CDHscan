# CDHscan image: the predict and train targets (Python, PyTorch, hic-straw, samtools, bedtools,
# Snakemake). The cosigt target runs cosigt's own containers and is best run from the conda
# environment on the host (see README).
FROM mambaorg/micromamba:1.5.10

WORKDIR /opt/cdhscan
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/cdhscan
RUN micromamba install -y -n base -f environment.yml && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /data
CMD ["cdhscan", "--help"]
