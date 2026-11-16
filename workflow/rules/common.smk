import pandas as pd

OUT = Path(config["output_dir"])
WORK = OUT / "work"
CHROMS = config.get("chromosomes") or LABELLED_CHROMS
REGION = config.get("region")


def region_option():
    return f"--region {REGION}" if REGION else ""


def read_samples(path):
    samples = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    assert {"sample", "hic", "alignment"} <= set(samples.columns), f"{path}: needs columns sample, hic, alignment"
    assert samples["sample"].is_unique, f"{path}: sample names must be unique"
    return samples.set_index("sample", drop=False)


def hic_fetch_command(source, local_file):
    """Shell commands that make source (local path or http(s)/s3 URL) available as local_file.

    Remote .hic files are downloaded first: long remote reads with hic-straw can hang or return
    empty blocks."""
    if source.startswith("s3://"):
        return f"aws s3 cp --no-sign-request --only-show-errors {source} {local_file}.part && mv {local_file}.part {local_file}"
    if source.startswith(("http://", "https://")):
        return f"curl -sSfL --retry 5 -o {local_file}.part {source} && mv {local_file}.part {local_file}"
    return f"ln -sf $(realpath {source}) {local_file}"
