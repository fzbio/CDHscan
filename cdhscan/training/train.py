"""Train the Hi-C + WGS network with one seed.

Every batch is PAIRS_PER_BATCH pairs, each a positive and a negative window of the same locus,
under binary cross-entropy over the 2 x PAIRS_PER_BATCH windows; an epoch is STEPS_PER_EPOCH
batches; positives are drawn without replacement within an epoch, each with a random negative of
its locus. Validation BCE, pooled AUC and pair-weighted within-locus AUC are logged after every
epoch. Training stops after PATIENCE epochs without a lower validation loss, at MAX_EPOCHS at the
latest. The weights with the lowest validation loss are saved after every gain, together with the
per-channel mean and sd of the training inputs (STANDARDISATION_WINDOWS random training windows)
that standardise every input. The checkpoint holds plain tensors only (torch.load(...,
weights_only=True) reads it), in the format that score.ReleasedModel loads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ..constants import BIN_BP, MAP_CHANNELS, WGS_CHANNELS, WINDOW_BINS
from ..model import JointCNN

LEARNING_RATE = 1e-3
PAIRS_PER_BATCH = 32
STEPS_PER_EPOCH = 100
MAX_EPOCHS = 200
PATIENCE = 10
PREDICTION_BATCH = 128
STANDARDISATION_WINDOWS = 2000
PREDICTION_COLUMNS = ["window_id", "sample", "role", "chrom", "start", "end", "label", "locus_id", "n_carriers"]


def load_hic_inputs(windows, hic_inputs_dir):
    """Hi-C map channels of the windows, one array per stored file holding only these windows' rows;
    adds the columns hic_key (the file) and hic_row (the row in its array)."""
    windows["hic_key"] = windows["input_file"]
    windows["hic_row"] = 0
    arrays = {}
    for input_file, group in windows.groupby("input_file", sort=False):
        # The whole file is read in one sequential pass; scattered reads are very slow on the shared file system.
        stored = np.load(Path(hic_inputs_dir) / input_file)
        assert stored.shape[1] == len(MAP_CHANNELS), f"{input_file}: unexpected channel count"
        arrays[input_file] = stored[group["input_row"].to_numpy()]
        del stored
        windows.loc[group.index, "hic_row"] = np.arange(len(group))
    return arrays


class WindowSet:
    """Windows with their stored inputs; batches come out standardised, with the locus masks added."""

    def __init__(self, windows, hic_arrays, tracks):
        self.windows = windows.reset_index(drop=True)
        self.hic_arrays, self.tracks = hic_arrays, tracks
        index = np.arange(WINDOW_BINS)
        first = np.clip((self.windows["locus_start"] - self.windows["start"]) // BIN_BP, 0, WINDOW_BINS - 1)
        last = np.clip((self.windows["locus_end"] - 1 - self.windows["start"]) // BIN_BP, 0, WINDOW_BINS - 1)
        self.in_locus = (index >= first.to_numpy()[:, None]) & (index <= last.to_numpy()[:, None])
        self.map_mean, self.map_sd, self.track_mean, self.track_sd = None, None, None, None

    def __len__(self):
        return len(self.windows)

    def maps(self, indices):
        """Stored Hi-C map channels (float32) of the windows at indices, in that order."""
        keys = self.windows["hic_key"].to_numpy()[indices]
        rows = self.windows["hic_row"].to_numpy()[indices]
        maps = np.empty((len(indices), len(MAP_CHANNELS), WINDOW_BINS, WINDOW_BINS), dtype=np.float32)
        for key in np.unique(keys):
            positions = np.flatnonzero(keys == key)
            maps[positions] = self.hic_arrays[key][rows[positions]]
        return maps

    def track_values(self, indices):
        """Stored WGS track channels of the windows at indices, in that order."""
        return self.tracks[self.windows["wgs_row"].to_numpy()[indices]]

    def batch(self, indices):
        """Model inputs: n x 13 x 100 x 100 map channels with the 2D mask, n x 13 x 100 track
        channels with the 1D mask."""
        in_locus = self.in_locus[indices]
        maps = (self.maps(indices) - self.map_mean[:, None, None]) / self.map_sd[:, None, None]
        mask = (in_locus[:, :, None] | in_locus[:, None, :]).astype(np.float32)
        contact_map = torch.from_numpy(np.concatenate([maps, mask[:, None]], axis=1))
        values = (self.track_values(indices) - self.track_mean[:, None]) / self.track_sd[:, None]
        tracks = torch.from_numpy(np.concatenate([values, in_locus[:, None].astype(np.float32)], axis=1))
        return contact_map, tracks


def standardisation_statistics(training, rng):
    """Mean and sd of each input channel over random training windows."""
    sample = np.sort(rng.choice(len(training), min(STANDARDISATION_WINDOWS, len(training)), replace=False))
    statistics = {}
    maps = training.maps(sample).astype(np.float64)
    statistics["map_mean"], statistics["map_sd"] = maps.mean(axis=(0, 2, 3)), maps.std(axis=(0, 2, 3))
    values = training.track_values(sample).astype(np.float64)
    statistics["track_mean"], statistics["track_sd"] = values.mean(axis=(0, 2)), values.std(axis=(0, 2))
    return {name: value.astype(np.float32) for name, value in statistics.items()}


def apply_standardisation(window_set, statistics):
    for name, value in statistics.items():
        setattr(window_set, name, value)


class PairSampler:
    """Random (positive, negative) pairs of training windows at the same locus."""

    def __init__(self, windows, rng):
        self.rng = rng
        locus_code = pd.factorize(windows["locus_id"])[0]
        labels = windows["label"].to_numpy()
        self.positives = np.flatnonzero(labels == 1)
        self.positive_locus = locus_code[self.positives]
        negatives = np.flatnonzero(labels == 0)
        order = np.argsort(locus_code[negatives], kind="stable")
        self.negatives_by_locus = negatives[order]
        counts = np.bincount(locus_code[negatives], minlength=locus_code.max() + 1)
        self.negative_count = counts
        self.negative_offset = np.concatenate([[0], np.cumsum(counts)[:-1]])
        assert (counts[self.positive_locus] > 0).all(), "a positive training window has no negative at its locus"

    def epoch_pairs(self, n_pairs):
        """n_pairs pairs; positives without replacement, each with a random negative of its locus."""
        positive_index = self.rng.choice(len(self.positives), n_pairs, replace=False)
        locus = self.positive_locus[positive_index]
        pick = self.negative_offset[locus] + (self.rng.random(n_pairs) * self.negative_count[locus]).astype(int)
        return self.positives[positive_index], self.negatives_by_locus[pick]


def predict_logits(model, window_set, indices):
    model.eval()
    with torch.no_grad():
        logits = [model(*window_set.batch(indices[i:i + PREDICTION_BATCH]))
                  for i in range(0, len(indices), PREDICTION_BATCH)]
    return torch.cat(logits)


def within_locus_pairs(windows):
    """Every (positive, negative) index pair of windows at the same locus."""
    positive_parts, negative_parts = [], []
    for _, locus in windows.groupby("locus_id", sort=False):
        positives = locus.index[locus["label"] == 1].to_numpy()
        negatives = locus.index[locus["label"] == 0].to_numpy()
        positive_parts.append(np.repeat(positives, len(negatives)))
        negative_parts.append(np.tile(negatives, len(positives)))
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def validation_scores(model, validation, pairs):
    """Validation BCE, pooled AUC and pair-weighted within-locus AUC."""
    logits = predict_logits(model, validation, np.arange(len(validation)))
    labels = torch.tensor(validation.windows["label"].to_numpy(), dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    difference = (logits[pairs[0]] - logits[pairs[1]]).numpy()
    within_auc = float((difference > 0).mean() + 0.5 * (difference == 0).mean())
    return loss, roc_auc_score(labels.numpy(), logits.numpy()), within_auc


def train(model, training, validation, statistics, rng, checkpoint_file, log_file):
    """Train until PATIENCE epochs pass without a lower validation loss, or MAX_EPOCHS."""
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = PairSampler(training.windows, rng)
    validation_pairs = within_locus_pairs(validation.windows)
    labels = torch.cat([torch.ones(PAIRS_PER_BATCH), torch.zeros(PAIRS_PER_BATCH)])
    best_loss, epochs_without_gain, log_rows = np.inf, 0, []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        positives, negatives = sampler.epoch_pairs(STEPS_PER_EPOCH * PAIRS_PER_BATCH)
        loss_sum = 0.0
        for step in range(STEPS_PER_EPOCH):
            part = slice(step * PAIRS_PER_BATCH, (step + 1) * PAIRS_PER_BATCH)
            logits = model(*training.batch(np.concatenate([positives[part], negatives[part]])))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_sum += loss.item()

        validation_loss, validation_auc, validation_within_auc = validation_scores(model, validation, validation_pairs)
        log_rows.append({"epoch": epoch, "train_loss": loss_sum / STEPS_PER_EPOCH, "val_loss": validation_loss,
                         "val_auc": validation_auc, "val_within_locus_auc": validation_within_auc})
        pd.DataFrame(log_rows).to_csv(log_file, sep="\t", index=False)
        print(f"epoch {epoch}: train loss {loss_sum / STEPS_PER_EPOCH:.4f}, validation loss {validation_loss:.4f}, "
              f"AUC {validation_auc:.4f}, within-locus AUC {validation_within_auc:.4f}", flush=True)

        if validation_loss < best_loss:
            best_loss, epochs_without_gain = validation_loss, 0
            save_checkpoint(model, statistics, epoch, checkpoint_file)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                break
    return best_loss


def predictions_table(model, window_set):
    table = window_set.windows[PREDICTION_COLUMNS].copy()
    table["probability"] = torch.sigmoid(predict_logits(model, window_set, np.arange(len(window_set)))).numpy()
    return table


def save_checkpoint(model, statistics, epoch, path):
    torch.save({"model": model.state_dict(), "epoch": epoch,
                "statistics": {name: torch.from_numpy(value) for name, value in statistics.items()},
                "map_channels": MAP_CHANNELS, "wgs_channels": WGS_CHANNELS}, path)


def train_model(windows_file, hic_inputs_dir, tracks_file, seed, threads, out_dir):
    """Train one seed; writes best_model.pt, training_log.tsv and predictions.tsv (validation windows)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "best_model.pt"

    windows = pd.read_csv(windows_file, sep="\t")
    # Row of each window in tracks.npy.
    windows["wgs_row"] = np.arange(len(windows))
    tracks = np.load(tracks_file)
    assert tracks.shape == (len(windows), len(WGS_CHANNELS), WINDOW_BINS), "tracks.npy does not match windows.tsv"
    for role, role_windows in windows.groupby("role"):
        print(f"{role}: {len(role_windows)} windows ({int(role_windows['label'].sum())} positive) "
              f"of {role_windows['sample'].nunique()} people at {role_windows['locus_id'].nunique()} loci", flush=True)

    window_sets = {}
    for role in ["train", "validation"]:
        role_windows = windows[windows["role"] == role].copy()
        window_sets[role] = WindowSet(role_windows, load_hic_inputs(role_windows, hic_inputs_dir), tracks)
    training, validation = window_sets["train"], window_sets["validation"]

    model = JointCNN(n_map_channels=len(MAP_CHANNELS) + 1, n_track_channels=len(WGS_CHANNELS) + 1)
    print(f"Hi-C + WGS network: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    statistics = standardisation_statistics(training, rng)
    for window_set in (training, validation):
        apply_standardisation(window_set, statistics)
    best_loss = train(model, training, validation, statistics, rng, checkpoint_file, out_dir / "training_log.tsv")
    print(f"training done: best validation loss {best_loss:.4f}", flush=True)

    checkpoint = torch.load(checkpoint_file, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    predictions_table(model, validation).to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
