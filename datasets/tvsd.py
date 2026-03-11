"""
TVSD (Temporal Visual Stimuli Dataset) loader for normalized MUA data.
Data from Papale et al. (Neuron 2025) — primate neural recordings across V1, V4, IT.

Files are HDF5/v7.3 MAT format — requires h5py.
"""

import h5py
import numpy as np
import os

# Electrode-to-area mapping (channel index ranges)
AREA_MAP = {
    'F': {'V1': (0, 512), 'IT': (512, 832), 'V4': (832, 1024)},
    'N': {'V1': (0, 512), 'V4': (512, 768), 'IT': (768, 1024)},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'TVSD')


def load_tvsd(monkey='N', area=None, quality_threshold=0.3):
    """
    Load TVSD normalized MUA data for a given monkey.

    Args:
        monkey: 'N' or 'F'
        area: None (all electrodes), or one of 'V1', 'V4', 'IT'
        quality_threshold: minimum oracle value to keep an electrode (default 0.3)

    Returns:
        train_mua: [n_train_stimuli × n_electrodes] array
        test_mua: [n_test_stimuli × n_electrodes] array
        metadata: dict with keys 'oracle', 'snr', 'electrode_indices', 'area_map',
                  'test_mua_reps', 'monkey', 'area'
    """
    mat_path = os.path.join(DATA_DIR, f'monkey{monkey}', 'THINGS_normMUA.mat')
    with h5py.File(mat_path, 'r') as f:
        train_mua = f['train_MUA'][:]  # [n_train_stimuli × 1024]
        test_mua = f['test_MUA'][:]    # [n_test_stimuli × 1024]
        test_mua_reps = f['test_MUA_reps'][:]  # [30 reps × 100 stimuli × 1024]
        oracle = f['oracle'][:].flatten()
        snr = f['SNR'][:]  # [4 × 1024]
        snr_max = f['SNR_max'][:].flatten()
        latency = f['lats'][:]  # [4 × 1024]

    n_electrodes = len(oracle)

    # Build electrode mask: area + quality
    if area is not None:
        start, end = AREA_MAP[monkey][area]
        area_mask = np.zeros(n_electrodes, dtype=bool)
        area_mask[start:end] = True
    else:
        area_mask = np.ones(n_electrodes, dtype=bool)

    quality_mask = oracle >= quality_threshold
    mask = area_mask & quality_mask

    electrode_indices = np.where(mask)[0]

    metadata = {
        'oracle': oracle,
        'snr': snr,
        'snr_max': snr_max,
        'latency': latency,
        'electrode_indices': electrode_indices,
        'area_map': AREA_MAP[monkey],
        'test_mua_reps': test_mua_reps[:, :, mask],
        'monkey': monkey,
        'area': area,
        'quality_threshold': quality_threshold,
        'n_electrodes_total': n_electrodes,
        'n_electrodes_selected': int(mask.sum()),
    }

    return train_mua[:, mask], test_mua[:, mask], metadata


def load_category_labels(monkey='N', split='test'):
    """
    Load THINGS category labels for stimuli.

    Args:
        monkey: 'N' or 'F'
        split: 'test' or 'train'

    Returns:
        labels: list of category name strings
        paths: list of image path strings
    """
    mat_path = os.path.join(DATA_DIR, f'monkey{monkey}', '_logs', 'things_imgs.mat')
    key = f'{split}_imgs'

    with h5py.File(mat_path, 'r') as f:
        group = f[key]
        n = group['class'].shape[0]

        labels = []
        paths = []
        for i in range(n):
            ref = group['class'][i, 0]
            name = ''.join(chr(c) for c in f[ref][:].flatten())
            labels.append(name)

            ref_p = group['things_path'][i, 0]
            path = ''.join(chr(c) for c in f[ref_p][:].flatten())
            paths.append(path)

    return labels, paths


def get_area_electrode_counts(monkey='N', quality_threshold=0.3):
    """Get electrode counts per area before and after quality filtering."""
    mat_path = os.path.join(DATA_DIR, f'monkey{monkey}', 'THINGS_normMUA.mat')
    with h5py.File(mat_path, 'r') as f:
        oracle = f['oracle'][:].flatten()

    counts = {}
    for area_name, (start, end) in AREA_MAP[monkey].items():
        total = end - start
        quality = int((oracle[start:end] >= quality_threshold).sum())
        counts[area_name] = {'total': total, 'quality': quality}

    return counts
