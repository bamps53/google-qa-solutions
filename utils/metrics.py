from scipy.stats import spearmanr
import numpy as np
from utils import OUTPUT_COLS
import warnings
warnings.filterwarnings('ignore')

def compute_spearmanr_ignore_nan(trues, preds, binarize=False, num_bins=18):
    rhos = []
    if binarize:
        preds = np.round((preds*num_bins))
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def compute_spearmanr_ignore_nan_each_targets(trues, preds, binarize=False, num_bins=18):
    rhos_dict = {}
    rhos = []

    if binarize:
        preds = np.round((preds*num_bins))
        rhos_dict['num_bins'] = num_bins
    for tcol, pcol, col in zip(np.transpose(trues), np.transpose(preds), OUTPUT_COLS):
        rhos.append(spearmanr(tcol, pcol).correlation)
        rhos_dict[col] = spearmanr(tcol, pcol).correlation
    rhos_dict['metric'] = np.nanmean(rhos)

    return rhos_dict