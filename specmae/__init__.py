"""SpecMultiMAE-AD: masked-autoencoder anomaly detection and absorption-wavelength
localisation for 1-D NIR spectra."""
from .config import ExperimentConfig
from .data import SpectralData, load_data, make_synthetic, one_class_folds
from .downstream import anomaly_score, localize, localize_batch
from .inference import MMIEResult, mmie
from .loss import compute_loss
from .masking import MaskSampler, cmsm_mask, random_mask
from .model import SpecMultiMAE
from .pipeline import SpecMultiMAEAD
from .preprocessing import preprocess

__all__ = [
    "ExperimentConfig", "SpectralData", "load_data", "make_synthetic", "one_class_folds",
    "anomaly_score", "localize", "localize_batch", "MMIEResult", "mmie", "compute_loss",
    "MaskSampler", "cmsm_mask", "random_mask", "SpecMultiMAE", "SpecMultiMAEAD", "preprocess",
]
