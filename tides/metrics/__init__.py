"""Event-stream metrics of Sec. 4.3.

The paper defers binning, windowing and normalisation to its supplementary
material, which we do not have. Every such choice below is a named, documented
parameter so it can be changed once the supplement is available.
"""
from .fidelity import chamfer_st, ig_nll
from .batching import fano_error, isi_spike_ratio, same_ts_fraction

__all__ = ["chamfer_st", "ig_nll", "fano_error", "isi_spike_ratio", "same_ts_fraction"]
