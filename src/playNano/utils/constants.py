"""Constants and default configurations for the playNano application."""

default_steps_with_kwargs = [
    ("remove_plane", {}),
    ("polynomial_flatten", {"order": 2}),
    ("mask_mean_offset", {"factor": 0.5}),
    ("row_median_align", {}),
    ("polynomial_flatten", {"order": 2}),
    ("zero_mean", {}),
]

# Number of bins for Z‑value histogram in the GUI
HIST_BINS = 100
