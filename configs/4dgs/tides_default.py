# 4DGS training config used for every TIDES scene (pass with --configs to the 4DGS train.py).
# Repository-chosen settings inspired by HyperNeRF; not verified paper hyperparameters.
# Temporal grid resolution: 4DGS recommends about half the number of training frames;
# edit "resolution"[3] per sequence if you train on many more or fewer than ~300 frames.
ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "resolution": [64, 64, 64, 150],
    },
    multires=[1, 2, 4],
    defor_depth=1,
    net_width=128,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    no_do=True,
    no_dshs=True,
    no_ds=False,
    no_dr=False,
    bounds=1.6,
)
OptimizationParams = dict(
    iterations=14000,
    batch_size=2,
    coarse_iterations=3000,
    densify_until_iter=10000,
    opacity_reset_interval=300000,
)

# Explicit upstream settings. Boolean CLI options are store_true: do not pass
# --eval False or --white_background False. Keep held-out views for diagnostics.
ModelParams = dict(
    white_background=True,
    eval=True,
    llffhold=8,
    resolution=1,
    render_process=False,
)
