"""Numbers reported in the TIDES paper (arXiv 2606.02058v1), for side-by-side tables."""

DATASETS = ["eds", "hsergb", "bsergb", "dsec"]
METHODS = ["esim", "v2e", "iebcs", "dvs_voltmeter", "tides_10x", "tides_adaptive"]
LABELS = {
    "esim": "ESIM", "v2e": "V2E", "iebcs": "ICNS/IEBCS", "dvs_voltmeter": "DVS-Voltmeter",
    "tides_10x": "TIDES (10x)", "tides_adaptive": "TIDES (adaptive)",
    "eds": "EDS", "hsergb": "HS-ERGB", "bsergb": "BS-ERGB", "dsec": "DSEC",
}

# Table 1: (IG-NLL, Chamfer)
TABLE1 = {
    "esim": {"eds": (0.004748, 0.046932), "hsergb": (0.006763, 0.021335), "bsergb": (0.007669, 0.039233), "dsec": (0.096768, 0.150414)},
    "v2e": {"eds": (0.004058, 0.044124), "hsergb": (0.005823, 0.016149), "bsergb": (0.006092, 0.035944), "dsec": (0.099792, 0.153248)},
    "iebcs": {"eds": (0.004575, 0.046710), "hsergb": (0.008341, 0.028272), "bsergb": (0.010608, 0.045028), "dsec": (0.092576, 0.142781)},
    "dvs_voltmeter": {"eds": (0.006050, 0.050914), "hsergb": (0.006388, 0.014871), "bsergb": (0.005163, 0.033895), "dsec": (0.080542, 0.133779)},
    "tides_10x": {"eds": (0.003840, 0.043245), "hsergb": (0.004984, 0.014892), "bsergb": (0.004392, 0.033002), "dsec": (0.068641, 0.121962)},
    "tides_adaptive": {"eds": (0.003733, 0.042622), "hsergb": (0.002343, 0.013772), "bsergb": (0.004251, 0.032260), "dsec": (0.068439, 0.120611)},
}

# Table 2: (Same-ts, Fano error, ISI spike)
TABLE2 = {
    "esim": {"eds": (0.281985, 0.506379, 0.096938), "hsergb": (0.166213, 3.305385, 0.064902), "bsergb": (0.279757, 1.689136, 0.111152), "dsec": (0.165757, 3.296069, 0.050237)},
    "v2e": {"eds": (0.076272, 4.284378, 0.047712), "hsergb": (0.126701, 6.409716, 0.081733), "bsergb": (0.155642, 6.473020, 0.089496), "dsec": (0.226925, 6.495386, 0.141100)},
    "iebcs": {"eds": (0.046909, 2.417609, 0.034290), "hsergb": (0.074536, 4.975071, 0.049699), "bsergb": (0.102743, 5.063936, 0.065669), "dsec": (0.114483, 4.957434, 0.073942)},
    "dvs_voltmeter": {"eds": (0.102028, 0.313285, 0.061608), "hsergb": (0.076948, 2.698272, 0.051943), "bsergb": (0.112803, 1.436576, 0.069397), "dsec": (0.060686, 3.155488, 0.039239)},
    "tides_10x": {"eds": (0.040882, 0.228440, 0.052497), "hsergb": (0.050215, 2.069280, 0.038454), "bsergb": (0.085554, 0.816516, 0.041768), "dsec": (0.057469, 2.332386, 0.024181)},
    "tides_adaptive": {"eds": (0.025234, 0.338062, 0.055599), "hsergb": (0.051124, 2.017263, 0.046340), "bsergb": (0.080981, 0.608790, 0.033215), "dsec": (0.056436, 2.199856, 0.021470)},
}

# Table 4 (EDS): (IG-NLL, Chamfer)
TABLE4 = {
    "full": ("Full TIDES", 0.00373, 0.04262),
    "finite_diff_slope": ("Finite-diff slope (L_dot by FD)", 0.00521, 0.04801),
    "no_multi_crossing": ("No multi-crossing (1 event max / step)", 0.00439, 0.04487),
    "fixed_step": ("Fixed step size (no controller)", 0.00486, 0.04672),
    "no_risk_refinement": ("No risk-gated refinement", 0.00458, 0.04593),
    "no_mixed_visibility_masking": ("No mixed-visibility masking", 0.00422, 0.04431),
    "no_alpha_cue": ("No alpha_dot cue", 0.00461, 0.04711),
    "no_readout": ("No readout model (ideal timestamps)", 0.00388, 0.04294),
    "compute_matched": ("Compute-matched TIDES (adaptivity off)", 0.00405, 0.04371),
}
