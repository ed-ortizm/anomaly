"""Store constants used thorughout the library"""
GALAXY_LINES = {
    # EMISSION
    "OII_3726": 3726.040,
    "H_delta_4101": 4101.734,
    "H_gamma_4340": 4340.472,
    "H_beta_4861": 4861.352,
    "OIII_4959": 4958.911,
    "OIII_5006": 5006.843,
    "NII_6548": 6548.041,
    "H_alpha_6562": 6562.787,
    "NII_6583": 6583.461,
    "SII_6716": 6716.440,
    "SII_6730": 6730.812,
    # ABSORPTION
}

# dictionary with score names and their description
scores_description = {
    "correlation": "Correlation",
    "correlation_filter_250kms": "Correlation (250 km s$^{-1}$ filter)",
    "cosine": "Cosine disimilarity score",
    "cosine_filter_250kms": "Cosine disimilarity (250 km s$^{-1}$ filter)",
    "lp_noRel100": "$L^p$",
    "lp_filter_250kms_noRel100": "$L^p$ (250 km s$^{-1}$ filter)",
    "lp_noRel97": "$L^p$ (ignore 3% of largest residuals)",
    "lp_filter_250kms_noRel97": "$L^p$ (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
    "lp_rel100": "$L^p$ relative",
    "lp_filter_250kms_rel100": "$L^p$ relative (250 km s$^{-1}$ filter)",
    "lp_rel97": "$L^p$ relative (ignore 3% of largest residuals)",
    "lp_filter_250kms_rel97": "$L^p$ relative (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
    "mad_noRel100": "MAD",
    "mad_filter_250kms_noRel100": "MAD (250 km s$^{-1}$ filter)",
    "mad_noRel97": "MAD (ignore 3% of largest residuals)",
    "mad_filter_250kms_noRel97": "MAD (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
    "mad_rel100": "MAD relative score",
    "mad_filter_250kms_rel100": "MAD relative (250 km s$^{-1}$ filter)",
    "mad_rel97": "MAD relative (ignore 3% of largest residuals)",
    "mad_filter_250kms_rel97": "MAD relative (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
    "mse_noRel100": "MSE",
    "mse_filter_250kms_noRel100": "MSE (250 km s$^{-1}$ filter)",
    "mse_noRel97": "MSE (ignore 3% of largest residuals)",
    "mse_filter_250kms_noRel97": "MSE (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
    "mse_rel100": "MSE relative score",
    "mse_filter_250kms_rel100": "MSE relative (250 km s$^{-1}$ filter)",
    "mse_rel97": "MSE relative (ignore 3% of largest residuals)",
    "mse_filter_250kms_rel97": "MSE relative (250 km s$^{-1}$ filter, ignore 3% of largest residuals)",
}
