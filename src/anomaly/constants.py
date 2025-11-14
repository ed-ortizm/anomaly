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
# convert previous dict to nanometers
GALAXY_LINES_NM = {
    'OII_3726': 372.6,
    'H_delta_4101': 410.17,
    'H_gamma_4340': 434.05,
    'H_beta_4861': 486.14,
    'OIII_4959': 495.89,
    'OIII_5006': 500.68,
    'NII_6548': 654.8,
    'H_alpha_6562': 656.28,
    'NII_6583': 658.35,
    'SII_6716': 671.64,
    'SII_6730': 673.08
}