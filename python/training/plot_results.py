"""
Process data collected from simulation (GRFs, legLenght) and calculate results like
- mean and std of the legLength over 100 hops
- Force Length Relationship

The data is collected during simulation in guro_env.py and has the following format:
data = {"grfs":[], "legLen":[], "legStif":[], "phases":[]},
where the dict items are lists (hops) of lists (data point per hop).

Example:
    data['legLen'][9][-1] returns the takeoff leg length of the tenth hop
"""

import numpy as np
from thesis_galljamov18.python import settings
from thesis_galljamov18.python.tools import plotMeanStdOverSeveralHops, plotForceLengthCurve

# load data collected in simulation
savePath = settings.PATH_THESIS_FOLDER + "python/training/sim_data/"
fileName = "PknHpET_wP5xwV_flKnee_10p20v_eplen8_64r_15208_splRew6_16LO_3M_LR3f5_s576_r12_l37_0.npy"
data = np.load(savePath+fileName)
data = data.item()

# extract data of interest
legLensAllHops = data['legLen']
grfsAllHops = data['grfs']

# plot the results
plotMeanStdOverSeveralHops(legLensAllHops)
plotForceLengthCurve(grfsAllHops, legLensAllHops, 2e3, cutFirstXTimestes=0, perturbed=True)

