# -*- coding: utf-8 -*-

"""
Madminer parameter and morphing setup code for a WH signal.

Includes only the CP-odd operator (oHWtil), with morphing done up to 2nd order (SM + SM-EFT interference term + EFT^2 term).

WARNING: events should ALWAYS be generated with proc cards which have at least the same order in the ME^2 as in the morphing (best: same order) - see gen_signal.py and gen_background.py

Ricardo Barrue (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse as ap
from madminer.plotting import plot_1d_morphing_basis

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

from madminer import MadMiner

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Creates MadMiner parameter and morphing setup file for a WH signal, with only the CP-odd operator (oHWtil), \
                               morphing up to second order (SM + SM-EFT interference + EFT^2 term).',
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',
                        required=True)

    args=parser.parse_args()

    # Instance of MadMiner core class
    miner = MadMiner()

    miner.add_parameter(
        lha_block='smeftcpv',
        lha_id=4,
        parameter_name='cHWtil',
        morphing_max_power=2, # interference + squared terms (matches what is in the run cards)
        parameter_range=(-1.2,1.2),
        param_card_transform='1.0*theta' # mandatory to avoid a crash due to a bug
    )

    # values originally obtained from morphing functionality
    # hardcoded to match what's on the paper
    miner.add_benchmark({'cHWtil':0.00},'sm')
    miner.add_benchmark({'cHWtil':1.15},'pos_chwtil')
    miner.add_benchmark({'cHWtil':-1.035},'neg_chwtil')

    # Morphing - automatic optimization to avoid large weights
    miner.set_morphing(max_overall_power=2,include_existing_benchmarks=True)

    miner.save(f'{args.main_dir}/setup.h5')

    morphing_basis=plot_1d_morphing_basis(miner.morpher,xlabel=r'$\tilde{c_{HW}}$',xrange=(-1.2,1.2))
    morphing_basis.savefig(f'{args.main_dir}/morphing_basis.png')

