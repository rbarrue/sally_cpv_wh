# -*- coding: utf-8 -*-

"""
gen_signal.py

Generates WH signal events WH(->l v b b~), divided by W decay channel and charge (250k events/submission)

Can use different morphing setups (default: CP-odd operator only).

Using the SALLY method around the SM -> generating events at that point
- sample contains weights for different benchmarks (from MG reweighting)

Can also generate events at the BSM benchmarks to populate regions of phase space not well populated by the SM sample
- smaller number than for SM point, 50k for each charge+flavour combination
- reweighted to other benchmarks (inc. SM point)

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import matplotlib
import os, sys
import math

from madminer.core import MadMiner
from madminer.lhe import LHEReader

import argparse as ap


# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Generates WH signal events WH(->l v b b~), divided by W decay channel and charge.',
                                formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',action='store_true',default=False)

    parser.add_argument('--mg_dir',help='Path for MadGraph installation',required=True)

    parser.add_argument('--init_command',help='Initial command to be ran before generation (for e.g. setting environment variables)',default=None)
    
    parser.add_argument('--skip_reweight',help='if running reweighting separately after generation', action='store_true',default=False)

    parser.add_argument('--nevents',help='number of total hard scattering events to generate (Madgraph-level)',type=float,default=30e6)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/setup.h5')
    lhe = LHEReader(f'{args.main_dir}/setup.h5')

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if args.auto_widths:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat'
    
    samples=['wph_mu','wph_e','wmh_mu','wmh_e']
    
    factor=math.ceil(args.nevents/1e6)
    # 250k for each of the 4 charge/flavor pairs -> 1M
    for sample in samples:

        # SM samples with MG (re)weights to BSM benchmark points
        process_directory=f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM'
        if os.path.exists(f'{process_directory}/Events'):
            logging.warning('folder with events already exists, skipping generation of Madgraph scripts')

        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/{sample}_smeftsim_SM',
            mg_process_directory=process_directory,
            proc_card_file=f'cards/signal_processes/proc_card_{sample}_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file='cards/pythia8_card.dat',
            sample_benchmarks=['sm'],
            is_background = args.skip_reweight,
            run_card_files=['cards/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            initial_command=args.init_command,
            only_prepare_script=args.prepare_scripts
        )

    os.remove('/tmp/generate.mg5')
