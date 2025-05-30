# -*- coding: utf-8 -*-

"""
gen_background.py

Generates background events: W + b b~, t t~, single top t b
- divided by W decay channel and charge (250k events for each combination)

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import math

from madminer.core import MadMiner
from madminer.lhe import LHEReader

import argparse as ap

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

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Generates background events: W + b b~, t t~, single top t b - divided by W decay channel and charge',formatter_class=ap.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',default=False,action='store_true')

    parser.add_argument('--mg_dir',help='Path for MadGraph installation',required=True)

    parser.add_argument('--init_command',help='Initial command to be ran before generation (for e.g. setting environment variables)',default=None)

    parser.add_argument('--nevents',help='number of total hard scattering events (for each background) to generate (Madgraph-level)',type=float,default=10e6)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/setup.h5')

    # samples split by lepton charge and flavor
    samples=['wpbb_mu','wpbb_e','wmbb_mu','wmbb_e'] # W + (b-)jets
    samples+=['tpb_mu','tpb_e','tmb_mu','tmb_e'] # single top production (tb channel)
    samples+=['tt_mupjj','tt_epjj','tt_mumjj','tt_emjj'] # semi-leptonic ttbar

    factor=math.ceil(args.nevents/1e6)
    for sample in samples:
    
        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/{sample}_background',
            mg_process_directory=f'{args.main_dir}/background_samples/{sample}_background',
            proc_card_file=f'cards/background_processes/proc_card_{sample}.dat',
            param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
            pythia8_card_file='cards/pythia8_card.dat',
            run_card_files=['cards/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            sample_benchmarks=['sm'],
            initial_command=args.init_command,
            is_background=True,
            only_prepare_script=args.prepare_scripts
        )

    os.remove('/tmp/generate.mg5')