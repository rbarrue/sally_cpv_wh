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

    parser.add_argument('--setup_file',help='name of setup file (without the .h5)',required=True)

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',action='store_true',default=False)

    parser.add_argument('--mg_dir',help='Path for MadGraph installation',required=True)

    parser.add_argument('--init_command',help='Initial command to be ran before generation (for e.g. setting environment variables)',default=None)

    # speeds up sample production, since reweighting can only run on multicore mode
    parser.add_argument('--reweight',help='if running reweighting alongside generation (doesnt work on multi-core mode)',action='store_true',default=False)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/{args.setup_file}.h5')
    lhe = LHEReader(f'{args.main_dir}/{args.setup_file}.h5')

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if args.auto_widths:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat'

    # signal
    miner.run_multiple(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/cutflow_wh_lvbb_smeftsim_SM',
        mg_process_directory=f'{args.main_dir}/cutflow/wh_lvbb_smeftsim_SM',
        proc_card_file=f'cards/cutflow/proc_card_wh_lvbb_smeftsim.dat',
        param_card_template_file=param_card_template_file,
        # added in signal cutflow, since there was a different MG vs. Pythia XS for signal samples (not for background), this way can check both
        pythia8_card_file='cards/pythia8_card.dat',
        sample_benchmarks=['sm'],
        is_background = not args.reweight,
        run_card_files=['cards/cutflow/run_card_250k_nocuts.dat','cards/cutflow/run_card_250k_WHMadminerCuts_1.dat','cards/cutflow/run_card_250k_WHMadminerCuts_2.dat','cards/cutflow/run_card_250k_WHMadminerCuts_3.dat','cards/cutflow/run_card_250k_WHMadminerCuts_4.dat','cards/cutflow/run_card_250k_WHMadminerCuts_5.dat','cards/cutflow/run_card_250k_WHMadminerCuts_6.dat'],
        initial_command=args.init_command,
        only_prepare_script=args.prepare_scripts
    )

    # W + (b-)jets
    miner.run_multiple(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/cutflow_wbb_lv_background',
        mg_process_directory=f'{args.main_dir}/cutflow/wbb_lv_background',
        proc_card_file=f'cards/cutflow/proc_card_wbb_lv.dat',
        param_card_template_file=param_card_template_file,
        sample_benchmarks=['sm'],
        is_background = True,
        run_card_files=['cards/cutflow/run_card_250k_nocuts.dat','cards/cutflow/run_card_250k_WHMadminerCuts_1.dat','cards/cutflow/run_card_250k_WHMadminerCuts_2.dat','cards/cutflow/run_card_250k_WHMadminerCuts_3.dat','cards/cutflow/run_card_250k_WHMadminerCuts_4.dat','cards/cutflow/run_card_250k_WHMadminerCuts_5.dat','cards/cutflow/run_card_250k_WHMadminerCuts_6.dat'],
        initial_command=args.init_command,
        only_prepare_script=args.prepare_scripts
    )

    # single top production (tb channel)
    miner.run_multiple(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/cutflow_tb_lv_background',
        mg_process_directory=f'{args.main_dir}/cutflow/tb_lv_background',
        proc_card_file=f'cards/cutflow/proc_card_tb_lv.dat',
        param_card_template_file=param_card_template_file,
        sample_benchmarks=['sm'],
        is_background = True,
        run_card_files=['cards/cutflow/run_card_250k_nocuts.dat','cards/cutflow/run_card_250k_WHMadminerCuts_1.dat','cards/cutflow/run_card_250k_WHMadminerCuts_2.dat','cards/cutflow/run_card_250k_WHMadminerCuts_3.dat','cards/cutflow/run_card_250k_WHMadminerCuts_4.dat','cards/cutflow/run_card_250k_WHMadminerCuts_5.dat','cards/cutflow/run_card_250k_WHMadminerCuts_6.dat'],
        initial_command=args.init_command,
        only_prepare_script=args.prepare_scripts
    )

    # semi-leptonic ttbar
    miner.run_multiple(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/cutflow_tt_semileptonic_background',
        mg_process_directory=f'{args.main_dir}/cutflow/tt_semileptonic_background',
        proc_card_file=f'cards/cutflow/proc_card_tt_semileptonic.dat',
        param_card_template_file=param_card_template_file,
        sample_benchmarks=['sm'],
        is_background = True,
        run_card_files=['cards/cutflow/run_card_250k_nocuts.dat','cards/cutflow/run_card_250k_WHMadminerCuts_1.dat','cards/cutflow/run_card_250k_WHMadminerCuts_2.dat','cards/cutflow/run_card_250k_WHMadminerCuts_3.dat','cards/cutflow/run_card_250k_WHMadminerCuts_4.dat','cards/cutflow/run_card_250k_WHMadminerCuts_5.dat','cards/cutflow/run_card_250k_WHMadminerCuts_6.dat'],
        initial_command=args.init_command,
        only_prepare_script=args.prepare_scripts
    )
    
    os.remove('/tmp/generate.mg5')
