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
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

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

    parser.add_argument('--setup_file',help='name of setup file (without the .h5)',required=True)

    parser.add_argument('--do_pythia',help='whether or not to run Pythia after Madgraph',default=False,action='store_true')

    parser.add_argument('--pythia_card',help='if running pythia, the path for the Pythia card',default='cards/pythia8_card.dat')

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',default=False,action='store_true')

    parser.add_argument('--mg_dir',help='Path for MadGraph installation',required=True)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/{args.setup_file}.h5')
    lhe = LHEReader(f'{args.main_dir}/{args.setup_file}.h5')

    init_command=None,

    # W + b b~, divided by W decay channel and charge

    # W+ -> mu+ vm 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wpbb_mu_background',
        mg_process_directory=f'{args.main_dir}/background_samples/wpbb_mu_background',
        proc_card_file='cards/background_processes/proc_card_wpbb_mu.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W+ -> e+ ve 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wpbb_e_background',
        mg_process_directory=f'{args.main_dir}/background_samples/wpbb_e_background',
        proc_card_file='cards/background_processes/proc_card_wpbb_e.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W- -> mu- vm~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wmbb_mu_background',
        mg_process_directory=f'{args.main_dir}/background_samples/wmbb_mu_background',
        proc_card_file='cards/background_processes/proc_card_wmbb_mu.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W- -> e- ve~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wmbb_e_background',
        mg_process_directory=f'{args.main_dir}/background_samples/wmbb_e_background',
        proc_card_file='cards/background_processes/proc_card_wmbb_e.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # tb production, divided by top (W) charge and W decay channel

    # t+, W+ -> mu+ vm 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tpb_mu_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tpb_mu_background',
        proc_card_file='cards/background_processes/proc_card_tpb_mu.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # t+, W+ -> e+ ve 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tpb_e_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tpb_e_background',
        proc_card_file='cards/background_processes/proc_card_tpb_e.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # t-, W- -> mu- vm~ 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tmb_mu_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tmb_mu_background',
        proc_card_file='cards/background_processes/proc_card_tmb_mu.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # t-, W- -> e- ve~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tmb_e_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tmb_e_background',
        proc_card_file='cards/background_processes/proc_card_tmb_e.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # semi-leptonic ttbar production, divided in possible decay channels

    # W+ -> mu+ vm, W- -> j j 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tt_mupjj_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tt_mupjj_background',
        proc_card_file='cards/background_processes/proc_card_tt_mupjj.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W+ -> e+ ve, W- -> j j 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tt_epjj_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tt_epjj_background',
        proc_card_file='cards/background_processes/proc_card_tt_epjj.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W+ -> j j, W- -> mu- vm~ 
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tt_mumjj_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tt_mumjj_background',
        proc_card_file='cards/background_processes/proc_card_tt_mumjj.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )

    # W+ -> j j, W- -> e- ve~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/tt_emjj_background',
        mg_process_directory=f'{args.main_dir}/background_samples/tt_emjj_background',
        proc_card_file='cards/background_processes/proc_card_tt_emjj.dat',
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        sample_benchmark='sm',
        initial_command=init_command,
        is_background=True,
        only_prepare_script=args.prepare_scripts
    )
    
    os.remove('/tmp/generate.mg5')