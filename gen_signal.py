# -*- coding: utf-8 -*-

"""
gen_signal.py

Generates WH signal events WH(->l v b b~), divided by W decay channel and charge (250k events each)

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

    parser.add_argument('--do_pythia',help='whether or not to run Pythia after Madgraph',default=False,action='store_true')

    parser.add_argument('--pythia_card',help='if running pythia, the path for the Pythia card',default='cards/pythia8_card.dat')

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',action='store_true',default=False)

    parser.add_argument('--generate_BSM',help='Generate additional events at the BSM benchmarks',action='store_true',default=False)

    parser.add_argument('--mg_dir',help='Path for MadGraph installation',required=True)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/{args.setup_file}.h5')
    lhe = LHEReader(f'{args.main_dir}/{args.setup_file}.h5')

    # List of BSM benchmarks - SM + 1 BSM benchmarks (from Madminer)
    list_BSM_benchmarks = [x for x in lhe.benchmark_names_phys if x != 'sm']

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if args.auto_widths:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat'

    init_command=None,

    # SM samples with MG (re)weights of BSM benchmarks
    # W+ -> mu+ vm
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wph_mu_smeftsim_SM',
        mg_process_directory=f'{args.main_dir}/signal_samples/wph_mu_smeftsim_SM',
        proc_card_file='cards/signal_processes/proc_card_wph_mu_smeftsim.dat',
        param_card_template_file=param_card_template_file,
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        sample_benchmark='sm',
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        initial_command=init_command,
        only_prepare_script=args.prepare_scripts
    )

    # W+ -> e+ ve
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wph_e_smeftsim_SM',
        mg_process_directory=f'{args.main_dir}/signal_samples/wph_e_smeftsim_SM',
        proc_card_file='cards/signal_processes/proc_card_wph_e_smeftsim.dat',
        param_card_template_file=param_card_template_file,
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        sample_benchmark='sm',
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        initial_command=init_command,
        only_prepare_script=args.prepare_scripts
    )

    # W- -> mu- vm~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wmh_mu_smeftsim_SM',
        mg_process_directory=f'{args.main_dir}/signal_samples/wmh_mu_smeftsim_SM',
        proc_card_file='cards/signal_processes/proc_card_wmh_mu_smeftsim.dat',
        param_card_template_file=param_card_template_file,
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        sample_benchmark='sm',
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        initial_command=init_command,
        only_prepare_script=args.prepare_scripts
    )

    # W- -> e- ve~
    miner.run(
        mg_directory=args.mg_dir,
        log_directory=f'{args.main_dir}/logs/wmh_e_smeftsim_SM',
        mg_process_directory=f'{args.main_dir}/signal_samples/wmh_e_smeftsim_SM',
        proc_card_file='cards/signal_processes/proc_card_wmh_e_smeftsim.dat',
        param_card_template_file=param_card_template_file,
        pythia8_card_file=args.pythia_card if args.do_pythia else None,
        sample_benchmark='sm',
        run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
        initial_command=init_command,
        only_prepare_script=args.prepare_scripts
    )

    # BSM samples with MG (re)weights of other benchmarks (inc. SM)
    if args.generate_BSM:
        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/wph_mu_smeftsim_BSM',
            mg_process_directory=f'{args.main_dir}/signal_samples/wph_mu_smeftsim_BSM',
            proc_card_file='cards/signal_processes/proc_card_wph_mu_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=args.pythia_card if args.do_pythia else None,
            sample_benchmarks=list_BSM_benchmarks,
            run_card_files=['cards/run_card_50k_WHMadminerCuts.dat'],
            initial_command=init_command,
            only_prepare_script=args.prepare_scripts
        )

        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/wph_e_smeftsim_BSM',
            mg_process_directory=f'{args.main_dir}/signal_samples/wph_e_smeftsim_BSM',
            proc_card_file='cards/signal_processes/proc_card_wph_e_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=args.pythia_card if args.do_pythia else None,
            sample_benchmarks=list_BSM_benchmarks,
            run_card_files=['cards/run_card_50k_WHMadminerCuts.dat'],
            initial_command=init_command,
            only_prepare_script=args.prepare_scripts
        )

        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/wmh_mu_smeftsim_BSM',
            mg_process_directory=f'{args.main_dir}/signal_samples/wmh_mu_smeftsim_BSM',
            proc_card_file='cards/signal_processes/proc_card_wmh_mu_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=args.pythia_card if args.do_pythia else None,
            sample_benchmarks=list_BSM_benchmarks,
            run_card_files=['cards/run_card_50k_WHMadminerCuts.dat'],
            initial_command=init_command,
            only_prepare_script=args.prepare_scripts
        )

        miner.run_multiple(
            mg_directory=args.mg_dir,
            log_directory=f'{args.main_dir}/logs/wmh_e_smeftsim_BSM',
            mg_process_directory=f'{args.main_dir}/signal_samples/wmh_e_smeftsim_BSM',
            proc_card_file='cards/signal_processes/proc_card_wmh_e_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=args.pythia_card if args.do_pythia else None,
            sample_benchmarks=list_BSM_benchmarks,
            run_card_files=['cards/run_card_50k_WHMadminerCuts.dat'],
            initial_command=init_command,
            only_prepare_script=args.prepare_scripts
        )

    os.remove('/tmp/generate.mg5')