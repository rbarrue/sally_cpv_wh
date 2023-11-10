# -*- coding: utf-8 -*-

"""
gen_signal.py

Reweights WH signal events WH(->l v b b~), divided by W decay channel and charge (250k events/submission)
- to be ran after signal generation if you want to run it multicore, since reweighting doesn't work in multi-core mode

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

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
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Reweights WH signal events WH(->l v b b~), divided by W decay channel and charge. Run after generation.',
                                formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--sample',help='sample to reweight', required=True)

    parser.add_argument('--setup_file',help='name of setup file (without the .h5)',required=True)

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    args=parser.parse_args()

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{args.main_dir}/{args.setup_file}.h5')
    lhe = LHEReader(f'{args.main_dir}/{args.setup_file}.h5')

    # List of benchmarks - SM + 2 BSM benchmarks (from Madminer)
    list_benchmarks = lhe.benchmark_names_phys

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if args.auto_widths:
        param_card_template_file='/lstore/titan/atlas/rbarrue/HWW_tensor_structure/sally_cpv_wh/cards/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file='/lstore/titan/atlas/rbarrue/HWW_tensor_structure/sally_cpv_wh/cards/param_card_template_SMEFTsim3_MwScheme.dat'
    
    # remove element from list, returning a new list
    def remove_element(lst, element):
        new_lst = [*lst]
        new_lst.remove(element)
        return new_lst
    
    # running reweightings serially for each sample (due to concurrent file access issues)
    for run in os.listdir(f'{args.main_dir}/signal_samples/{args.sample}_smeftsim_SM/Events'):
        miner.reweight_existing_sample(
            mg_process_directory=f'{args.main_dir}/signal_samples/{args.sample}_smeftsim_SM',
            run_name=run,
            sample_benchmark='sm',
            reweight_benchmarks=remove_element(list_benchmarks,'sm'), 
            param_card_template_file=param_card_template_file,
            log_directory=f'{args.main_dir}/logs/{args.sample}_smeftsim_SM_reweight/{run}',
        )