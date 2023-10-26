# -*- coding: utf-8 -*-

"""
analysis_sample_combiner.py

Combines analyzed samples from the output of the analysis scripts.

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

import logging
import os, sys
import argparse as ap

from itertools import product
from madminer.sampling import combine_and_shuffle

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
  
  parser = ap.ArgumentParser(description='Combines and shuffles different samples, depending on the purposes.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

  parser.add_argument('--combine_individual',help='combine samples for each of the charge+flavor combination separately (should be done once before all other combination possibilities)',action='store_true',default=False)

  parser.add_argument('--combine_flavors',help='combine muon and electron events for each charges separately', action='store_true',default=False)

  parser.add_argument('--combine_charges',help='combine w+ and w- events for each flavor separately', action='store_true',default=False)

  parser.add_argument('--combine_all',help='combine all charges + flavors', action='store_true',default=False)

  args=parser.parse_args()

  if not args.combine_individual and (args.combine_flavors or args.combine_charges or args.combine_all):
    logging.warning('not asking to combine samples for each of the charge+flavor combination separately, but asking to make a combination of samples from different charges or flavors. watch out for possible crash from missing samples')
  
  if args.combine_flavors and args.combine_charges:
    if not args.combine_all:
      logging.warning('asked to combine samples in flavor and in charge separately but combine_all=False, won\'t create single WH sample')
  
  flavor_charge_combinations = list(product(('mu','e'),('p','m')))

  if args.combine_individual:
    # combining individual run samples
    signal_samples = [f'w{charge}h_{flavor}' for (flavor,charge) in flavor_charge_combinations]
    for sample in signal_samples:
      event_folder = f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM/Events'
      list_samples_to_combine = [f'{event_folder}/{run}/analysed_events.h5' for run in os.listdir(event_folder)]
      combine_and_shuffle(list_samples_to_combine,f'{args.main_dir}/{sample}_signalOnly.h5')

    background_samples=[f't{charge}b_{flavor}' for (flavor,charge) in flavor_charge_combinations]
    background_samples+=[f'tt_{flavor}{charge}jj' for (flavor,charge) in flavor_charge_combinations]
    background_samples+=[f'w{charge}bb_{flavor}' for (flavor,charge) in flavor_charge_combinations]
    
    for sample in background_samples:
      event_folder = f'{args.main_dir}/background_samples/{sample}_background/Events'
      list_samples_to_combine = [f'{event_folder}/{run}/analysed_events.h5' for run in os.listdir(event_folder)]
      combine_and_shuffle(list_samples_to_combine,f'{args.main_dir}/background_samples/{sample}_background.h5')

    for (flavor,charge) in flavor_charge_combinations:
      combine_and_shuffle([
        f'{args.main_dir}/background_samples/t{charge}b_{flavor}_background.h5',
        f'{args.main_dir}/background_samples/tt_{flavor}{charge}jj_background.h5',
        f'{args.main_dir}/background_samples/w{charge}bb_{flavor}_background.h5'],
        f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'
      )

      combine_and_shuffle([
        f'{args.main_dir}/w{charge}h_{flavor}_signalOnly.h5',
        f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'],
        f'{args.main_dir}/w{charge}h_{flavor}_withBackgrounds.h5'
      )

    logging.info('finished standard sample combination for training of ML methods, will now start the combinations used for plotting')

  if args.combine_flavors:
    for charge in ('p','m'):
      combine_and_shuffle([
        f'{args.main_dir}/w{charge}h_e_signalOnly.h5',
        f'{args.main_dir}/w{charge}h_mu_signalOnly.h5'],
        f'{args.main_dir}/w{charge}h_signalOnly.h5'
      )

      combine_and_shuffle([
        f'{args.main_dir}/w{charge}h_e_backgroundOnly.h5',
        f'{args.main_dir}/w{charge}h_mu_backgroundOnly.h5'],
        f'{args.main_dir}/w{charge}h_backgroundOnly.h5'
      )

      combine_and_shuffle([
        f'{args.main_dir}/w{charge}h_signalOnly.h5',
        f'{args.main_dir}/w{charge}h_backgroundOnly.h5'],
        f'{args.main_dir}/w{charge}h_withBackgrounds.h5'
      )

  if args.combine_charges:
    for flavor in ('e','mu'):
      combine_and_shuffle([
        f'{args.main_dir}/wph_{flavor}_signalOnly.h5',
        f'{args.main_dir}/wmh_{flavor}_signalOnly.h5'],
        f'{args.main_dir}/wh_{flavor}_signalOnly.h5'
      )

      combine_and_shuffle([
        f'{args.main_dir}/wph_{flavor}_backgroundOnly.h5',
        f'{args.main_dir}/wmh_{flavor}_backgroundOnly.h5'],
        f'{args.main_dir}/wh_{flavor}_backgroundOnly.h5'
      )

      combine_and_shuffle([
        f'{args.main_dir}/wh_{flavor}_signalOnly.h5',
        f'{args.main_dir}/wh_{flavor}_backgroundOnly.h5'],
        f'{args.main_dir}/wh_{flavor}_withBackgrounds.h5'
      )
  
  if args.combine_all:
    combine_and_shuffle([f'{args.main_dir}/w{charge}h_{flavor}_signalOnly.h5'
                         for (flavor,charge) in flavor_charge_combinations],
                        f'{args.main_dir}/wh_signalOnly.h5')
    
    combine_and_shuffle([f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'
                      for (flavor,charge) in flavor_charge_combinations],
                    f'{args.main_dir}/wh_backgroundOnly.h5')
    
    combine_and_shuffle([f'{args.main_dir}/wh_signalOnly.h5',f'{args.main_dir}/wh_backgroundOnly.h5'],
                f'{args.main_dir}/wh_withBackgrounds.h5')