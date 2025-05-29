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

  parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

  parser.add_argument('-f','--combine_flavors',help='combine muon and electron events for each charges separately', action='store_true',default=False)

  parser.add_argument('-c', '--combine_charges',help='combine w+ and w- events for each flavor separately', action='store_true',default=False)

  args=parser.parse_args()

  flavor_charge_combinations = list(product(('mu','e'),('p','m')))
  
  # signal samples
  signal_samples = [f'w{charge}h_{flavor}' for (flavor,charge) in flavor_charge_combinations]
  for sample in signal_samples:
    event_folder = f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM/Events'
    list_samples_to_combine = [f'{event_folder}/{run}/analysed_events.h5' for run in os.listdir(event_folder)]
    logging.warning(f'found {len(list_samples_to_combine)} runs for {sample}. weighting each by the inverse of the number of runs')
    combine_and_shuffle(list_samples_to_combine,f'{args.main_dir}/{sample}_signalOnly.h5',k_factors=1.0/len(list_samples_to_combine))

  # background samples
  background_samples=[f't{charge}b_{flavor}' for (flavor,charge) in flavor_charge_combinations]
  background_samples+=[f'tt_{flavor}{charge}jj' for (flavor,charge) in flavor_charge_combinations]
  background_samples+=[f'w{charge}bb_{flavor}' for (flavor,charge) in flavor_charge_combinations]
  
  for sample in background_samples:
    event_folder = f'{args.main_dir}/background_samples/{sample}_background/Events'
    list_samples_to_combine = [f'{event_folder}/{run}/analysed_events.h5' for run in os.listdir(event_folder)]
    logging.warning(f'found {len(list_samples_to_combine)} runs for {sample}. weighting each by the inverse of the number of runs')
    combine_and_shuffle(list_samples_to_combine,f'{args.main_dir}/background_samples/{sample}_background.h5',k_factors=1.0/len(list_samples_to_combine))

  for (flavor,charge) in flavor_charge_combinations:
    combine_and_shuffle([
      f'{args.main_dir}/background_samples/t{charge}b_{flavor}_background.h5',
      f'{args.main_dir}/background_samples/tt_{flavor}{charge}jj_background.h5',
      f'{args.main_dir}/background_samples/w{charge}bb_{flavor}_background.h5'],
      f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'
    )

  # combining signals and backgrounds for each charge/flavor pair
  combine_and_shuffle([
    f'{args.main_dir}/w{charge}h_{flavor}_signalOnly.h5',
    f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'],
    f'{args.main_dir}/w{charge}h_{flavor}_withBackgrounds.h5'
  )

  # combining charge/flavor pairs - used to train the SALLY models 
  combine_and_shuffle([f'{args.main_dir}/w{charge}h_{flavor}_signalOnly.h5'
                      for (flavor,charge) in flavor_charge_combinations],
                      f'{args.main_dir}/wh_signalOnly.h5')

  combine_and_shuffle([f'{args.main_dir}/w{charge}h_{flavor}_backgroundOnly.h5'
                    for (flavor,charge) in flavor_charge_combinations],
                  f'{args.main_dir}/wh_backgroundOnly.h5')
  

  combine_and_shuffle([f'{args.main_dir}/wh_signalOnly.h5',f'{args.main_dir}/wh_backgroundOnly.h5'],
                f'{args.main_dir}/wh_withBackgrounds.h5')

  logging.info('finished standard sample combination for training of ML methods, will now start the combinations used for plotting')

  if args.combine_flavors:
    logging.info("Combining samples with same charge and different flavors")
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
    logging.info("Combining samples with opposite charge and same flavor")
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
