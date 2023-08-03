# -*- coding: utf-8 -*-

"""
analysis_sample_combiner.py

Combines analyzed samples from the output of the analysis scripts.

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from email.policy import default
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os, shutil, sys
import argparse as ap

from madminer.core import MadMiner
from madminer.lhe import LHEReader
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

  parser.add_argument('--do_signal',help='analyze/combine signal samples', action='store_true',default=False)

  parser.add_argument('--do_backgrounds',help='combine background samples', action='store_true',default=False)

  parser.add_argument('--observable_set',help="using full (including unobservable degrees of freedom) or met (only observable degrees of freedom) observable set",required=True,choices=['full','met'])

  parser.add_argument('--combine_individual',help='combine samples for each of the charge+flavor combination separately (should be done once before all other combination possibilities)',action='store_true',default=False)

  parser.add_argument('--combine_flavors',help='combine muon and electron events for each charges separately', action='store_true',default=False)

  parser.add_argument('--combine_charges',help='combine w+ and w- events for each flavor separately', action='store_true',default=False)

  parser.add_argument('--combine_all',help='combine all charges + flavors', action='store_true',default=False)

  parser.add_argument('--combine_BSM',help='combine samples generated at the BSM benchmarks with those generated at the SM point', action='store_true',default=False)

  args=parser.parse_args()

  if not (args.do_signal or args.do_backgrounds):
   sys.exit('asking to process events but not asking for either signal or background ! Exiting...')

  if not args.combine_individual and (args.combine_flavors or args.combine_charges or args.combine_all):
    logging.warning('not asking to combine samples for each of the charge+flavor combination separately, but asking to make a combination of samples from different charges or flavors. watch out for possible crash from missing samples')
  
  if args.combine_flavors and args.combine_charges:
    if not args.combine_all:
      logging.warning('asked to combine samples in flavor and in charge separately but combine_all=False, won\'t create single WH sample')

  full_proc_dir=f'{args.main_dir}/{args.observable_set}/'
  
  if args.combine_individual:
      
    #### mu+ vm
    if args.do_signal:

      # generated SM samples
      os.symlink(f'{full_proc_dir}/signal/wph_mu_smeftsim_SM_lhe.h5'.format(),f'{full_proc_dir}/wph_mu_signalOnly_SMonly_noSysts_lhe.h5'.format())

      # generated SM samples + generated BSM samples      
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/signal/wph_mu_smeftsim_SM_lhe.h5',
          f'{full_proc_dir}/signal/wph_mu_smeftsim_BSM_lhe.h5'],
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5'
        )

    if args.do_backgrounds:

      combine_and_shuffle([
        f'{full_proc_dir}/background/tpb_mu_background_lhe.h5',
        f'{full_proc_dir}/background/tt_mupjj_background_lhe.h5',
        f'{full_proc_dir}/background/wpbb_mu_background_lhe.h5'],
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:

      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/signal/wph_mu_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wph_mu_withBackgrounds_SMonly_noSysts_lhe.h5'
      )   

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wph_mu_withBackgrounds_noSysts_lhe.h5'
        )

    #### e+ ve
    if args.do_signal:

      # generated SM samples
      os.symlink(f'{full_proc_dir}/signal/wph_e_smeftsim_SM_lhe.h5',f'{full_proc_dir}/wph_e_signalOnly_SMonly_noSysts_lhe.h5')

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/signal/wph_e_smeftsim_SM_lhe.h5',
          f'{full_proc_dir}/signal/wph_e_smeftsim_BSM_lhe.h5'],
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5'
        )

    if args.do_backgrounds:

      combine_and_shuffle([
        f'{full_proc_dir}/background/tpb_e_background_lhe.h5',
        f'{full_proc_dir}/background/tt_epjj_background_lhe.h5',
        f'{full_proc_dir}/background/wpbb_e_background_lhe.h5'],
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:

      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/signal/wph_e_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wph_e_withBackgrounds_SMonly_noSysts_lhe.h5'
      ) 

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wph_e_withBackgrounds_noSysts_lhe.h5'
        )

    ##### mu- vm~
    if args.do_signal:

      # generated SM samples
      os.symlink(f'{full_proc_dir}/signal/wmh_mu_smeftsim_SM_lhe.h5',f'{full_proc_dir}/wmh_mu_signalOnly_SMonly_noSysts_lhe.h5')

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/signal/wmh_mu_smeftsim_SM_lhe.h5',
          f'{full_proc_dir}/signal/wmh_mu_smeftsim_BSM_lhe.h5'],
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5'
        )

    if args.do_backgrounds:

      combine_and_shuffle([
        f'{full_proc_dir}/background/tmb_mu_background_lhe.h5',
        f'{full_proc_dir}/background/tt_mumjj_background_lhe.h5',
        f'{full_proc_dir}/background/wmbb_mu_background_lhe.h5'],
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:

      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/signal/wmh_mu_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wmh_mu_withBackgrounds_SMonly_noSysts_lhe.h5'
      )     

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wmh_mu_withBackgrounds_noSysts_lhe.h5'
        )

    #### e- ve~
    if args.do_signal:

      # generated SM samples
      os.symlink(f'{full_proc_dir}/signal/wmh_e_smeftsim_SM_lhe.h5',f'{full_proc_dir}/wmh_e_signalOnly_SMonly_noSysts_lhe.h5')

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/signal/wmh_e_smeftsim_SM_lhe.h5',
          f'{full_proc_dir}/signal/wmh_e_smeftsim_BSM_lhe.h5'],
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5'
        )

    if args.do_backgrounds:

      combine_and_shuffle([
        f'{full_proc_dir}/background/tmb_e_background_lhe.h5',
        f'{full_proc_dir}/background/tt_emjj_background_lhe.h5',
        f'{full_proc_dir}/background/wmbb_e_background_lhe.h5'],
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:

      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/signal/wmh_e_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wmh_e_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wmh_e_withBackgrounds_noSysts_lhe.h5'
        )

  logging.info('finished standard sample combination for training of ML methods, will now start the combinations used for plotting')
  
  # combining electron and muon channels for each of the charges separately
  if args.combine_flavors:

    if args.do_signal:
      
      #### w+
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wph_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wph_signalOnly_noSysts_lhe.h5'
        )

      #### w-
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wmh_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wmh_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wmh_signalOnly_noSysts_lhe.h5'
        )
    
    if args.do_backgrounds:

      #### w+
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wph_backgroundOnly_noSysts_lhe.h5'
      )

      #### w-
      combine_and_shuffle([
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wmh_backgroundOnly_noSysts_lhe.h5'
      )
    
    if args.do_signal and args.do_backgrounds:
      
      #### w+
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wph_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wph_withBackgrounds_noSysts_lhe.h5'
        )

      #### w-
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wmh_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wmh_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wmh_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wmh_withBackgrounds_noSysts_lhe.h5'
        )
  
  # combining positive and negative charge channels for each of the flavors separately
  if args.combine_charges:

    if args.do_signal:
      
      #### e
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_e_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_e_signalOnly_noSysts_lhe.h5'
        )

      #### mu
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_mu_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_mu_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_mu_signalOnly_noSysts_lhe.h5'
        )
    
    if args.do_backgrounds:

      #### e
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_e_backgroundOnly_noSysts_lhe.h5'
      )

      #### mu
      combine_and_shuffle([
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_mu_backgroundOnly_noSysts_lhe.h5'
      )
    
    if args.do_signal and args.do_backgrounds:
      #### e
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wh_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wh_e_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_e_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wh_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wh_e_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_e_withBackgrounds_noSysts_lhe.h5'
        )

      #### mu
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wh_mu_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_mu_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wh_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wh_mu_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_mu_withBackgrounds_noSysts_lhe.h5'
        )
  
  # combining positive and negative charge channels and all the flavors
  if args.combine_all:
    if args.do_signal:
      
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_signalOnly_noSysts_lhe.h5'
        )
    
    if args.do_backgrounds:
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wh_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_withBackgrounds_SMonly_noSysts_lhe.h5'
      )
      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wh_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_withBackgrounds_noSysts_lhe.h5'
        )