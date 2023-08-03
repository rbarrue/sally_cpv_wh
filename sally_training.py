# -*- coding: utf-8 -*-

"""
sally_training.py

Handles extraction of joint score from event samples and training of SALLY method

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import sys
import os
from time import strftime
import argparse as ap
import numpy as np

from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ScoreEstimator, Ensemble

# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)


# timestamp for model saving
timestamp = strftime("%d%m%y")

def augment_and_train(input_dir,sample_name,nsamples=-1,observable_set='met',training_observables='kinematic_only',nestimators=5,mode='augment_only',model_name=''):

  """  
  Creates training samples for the a local score-based method (SALLY), using SampleAugmenter to extract training (and test) samples and joint score.

  Trains an ensemble of NN as score estimators (to have an idea of the uncertainty from the different NN trainings).

  Parameters:

  input_dir: folder where h5 files and samples are stored

  sample_name: .h5 sample name to augment/train

  observable_set: 'full' (including unobservable degrees of freedom) or 'met' (only observable degrees of freedom)

  training_observables: which observables use to do the training, 'kinematic_only' (for only kinematic observables in full and met observable set), all_observables (kinematic + angular observables in met observable set)

  nestimators: number of estimators for SALLY method NN ensemble

  mode: either create only the augmented data samples (mode=augment_only), only do the SALLY method training (mode=train_only), or augment+training (mode=augment_and_train)

  model_name: model name, given to differentiate between, e.g. different SALLY NN configurations
  """

  # access to the .h5 file with MadMiner settings
  madminer_settings=load_madminer_settings(f'{input_dir}/{observable_set}/{sample_name}.h5',include_nuisance_benchmarks=False)

  if nsamples==-1:
    nsamples=madminer_settings[6]

  logging.info(f'running mode: {mode}; sample_name: {sample_name}; observable set: {observable_set}; training observables: {training_observables}; nsamples: {nsamples}')

  if mode.lower() in ['augment_only','augment_and_train']:

    ######### Outputting training variable index for training step ##########
    observable_dict=madminer_settings[5]
    for i_obs, obs_name in enumerate(observable_dict):
      logging.info(f'index: {i_obs}; name: {obs_name};')

    ########## Sample Augmentation ###########

    # object to create the augmented training samples
    sampler=SampleAugmenter(f'{input_dir}/{observable_set}/{sample_name}.h5')

    # Creates a set of training data (as many as the number of estimators) - centered around the SM
    # Samples from all benchmarks to take into account the different kinematics from CP-odd operator
    for i_estimator in range(nestimators):
      _,_,_,eff_n_samples = sampler.sample_train_local(theta=sampling.benchmark('sm'),
                                        n_samples=int(nsamples),
                                        folder=f'{input_dir}/{observable_set}/training_samples/',
                                        filename=f'train_score_{sample_name}_{i_estimator}',
                                        sample_only_from_closest_benchmark=False)
    
    logging.info(f'effective number of samples for estimator {i_estimator}: {eff_n_samples}')

  if mode.lower() in ['train_only','augment_and_train']:

    ########## Training ###########

    # Choose which features to train on 
    # If 'met' or 'full', we use all of them (None), otherwise we select the correct indices
    # for the full observable set we always use all of the variables in the sample
    if observable_set == 'full':
      if training_observables!='kinematic_only':
        logging.warning('for the full observable set, always training with the kinematic_only observable set, which includes all features')
      training_observables='kinematic_only'
      my_features = None
    else:
      if training_observables == 'kinematic_only':
        my_features = list(range(48))
      if training_observables == 'all_observables':
        my_features = None
      # removing non-charge-weighted cosDelta (50,51) and charge-weighted cosThetaStar (53)
      elif training_observables == 'all_observables_remove_redundant_cos':
        my_features = [*range(48),48,49,52,54,55]
      elif training_observables == 'ptw_ql_cos_deltaPlus':
        my_features = [18,54]      
      elif training_observables == 'mttot_ql_cos_deltaPlus':
        my_features = [39,54]
    
    #Create a list of ScoreEstimator objects to add to the ensemble
    estimators = [ ScoreEstimator(features=my_features, n_hidden=(100,50,),activation="relu") for _ in range(nestimators) ]
    ensemble = Ensemble(estimators)

    # Run the training of the ensemble
    # result is a list of N tuples, where N is the number of estimators,
    # and each tuple contains two arrays, the first with the training losses, the second with the validation losses
    result = ensemble.train_all(method='sally',
      x=[f'{input_dir}/{observable_set}/training_samples/x_train_score_{sample_name}_{i_estimator}.npy' for i_estimator in range(nestimators)],
      t_xz=[f'{input_dir}/{observable_set}/training_samples/t_xz_train_score_{sample_name}_{i_estimator}.npy' for i_estimator in range(nestimators)],
      memmap=True,verbose="none",n_workers=4,limit_samplesize=nsamples,n_epochs=100,batch_size=1024,
    )    

    # saving ensemble state dict and training and validation losses
    ensemble.save(f'{input_dir}/{observable_set}/models/{training_observables}/{model_name}/sally_ensemble_{sample_name}')
    np.savez(f'{input_dir}/{observable_set}/models/{training_observables}/{model_name}/losses_{sample_name}',result)
  
if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Creates augmented (unweighted) training samples for a local score-based method (SALLY). Trains an ensemble of NNs as score estimators.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--run_mode',help="running mode: 'augment_only' creates only training samples; 'train_only' does only the training; 'augment_and_train': does augmentation and training in one go",required=True,choices=['augment_only','train_only','augment_and_train'])

    parser.add_argument('--observable_set',help="which observable sets to process in one run: full (including unobservable degrees of freedom), met (only observable degrees of freedom), or both sequentially",default=['full','met'],choices=['full','met'],nargs="+")

    parser.add_argument('--training_observables',help="observables used for the training: all observables for the full observable set and simple kinematic observables for the met observable set",default='kinematic_only',choices=['kinematic_only','all_observables_remove_redundant_cos'])

    parser.add_argument('--nsamples',help="number of events in augmented data sample/number of events on which to train on. Note: if running augmentation and training in separate jobs, these can be different, although number of events in training <= number of events in augmented data sample",type=int,default=-1)

    parser.add_argument('--channel',help='lepton+charge flavor channels to augment/train. included to allow parallel training of the different channels',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],nargs="+",default=['wh_mu','wh_e'])
    
    parser.add_argument('--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds. included to allow sequential training for the different possibilities',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],nargs="+",default=['signalOnly_SMonly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe'])

    parser.add_argument('--model_name',help='model name, given to differentiate between, e.g. different SALLY NN configurations',default=timestamp)

    args=parser.parse_args()

    for observable_set in args.observable_set:
        for channel in args.channel:
            for sample_type in args.sample_type:
                
                logging.info(f'observable set: {observable_set}; channel: {channel}; sample type: {sample_type}')
                
                augment_and_train(input_dir=args.main_dir,observable_set=observable_set,training_observables=args.training_observables,model_name=args.model_name,sample_name=f'{channel}_{sample_type}',mode=args.run_mode,nsamples=args.nsamples)
