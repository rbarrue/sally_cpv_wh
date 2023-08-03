# -*- coding: utf-8 -*-

"""
compute_FisherInfo_histograms.py

Computes Fisher Information matrices from the full kinematics using the score derived by the SALLY method.

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import argparse as ap

# central object to extract the Fisher information from observables
from madminer.fisherinformation import FisherInformation

from pandas import DataFrame # to write the FI matrices

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

def get_FisherInfo_in_sally(main_proc_dir,sample_type,lumi=300,labels=['cHW~'],training_observables='kinematic_only',model_name='one_hidden_layer',do_charge_inclusive=False):

    logging.info(f'Calculating information in SALLY estimator; target lumi: {lumi} fb^-1')

    if do_charge_inclusive:
        logging.info('Calculating information in charge-inclusive samples, still separating by flavor.')
        fi_wh_mu_mean, fi_wh_mu_covariance=fisher_wh_mu.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wh_mu_{sample_type}'
        )
        fi_wh_e_mean, fi_wh_e_covariance=fisher_wh_e.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wh_e_{sample_type}'
        )   

        fi_matrix_dataframe=DataFrame(
        data=fi_wh_mu_mean+fi_wh_e_mean,
        columns=labels,
        index=labels)

        fi_list = [
            fi_wh_mu_mean, 
            fi_wh_e_mean,
        ]

        fi_cov_list = [
            fi_wh_mu_covariance, 
            fi_wh_e_covariance,
        ]

    else:
        fi_wph_mu_mean, fi_wph_mu_covariance=fisher_wph_mu.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wph_mu_{sample_type}'
        )

        fi_wph_e_mean, fi_wph_e_covariance=fisher_wph_e.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wph_e_{sample_type}'
        )

        fi_wmh_mu_mean, fi_wmh_mu_covariance=fisher_wmh_mu.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wmh_mu_{sample_type}'  
        )

        fi_wmh_e_mean, fi_wmh_e_covariance=fisher_wmh_e.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_proc_dir}/models/{training_observables}/{model_name}/sally_ensemble_wmh_e_{sample_type}'    
        )

        fi_matrix_dataframe=DataFrame(
        data=fi_wph_mu_mean+fi_wph_e_mean+fi_wmh_mu_mean+fi_wmh_e_mean,
        columns=labels,
        index=labels)

        fi_list = [
            fi_wph_mu_mean, 
            fi_wph_e_mean,
            fi_wmh_mu_mean,
            fi_wmh_e_mean 
        ]

        fi_cov_list = [
            fi_wph_mu_covariance, 
            fi_wph_e_covariance,
            fi_wmh_mu_covariance,
            fi_wmh_e_covariance  
        ]

        #logging.info('\n Information matrix: \n'+fi_matrix_dataframe.to_string())

    return (fi_matrix_dataframe,fi_list,fi_cov_list)

if __name__== "__main__":

    parser = ap.ArgumentParser(description='Computes Fisher information in SALLY estimator.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--observable_set',help="which observable sets to process in one run: full (with full neutrino 4-vector) or met (only observable degrees of freedom)",choices=['full','met'],required=True)

    parser.add_argument('--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],default=['signalOnly_SMonly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe'],nargs="+")

    parser.add_argument('--training_observables',help="observables used for the training: all observables for the full observable set and simple kinematic observables for the met observable set",default='kinematic_only',choices=['kinematic_only','all_observables_remove_redundant_cos'])

    parser.add_argument('--model_name',help='model name, given to differentiate between, e.g. different SALLY NN configurations',required=True)

    parser.add_argument('--do_charge_inclusive',help='if processing charge-inclusive or charge separate samples (still splitting in flavour)',action='store_true',default=False)

    args=parser.parse_args()

    # Partition to store h5 files and samples + path to setup file
    main_proc_dir = f'{args.main_dir}/{args.observable_set}'
    main_plot_dir = f'{args.main_dir}/plots/{args.observable_set}'
  
    for sample_type in args.sample_type:

        logging.debug(f'Main directory: {args.main_dir}; observable set: {args.observable_set}; sample type: {sample_type}')

        # Setting up the Fisher Information objects (loaded outside of function, since need to be done once per signal type and take some time to load)
        if args.do_charge_inclusive:
            fisher_wh_mu=FisherInformation(f'{main_proc_dir}/wh_mu_{sample_type}.h5',include_nuisance_parameters=False)
            fisher_wh_e=FisherInformation(f'{main_proc_dir}/wh_e_{sample_type}.h5',include_nuisance_parameters=False)
        else:
            fisher_wph_mu=FisherInformation(f'{main_proc_dir}/wph_mu_{sample_type}.h5',include_nuisance_parameters=False)
            fisher_wph_e=FisherInformation(f'{main_proc_dir}/wph_e_{sample_type}.h5',include_nuisance_parameters=False)
            fisher_wmh_mu=FisherInformation(f'{main_proc_dir}/wmh_mu_{sample_type}.h5',include_nuisance_parameters=False)
            fisher_wmh_e=FisherInformation(f'{main_proc_dir}/wmh_e_{sample_type}.h5',include_nuisance_parameters=False)

        # store FI matrix information
        os.makedirs(f'{main_plot_dir}/',exist_ok=True)
        os.makedirs(f'{main_proc_dir}/fisher_info/sally_{args.training_observables}/{args.model_name}',exist_ok=True)
        fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_in_sally(main_proc_dir,sample_type,lumi=300,labels=['cHW~'],training_observables=args.training_observables,model_name=args.model_name,do_charge_inclusive=args.do_charge_inclusive)
        
        if args.do_charge_inclusive:
            sample_type+='_charge_inclusive'

        log_file_path=f'{main_plot_dir}/fisherInfo_sally_{sample_type}.csv'
        log_file=open(f'{log_file_path}','a')
        if os.path.getsize(log_file_path) == 0:
            log_file.write('training observables, model_name, I_00 (info on cHWtil), 68% CL, 95% CL \n')

        log_file.write(f'{args.training_observables},{args.model_name},{fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
        np.savez(f'{main_proc_dir}/fisher_info/sally_{args.training_observables}/{args.model_name}/fi_{sample_type}.npz', fi_list, allow_pickle=False)
        np.savez(f'{main_proc_dir}/fisher_info/sally_{args.training_observables}/{args.model_name}/fi_cov_{sample_type}.npz', fi_cov_list, allow_pickle=False)

        log_file.close()