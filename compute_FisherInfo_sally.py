# -*- coding: utf-8 -*-

"""
compute_FisherInfo_histograms.py

Computes Fisher Information matrices from the full kinematics using the score derived by the SALLY method.

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import argparse as ap
from compute_FisherInfo_histograms import get_FisherInfo_dict

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

def get_FisherInfo_sally(fisher_info_dict,main_dir,sample_type,lumi=300,labels=['cHW~'],sally_observables='kinematic_only',sally_model='one_layer_50_neurons_50_epochs'):

    logging.info(f'Calculating information in SALLY estimator; target lumi: {lumi} fb^-1')

    fi_list=[]
    fi_cov_list=[]

    for sample, fisher_info in fisher_info_dict.items():
        if fisher_info is None:
            raise RuntimeError(f'Fisher information object for sample {sample} not found.')
        
        fi_mean,fi_covariance = fisher_info.full_information(
            theta=[0.0],luminosity=lumi*1e3,
            model_file=f'{main_dir}/models/{sally_observables}/{sally_model}/sally_ensemble_{sample}_{sample_type}'
        )

        fi_list.append(fi_mean)
        fi_cov_list.append(fi_covariance)

    fi_matrix_dataframe=DataFrame(
        data=np.sum(fi_list,axis=0),
        columns=labels,
        index=labels)
    
    return (fi_matrix_dataframe,fi_list,fi_cov_list)

if __name__== "__main__":

    parser = ap.ArgumentParser(description='Computes Fisher information in SALLY estimator.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('-odir','--out_dir',help='folder where to save results to',default=None)

    parser.add_argument('-s','--sample_type',help='sample types to process, without/with backgrounds.',choices=['signalOnly','withBackgrounds'],default='withBackgrounds')

    parser.add_argument('-so','--sally_observables',help="observables used for the training: all observables for the full observable set and simple kinematic observables for the met observable set",default='kinematic_only',choices=['kinematic_only','all_observables','all_observables_remove_qlCosDeltaMinus'])

    parser.add_argument('-sm','--sally_model',help='which of the SALLY models (for each of the input variable configurations) to use',required=True)

    parser.add_argument('-ci','--charge_inclusive',help='process charge-inclusive samples (split in flavour), combine after individual calculations',action='store_true',default=False)

    parser.add_argument('-i','--inclusive',help='process lepton+flavor inclusive samples',action='store_true',default=False)

    parser.add_argument('-l','--lumi',help='luminosity with which to extract the limits',type=int,default=300.)    

    args=parser.parse_args()
    
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.main_dir

    os.makedirs(f'{out_dir}/limits/',exist_ok=True)

    logging.debug(args)

    # Setting up the Fisher Information object (loaded outside of function, since need to be done once per signal type and take some time to load)
    fisher_info_dict=get_FisherInfo_dict(args.main_dir,args.sample_type,charge_inclusive=args.charge_inclusive,inclusive=args.inclusive)

    # store FI matrix information
    os.makedirs(f'{args.main_dir}/fisher_info/sally_{args.sally_observables}/{args.sally_model}',exist_ok=True)
    fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_sally(fisher_info_dict,args.main_dir,args.sample_type,lumi=300,labels=['cHW~'],
                                                                 sally_observables=args.sally_observables,sally_model=args.sally_model)
    
    log_file_path=f'{out_dir}/limits/fisherInfo_sally_{args.sample_type}_lumi{args.lumi}'
    
    if args.inclusive:
        log_file_path+='_inclusive'
    elif args.charge_inclusive:
        log_file_path+='_charge_inclusive'

    log_file=open(f'{log_file_path}.csv','a')
    if os.path.getsize(f'{log_file_path}.csv') == 0:
        log_file.write('observables, model, I_00 (info on cHWtil), 68% CL, 95% CL \n')

    fisher_info=fi_matrix_dataframe.iloc[0,0]
    log_file.write(f'{args.sally_observables},{args.sally_model},{fisher_info},{1./np.sqrt(fisher_info)},{1.69/np.sqrt(fisher_info)} \n')
    np.savez(f'{args.main_dir}/fisher_info/sally_{args.sally_observables}/{args.sally_model}/fi_{args.sample_type}.npz', fi_list, allow_pickle=False)
    np.savez(f'{args.main_dir}/fisher_info/sally_{args.sally_observables}/{args.sally_model}/fi_cov_{args.sample_type}.npz', fi_cov_list, allow_pickle=False)

    log_file.close()