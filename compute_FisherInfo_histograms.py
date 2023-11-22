# -*- coding: utf-8 -*-

"""
compute_FisherInfo_histograms.py

Computes Fisher Information matrices from several quantities:

- complete truth-level information (as if one could measure the latent variables)

- rate (to see what is the sensitivity of the total rate)

- 1D or 2D histograms (starting from pTW ala VHbb STXS and moving to mTtot to angular observables)

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

import logging
import os
import argparse as ap
import numpy as np
from itertools import product

# central object to extract the Fisher information from observables
from madminer.fisherinformation import FisherInformation

from pandas import DataFrame # to write the FI matrices

#MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
      logging.getLogger(key).setLevel(logging.WARNING)

# function to fetch and load dict of FisherInformation objects
def get_FisherInfo_dict(main_dir,sample_type,include_nuisance_parameters=False,charge_inclusive=False,inclusive=False):

    fisher_info_dict={}

    if inclusive:
        charge_flavor_combinations=['wh']
    elif charge_inclusive:
        charge_flavor_combinations=[f'wh_{flavor}' for flavor in ('mu','e')]
    else:
        charge_flavor_combinations=[f'w{charge}h_{flavor}' for (charge,flavor) in product(('p','m'),('mu','e'))]

    for charge_flavor_pair in charge_flavor_combinations:
        logging.info(f'Loading Fisher information for {charge_flavor_pair}...')
        fisher_info_dict[charge_flavor_pair]=FisherInformation(f'{main_dir}/{charge_flavor_pair}_{sample_type}.h5',include_nuisance_parameters)

    return fisher_info_dict

def get_FisherInfo_parton(fisher_info_dict,lumi=300,labels=['cHW~']):

    fi_list=[]
    fi_cov_list=[]

    for _,fisher_info in fisher_info_dict.items():
        
        fi_mean, fi_covariance=fisher_info.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_list.append(fi_mean)
        fi_cov_list.append(fi_covariance)

    fi_matrix_dataframe=DataFrame(
        data=np.sum(fi_list,axis=0),
        columns=labels,
    index=labels)

    return (fi_matrix_dataframe,fi_list,fi_cov_list)


def get_FisherInfo_rate(fisher_info_dict,lumi=300,labels=['cHW~']):

    fi_list=[]
    fi_cov_list=[]

    for _,fisher_info in fisher_info_dict.items():
        
        fi_mean, fi_covariance=fisher_info.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_list.append(fi_mean)
        fi_cov_list.append(fi_covariance)

    fi_matrix_dataframe=DataFrame(
        data=np.sum(fi_list,axis=0),
        columns=labels,
    index=labels)

    return (fi_matrix_dataframe,fi_list,fi_cov_list)


def get_FisherInfo_histograms(fisher_info_dict,lumi=300,observable='pt_w',bins_observable=[0.,75.,150.,250.,400.],observable2=None,bins_observable2=None,labels=['cHW~']):

    fi_list=[]
    fi_cov_list=[]

    for _,fisher_info in fisher_info_dict.items():

        ### 1D histograms
        if observable2 is None:
            fi_mean, fi_covariance=fisher_info.histo_information(
                theta=[0.0],luminosity=lumi*1e3,
                observable=observable, bins=bins_observable,
                histrange=(bins_observable[0],bins_observable[-1])
            )

        ### 2D histograms
        else:
            
            fi_mean, fi_covariance=fisher_info.histo_information_2d(
                theta=[0.0],luminosity=lumi*1e3,
                observable1=observable, bins1=bins_observable,
                observable2=observable2,bins2=bins_observable2,
                histrange1=(bins_observable[0],bins_observable[-1]),
                histrange2=(bins_observable2[0],bins_observable2[-1])
            )
        
        
        fi_list.append(fi_mean)
        fi_cov_list.append(fi_covariance)

    fi_matrix_dataframe=DataFrame(
        data=np.sum(fi_list,axis=0),
        columns=labels,
    index=labels)

    return (fi_matrix_dataframe,fi_list,fi_cov_list)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Computes Fisher information in histograms of different variables.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('-pdir','--plot_dir',help='folder where to save plots to',required=True)

    parser.add_argument('-s','--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',choices=['signalOnly_SMonly','signalOnly','withBackgrounds_SMonly','withBackgrounds'],default='signalOnly')

    parser.add_argument('-m','--mode',help='what to use to extract the limits, given as input to the expected_limits function',choices=['parton','rate','histo'],required=True)

    parser.add_argument('-x','--observable_x',help='which of the observables to use in the x-axis if extracting limits from histograms',default=None)

    parser.add_argument('-y','--observable_y',help='which of the observables to use in the y-axis if extracting limits from histograms',default=None)

    parser.add_argument('-bx','--binning_x',help='binning of the variable in the x-axis (can either be a standard observable or output of the SALLY network)',nargs="*",type=float,default=None)

    parser.add_argument('-by','--binning_y',help='binning of the variable in the y-axis',nargs="*",type=float,default=None)

    parser.add_argument('-ci','--charge_inclusive',help='process charge-inclusive samples (still split in flavour)',action='store_true',default=False)

    parser.add_argument('-i','--inclusive',help='process charge+flavor inclusive samples',action='store_true',default=False)

    args=parser.parse_args()
    
    # store FI matrix information
    os.makedirs(f'{args.main_dir}/fisher_info/',exist_ok=True)
    os.makedirs(f'{args.plot_dir}/limits/',exist_ok=True)

    logging.debug(args)

    # Loading FisherInformation objects
    fisher_info_dict=get_FisherInfo_dict(args.main_dir,args.sample_type,include_nuisance_parameters=False,charge_inclusive=args.charge_inclusive,inclusive=args.inclusive)  

    log_file_path=f'{args.plot_dir}/limits/fisherInfo_histograms_{args.sample_type}'

    if args.inclusive:
        log_file_path+='_inclusive'
    elif args.charge_inclusive:
        log_file_path+='_charge_inclusive'

    log_file=open(f'{log_file_path}.csv','a')
    if os.path.getsize(f'{log_file_path}.csv') == 0:
        log_file.write('observable_x, binning_x, observable_y, binning_y, I_00 (info on cHWtil), 68% CL, 95% CL \n')

    # Complete truth information (from parton-level weights)
    if args.mode=='parton':
        if 'signalOnly' in args.sample_type:                                       
            fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_parton(fisher_info_dict)    
            log_file.write( f'truth, None, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
            np.savez(f'{args.main_dir}/fisher_info/fi_truth_{args.sample_type}.npz', fi_list, allow_pickle=False)
            np.savez(f'{args.main_dir}/fisher_info/fi_cov_truth_{args.sample_type}.npz', fi_cov_list, allow_pickle=False)
        else:
            logging.warning('Asking for parton-level limits in a sample which is not signal-only, skipping...')

    # Rate information
    if args.mode=='rate':
        fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_rate(fisher_info_dict)
        log_file.write( f'rate, None, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
        np.savez(f'{args.main_dir}/fisher_info/fi_rate_{args.sample_type}.npz', fi_list, allow_pickle=False)
        np.savez(f'{args.main_dir}/fisher_info/fi_cov_rate_{args.sample_type}.npz', fi_cov_list, allow_pickle=False)

    # histogram information
    if args.observable_x!=None:
        # 1D 
        if args.observable_y==None:
            fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_histograms(fisher_info_dict,observable=args.observable_x,bins_observable=args.binning_x)
            log_file.write( f'{args.observable_x}, {args.binning_x}, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
            np.savez(f'{args.main_dir}/fisher_info/fi_{args.observable_x}_{len(args.binning_x)}bins_{args.sample_type}.npz', fi_list, allow_pickle=False)
            np.savez(f'{args.main_dir}/fisher_info/fi_cov_{args.observable_x}_{len(args.binning_x)}bins_{args.sample_type}.npz', fi_cov_list, allow_pickle=False)
        # 2D
        else:                
            fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_histograms(fisher_info_dict,observable=args.observable_x,bins_observable=args.binning_x,
                                                                                    observable2=args.observable_y,bins_observable2=args.binning_y)
            log_file.write( f'{args.observable_x}, {args.binning_x}, {args.observable_y}, {args.binning_y}, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')   
            np.savez(f'{args.main_dir}/fisher_info/fi_{args.observable_x}_{len(args.binning_x)}bins_{args.observable_y}_{len(args.observable_y)}bins_{args.sample_type}.npz', fi_list, allow_pickle=False)
            np.savez(f'{args.main_dir}/fisher_info/fi_cov_{args.observable_x}_{len(args.binning_x)}bins_{args.observable_y}_{len(args.observable_y)}bins_{args.sample_type}.npz', fi_cov_list, allow_pickle=False)
    else:
        logging.warning('did not specify any of the possible histogram options, skipping...')

    log_file.close()