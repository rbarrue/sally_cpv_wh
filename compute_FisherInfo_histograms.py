# -*- coding: utf-8 -*-

"""
compute_FisherInfo_histograms.py

Computes Fisher Information matrices from several quantities:

- complete truth-level information (as if one could measure the latent variables)

- rate (to see what is the sensitivity of the total rate)

- 1D or 2D histograms (starting from pTW ala VHbb STXS and moving to mTtot to angular observables)

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from threading import activeCount
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os, sys
import argparse as ap

# central object to extract the Fisher information from observables
from madminer.fisherinformation import FisherInformation
from madminer.fisherinformation import project_information, profile_information # for projecting/profiling over systematics

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

# TODO: introduce documentation
def get_parton_FisherInfo(lumi=300,do_charge_inclusive=False,labels=['cHW~']):
    
    logging.info(f'Calculating information in the complete truth-level information, target lumi: {lumi} fb^-1')

    if do_charge_inclusive:
        fi_wh_mu_mean, fi_wh_mu_covariance=fisher_wh_mu.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wh_e_mean, fi_wh_e_covariance=fisher_wh_e.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )
    else:
        fi_wph_mu_mean, fi_wph_mu_covariance=fisher_wph_mu.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wph_e_mean, fi_wph_e_covariance=fisher_wph_e.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wmh_mu_mean, fi_wmh_mu_covariance=fisher_wmh_mu.truth_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wmh_e_mean, fi_wmh_e_covariance=fisher_wmh_e.truth_information(
        theta=[0.0],luminosity=lumi*1e3,
        )

    if do_charge_inclusive:
        fi_matrix_dataframe=DataFrame(
        data=fi_wh_mu_mean+fi_wh_e_mean,
        columns=labels,
        index=labels)
    else:
        fi_matrix_dataframe=DataFrame(
        data=fi_wph_mu_mean+fi_wph_e_mean+fi_wmh_mu_mean+fi_wmh_e_mean,
        columns=labels,
        index=labels)

    if do_charge_inclusive:
        fi_list = [
            fi_wh_mu_mean, 
            fi_wh_e_mean
        ]

        fi_cov_list = [
            fi_wh_mu_covariance, 
            fi_wh_e_covariance
        ]
    else:
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
    
    return (fi_matrix_dataframe,fi_list,fi_cov_list)

def get_rate_FisherInfo(lumi=300,do_charge_inclusive=False,labels=['cHW~']):
  
    logging.info(f'Calculating information in a rate-only measurement, target lumi: {lumi} fb^-1')

    if do_charge_inclusive:
        fi_wh_mu_mean, fi_wh_mu_covariance=fisher_wh_mu.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wh_e_mean, fi_wh_e_covariance=fisher_wh_e.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )
    else:
        fi_wph_mu_mean, fi_wph_mu_covariance=fisher_wph_mu.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wph_e_mean, fi_wph_e_covariance=fisher_wph_e.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wmh_mu_mean, fi_wmh_mu_covariance=fisher_wmh_mu.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )

        fi_wmh_e_mean, fi_wmh_e_covariance=fisher_wmh_e.rate_information(
            theta=[0.0],luminosity=lumi*1e3,
        )
    
    if do_charge_inclusive:
        fi_matrix_dataframe=DataFrame(
        data=fi_wh_mu_mean+fi_wh_e_mean,
        columns=labels,
        index=labels)
    else:
        fi_matrix_dataframe=DataFrame(
        data=fi_wph_mu_mean+fi_wph_e_mean+fi_wmh_mu_mean+fi_wmh_e_mean,
        columns=labels,
        index=labels)

    if do_charge_inclusive:
        fi_list = [
            fi_wh_mu_mean, 
            fi_wh_e_mean
        ]

        fi_cov_list = [
            fi_wh_mu_covariance, 
            fi_wh_e_covariance
        ]
    else:
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
    
    return (fi_matrix_dataframe,fi_list,fi_cov_list)

def get_FisherInfo_1D_histograms(lumi=300,observable='pt_w',bins_observable=[0.,75.,150.,250.,400.],labels=['cHW~'],do_charge_inclusive=False):

    if do_charge_inclusive:
        logging.info('Calculating information in charge-inclusive samples, still separating by flavor')
    
    ########## 1D histograms #########
    #if observable2 is None and bins_observable2 is None:
        
    logging.info(f'Calculating 1D histogram information in {len(bins_observable)} bins of observable {observable}, {bins_observable}; target lumi: {lumi} fb^-1')

    if do_charge_inclusive:
        fi_wh_mu_mean, fi_wh_mu_covariance=fisher_wh_mu.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )

        fi_wh_e_mean, fi_wh_e_covariance=fisher_wh_e.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )
    else:
        fi_wph_mu_mean, fi_wph_mu_covariance=fisher_wph_mu.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )

        fi_wph_e_mean, fi_wph_e_covariance=fisher_wph_e.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )

        fi_wmh_mu_mean, fi_wmh_mu_covariance=fisher_wmh_mu.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )

        fi_wmh_e_mean, fi_wmh_e_covariance=fisher_wmh_e.histo_information(
            theta=[0.0],luminosity=lumi*1e3,
            observable=observable, bins=bins_observable,
            histrange=(bins_observable[0],bins_observable[-1])
        )

    if do_charge_inclusive:
        fi_matrix_dataframe=DataFrame(
        data=fi_wh_mu_mean+fi_wh_e_mean,
        columns=labels,
        index=labels)
    else:
        fi_matrix_dataframe=DataFrame(
        data=fi_wph_mu_mean+fi_wph_e_mean+fi_wmh_mu_mean+fi_wmh_e_mean,
        columns=labels,
        index=labels)

    if do_charge_inclusive:
        fi_list = [
            fi_wh_mu_mean, 
            fi_wh_e_mean
        ]

        fi_cov_list = [
            fi_wh_mu_covariance, 
            fi_wh_e_covariance
        ]
    else:
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
    
    return (fi_matrix_dataframe,fi_list,fi_cov_list)

def get_FisherInfo_2D_histograms(lumi=300,observable='pt_w',bins_observable=[0.,75.,150.,250.,400.],observable2='mt_tot',bins_observable2=[0.,400.,800.],labels=['cHW~'],do_charge_inclusive=False):

    logging.info(f'Calculating 2D histogram information in {len(bins_observable)} bins of observable {observable}, {bins_observable} x {len(bins_observable2)} bins of observable {observable2}, {bins_observable2}; target lumi: {lumi} fb^-1')

    if do_charge_inclusive:
        fi_wh_mu_mean, fi_wh_mu_covariance=fisher_wh_mu.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

        fi_wh_e_mean, fi_wh_e_covariance=fisher_wh_e.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

    else:
        fi_wph_mu_mean, fi_wph_mu_covariance=fisher_wph_mu.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

        fi_wph_e_mean, fi_wph_e_covariance=fisher_wph_e.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

        fi_wmh_mu_mean, fi_wmh_mu_covariance=fisher_wmh_mu.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

        fi_wmh_e_mean, fi_wmh_e_covariance=fisher_wmh_e.histo_information_2d(
            theta=[0.0],luminosity=lumi*1e3,
            observable1=observable, bins1=bins_observable,
            observable2=observable2,bins2=bins_observable2,
            histrange1=(bins_observable[0],bins_observable[-1]),
            histrange2=(bins_observable2[0],bins_observable2[-1])
        )

    if do_charge_inclusive:
        fi_matrix_dataframe=DataFrame(
        data=fi_wh_mu_mean+fi_wh_e_mean,
        columns=labels,
        index=labels)
    else:
        fi_matrix_dataframe=DataFrame(
        data=fi_wph_mu_mean+fi_wph_e_mean+fi_wmh_mu_mean+fi_wmh_e_mean,
        columns=labels,
        index=labels)

    if do_charge_inclusive:
        fi_list = [
            fi_wh_mu_mean, 
            fi_wh_e_mean
        ]

        fi_cov_list = [
            fi_wh_mu_covariance, 
            fi_wh_e_covariance
        ]
    else:
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
    
    return (fi_matrix_dataframe,fi_list,fi_cov_list)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Computes Fisher information in histograms of different variables.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--observable_set',help="which observable sets to process in one run: full (including unobservable degrees of freedom), met (only observable degrees of freedom), or both sequentially",choices=['full','met'],required=True)

    parser.add_argument('--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],default='withBackgrounds_SMonly_noSysts_lhe',nargs="+")

    parser.add_argument('--do_parton',help='calculate information contained in the parton-level kinematics (optimal limits)',action='store_true')

    parser.add_argument('--do_rate',help='calculate information contatined in the total rate',action='store_true')

    parser.add_argument('--observable_x',help='which of the observables to use in the x-axis if extracting limits from histograms',default=None)

    parser.add_argument('--observable_y',help='which of the observables to use in the y-axis if extracting limits from histograms',default=None)

    parser.add_argument('--binning_x',help='binning of the variable in the x-axis (can either be a standard observable or output of the SALLY network)',nargs="*",type=float,default=None)

    parser.add_argument('--binning_y',help='binning of the variable in the y-axis',nargs="*",type=float,default=None)

    parser.add_argument('--do_charge_inclusive',help='if processing charge-inclusive or charge separate samples (still splitting in flavour)',action='store_true',default=False)

    args=parser.parse_args()

    # Partition to store h5 files and samples + path to setup file
    main_proc_dir = f'{args.main_dir}/{args.observable_set}'
    main_plot_dir = f'{args.main_dir}/plots/{args.observable_set}'
  
    for sample_type in args.sample_type:

        logging.debug(f'Main directory: {args.main_dir}; observable set: {args.observable_set}; sample type: {sample_type}; charge-inclusive ? {args.do_charge_inclusive}')

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
        os.makedirs(f'{main_proc_dir}/fisher_info/',exist_ok=True)

        if args.do_charge_inclusive:
            sample_type+='_charge_inclusive'
        
        log_file_path=f'{main_plot_dir}/fisherInfo_histograms_{sample_type}.csv'
        log_file=open(log_file_path,'a')
        if os.path.getsize(log_file_path) == 0:
            log_file.write('observable_x, binning_x, observable_y, binning_y, I_00 (info on cHWtil), 68% CL, 95% CL \n')

        if 'signalOnly' in sample_type and args.do_parton:
            # Complete truth information (from parton-level weights)
            fi_matrix_dataframe,fi_list,fi_cov_list=get_parton_FisherInfo(do_charge_inclusive=args.do_charge_inclusive)    
            log_file.write( f'truth, None, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
            np.savez(f'{main_proc_dir}/fisher_info/fi_truth_{sample_type}.npz', fi_list, allow_pickle=False)
            np.savez(f'{main_proc_dir}/fisher_info/fi_cov_truth_{sample_type}.npz', fi_cov_list, allow_pickle=False)

        if args.do_rate:
            # Rate information
            fi_matrix_dataframe,fi_list,fi_cov_list=get_rate_FisherInfo(do_charge_inclusive=args.do_charge_inclusive)
            log_file.write( f'rate, None, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
            np.savez(f'{main_proc_dir}/fisher_info/fi_rate_{sample_type}.npz', fi_list, allow_pickle=False)
            np.savez(f'{main_proc_dir}/fisher_info/fi_cov_rate_{sample_type}.npz', fi_cov_list, allow_pickle=False)   
        
        if args.observable_x!=None:
            if args.observable_y==None:
                # 1D histogram information
                fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_1D_histograms(observable=args.observable_x,bins_observable=args.binning_x,do_charge_inclusive=args.do_charge_inclusive)
                log_file.write( f'{args.observable_x}, {args.binning_x}, None, None, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')
                np.savez(f'{main_proc_dir}/fisher_info/fi_{args.observable_x}_{len(args.binning_x)}bins_{sample_type}.npz', fi_list, allow_pickle=False)
                np.savez(f'{main_proc_dir}/fisher_info/fi_cov_{args.observable_x}_{len(args.binning_x)}bins_{sample_type}.npz', fi_cov_list, allow_pickle=False)
            else:                
                # 2D histogram information
                fi_matrix_dataframe,fi_list,fi_cov_list=get_FisherInfo_2D_histograms(observable=args.observable_x,bins_observable=args.binning_x,observable2=args.observable_y,bins_observable2=args.binning_y,do_charge_inclusive=args.do_charge_inclusive)
                log_file.write( f'{args.observable_x}, {args.binning_x}, {args.observable_y}, {args.binning_y}, {fi_matrix_dataframe.iloc[0,0]},{1./np.sqrt(fi_matrix_dataframe.iloc[0,0])},{1.69/np.sqrt(fi_matrix_dataframe.iloc[0,0])} \n')   
                np.savez(f'{main_proc_dir}/fisher_info/fi_{args.observable_x}_{len(args.binning_x)}bins_{args.observable_y}_{len(args.observable_y)}bins_{sample_type}.npz', fi_list, allow_pickle=False)
                np.savez(f'{main_proc_dir}/fisher_info/fi_cov_{args.observable_x}_{len(args.binning_x)}bins_{args.observable_y}_{len(args.observable_y)}bins_{sample_type}.npz', fi_cov_list, allow_pickle=False)
        else:
            logging.warning('did not specify any of the possible histogram options, skipping...')


        log_file.close()