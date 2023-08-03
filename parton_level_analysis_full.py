# -*- coding: utf-8 -*-

"""
parton_level_analysis_full.py

Parton-level analysis of signal and background events with a complete set of observables (including leading neutrino 4-vector).

Includes a set of transfer functions (TF) to approximate detector response.

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

# "Full" set of observables
# Elements in the lists will be given as input to the add_observable function
observable_names = [
    'b1_px', 'b1_py', 'b1_pz', 'b1_e',
    'b2_px', 'b2_py', 'b2_pz', 'b2_e',
    'l_px', 'l_py', 'l_pz', 'l_e',
    'v_px', 'v_py', 'v_pz', 'v_e',
    'pt_b1', 'pt_b2', 'pt_l1', 'pt_l2', 'pt_w', 'pt_h',
    'eta_b1', 'eta_b2', 'eta_l', 'eta_v', 'eta_w', 'eta_h',
    'phi_b1', 'phi_b2', 'phi_l', 'phi_v', 'phi_w', 'phi_h',
    'theta_b1', 'theta_b2', 'theta_l', 'theta_v', 'theta_w', 'theta_h',
    'dphi_bb', 'dphi_lv', 'dphi_wh',
    'm_bb', 'm_lv', 'm_tot',
    'q_l', 'q_v', 'q_b1', 'q_b2',
    'dphi_lb1', 'dphi_lb2', 'dphi_vb1', 'dphi_vb2',
    'dR_bb', 'dR_lv', 'dR_lb1', 'dR_lb2', 'dR_vb1', 'dR_vb2'
]

list_of_observables = [
    'j[0].px', 'j[0].py', 'j[0].pz', 'j[0].e',
    'j[1].px', 'j[1].py', 'j[1].pz', 'j[1].e',
    'l[0].px', 'l[0].py', 'l[0].pz', 'l[0].e',
    'v[0].px', 'v[0].py', 'v[0].pz', 'v[0].e',
    'j[0].pt', 'j[1].pt', 'l[0].pt', 'v[0].pt', '(l[0] + v[0]).pt', '(j[0] + j[1]).pt',
    'j[0].eta', 'j[1].eta', 'l[0].eta', 'v[0].eta', '(l[0] + v[0]).eta', '(j[0] + j[1]).eta',
    'j[0].phi', 'j[1].phi', 'l[0].phi', 'v[0].phi', '(l[0] + v[0]).phi', '(j[0] + j[1]).phi',
    'j[0].theta', 'j[1].theta', 'l[0].theta', 'v[0].theta', '(l[0] + v[0]).theta', '(j[0] + j[1]).theta',
    'j[0].deltaphi(j[1])', 'l[0].deltaphi(v[0])', '(l[0] + v[0]).deltaphi(j[0] + j[1])',
    '(j[0] + j[1]).m', '(l[0] + v[0]).m', '(j[0] + j[1] + l[0] + v[0]).m',
    'l[0].charge', 'v[0].charge', 'j[0].charge', 'j[1].charge',
    'l[0].deltaphi(j[0])', 'l[0].deltaphi(j[1])', 'v[0].deltaphi(j[0])', 'v[0].deltaphi(j[1])',
    'j[0].deltaR(j[1])', 'l[0].deltaR(v[0])', 'l[0].deltaR(j[0])', 'l[0].deltaR(j[1])', 'v[0].deltaR(j[0])', 'v[0].deltaR(j[1])',
]

# Function to process the events, run on each separate data sample
def process_events(event_path, setup_file_path,output_file_path,is_background_process=False,is_SM=True):

    # Load Madminer setup
    lhe = LHEReader(setup_file_path)

    # Smear b-quark energies using a (relative) Gaussian TF with width sigmaE/E = 0.1
    # Approximates the mBB distribution in CMS H(->bb) observation paper (1808.08242)
    lhe.set_smearing(
        pdgids=[5,-5],
        energy_resolution_abs=0,
        energy_resolution_rel=0.1,
        pt_resolution_abs=None, # can use this since I used massive b-quarks
        pt_resolution_rel=None  # calculating pt resolutions from an on-shell condition
    )

    # Smear the neutrino momenta using a (absolute) Gaussian TF with sigmaMET = 12.5 GeV
    # Approximates the Run 2 ATLAS MET performance paper (1802.08168)
    lhe.set_smearing(
        pdgids=[12,-12,14,-14,16,-16],
        energy_resolution_abs=12.5,
        energy_resolution_rel=0.0,
        pt_resolution_abs=None,
        pt_resolution_rel=None
    )

    # Add events
    if(is_SM):
        lhe.add_sample(
            f'{event_path}/Events/run_01/unweighted_events.lhe.gz',
            sampled_from_benchmark='sm',
            is_background=is_background_process
        )
    else:
        list_BSM_benchmarks = [x for x in lhe.benchmark_names_phys if x != 'sm']
        for i,benchmark in enumerate(list_BSM_benchmarks,start=1):
            run_str = str(i)
            if len(run_str) < 2:
                run_str = '0' + run_str
            lhe.add_sample(
                f'{event_path}/Events/run_{run_str}/unweighted_events.lhe.gz',
                sampled_from_benchmark=benchmark,
                is_background=is_background_process
            )

    # Adding observables
    for i, name in enumerate(observable_names):
        lhe.add_observable( name, list_of_observables[i], required=True )

    # Emulates double b-tagging efficiency (flat 70% b-tagging probability for each)
    lhe.add_efficiency('0.7')
    lhe.add_efficiency('0.7')

    # Analyse samples and save the processed events as an .h5 file for later use
    lhe.analyse_samples()

    lhe.save(output_file_path)

if __name__ == "__main__":
  
  parser = ap.ArgumentParser(description='Parton-level analysis of signal and background events with a complete set of observables (including leading neutrino 4-vector).',formatter_class=ap.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

  parser.add_argument('--setup_file',help='name of setup file (without the .h5)',required=True)

  parser.add_argument('--do_signal',help='analyze signal events', action='store_true',default=False)

  parser.add_argument('--do_backgrounds',help='analyze background events', action='store_true',default=False)

  parser.add_argument('--do_BSM',help='analyze samples generated at the BSM benchmarks', action='store_true',default=False)

  args=parser.parse_args()
  
  if not (args.do_signal or args.do_backgrounds):
    logging.warning('not asking for either signal or background !')

  output_dir=f'{args.main_dir}/full'

  ############## Signal (W H -> l v b b ) ###############
  
  if args.do_signal:

    os.makedirs(f'{output_dir}/signal/',exist_ok=True)
    
    # W+ -> mu+ vm
    process_events(
        event_path=f'{args.main_dir}/signal_samples/wph_mu_smeftsim_SM',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=False,
        is_SM=True,
        output_file_path=f'{output_dir}/signal/wph_mu_smeftsim_SM_lhe.h5',
    )
    if args.do_BSM:
        process_events(
            event_path=f'{args.main_dir}/signal_samples/wph_mu_smeftsim_BSM',
            setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
            is_background_process=False,
            is_SM=False,
            output_file_path=f'{output_dir}/signal/wph_mu_smeftsim_BSM_lhe.h5',
        )
    
    # W+ -> e+ ve
    process_events(
        event_path=f'{args.main_dir}/signal_samples/wph_e_smeftsim_SM',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=False,
        is_SM=True,
        output_file_path=f'{output_dir}/signal/wph_e_smeftsim_SM_lhe.h5',
    )

    if args.do_BSM:
        process_events(
            event_path=f'{args.main_dir}/signal_samples/wph_e_smeftsim_BSM',
            setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
            is_background_process=False,
            is_SM=False,
            output_file_path=f'{output_dir}/signal/wph_e_smeftsim_BSM_lhe.h5',
        )

    # W- -> mu- vm~
    process_events(
        event_path=f'{args.main_dir}/signal_samples/wmh_mu_smeftsim_SM',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=False,
        is_SM=True,
        output_file_path=f'{output_dir}/signal/wmh_mu_smeftsim_SM_lhe.h5',
    )

    if args.do_BSM:
        process_events(
            event_path=f'{args.main_dir}/signal_samples/wmh_mu_smeftsim_BSM',
            setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
            is_background_process=False,
            is_SM=False,
            output_file_path=f'{output_dir}/signal/wmh_mu_smeftsim_BSM_lhe.h5',
        )

    # W- -> e- ve~
    process_events(
        event_path=f'{args.main_dir}/signal_samples/wmh_e_smeftsim_SM',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=False,
        is_SM=True,
        output_file_path=f'{output_dir}/signal/wmh_e_smeftsim_SM_lhe.h5',
    )

    if args.do_BSM:
        process_events(
            event_path=f'{args.main_dir}/signal_samples/wmh_e_smeftsim_BSM',
            setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
            is_background_process=False,
            is_SM=False,
            output_file_path=f'{output_dir}/signal/wmh_e_smeftsim_BSM_lhe.h5',
        )

  ############## Backgrounds ###############
  if args.do_backgrounds:

    os.makedirs(f'{output_dir}/background/',exist_ok=True)

    ############## Single top tb -> W b b -> l v b b ###############

    # W+ -> mu+ vm
    process_events(
        event_path=f'{args.main_dir}/background_samples/tpb_mu_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tpb_mu_background_lhe.h5',
    )

    # W+ -> e+ ve
    process_events(
        event_path=f'{args.main_dir}/background_samples/tpb_e_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tpb_e_background_lhe.h5',
    )

    # W- -> mu- vm~
    process_events(
        event_path=f'{args.main_dir}/background_samples/tmb_mu_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tmb_mu_background_lhe.h5',
    )

    # W- -> e- ve~
    process_events(
        event_path=f'{args.main_dir}/background_samples/tmb_e_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tmb_e_background_lhe.h5',
    )

    ############## ttbar -> W b W b -> l v j j b b ###############

    # W+ -> mu+ vm
    process_events(
        event_path=f'{args.main_dir}/background_samples/tt_mupjj_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tt_mupjj_background_lhe.h5',
    )

    # W+ -> e+ ve
    process_events(
        event_path=f'{args.main_dir}/background_samples/tt_epjj_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tt_epjj_background_lhe.h5',
    )

    # W- -> mu- vm~
    process_events(
        event_path=f'{args.main_dir}/background_samples/tt_mumjj_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tt_mumjj_background_lhe.h5',
    )

    # W- -> e- ve~
    process_events(
        event_path=f'{args.main_dir}/background_samples/tt_emjj_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/tt_emjj_background_lhe.h5',
    )

    ############## W + b-jets ###############

    # W+ -> mu+ vm
    process_events(
        event_path=f'{args.main_dir}/background_samples/wpbb_mu_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/wpbb_mu_background_lhe.h5',
    )

    # W+ -> e+ ve
    process_events(
        event_path=f'{args.main_dir}/background_samples/wpbb_e_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/wpbb_e_background_lhe.h5',
    )

    # W- -> mu- vm~
    process_events(
        event_path=f'{args.main_dir}/background_samples/wmbb_mu_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/wmbb_mu_background_lhe.h5',
    )

    # W- -> e- ve~
    process_events(
        event_path=f'{args.main_dir}/background_samples/wmbb_e_background',
        setup_file_path=f'{args.main_dir}/{args.setup_file}.h5',
        is_background_process=True,
        is_SM=True,
        output_file_path=f'{output_dir}/background/wmbb_e_background_lhe.h5',
    )