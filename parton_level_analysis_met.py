# -*- coding: utf-8 -*-

"""
parton_level_analysis.py

Parton-level analysis of signal and background events using as observables only "physical" degrees of freedom (MET, mTW, q_l, etc.)

Includes a set of transfer functions (TF) to approximate detector response

Includes the calculation of the neutrino pZ in the same way as done in the ATLAS VH(bb) analyses

Includes the calculation of a set of angular observables.

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from math import sqrt, fabs

import os

import argparse as ap

from madminer.lhe import LHEReader
from madminer.utils.particle import MadMinerParticle
import vector

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
        
# "MET" set of observables
# Elements in the lists will be given as input to the add_observable function
observable_names = [
    'b1_px', 'b1_py', 'b1_pz', 'b1_e',
    'b2_px', 'b2_py', 'b2_pz', 'b2_e',
    'l_px', 'l_py', 'l_pz', 'l_e',
    'v_px', 'v_py',
    'pt_b1', 'pt_b2', 'pt_l', 'met', 'pt_w', 'pt_h',
    'eta_b1', 'eta_b2', 'eta_l', 'eta_h',
    'phi_b1', 'phi_b2', 'phi_l', 'phi_v', 'phi_w', 'phi_h',
    'theta_b1', 'theta_b2', 'theta_l', 'theta_h',
    'dphi_bb', 'dphi_lv', 'dphi_wh',
    'm_bb', 'mt_lv', 'mt_tot',
    'q_l',
    'dphi_lb1', 'dphi_lb2', 'dphi_vb1', 'dphi_vb2',
    'dR_bb', 'dR_lb1', 'dR_lb2'
]

list_of_observables = [
    'j[0].px', 'j[0].py', 'j[0].pz', 'j[0].e',
    'j[1].px', 'j[1].py', 'j[1].pz', 'j[1].e',
    'l[0].px', 'l[0].py', 'l[0].pz', 'l[0].e',
    'met.px', 'met.py',   
    'j[0].pt', 'j[1].pt', 'l[0].pt', 'met.pt', '(l[0] + met).pt', '(j[0] + j[1]).pt',
    'j[0].eta', 'j[1].eta', 'l[0].eta', '(j[0] + j[1]).eta',
    'j[0].phi', 'j[1].phi', 'l[0].phi', 'met.phi', '(l[0] + met).phi', '(j[0] + j[1]).phi',
    'j[0].theta', 'j[1].theta', 'l[0].theta', '(j[0] + j[1]).theta',
    'j[0].deltaphi(j[1])', 'l[0].deltaphi(met)', '(l[0] + met).deltaphi(j[0] + j[1])',
    '(j[0] + j[1]).m', '(l[0] + met).mt', '(j[0] + j[1] + l[0] + met).mt',
    'l[0].charge',
    'l[0].deltaphi(j[0])', 'l[0].deltaphi(j[1])', 'met.deltaphi(j[0])', 'met.deltaphi(j[1])',
    'j[0].deltaR(j[1])', 'l[0].deltaR(j[0])', 'l[0].deltaR(j[1])'
]

# solves quadratic equation to get two possible solutions for pz_nu
# calculation similar to what is done in the ATLAS VHbb analyses (getNeutrinoPz function in AnalysisReader.cxx)
def get_neutrino_pz(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  if (met is None) or leptons==[] or jets==[]:
      raise TypeError("one of the required inputs of wrong type (not found)")
  
  # W boson mass (GeV)
  m_w=80.379
  
  pz_nu_1=0.0
  pz_nu_2=0.0

  if debug:
    logging.debug(f'leading lepton 4-vector: {leptons[0]}')
    logging.debug(f'leading jet 4-vector: {jets[0]}')
    logging.debug(f'subleading jet 4-vector: {jets[1]}')
    logging.debug(f'met 4-vector: {met}')

  # auxiliary quantities
  mu = 0.5 * pow(m_w,2) + (leptons[0].px*met.px + leptons[0].py*met.py) # = tmp/2
  delta = pow(mu*leptons[0].pz / pow(leptons[0].pt,2) ,2) - (pow(leptons[0].e,2)*pow(met.pt,2) - pow(mu,2))/pow(leptons[0].pt,2) 

  # neglecting any imaginary parts (equivalent to setting mtw = mw)
  if delta < 0.0:
    delta=0.0
  
  pz_nu_1=mu*leptons[0].pz/pow(leptons[0].pt,2) + sqrt(delta)
  pz_nu_2=mu*leptons[0].pz/pow(leptons[0].pt,2) - sqrt(delta)

  # NOTE: MadMinerParticle inherits from Vector class in scikit-hep)
  nu_1_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu_1,m=0.0)
  nu_2_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu_2,m=0.0)

  nu_1=MadMinerParticle(nu_1_vec.azimuthal,nu_1_vec.longitudinal,nu_1_vec.temporal)
  nu_2=MadMinerParticle(nu_2_vec.azimuthal,nu_2_vec.longitudinal,nu_2_vec.temporal)

  if debug:
    logging.debug(f'nu_1 4-vector: {nu_1}; nu_2 4-vector: {nu_2}')

  h_candidate=jets[0]+jets[1]

  w_candidate_1=leptons[0]+nu_1
  w_candidate_2=leptons[0]+nu_2

  # correct definition of betaZ, but neutrino pZ reconstruction studies in NLO SM sample showed that this is not optimal
  """
  h_candidate_betaZ=h_candidate.to_beta3().z

  w_candidate_1_betaZ=w_candidate_1.to_beta3().z
  w_candidate_2_betaZ=w_candidate_2.to_beta3().z
  """
  h_candidate_betaZ=h_candidate.pz/np.sqrt(pow(h_candidate.pz,2)+h_candidate.M2) 
  w_candidate_1_betaZ=w_candidate_1.pz/np.sqrt(pow(w_candidate_1.pz,2)+w_candidate_1.M2) 
  w_candidate_2_betaZ=w_candidate_2.pz/np.sqrt(pow(w_candidate_2.pz,2)+w_candidate_2.M2) 

  
  if debug:
    logging.debug(f'h_candidate 4-vector: {h_candidate}')
    logging.debug(f'w_candidate_1 4-vector: {w_candidate_1}; w_candidate_2 4-vector: {w_candidate_2}')

    logging.debug(f'h_candidate_BetaZ: {h_candidate_betaZ}')

    logging.debug(f'w_candidate_1_BetaZ: {w_candidate_1_betaZ}')
    logging.debug(f'w_candidate_2_BetaZ: {w_candidate_2_betaZ}')
  

  dBetaZ_1=fabs(w_candidate_1_betaZ - h_candidate_betaZ)
  dBetaZ_2=fabs(w_candidate_2_betaZ - h_candidate_betaZ)

  if debug:
    logging.debug(f'dBetaZ_1: {dBetaZ_1}; dBetaZ_2: {dBetaZ_2}')

  if dBetaZ_1 <= dBetaZ_2:
    if debug:
      logging.debug(f'pz_nu: {pz_nu_1}')
    return pz_nu_1
  else:
    if debug:
      print(f'pz_nu: {pz_nu_2}')
    return pz_nu_2

def get_cos_thetaStar(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu
  # equivalent to boostCM_of_p4(w_candidate), which hasn't been implemented yet for MomentumObject4D
  # negating spatial part only to boost *into* the CM, default would be to boost *away* from the CM
  lead_lepton_w_centerOfMass=leptons[0].boost_beta3(-w_candidate.to_beta3())

  if debug:

    logging.debug(f'W candidate 4-vector: {w_candidate}')
    w_candidate_w_centerOfMass=w_candidate.boost_beta3(-w_candidate.to_beta3())
    print(f'W candidate 4-vector boosted to the CM of the W candidate (expect[0.0,0.0,0.0,m_w]): {w_candidate_w_centerOfMass}')
    print(f'leading lepton 4-vector boosted to the CM of the W candidate: {lead_lepton_w_centerOfMass}')  
  
  cos_thetaStar=lead_lepton_w_centerOfMass.to_Vector3D().dot(w_candidate.to_Vector3D())/(fabs(lead_lepton_w_centerOfMass.p) * fabs(w_candidate.p))

  if debug:
    print(f'cos_thetaStar= {cos_thetaStar}')

  return cos_thetaStar

def get_ql_cos_thetaStar(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  cos_thetaStar=get_cos_thetaStar(leptons=leptons,jets=jets,met=met)

  if debug:
    logging.debug(f'ql_cos_thetaStar = {leptons[0].charge * cos_thetaStar}')
  return leptons[0].charge * cos_thetaStar

def get_cos_deltaPlus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):
  
  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu
  # equivalent to boostCM_of_p4(w_candidate), which hasn't been implemented yet for MomentumObject4D
  # negating spatial part only to boost *into* the CM, default would be to boost *away* from the CM
  lead_lepton_w_centerOfMass=leptons[0].boost_beta3(-w_candidate.to_beta3())

  if debug:
    logging.debug(f'W candidate 4-vector: {w_candidate}')
    w_candidate_w_centerOfMass=w_candidate.boost_beta3(-w_candidate.to_beta3())
    logging.debug(f'W candidate 4-vector boosted to the CM of the W candidate (expect[0.0,0.0,0.0,m_w]): {w_candidate_w_centerOfMass}')
    logging.debug(f'leading lepton 4-vector boosted to the CM of the W candidate: {lead_lepton_w_centerOfMass}')
  
  h_candidate=jets[0]+jets[1]

  h_cross_w=h_candidate.to_Vector3D().cross(w_candidate.to_Vector3D())

  cos_deltaPlus=lead_lepton_w_centerOfMass.to_Vector3D().dot(h_cross_w)/(fabs(lead_lepton_w_centerOfMass.p) * fabs(h_cross_w.mag))

  if debug:
    logging.debug(f'cos_deltaPlus = {cos_deltaPlus}')

  return cos_deltaPlus

def get_ql_cos_deltaPlus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  cos_deltaPlus=get_cos_deltaPlus(leptons=leptons,jets=jets,met=met)
  if debug:
    logging.debug(f'ql_cos_deltaPlus = {leptons[0].charge * cos_deltaPlus}')
  return leptons[0].charge * cos_deltaPlus

def get_cos_deltaMinus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu

  h_candidate=jets[0]+jets[1]

  if debug:

    logging.debug(f'H candidate 4-vector: {h_candidate}')
    h_candidate_h_centerOfMass=h_candidate.boost_beta3(-h_candidate.to_beta3())
    logging.debug(f'H candidate 4-vector boosted to the CM of the H candidate (expect[0.0,0.0,0.0,m_H]): {h_candidate_h_centerOfMass}')
    
  lead_lepton_inv_h_centerOfMass=leptons[0].boost_beta3(h_candidate.to_beta3())
  nu_inv_h_centerOfMass=nu.boost_beta3(h_candidate.to_beta3())

  if debug:  
    logging.debug(f'leading lepton 4-vector boosted to the CM of the H candidate with an inverted momentum: {lead_lepton_inv_h_centerOfMass}')
    logging.debug(f'neutrino 4-vector boosted to the CM of the H candidate with an inverted momentum: {nu_inv_h_centerOfMass}')

  lep_cross_nu_inv_h_centerOfMass=lead_lepton_inv_h_centerOfMass.to_Vector3D().cross(nu_inv_h_centerOfMass.to_Vector3D())

  cos_deltaMinus=lep_cross_nu_inv_h_centerOfMass.dot(w_candidate.to_Vector3D())/(fabs(lep_cross_nu_inv_h_centerOfMass.mag)*fabs(w_candidate.p))

  if debug:
    logging.debug(f'cos_deltaMinus = {cos_deltaMinus}')

  return cos_deltaMinus

def get_ql_cos_deltaMinus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):
  cos_deltaMinus=get_cos_deltaMinus(leptons=leptons,jets=jets,met=met)
  if debug:
    logging.debug(f'ql_cos_deltaMinus = {leptons[0].charge * cos_deltaMinus}')
  return leptons[0].charge * cos_deltaMinus

def get_deltaphi_lv(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):
  
  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu
  lead_lepton_w_centerOfMass=leptons[0].boost_beta3(-w_candidate.to_beta3())

  if debug:

    logging.debug(f'W candidate 4-vector: {w_candidate}')
    w_candidate_w_centerOfMass=w_candidate.boost_beta3(-w_candidate.to_beta3())
    logging.debug(f'W candidate 4-vector boosted to the CM of the W candidate (expect[0.0,0.0,0.0,m_w]): {w_candidate_w_centerOfMass}')
    logging.debug(f'leading lepton 4-vector boosted to the CM of the W candidate: {lead_lepton_w_centerOfMass}')

  deltaphi_lv=lead_lepton_w_centerOfMass.deltaphi(w_candidate)

  if debug:
    logging.debug(f'deltaphi_lv = {deltaphi_lv}')

  return deltaphi_lv

# Function to process the events, run on each separate data sample
def process_events(event_path, setup_file_path, output_file_path, is_background_process=False, is_SM=True):

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

    # Smear MET using a (absolute) Gaussian TF with sigmaMET = 12.5 GeV
    # Approximates the Run 2 ATLAS MET performance paper (1802.08168)
    lhe.set_met_noise(abs_=12.5, rel=0.0)

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
    # "MET" observables    
    for i, name in enumerate(observable_names):
        lhe.add_observable( name, list_of_observables[i], required=True )
    
    lhe.add_observable_from_function('pz_nu',get_neutrino_pz,required=True)
    lhe.add_observable_from_function('cos_thetaStar',get_cos_thetaStar,required=True)
    lhe.add_observable_from_function('cos_deltaPlus',get_cos_deltaPlus,required=True)
    lhe.add_observable_from_function('cos_deltaMinus',get_cos_deltaMinus,required=True)
    lhe.add_observable_from_function('deltaphi_lv',get_deltaphi_lv,required=True)
    lhe.add_observable_from_function('ql_cos_thetaStar',get_ql_cos_thetaStar,required=True)
    lhe.add_observable_from_function('ql_cos_deltaPlus',get_ql_cos_deltaPlus,required=True)
    lhe.add_observable_from_function('ql_cos_deltaMinus',get_ql_cos_deltaMinus,required=True)

    # Emulates double b-tagging efficiency (flat 70% b-tagging probability for each)
    lhe.add_efficiency('0.7')
    lhe.add_efficiency('0.7')

    # Analyse samples and save the processed events as an .h5 file for later use
    lhe.analyse_samples()

    lhe.save(output_file_path,shuffle=False)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Parton-level analysis of signal and background events using as observables only "physical" degrees of freedom (MET, mTW, q_l, etc.). Includes also the computation of the pZ of the neutrino and several angular observables',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--setup_file',help='name of setup file (without the .h5)',required=True)

    parser.add_argument('--do_signal',help='analyze signal samples', action='store_true',default=False)

    parser.add_argument('--do_backgrounds',help='analyze background samples', action='store_true',default=False)

    parser.add_argument('--do_BSM',help='analyze samples generated at the BSM benchmarks', action='store_true',default=False)

    args=parser.parse_args()

    if not (args.do_signal or args.do_backgrounds):
        logging.warning('not asking for either signal or background !')

    output_dir=f'{args.main_dir}/met/'

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