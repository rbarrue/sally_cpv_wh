from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from madminer.delphes import DelphesReader
import argparse as ap
import math

import parton_level_analysis_met as parton

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

# redefining here since for Delphes files, the particles input variable (existent in LHE files) is not used anymore
def get_neutrino_pz(leptons=[],photons=[],jets=[],met=None,debug=False):
    
    return parton.get_neutrino_pz([],leptons,photons,jets,met,debug)

def get_cos_deltaPlus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_cos_deltaPlus([],leptons,photons,jets,met,debug)

def get_ql_cos_deltaPlus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_ql_cos_deltaPlus([],leptons,photons,jets,met,debug)

def process_events(event_path, setup_file_path, is_background_process=False,k_factor=1.0):
    
    reader=DelphesReader(setup_file_path)
    
    logging.info(f'event_path: {event_path}, is_background: {is_background_process}')

    reader.add_sample(hepmc_filename=f'{event_path}/tag_1_pythia8_events.hepmc',
                    sampled_from_benchmark='sm',
                    is_background=is_background_process,
                    lhe_filename=f'{event_path}/unweighted_events.lhe.gz',
                    delphes_filename=f'{event_path}/delphes_events.root',
                    k_factor=k_factor,
                    weights='lhe')

    # this will have to be changed if I am to change 
    for i, name in enumerate(parton.observable_names):
        reader.add_observable( name, parton.list_of_observables[i], required=True )

    reader.add_observable_from_function('pz_nu', get_neutrino_pz,required=True)
    reader.add_observable_from_function('cos_deltaPlus', get_cos_deltaPlus,required=True)
    reader.add_observable_from_function('ql_cos_deltaPlus', get_ql_cos_deltaPlus,required=True)

    # requiring the two leading jets to be b-tagged
    reader.add_cut('j[0].b_tag',required=True)
    reader.add_cut('j[1].b_tag',required=True)

    reader.analyse_delphes_samples()

    reader.save(f'{event_path}/analysed_events.h5')


if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Detector-level analysis of signal and background events (with Delphes). Includes the computation of the pZ of the neutrino and several angular observables',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--sample_dir',help='folder where the individual sample is',required=True)

    parser.add_argument('--debug',help='output debug information',default=False,action="store_true")

    args=parser.parse_args()

    if args.debug:
        logging.getLogger("madminer").setLevel(logging.DEBUG)

    if 'background' in args.sample_dir:
        process_events(f'{args.sample_dir}',f'{args.main_dir}/setup.h5',is_background_process=True)
    else:
        process_events(f'{args.sample_dir}',f'{args.main_dir}/setup.h5',is_background_process=False)
    