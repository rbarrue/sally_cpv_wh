#!/bin/bash

export MAIN_DIR='output/'

# extracting limits with linearized likelihood ratio using Fisher Information and Local Fisher Distance
# parton-level (optimal limits) and rate-only 
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode parton
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode rate

# energy-dependent observables
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 150. 250. 1e9
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 1e9

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9

# angular observables
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 0. 1.0
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.5 0. 0.5 1.0

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 0. 1.0
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.5 0. 0.5 1.0

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0
python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9

python compute_FisherInfo_histograms.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9

# SALLY
python compute_FisherInfo_sally.py --main_dir ${MAIN_DIR} -so kinematic_only -sm one_layer_50_neurons_50_epochs -i
python compute_FisherInfo_sally.py --main_dir ${MAIN_DIR} -so all_observables -sm one_layer_50_neurons_50_epochs -i
python compute_FisherInfo_sally.py --main_dir ${MAIN_DIR} -so all_observables_remove_qlCosDeltaMinus -sm one_layer_50_neurons_50_epochs -i