#!/bin/bash

MAIN_DIR='output/'

# plotting the distributions of the relevant kinematic and angular observables
python compute_distributions.py -dir ${MAIN_DIR} -o pt_w mt_tot pz_nu cos_deltaPlus cos_deltaMinus ql_cos_deltaPlus ql_cos_deltaMinus -s signalOnly -shape 

# plotting the SALLY distributions (inc. losses)
python compute_distributions.py -dir ${MAIN_DIR} -c wh -sally -so kinematic_only -sm one_layer_50_neurons_50_epochs -s signalOnly -shape -st withBackgrounds -l