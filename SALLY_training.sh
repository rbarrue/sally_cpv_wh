#!/bin/bash

MAIN_DIR='output/'

# creating the training dataset, augmented with the joint score information
python sally_training.py --main_dir ${MAIN_DIR} --mode augment_only -c wh -s withBackgrounds

# training the different models
python sally_training.py --main_dir ${MAIN_DIR} --mode train_only -c wh -s withBackgrounds -o kinematic_only -n one_layer_50_neurons_50_epochs

python sally_training.py --main_dir ${MAIN_DIR} --mode train_only -c wh -s withBackgrounds -o all_observables -n one_layer_50_neurons_50_epochs

python sally_training.py --main_dir ${MAIN_DIR} --mode train_only -c wh -s withBackgrounds -o all_observables_remove_qlCosDeltaMinus -n one_layer_50_neurons_50_epochs