#!/bin/bash

MAIN_DIR='output/'
    
# extracting limits with the full likelihood ratio and the asymptotic formalism (including rate term)
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 150. 250. 1e9 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 1e9 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9  
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9  
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 0. 1.0   
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.5 0. 0.5 1.0  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0  
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 0. 1.0   
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.5 0. 0.5 1.0  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0  
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9  

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so kinematic_only -sm one_layer_50_neurons_50_epochs # no binning = 25 bins

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so kinematic_only -sm one_layer_50_neurons_50_epochs  --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables -sm one_layer_50_neurons_50_epochs

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables -sm one_layer_50_neurons_50_epochs  --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables_remove_qlCosDeltaMinus -sm one_layer_50_neurons_50_epochs

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables_remove_qlCosDeltaMinus -sm one_layer_50_neurons_50_epochs  --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0

# extracting limits with the full likelihood ratio and the asymptotic formalism (excluding rate term, i.e. shape-only)
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 150. 250. 1e9 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x pt_w --binning_x 0. 75. 150. 250. 400. 600. 1e9 --observable_y mt_tot --binning_y 0. 400. 800. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 0. 1.0 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.5 0. 0.5 1.0 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaPlus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 0. 1.0 --shape  
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.5 0. 0.5 1.0 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --shape 
python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y pt_w --binning_y 0. 75. 150. 250. 400. 600. 1e9  --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode histo --observable_x ql_cos_deltaMinus --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --observable_y mt_tot --binning_y 0. 400. 800. 1e9 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so kinematic_only -sm one_layer_50_neurons_50_epochs --shape # no binning = 25 bins

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so kinematic_only -sm one_layer_50_neurons_50_epochs --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables -sm one_layer_50_neurons_50_epochs --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables -sm one_layer_50_neurons_50_epochs --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables_remove_qlCosDeltaMinus -sm one_layer_50_neurons_50_epochs --shape 

python compute_full_limits.py --main_dir ${MAIN_DIR} --mode sally -so all_observables_remove_qlCosDeltaMinus -sm one_layer_50_neurons_50_epochs --binning_x -1.0 -0.666666 -0.333333 0. 0.333333 0.666666 1.0 --shape 
