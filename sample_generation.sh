#!/bin/bash

# file to run all the steps in the generation

export DELPHES_DIR='/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes/'
export MADGRAPH_DIR='/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/'
export MAIN_DIR='output/'

# run setup
python setup.py --main_dir ${MAIN_DIR}

# setting up the SMEFTsim model
if [ -d "./models/SMEFTsim_U35_MwScheme_UFO/" ]; then
    echo "SMEFTsim model found, setting up the environment variable"
    export SMEFTSIM_FOLDER=${PWD}/models/
else
    echo "SMEFTsim model not found, downloading it and then setting up the environment variable"
    wget http://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/SMEFT/SMEFTsim_U35_MwScheme_UFO.tar.gz
    tar -xzvf SMEFTsim_U35_MwScheme_UFO.tar.gz
    rm -r SMEFTsim_U35_MwScheme_UFO.tar.gz
    mkdir -p models
    mv -n SMEFTsim_U35_MwScheme_UFO models/
    cp cards/restrict_all_WHbb_massless.dat models/SMEFTsim_U35_MwScheme_UFO/
    export SMEFTSIM_FOLDER=${PWD}/models/
fi

# generate signal and background samples 

# MadMiner allows to just prepare the Madgraph scripts for each sample,
# which can later be run in parallel, speeding up event generation a lot
# see --prepare_scripts argument

python gen_signal.py --main_dir ${MAIN_DIR} --mg_dir ${MADGRAPH_DIR} --nevents 30000000

python gen_background.py --main_dir ${MAIN_DIR} --mg_dir ${MADGRAPH_DIR} --nevents 10000000

# run Delphes and analysis code, taking as input an individual sample run
# to allow parallel execution since this step takes longer than event generation
for SAMPLE_DIR in output/signal_samples/*/Events/*; do
    echo "$SAMPLE_DIR"  
    python delphes_analysis.py --main_dir ${MAIN_DIR} --delphes_dir ${DELPHES_DIR} --sample_dir ${SAMPLE_DIR}
done

for SAMPLE_DIR in output/background_samples/*/Events/*; do
    echo "$SAMPLE_DIR"
    python delphes_analysis.py --main_dir ${MAIN_DIR} --delphes_dir ${DELPHES_DIR} --sample_dir ${SAMPLE_DIR}  
done

# combining the individual samples for training ML methods
python analysis_sample_combiner.py --main_dir ${MAIN_DIR}

############### AUXILIARY TASKS ##################

# generate samples to derive cutflow
# python gen_cutflow.py --main_dir ${MAIN_DIR} --mg_dir ${MADGRAPH_DIR}

# separating signal sample generation and reweighting
# since reweighting cannot run in multicore mode (https://answers.launchpad.net/mg5amcnlo/+question/691641)
# this option is DISCOURAGED, since it led to some transient errors and generation is not the main time bottleneck

# python gen_signal.py --main_dir ${MAIN_DIR} --mg_dir ${MADGRAPH_DIR} --nevents 30000000 --skip_reweighting
# python rw_signal.py --main_dir ${MAIN_DIR}

