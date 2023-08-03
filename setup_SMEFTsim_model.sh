#!/bin/bash

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