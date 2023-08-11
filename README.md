# Simulation-based inference in the search for CP violation in leptonic WH production

This is the companion code to https://arxiv.org/abs/2308.02882, for benchmarking different techniques to search for CP violation in the HWW vertex in leptonic WH production.

Compares energy-related observables, angular observables and combinations thereof with a simulation-based inference method, SALLY (Score Approximates Likelihood LocallY), that aims to reconstruct a detector-level optimal observable around the SM point.

Compares exclusion limits obtained with linearized and full likelihood ratios.

Work uses version 0.9.3 of MadMiner (https://github.com/madminer-tool/madminer), installed simply by running `pip install -r requirements.txt`

Requires version >= 3.3.1 of Madgraph if generating samples.

Everytime you want to generate samples, either locally or in a batch system, you should set up the download and/or linkage to the SMEFTsim model and restrict card used, by running the following command:
> source setup_SMEFTsim_model.sh

To run the analysis chain the programs must be ran in a certain order:

1. _setup.py_: defines the Wilson coefficient and morphing setup and creates setup file.

2. _gen\_signal/background.py_: generate signal and background samples or prepare files to run sample generation on batch systems

3. _parton\_level\_analysis\_full.py_: performs analysis (simulating detector response via smearing of parton-level quantities) for the observable set with the full neutrino 4-vector ('full')

4. _parton\_level\_analysis\_met.py_: performs analysis (simulating detector response via smearing of parton-level quantities) for the observable set ('met') with observable degrees of freedom, inc. neutrino pZ and CP-sensitive angular observable reconstruction

5. _analysis\_sample\_combiner.py_: combines the analysed samples

6. _sally\_training.py_: train a multivariate method based on the score (SALLY - other methods may be added in the future)

7. _compute\_FisherInfo\_sally.py_: compute the Fisher information matrices and derive linearized limits from the SALLY method 

8. _compute\_FisherInfo\_histograms.py_: compute the Fisher information matrices and derive linearized limits from the complete truth-level information, the rate or 1D/2D histograms 

9. _compute\_full\_limits.py_: compute limits with the full likelihood ratio (in the asymptotic limit) from 1D/2D histograms or SALLY

Steps 3 and 4 can be run simultaneously. Steps 5 has to run before step 6, but after that, step 6, 7, 8 and 9 can be run simultaneously.

All of the scripts have an argument parser describing their API, just type _script_name_ -h to see the list of available options.
