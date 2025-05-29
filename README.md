# Simulation-based inference in the search for CP violation in leptonic WH production

This is the companion code to "Simulation-based inference in the search for CP violation in leptonic WH production" (https://link.springer.com/article/10.1007/JHEP04(2024)014). In this paper, we benchmarked different techniques to search for CP violation in the HWW vertex in leptonic WH production.

Compares energy-related observables, angular observables and combinations thereof with a simulation-based inference method, SALLY (Score Approximates Likelihood LocallY), that aims to reconstruct a detector-level optimal observable around the SM point.

Compares exclusion limits obtained with linearized and full likelihood ratios (without and with rate information).

Everytime you want to generate samples, either locally or in a batch system, you should set up the download and/or linkage to the SMEFTsim model and restrict card used, by running the following command:
> source setup_SMEFTsim_model.sh

Work uses version 0.9.3 of MadMiner (https://github.com/madminer-tool/madminer), installed simply by running `pip install -r requirements.txt`

Requires version >= 3.3.1 of Madgraph if generating samples.

There are different files in the codebase, designed to maximize the parallelizability of event generation and analysis:

- `setup.py`: defines the Wilson coefficient and morphing setup and creates setup file.

- `gen_signal/gen_background.py`: generate signal and background samples or prepare files to run sample generation on batch systems
    - requires pointing the code to your Madgraph installation

- `delphes_level_analysis.py`: simulates detector response with Delphes (with the ATLAS card) and performs the analysis (inc. reconstructing the relevant observables)

- `analysis_sample_combiner.py`: combines the analysed samples

- `sally_training.py`: creates unweighted training sample, augmented with joint score + trains the neural networks for the SALLY method, with a few different input variable sets

- `compute_FisherInfo_sally.py`: compute the Fisher information matrices and derive linearized limits from the SALLY method 

- `compute_FisherInfo_histograms.py`: compute the Fisher information matrices and derive linearized limits from the complete truth-level information, the rate or 1D/2D histograms 

- `compute_full_limits.py`: compute limits with the full likelihood ratio (in the asymptotic limit) from 1D/2D histograms or SALLY

All of the scripts have an argument parser describing their API, just type `_script_ -h` to see the list of available options.

If you want to reproduce the results, we added is a series of shell scripts with the exact parameters used for this work, to be run in the following order `sample_generation.sh` &rarr; `SALLY_training.sh` &rarr; `plotting.sh`/`linearized_limits.sh`/`full_limits.sh` (you can run these simultaneously). 