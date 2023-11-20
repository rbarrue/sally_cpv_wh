# -*- coding: utf-8 -*-

"""
compute_distributions.py

Computes distributions of different variables for signal and backgrounds, either S+B vs. B or S vs. B with signal at the different generated benchmarks

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import matplotlib
import os, sys
import argparse as ap
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from madminer.plotting.distributions import *
from madminer.ml import ScoreEstimator, Ensemble
from madminer import sampling

#from operator import 
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

# modification of plot_distributions function in MadMiner to be able to plot signal and backgrounds separated in one go
def plot_distributions_split_backgrounds(
    filename,
    filename_bkgonly,
    observables=None,
    parameter_points=None,
    parameter_points_bkgs=['sm'],
    uncertainties="nuisance",
    nuisance_parameters=None,
    draw_nuisance_toys=None,
    normalize=False,
    log=False,
    observable_labels=None,
    n_bins=50,
    line_labels=None,
    line_labels_bkgs=['Backgrounds (SM)'],
    colors=None,
    linestyles=None,
    linewidths=1.5,
    toy_linewidths=0.5,
    alpha=0.15,
    toy_alpha=0.75,
    n_events=None,
    n_toys=100,
    n_cols=3,
    quantiles_for_range=(0.001, 0.999),
    sample_only_from_closest_benchmark=True,
    remove_negative_weights=False
):
    """
    Plots one-dimensional histograms of observables in a MadMiner file for a given set of benchmarks 
    as well as the background-only histograms for another given set of benchmarks in the same canvas
    (useful to see if desired operators affect the background).

    Parameters
    ----------
    filename : str
        Filename of a MadMiner HDF5 file.

    filename_bkgonly : str
        Filename of the MadMiner HDF5 file which contains the background-only information.

    observables : list of str or None, optional
        Which observables to plot, given by a list of their names. If None, all observables in the file
        are plotted. Default value: None.

    parameter_points : list of (str or ndarray) or None, optional
        Which parameter points to use for histogramming the data. Given by a list, each element can either be the name
        of a benchmark in the MadMiner file, or an ndarray specifying any parameter point in a morphing setup. If None,
        all physics (non-nuisance) benchmarks defined in the MadMiner file are plotted. Default value: None.
    
    parameter_points_bkgs : list of (str or ndarray) or None, optional
        Same as parameter_points but for the background-only sample

    uncertainties : {"nuisance", "none"}, optional
        Defines how uncertainty bands are drawn. With "nuisance", the variation in cross section from all nuisance
        parameters is added in quadrature. With "none", no error bands are drawn.

    nuisance_parameters : None or list of int, optional
        If uncertainties is "nuisance", this can restrict which nuisance parameters are used to draw the uncertainty
        bands. Each entry of this list is the index of one nuisance parameter (same order as in the MadMiner file).

    draw_nuisance_toys : None or int, optional
        If not None and uncertainties is "nuisance", sets the number of nuisance toy distributions that are drawn
        (in addition to the error bands).

    normalize : bool, optional
        Whether the distribution is normalized to the total cross section. Default value: False.

    log : bool, optional
        Whether to draw the y axes on a logarithmic scale. Default value: False.

    observable_labels : None or list of (str or None), optional
        x-axis labels naming the observables. If None, the observable names from the MadMiner file are used. Default
        value: None.

    n_bins : int, optional
        Number of histogram bins. Default value: 50.

    line_labels : None or list of (str or None), optional
        Labels for the different parameter points. If None and if parameter_points is None, the benchmark names from
        the MadMiner file are used. Default value: None.

    line_labels_bkgs : None or list of (str or None), optional
        Same as line_labels but for the background-only sample.

    colors : None or str or list of str, optional
        Matplotlib line (and error band) colors for the distributions. If None, uses default colors. Default value:
        None.

    linestyles : None or str or list of str, optional
        Matplotlib line styles for the distributions. If None, uses default linestyles. Default value: None.

    linewidths : float or list of float, optional
        Line widths for the contours. Default value: 1.5.

    toy_linewidths : float or list of float or None, optional
        Line widths for the toy replicas, if uncertainties is "nuisance" and draw_nuisance_toys is not None. If None,
        linewidths is used. Default value: 1.

    alpha : float, optional
        alpha value for the uncertainty bands. Default value: 0.25.

    toy_alpha : float, optional
        alpha value for the toy replicas, if uncertainties is "nuisance" and draw_nuisance_toys is not None. Default
        value: 0.75.

    n_events : None or int, optional
        If not None, sets the number of events from the MadMiner file that will be analyzed and plotted. Default value:
        None.

    n_toys : int, optional
        Number of toy nuisance parameter vectors used to estimate the systematic uncertainties. Default value: 100.

    n_cols : int, optional
        Number of columns of subfigures in the plot. Default value: 3.

    quantiles_for_range : tuple of two float, optional
        Tuple `(min_quantile, max_quantile)` that defines how the observable range is determined for each panel.
        Default: (0.025, 0.075).

    sample_only_from_closest_benchmark : bool, optional
        If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

    remove_negative_weights : bool, optional
        If True, events with negative weights are removed from the distributions. Default value: False

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    logger.warning(
            "Careful: this tool assumes that the signal and background samples are generated with the same setup, including"
            " identical observables and identical nuisance parameters. If it is used with "
            "samples with different"
            " settings, there will be wrong results! There are no explicit cross checks in place yet.")

    # Load data
    sa = SampleAugmenter(filename, include_nuisance_parameters=True)
        # bkg-only sample
    sa_bkgs=SampleAugmenter(filename_bkgonly,include_nuisance_parameters=True)

    if uncertainties == "nuisance":
        nuisance_morpher = NuisanceMorpher(
            sa.nuisance_parameters, list(sa.benchmarks.keys()), reference_benchmark=sa.reference_benchmark
        )
        # bkg-only sample
        nuisance_morpher_bkgs = NuisanceMorpher(
            sa_bkgs.nuisance_parameters, list(sa_bkgs.benchmarks.keys()), reference_benchmark=sa_bkgs.reference_benchmark
        )

    if np.count_nonzero(sa.n_events_generated_per_benchmark)==0:
        logger.warning("(Supposed) signal file contains only backgrounds, will plot only the bkg distribution")
    elif sa.n_events_backgrounds>0:
        logger.info("Signal file contains backgrounds, will plot signal+bkgs for the different signal benchmark points vs. background-only")
    else:
        logger.info("Signal file doesn't contain backgrounds, will plot signal at the different signal benchmark points vs. background-only")

    # Default settings
    logger.debug("Benchmarks: %s",sa.benchmarks)
    if parameter_points is None:
        parameter_points = []
        for key, is_nuisance in zip(sa.benchmarks, sa.benchmark_nuisance_flags):
            if not is_nuisance:
                parameter_points.append(key)
    
    # RB: if not setting the benchmark point at which to show the bkgs, showing the bkg-only distribution for the same benchmarks as for signal
    if parameter_points_bkgs is None:
        parameter_points_bkgs=parameter_points
        
    if line_labels is None:
        line_labels = parameter_points
    
    if line_labels_bkgs is None:
        line_labels_bkgs=[f"Backgrounds ({parameter_point})" for parameter_point in parameter_points_bkgs]

    line_labels=[*line_labels,*line_labels_bkgs]

    n_parameter_points = len(parameter_points) + len(parameter_points_bkgs)

    # RB: n_parameter_points is only used in the visual setting of the plots
    if colors is None:
        colors = [f"C{i}" for i in range(10)] * (n_parameter_points // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_parameter_points)]

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot"] * (n_parameter_points // 4 + 1)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_parameter_points)]

    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_parameter_points)]

    if toy_linewidths is None:
        toy_linewidths = linewidths
    if not isinstance(toy_linewidths, list):
        toy_linewidths = [toy_linewidths for _ in range(n_parameter_points)]

    # Observables (assuming that the observables are the same in both files)
    observable_indices = []
    if observables is None:
        observable_indices = list(range(len(sa.observables)))
    else:
        all_observables = list(sa.observables.keys())
        for obs in observables:
            try:
                observable_indices.append(all_observables.index(str(obs)))
            except ValueError:
                logging.warning("Ignoring unknown observable %s", obs)

    logger.debug("Observable indices: %s", observable_indices)

    n_observables = len(observable_indices)

    if observable_labels is None:
        all_observables = list(sa.observables.keys())
        observable_labels = [all_observables[obs] for obs in observable_indices]
    
    # Parse thetas
    theta_values = [sa._get_theta_value(theta) for theta in parameter_points]
    theta_matrices = [sa._get_theta_benchmark_matrix(theta) for theta in parameter_points]
    logger.debug("Calculated %s theta matrices", len(theta_matrices))

    # bkg-only samples
    theta_values_bkgs = [sa_bkgs._get_theta_value(theta) for theta in parameter_points_bkgs]
    theta_matrices_bkgs = [sa_bkgs._get_theta_benchmark_matrix(theta) for theta in parameter_points_bkgs]
    logger.debug("Calculated %s theta matrices for background-only", len(theta_matrices_bkgs))

    # Get event data (observations and weights)
    all_x, all_weights_benchmarks = sa.weighted_events(generated_close_to=None)
    logger.debug("Loaded raw data with shapes %s, %s", all_x.shape, all_weights_benchmarks.shape)

    indiv_x, indiv_weights_benchmarks = [], []
    if sample_only_from_closest_benchmark:
        for theta in theta_values:
            this_x, this_weights = sa.weighted_events(generated_close_to=theta)
            indiv_x.append(this_x)
            indiv_weights_benchmarks.append(this_weights)

        # bkg-only samples
    all_x_bkgs, all_weights_benchmarks_bkgs = sa_bkgs.weighted_events(generated_close_to=None)
    logger.debug("Loaded raw background-only data with shapes %s, %s", all_x_bkgs.shape, all_weights_benchmarks_bkgs.shape)

    indiv_x_bkgs, indiv_weights_benchmarks_bkgs = [], []
    if sample_only_from_closest_benchmark:
        for theta in theta_values_bkgs:
            this_x_bkgs, this_weights_bkgs = sa_bkgs.weighted_events(generated_close_to=theta)
            indiv_x_bkgs.append(this_x_bkgs)
            indiv_weights_benchmarks_bkgs.append(this_weights_bkgs)

    if remove_negative_weights:
        # Remove negative weights
        sane_event_filter = np.all(all_weights_benchmarks >= 0.0, axis=1)

        n_events_before = all_weights_benchmarks.shape[0]
        all_x = all_x[sane_event_filter]
        all_weights_benchmarks = all_weights_benchmarks[sane_event_filter]
        n_events_removed = n_events_before - all_weights_benchmarks.shape[0]

        if int(np.sum(sane_event_filter, dtype=int)) < len(sane_event_filter):
            logger.warning("Removed %s / %s events with negative weights", n_events_removed, n_events_before)

        for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
            sane_event_filter = np.all(weights >= 0.0, axis=1)
            indiv_x[i] = x[sane_event_filter]
            indiv_weights_benchmarks[i] = weights[sane_event_filter]

            # bkg-only sample
        sane_event_filter = np.all(all_weights_benchmarks_bkgs >= 0.0, axis=1)

        n_events_before = all_weights_benchmarks_bkgs.shape[0]
        all_x_bkgs = all_x_bkgs[sane_event_filter]
        all_weights_benchmarks_bkgs = all_weights_benchmarks_bkgs[sane_event_filter]
        n_events_removed = n_events_before - all_weights_benchmarks_bkgs.shape[0]

        if int(np.sum(sane_event_filter, dtype=int)) < len(sane_event_filter):
            logger.warning("Removed %s / %s background-only events with negative weights", n_events_removed, n_events_before)

        for i, (x, weights) in enumerate(zip(indiv_x_bkgs, indiv_weights_benchmarks_bkgs)):
            sane_event_filter = np.all(weights >= 0.0, axis=1)
            indiv_x_bkgs[i] = x[sane_event_filter]
            indiv_weights_benchmarks_bkgs[i] = weights[sane_event_filter]

    # Shuffle events
    all_x, all_weights_benchmarks = shuffle(all_x, all_weights_benchmarks)

    for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
        indiv_x[i], indiv_weights_benchmarks[i] = shuffle(x, weights)

    # bkg-only
    all_x_bkgs, all_weights_benchmarks_bkgs = shuffle(all_x_bkgs, all_weights_benchmarks_bkgs)
    for i, (x, weights) in enumerate(zip(indiv_x_bkgs, indiv_weights_benchmarks_bkgs)):
        indiv_x_bkgs[i], indiv_weights_benchmarks_bkgs[i] = shuffle(x, weights)

    # Only analyze n_events
    if n_events is not None and n_events < all_x.shape[0]:
        logger.debug("Only analyzing first %s / %s events", n_events, all_x.shape[0])

        all_x = all_x[:n_events]
        all_weights_benchmarks = all_weights_benchmarks[:n_events]

        for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
            indiv_x[i] = x[:n_events]
            indiv_weights_benchmarks[i] = weights[:n_events]

        # bkg-only sample
        all_x_bkgs = all_x_bkgs[:n_events]
        all_weights_benchmarks_bkgs = all_weights_benchmarks_bkgs[:n_events]

        for i, (x, weights) in enumerate(zip(indiv_x_bkgs, indiv_weights_benchmarks_bkgs)):
            indiv_x_bkgs[i] = x[:n_events]
            indiv_weights_benchmarks_bkgs[i] = weights[:n_events]

    if uncertainties != "nuisance":
        n_toys = 0

    n_nuisance_toys_drawn = 0
    if draw_nuisance_toys is not None:
        n_nuisance_toys_drawn = draw_nuisance_toys

    # Nuisance parameters
    nuisance_toy_factors = []

    if uncertainties == "nuisance":
        n_nuisance_params = sa.n_nuisance_parameters

        if not n_nuisance_params > 0:
            raise RuntimeError("Cannot draw systematic uncertainties -- no nuisance parameters found!")

        logger.debug("Drawing nuisance toys")

        nuisance_toys = np.random.normal(loc=0.0, scale=1.0, size=n_nuisance_params * n_toys)
        nuisance_toys = nuisance_toys.reshape(n_toys, n_nuisance_params)

        # Restrict nuisance parameters
        if nuisance_parameters is not None:
            for i in range(n_nuisance_params):
                if i not in nuisance_parameters:
                    nuisance_toys[:, i] = 0.0

        logger.debug("Drew %s toy values for nuisance parameters", n_toys * n_nuisance_params)

        nuisance_toy_factors = np.array(
            [
                nuisance_morpher.calculate_nuisance_factors(nuisance_toy, all_weights_benchmarks)
                for nuisance_toy in nuisance_toys
            ]
        )  # Shape (n_toys, n_events)

        nuisance_toy_factors = sanitize_array(nuisance_toy_factors, min_value=1.0e-2, max_value=100.0)
        # Shape (n_toys, n_events)

            # bkg-only sample
        nuisance_toy_factors_bkgs = np.array(
            [
                nuisance_morpher_bkgs.calculate_nuisance_factors(nuisance_toy, all_weights_benchmarks_bkgs)
                for nuisance_toy in nuisance_toys
            ]
        )  # Shape (n_toys, n_events)

        nuisance_toy_factors = sanitize_array(nuisance_toy_factors_bkgs, min_value=1.0e-2, max_value=100.0)
        # Shape (n_toys, n_events)


    # Preparing plot
    n_rows = (n_observables + n_cols - 1) // n_cols
    n_events_for_range = 10000 if n_events is None else min(10000, n_events)

    fig = plt.figure(figsize=(4.0 * n_cols, 4.0 * n_rows))

    # loop over observables
    for i_panel, (i_obs, xlabel) in enumerate(zip(observable_indices, observable_labels)):
        logger.debug("Plotting panel %s: observable %s, label %s", i_panel, i_obs, xlabel)

        # Figure out x range
        xmins, xmaxs = [], []
        for theta_matrix in theta_matrices:
            x_small = all_x[:n_events_for_range]
            weights_small = mdot(theta_matrix, all_weights_benchmarks[:n_events_for_range])

            xmin = weighted_quantile(x_small[:, i_obs], quantiles_for_range[0], weights_small)
            xmax = weighted_quantile(x_small[:, i_obs], quantiles_for_range[1], weights_small)
            xwidth = xmax - xmin
            xmin -= xwidth * 0.1
            xmax += xwidth * 0.1

            xmin = max(xmin, np.min(all_x[:, i_obs]))
            xmax = min(xmax, np.max(all_x[:, i_obs]))

            xmins.append(xmin)
            xmaxs.append(xmax)
            
            # bkg-only sample
        for theta_matrix in theta_matrices_bkgs:
            x_small = all_x_bkgs[:n_events_for_range]
            weights_small = mdot(theta_matrix, all_weights_benchmarks_bkgs[:n_events_for_range])
            xmin = weighted_quantile(x_small[:, i_obs], quantiles_for_range[0], weights_small)
            xmax = weighted_quantile(x_small[:, i_obs], quantiles_for_range[1], weights_small)
            xwidth = xmax - xmin
            xmin -= xwidth * 0.1
            xmax += xwidth * 0.1

            xmin = max(xmin, np.min(all_x[:, i_obs]))
            xmax = min(xmax, np.max(all_x[:, i_obs]))

            xmins.append(xmin)
            xmaxs.append(xmax)            

        xmin = min(xmins)
        xmax = max(xmaxs)
        x_range = (xmin, xmax)

        logger.debug("Ranges for observable %s: min = %s, max = %s", xlabel, xmins, xmaxs)

        # Subfigure
        ax = plt.subplot(n_rows, n_cols, i_panel + 1)

        # Calculate histograms and append them to the list
        bin_edges = None
        histos = []
        histos_up = []
        histos_down = []
        histos_toys = []

        for i_theta, theta_matrix in enumerate(theta_matrices):
            theta_weights = mdot(theta_matrix, all_weights_benchmarks)  # Shape (n_events,)

            if sample_only_from_closest_benchmark:
                indiv_theta_weights = mdot(theta_matrix, indiv_weights_benchmarks[i_theta])  # Shape (n_events,)
                histo, bin_edges = np.histogram(
                    indiv_x[i_theta][:, i_obs],
                    bins=n_bins,
                    range=x_range,
                    weights=indiv_theta_weights,
                    density=normalize,
                )
            else:
                histo, bin_edges = np.histogram(
                    all_x[:, i_obs], bins=n_bins, range=x_range, weights=theta_weights, density=normalize
                )
            histos.append(histo)

            if uncertainties == "nuisance":
                histos_toys_this_theta = []
                for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
                    toy_histo, _ = np.histogram(
                        all_x[:, i_obs],
                        bins=n_bins,
                        range=x_range,
                        weights=theta_weights * nuisance_toy_factors_this_toy,
                        density=normalize,
                    )
                    histos_toys_this_theta.append(toy_histo)

                histos_up.append(np.percentile(histos_toys_this_theta, 84.0, axis=0))
                histos_down.append(np.percentile(histos_toys_this_theta, 16.0, axis=0))
                histos_toys.append(histos_toys_this_theta[:n_nuisance_toys_drawn])

        # bkg-only histograms
        for i_theta, theta_matrix in enumerate(theta_matrices_bkgs):
            theta_weights = mdot(theta_matrix, all_weights_benchmarks_bkgs)  # Shape (n_events,)

            if sample_only_from_closest_benchmark:
                indiv_theta_weights = mdot(theta_matrix, indiv_weights_benchmarks_bkgs[i_theta])  # Shape (n_events,)
                histo, bin_edges = np.histogram(
                    indiv_x_bkgs[i_theta][:, i_obs],
                    bins=n_bins,
                    range=x_range,
                    weights=indiv_theta_weights,
                    density=normalize,
                )
            else:
                histo, bin_edges = np.histogram(
                    all_x_bkgs[:, i_obs], bins=n_bins, range=x_range, weights=theta_weights, density=normalize
                )
            histos.append(histo)

            if uncertainties == "nuisance":
                histos_toys_this_theta = []
                for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors_bkgs):
                    toy_histo, _ = np.histogram(
                        all_x_bkgs[:, i_obs],
                        bins=n_bins,
                        range=x_range,
                        weights=theta_weights * nuisance_toy_factors_this_toy,
                        density=normalize,
                    )
                    histos_toys_this_theta.append(toy_histo)

                histos_up.append(np.percentile(histos_toys_this_theta, 84.0, axis=0))
                histos_down.append(np.percentile(histos_toys_this_theta, 16.0, axis=0))
                histos_toys.append(histos_toys_this_theta[:n_nuisance_toys_drawn])
        
        # Draw error bands
        if uncertainties == "nuisance":
            for histo_up, histo_down, lw, color, label, ls in zip(
                histos_up, histos_down, linewidths, colors, line_labels, linestyles
            ):
                bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
                histo_down_ = np.repeat(histo_down, 2)
                histo_up_ = np.repeat(histo_up, 2)

                plt.fill_between(bin_edges_, histo_down_, histo_up_, facecolor=color, edgecolor="none", alpha=alpha)

            # Draw some toys
            for histo_toys, lw, color, ls in zip(histos_toys, toy_linewidths, colors, linestyles):
                for k in range(n_nuisance_toys_drawn):
                    bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
                    histo_ = np.repeat(histo_toys[k], 2)

                    plt.plot(bin_edges_, histo_, color=color, alpha=toy_alpha, lw=lw, ls=ls)

        # Draw central lines
        for histo, lw, color, label, ls in zip(histos, linewidths, colors, line_labels, linestyles):
            bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
            histo_ = np.repeat(histo, 2)
            # in NP histos, "sum of the histogram values will not be equal to 1 unless bins of unity width are chosen"
            # binning is the same for all histograms of the same variable, this is only an additional scale factor
            if normalize:
                histo_*=1.0/np.sum(histo)

            plt.plot(bin_edges_, histo_, color=color, lw=lw, ls=ls, label=label, alpha=1.0)


        plt.legend()
        
        plt.xlabel(xlabel,loc='right')
        if normalize:
            plt.ylabel("Normalized distribution")
        else:
            plt.ylabel(r"$\frac{d\sigma}{dx}$ [pb / bin]")

        plt.xlim(x_range[0], x_range[1])
        if log:
            ax.set_yscale("log", nonposy="clip")
        else:
            plt.ylim(0.0, None)



    plt.tight_layout()

    return fig

def plot_sally_distributions_split_backgrounds(filename,filename_bkgonly,model_path,sally_component=0,n_bins=50,normalize=True,parameter_points=None,parameter_points_bkgs=['sm'],line_labels=None,line_labels_bkgs=['Backgrounds (SM)'],log=False,colors=None,linestyles=None,linewidths=1.5,quantiles_for_range=(0.01, 0.99)):

    score_estimator=Ensemble()
    score_estimator.load(model_path)

    # Load data
    sa = SampleAugmenter(filename, include_nuisance_parameters=True)

    # bkg-only sample
    sa_bkgs=SampleAugmenter(filename_bkgonly,include_nuisance_parameters=True)  

    logger.debug("Benchmarks: %s",sa.benchmarks)

    if parameter_points is None:
        parameter_points = []
        for key, is_nuisance in zip(sa.benchmarks, sa.benchmark_nuisance_flags):
            if not is_nuisance:
                parameter_points.append(key)

    if line_labels==None:
        line_labels=parameter_points
    else:
        if len(line_labels)!=len(parameter_points):
            logging.warning("different number of parameter points and line labels !")

    fig = plt.figure()
    n_parameter_points = len(parameter_points) + len(parameter_points_bkgs)

    # RB: n_parameter_points is only used in the visual setting of the plots
    if colors is None:
        colors = [f"C{i}" for i in range(10)] * (n_parameter_points // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_parameter_points)]

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot"] * (n_parameter_points // 4 + 1)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_parameter_points)]

    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_parameter_points)]

    histos=[]

    logging.info('Plotting only events in the test partition')
    for parameter_point in parameter_points:
        start_event_test, end_event_test, correction_factor_test = sa._train_validation_test_split('test',validation_split=0.2,test_split=0.2)
        x,weights=sa.weighted_events(theta=parameter_point,start_event=start_event_test,end_event=end_event_test)
        weights*=correction_factor_test # scale the events by the correction factor
        t_hat,_=score_estimator.evaluate_score(x=x)
        xmin = weighted_quantile(t_hat[:,sally_component], quantiles_for_range[0], weights)
        xmax = weighted_quantile(t_hat[:,sally_component], quantiles_for_range[1], weights)
        histo,bin_edges=np.histogram(
                    t_hat[:,sally_component],
                    bins=n_bins,
                    range=(xmin,xmax),
                    weights=weights,
                    density=normalize,
                )
        if normalize:
            histo*=1.0/np.sum(histo)
        histos.append(histo)

    for parameter_point in parameter_points_bkgs:
        start_event_test, end_event_test, correction_factor_test = sa_bkgs._train_validation_test_split('test',validation_split=0.2,test_split=0.2)
        x_bkgs,weights_bkg=sa_bkgs.weighted_events(theta=parameter_point,start_event=start_event_test,end_event=end_event_test)
        weights_bkg*=correction_factor_test # scale the events by the correction factor
        t_hat_bkg,_ = score_estimator.evaluate_score(x=x_bkgs)
        xmin = weighted_quantile(t_hat_bkg[:,sally_component], quantiles_for_range[0], weights_bkg)
        xmax = weighted_quantile(t_hat_bkg[:,sally_component], quantiles_for_range[1], weights_bkg)
        histo,bin_edges=np.histogram(
            t_hat_bkg[:,sally_component],
            bins=n_bins,
            range=(xmin,xmax),
            weights=weights_bkg,
            density=normalize,
        )

        if normalize:
            histo*=1.0/np.sum(histo)
        histos.append(histo)
    
    for histo, lw, color, label, ls in zip(histos, linewidths, colors, line_labels+line_labels_bkgs, linestyles):
        bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
        histo_ = np.repeat(histo, 2)
        plt.plot(bin_edges_, histo_, color=color, lw=lw, ls=ls, label=label, alpha=1.0)

    if log:
        plt.yscale("log")
    else:
        plt.ylim(0.0, None)

    if normalize:
        plt.ylabel("Normalized distribution")
    else:
        plt.ylabel(r"$\frac{d\sigma}{dx}$ [pb / bin]")
    
    xmin_plot = xmin*1.2 if xmin < 0 else xmin/1.2
    xmax_plot = xmax*1.2 if xmax > 0 else xmax/1.2
    plt.xlim(xmin_plot,xmax_plot)

    plt.legend()
    plt.tight_layout()
    plt.xlabel("SALLY",loc='right')

    return fig

def plot_sally_train_val_losses(losses_filename):
    result=np.load(losses_filename)

    fig=plt.figure()

    for i_estimator,(train_loss,val_loss) in enumerate(result['arr_0']):
        plt.plot(train_loss,lw=1.5,ls='solid',label=f'estimator {i_estimator+1} (train)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
        plt.plot(val_loss,lw=1.5,ls='dashed',label=f'estimator {i_estimator+1} (val)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
    
    plt.legend()
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Computes distributions of different variables for signal and backgrounds.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('-pdir','--plot_dir',help='folder where to save plots to',required=True)

    parser.add_argument('-s','--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',choices=['signalOnly_SMonly','signalOnly','withBackgrounds_SMonly','withBackgrounds','backgroundOnly'],default='signalOnly') # to stay like this until we reimplement the BSM sample features in the other scripts
        
    parser.add_argument('-c','--channel',help='lepton+charge flavor channels to plot.',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],default=['wh'],nargs="+")

    parser.add_argument('-o','--observables',help='which of the observables to plot. If None, plots all the observables in the file.',nargs="*",default=None)

    parser.add_argument('-sally','--plot_sally',help='whether to plot the distribution of the output of SALLY. uses all the weighted events in the given sample.',default=False,action='store_true')

    parser.add_argument('-so','--sally_observables',help='which of the SALLY training input variable set models to use',required='sally' in sys.argv)

    parser.add_argument('-sm','--sally_model',help='which of the SALLY models (for each of the input variable configurations) to use.',required='sally' in sys.argv)

    parser.add_argument('-shape','--do_shape_only',help='whether or not to do shape-only plots.',default=False,action='store_true')

    parser.add_argument('-log','--do_log',help='do plots in log scale',default=False,action='store_true')

    parser.add_argument('-nb','--n_bins',help='number of evenly spaced bins in the histogram',type=int,default=20)

    parser.add_argument('-stem','--plot_stem',help='string to add to end of the observable plot name.',default='')

    parser.add_argument('--remove_negative_weights',help='removes events with negative weights from entering the plot',default=False,action='store_true')

    parser.add_argument('-l','--plot_losses',help='plots losses',default=False,action='store_true')

    args=parser.parse_args()

    os.makedirs(f'{args.plot_dir}/',exist_ok=True)

    obs_xlabel_dict={
        'pt_w':r'$p_T^W ~ [GeV]$',
        #'pt_l':r'$p_T^\ell ~ [GeV]$',
        'mt_tot':r'$m_T^{\ell \nu b \bar{b}} ~ [GeV]$',
        'pz_nu':r'$p_z^{\nu}$',
        'cos_deltaPlus':r'$\cos \delta^+$',
        'cos_deltaMinus':r'$\cos \delta^-$',
        'ql_cos_deltaPlus':r'$Q_{\ell} ~\cos \delta^+$',
        'ql_cos_deltaMinus':r'$Q_{\ell} ~\cos \delta^-$',
    }

    if args.sample_type=='backgroundOnly':
        logging.warning("Plotting only background distributions")

    for channel in args.channel:

        plot_stem=args.plot_stem
        if not args.plot_sally:
            
            if plot_stem != '':
                plot_stem=f'_{plot_stem}'
            else:                
                if args.observables is not None and len(args.observables)==1:
                    plot_stem=f'_{args.observables[0]}'
                elif args.observables is None:
                    plot_stem='_all_observables'
                else:
                    logging.warning('plot has no plot stem to identify it !')
                    plot_stem=''
            
            if args.do_shape_only:
                plot_stem+='_shape_only'
            if args.do_log:
                plot_stem+='_log'
            
            histo_observables=plot_distributions_split_backgrounds(filename=f'{args.main_dir}/{channel}_{args.sample_type}.h5',
            parameter_points=None if args.sample_type!='backgroundOnly' else [],
            filename_bkgonly=f'{args.main_dir}/{channel}_backgroundOnly.h5',
            observables=args.observables,
            observable_labels=[obs_xlabel_dict[obs] if obs in obs_xlabel_dict else obs for obs in args.observables] if args.observables!= None else None,
            line_labels=['SM',r'$c_\tilde{HW} = 1.15$',r'$c_\tilde{HW} = -1.035$'] if args.sample_type!='backgroundOnly' else [],
            log=args.do_log,
            linestyles=None if args.sample_type!='backgroundOnly' else ['dashdot'],
            normalize=args.do_shape_only,n_bins=args.n_bins,uncertainties='None',
            remove_negative_weights=args.remove_negative_weights,n_cols=3)

            histo_observables.savefig(f'{args.plot_dir}/{channel}_{args.sample_type}{plot_stem}.pdf')
        
        else:
            
            histo_sally=plot_sally_distributions_split_backgrounds(filename=f'{args.main_dir}/{channel}_{args.sample_type}.h5',
            parameter_points=None if args.sample_type!='backgroundOnly' else [],
            filename_bkgonly=f'{args.main_dir}/{channel}_backgroundOnly.h5',
            model_path=f'{args.main_dir}/models/{args.sally_observables}/{args.sally_model}/sally_ensemble_{channel}_{args.sample_type}',
            line_labels=['SM',r'$c_\tilde{HW} = 1.15$',r'$c_\tilde{HW} = -1.035$'] if args.sample_type!='backgroundOnly' else [],
            linestyles=None if args.sample_type!='backgroundOnly' else ['dashdot'],
            normalize=args.do_shape_only,log=args.do_log)
            
            if args.do_shape_only:
                plot_stem+='_shape_only'
            if args.do_log:
                plot_stem+='_log'

            histo_sally.savefig(f'{args.plot_dir}/sally_{channel}_{args.sample_type}_{args.sally_observables}_{args.sally_model}{plot_stem}.pdf')

            if args.plot_losses:
                histo_sally_losses=plot_sally_train_val_losses(f'{args.main_dir}/models/{args.sally_observables}/{args.sally_model}/losses_{channel}_{args.sample_type}.npz')
                histo_sally_losses.savefig((f'{args.plot_dir}/sally_losses_{channel}_{args.sample_type}_{args.sally_observables}_{args.sally_model}.pdf'))