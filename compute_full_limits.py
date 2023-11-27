# -*- coding: utf-8 -*-

"""
compute_asymptotic_limits.py

Extracts limits based on the asymptotic properties of the likelihood ratio as test statistics.

This allows taking into account the effect of square terms in a consistent way (not possible with the Fisher information formalism)

Ricardo Barrué (LIP/IST/CERN-ATLAS), 3/8/2023

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os,sys
import argparse as ap
from madminer.limits import AsymptoticLimits
from madminer import sampling
from madminer.plotting import plot_histograms
from madminer.utils.histo import Histo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from madminer.sampling import combine_and_shuffle

#MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)

def plot_likelihood_ratio(parameter_grid,log_r,xlabel,ylabel,do_log):

  fig = plt.figure()

  plt.plot(parameter_grid,log_r,color='black',lw=0.5)

  if do_log:
    plt.yscale ('log')
  plt.axhline(y=1.64,linestyle='--',color='blue',label='95%CL')
  plt.axhline(y=1.0,linestyle='--',color='red',label='68%CL')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  parameter_points_to_plot=[parameter_grid[i] for i,log_r_val in enumerate(log_r) if log_r_val<3.28]
  plt.xlim(-max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])),+max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])))
  plt.ylim(-1,3.28)
  plt.legend()
  plt.tight_layout()

  return fig

def extract_limits_single_parameter(parameter_grid,p_values,index_central_point):
  
  n_parameters=len(parameter_grid[0])
  list_points_inside_68_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.32 ]
  list_points_inside_95_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.05 ]  
  return parameter_grid[index_central_point],[round(list_points_inside_68_cl[0],5),round(list_points_inside_68_cl[-1],5)],[round(list_points_inside_95_cl[0],5),round(list_points_inside_95_cl[-1],5)]

# alternative to setting the points where to calculate the likelihoods for a single coefficient
def get_thetas_eval(theta_min,theta_max,spacing):
  theta_max=1.2
  theta_min=-1.2
  spacing=0.1
  thetas_eval=np.array([[round(i*spacing - (theta_max-theta_min)/2,4)] for i in range(int((theta_max-theta_min)/spacing)+1)])
  if np.array(0.0) not in thetas_eval:
    thetas_eval=np.append(thetas_eval,np.array(0.0))
  
  return thetas_eval

# get indices of likelihood histograms to plot for a single coefficient
def get_indices_log_r_histograms(parameter_grid,npoints_plotting,plot_parameter_spacing=None,index_best_point=None):
  
  if plot_parameter_spacing is not None:
    npoints_plotting=int((parameter_grid[0,0]-parameter_grid[-1,0])/plot_parameter_spacing)
  
  if npoints_plotting>len(parameter_grid):
    logging.warning('asking for more points than what the parameter grid has, will plot all points')
    npoints_plotting=len(parameter_grid)
  
  # span the entire parameter grid
  if index_best_point is None:
    indices = list(range(0,len(parameter_grid),len(parameter_grid)/npoints_plotting))
  # span a small region around the best fit point
  else:
    indices = list(range(max(0,index_best_point-int(npoints_plotting/2)),min(len(parameter_grid),index_best_point+int(npoints_plotting/2))))
  
  # make sure that the SM point is always included
  sm_point_index=np.where(parameter_grid==[0.0])[0][0] # 1D case
  if sm_point_index not in indices:
    indices.append(sm_point_index)

  return list(indices)

def get_minmax_y_histograms(histos,epsilon=0.02):

  min_value=np.min([np.min(histo.histo) for histo in histos])*(1-epsilon)
  max_value=np.max([np.max(histo.histo) for histo in histos])*(1+epsilon)

  return (min_value,max_value)

if __name__ == "__main__":

  parser = ap.ArgumentParser(description='Computes limits using asymptotic (large sample size) limit of the likelihood ratio as a chi-square distribution.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

  parser.add_argument('-pdir','--plot_dir',help='folder where to save plots to',required=True)

  parser.add_argument('-c','--channel',help='lepton/charge flavor channels to derive limits on',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],default='wh')

  parser.add_argument('-s','--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',default='signalOnly')

  parser.add_argument('-m','--mode',help='what to use to extract the limits, given as input to the expected_limits function',choices=['rate','histo','sally'],required=True)

  parser.add_argument('-shape','--shape_only',help='if the limits are derived in shape-only histograms (i.e. without Poisson likelihood term)',action='store_true',default=False)

  parser.add_argument('-so','--sally_observables',help='which of the SALLY training observable sets to use when extracting limits from a SALLY-like method',choices=['all_observables_remove_redundant_cos','kinematic_only'],required='sally' in sys.argv)

  parser.add_argument('-sm','--sally_model',help='which of the SALLY models (for each of the input variable configurations) to use',required='sally' in sys.argv)

  parser.add_argument('-x','--observable_x',help='which of the observables to use in the x-axis if extracting limits from histograms',required='histo' in sys.argv)

  parser.add_argument('-y','--observable_y',help='which of the observables to use in the y-axis if extracting limits from histograms',default=None)

  parser.add_argument('-bx','--binning_x',help='binning of the variable in the x-axis (can either be a standard observable or output of the SALLY network)',nargs="*",type=float,default=None)

  parser.add_argument('-by','--binning_y',help='binning of the variable in the y-axis',nargs="*",type=float,default=None)

  parser.add_argument('-log','--do_log',help='whether or not to do histograms of likelihood in log_scale',default=False,action='store_true')

  parser.add_argument('-nf','--n_fits',help='number of times to shuffle the input dataset and redo the fit',type=int,default=1)

  parser.add_argument('-l','--lumi',help='process charge+flavor inclusive samples',type=int,default=300.)    

  parser.add_argument('-dbg','--debug',help='turns on debug functions',default=False,action='store_true')

  args=parser.parse_args()
  
  os.makedirs(f'{args.plot_dir}/limits/',exist_ok=True)
  
  if args.observable_y is None:
    hist_vars=[args.observable_x]
    hist_bins=[args.binning_x] if args.binning_x is not None else None
  else:
    hist_vars=[args.observable_x,args.observable_y]
    hist_bins=[args.binning_x,args.binning_y] if (args.binning_x is not None and args.binning_y is not None) else None
  
  logging.debug(f'hist variables: {str(hist_vars)}; hist bins: {str(hist_bins)}')

  #for sample_type in args.sample_type:
  logging.info(f"sample type: {args.sample_type}")
  list_central_values=[]

  if 'sally' in args.mode:
    log_file_path=f"{args.plot_dir}/limits/full_sally_{args.sample_type}_lumi{args.lumi}"  
  else:
    log_file_path=f"{args.plot_dir}/limits/full_histograms_{args.sample_type}_lumi{args.lumi}"

  if args.shape_only:
    log_file_path += '_shape_only'

  log_file=open(log_file_path+".csv",'a')
  if os.path.getsize(log_file_path+".csv") == 0:
    
    if 'sally' in args.mode:
      log_file.write('observables, model, binning_x, central value, 68% CL, 95% CL \n')
    else:
      log_file.write('observable_x, binning_x, observable_y, binning_y, central value, 68% CL, 95% CL \n')

  for i in range(args.n_fits):
    
    # object/class to perform likelihood/limit extraction, shuffling to have different datasets
    if i==0:
      limits_file=AsymptoticLimits(f'{args.main_dir}/{args.channel}_{args.sample_type}.h5')
    else:
      combine_and_shuffle([f'{args.main_dir}/{args.channel}_{args.sample_type}.h5'],
        f'{args.main_dir}/{args.channel}_{args.sample_type}_shuffled.h5',recalculate_header=False)
      limits_file=AsymptoticLimits(f'{args.main_dir}/{args.channel}_{args.sample_type}_shuffled.h5')

    parameter_grid,p_values,index_best_point,log_r_kin,log_r_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode=args.mode,theta_true=[0.0], include_xsec=not args.shape_only,
    model_file=f'{args.main_dir}/models/{args.sally_observables}/{args.sally_model}/sally_ensemble_{args.channel}_{args.sample_type}' if 'sally' in args.mode else None,
    hist_vars=hist_vars if 'histo' in args.mode else None,
    hist_bins=hist_bins,
    luminosity=args.lumi*1000.0,
    return_asimov=True,test_split=0.5,n_histo_toys=None,
    grid_ranges=[(-1.2,1.2)],grid_resolutions=[101])
    #thetas_eval=thetas_eval,grid_resolutions=None) # can be set given min, max and spacing using the get_thetas_eval funcion

    logging.debug(f'parameter grid: {parameter_grid}')
    if i%5==0:

      # plotting a subset of the likelihood histograms for debugging
      if args.debug:
        
        indices=get_indices_log_r_histograms(parameter_grid,npoints_plotting=4,index_best_point=index_best_point)

        likelihood_histograms = plot_histograms(
            histos=[histos[i] for i in indices],
            observed=[observed[i] for i in indices],
            observed_weights=observed_weights,
            histo_labels=[f"$cHWtil = {parameter_grid[i,0]:.2f}$" for i in indices],
            xlabel=args.observable_x,
            xrange=(hist_bins[0][0],hist_bins[0][-1]) if hist_bins is not None else None,
            yrange=get_minmax_y_histograms([histos[i] for i in indices],epsilon=0.05) ,
            log=args.do_log,
        )
        if 'sally' in args.mode:
          likelihood_histograms.savefig(f'{args.plot_dir}/limits/likelihoods_sally_{args.channel}_{args.sample_type}_{args.sally_observables}_{args.sally_model}_{i}.pdf')
        else:
          if args.observable_y is None:
            likelihood_histograms.savefig(f'{args.plot_dir}/limits/likelihoods_{args.channel}_{args.sample_type}_{args.observable_x}_{len(args.binning_x)-1}bins_lumi{args.lumi}_{i}.pdf')
          else:
            likelihood_histograms.savefig(f'{args.plot_dir}/limits/likelihoods_{args.channel}_{args.sample_type}_{args.observable_x}_{len(args.binning_x)-1}bins_{args.observable_y}_{len(args.binning_y)-1}bins_lumi{args.lumi}_{i}.pdf')

      # AsymptoticLimits returs unscaled individual pieces, (re)rescaling them for plotting
      rescaled_log_r = log_r_kin+log_r_rate
      rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])
      log_r_histo= plot_likelihood_ratio(parameter_grid[:,0],rescaled_log_r,xlabel='cHWtil',ylabel='Rescaled -2*log_r',do_log=False)

      if 'sally' in args.mode:
        log_r_histo.savefig(f'{args.plot_dir}/limits/log_r_curve_sally_{args.channel}_{args.sample_type}_{args.sally_observables}_lumi{args.lumi}_{i}.pdf')
      else:
        if args.observable_y is None:
          log_r_histo.savefig(f'{args.plot_dir}/limits/log_r_curve_{args.channel}_{args.sample_type}_{args.observable_x}_{len(args.binning_x)-1}bins_lumi{args.lumi}_{i}.pdf')
        else:
          log_r_histo.savefig(f'{args.plot_dir}/limits/log_r_curve_{args.channel}_{args.sample_type}_{args.observable_x}_{len(args.binning_x)-1}bins_{args.observable_y}_{len(args.binning_y)-1}bins_lumi{args.lumi}_{i}.pdf')
    
    central_value,cl_68,cl_95=extract_limits_single_parameter(parameter_grid,p_values,index_best_point)
    logging.debug(f'n_fit: {str(i)}; central value: {str(central_value)}; 68% CL: {str(cl_68)}; 95% CL: {str(cl_95)}')
    list_central_values.append(central_value[0]) 

    if 'sally' in args.mode:
      log_file.write(f"{args.sally_observables},{args.sally_model},{str(args.binning_x).replace(',',' ')},{str(central_value[0])}, {str(cl_68).replace(',',' ')},{str(cl_95).replace(',',' ')} \n") 
    else:
      log_file.write(f"{args.observable_x},{str(args.binning_x).replace(',',' ')},{args.observable_y},{str(args.binning_y).replace(',',' ')},{str(central_value[0]).replace(',',' ')},{str(cl_68).replace(',',' ')},{str(cl_95).replace(',',' ')} \n") 

  logging.debug("list of central values : "+str(list_central_values))

  log_file.close()