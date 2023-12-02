import csv
import pandas as pd
pd.set_option('display.max_colwidth', None)

def clean_whitespace(df):
    df.columns=[col.strip() for col in df.columns]
    df["observable_x"]=[col.strip() for col in df["observable_x"]]
    df["observable_y"]=[col.strip() for col in df["observable_y"]]
    return df

def format_observable_latex_histograms(df_series):

    dict_text_to_latex={
        "pt_w" : r"\ptw",
        "mt_tot" : r"\mttot",
        "ql_cos_deltaPlus" : r"\qlCDPlus",
        "ql_cos_deltaMinus" : r"\qlCDMinus",
    }

    if df_series['observable_x'] not in dict_text_to_latex:
        return df_series['observable_x']

    str_latex=fr"${dict_text_to_latex[df_series['observable_x']]}"
    str_latex+=fr" \in {df_series['binning_x']}"
    if df_series['observable_x']=='mt_tot' or df_series['observable_x']=='pt_w':
        str_latex+=fr" \:\GeV"
    
    if df_series['observable_y'].strip() == "None":
        str_latex+="$"
        return str_latex.replace("[","\left[").replace("]","\right]")
    
    str_latex='\makecell[l]{'+str_latex
    str_latex+=fr" \otimes$ \\ ${dict_text_to_latex[df_series['observable_y']]}"
    str_latex+=fr" \in {df_series['binning_y']}"

    if df_series['observable_y']=='mt_tot' or df_series['observable_y']=='pt_w':
        str_latex+=fr" \:\GeV"

    str_latex+="$}"
    return str_latex.replace("[","\left[").replace("]","\right]")

def format_observable_latex_sally(df_series):

    dict_sally_model_to_latex={
        "kinematic_only" : "SALLY, w/ detector-level observables",
        "all_observables_remove_redundant_cos" : r"\makecell[l]{SALLY, w detector-level observables + \\ $\pznu$ and $\qlCDPlus$}",
    }

    str_latex=fr"${dict_sally_model_to_latex[df_series['observables']]}"
    str_latex+=fr" \in {df_series['binning_x']}"
    str_latex.replace("[","\left[").replace("]","\right]")

    #if df_series['observable_y'].strip() == "None":
    str_latex+="$"
    
    return str_a



plot_dir="/lstore/titan/atlas/rbarrue/HWW_tensor_structure/HWW_tensor_structure_cernbox/madminer_WH/CPoddOnly_fullME2_PythiaDelphes/"
sample_type="withBackgrounds"
lumi=300

df_FI_histograms=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/fisherInfo_histograms_{sample_type}_lumi{lumi}_inclusive.csv'))
df_full_histograms=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/full_histograms_{sample_type}_lumi{lumi}.csv'))
df_full_histograms_shape_only=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/full_histograms_{sample_type}_lumi{lumi}_shape_only.csv'))

df_limits_latex_histograms=pd.DataFrame()

df_limits_latex_histograms["Observable"]=[format_observable_latex_histograms(row) for _,row in df_FI_histograms.iterrows()]

df_limits_latex_histograms.drop(0,inplace=True) # remove the 'rate' column
df_limits_latex_histograms.reset_index(drop=True, inplace=True)
df_limits_latex_histograms["Linearized limits"]=[f'${limit}$' for limit in df_full_histograms["95% CL"]]
df_limits_latex_histograms["Full limits"]=[f'${limit}$' for limit in df_full_histograms["95% CL"]]
df_limits_latex_histograms["Full limits (shape-only)"]=[f'${limit}$' for limit in df_full_histograms_shape_only["95% CL"]]

latex_file_histograms=open(f"{plot_dir}/limits/limits_histograms_{sample_type}_lumi{lumi}.tex","w")
latex_file_histograms.write(df_limits_latex_histograms.to_latex(index=False,escape=False,bold_rows=True,multirow=False))

df_FI_sally=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/fisherInfo_sally_{sample_type}_lumi{lumi}_inclusive.csv'))
df_full_sally=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/full_sally_{sample_type}_lumi{lumi}.csv'))
df_full_sally_shape_only=clean_whitespace(pd.read_csv(f'{plot_dir}/limits/full_sally_{sample_type}_lumi{lumi}_shape_only.csv'))