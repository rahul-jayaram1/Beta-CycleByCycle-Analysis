from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
from bycycle.filt import bandpass_filter
from bycycle.cyclepoints import _fzerorise, _fzerofall, find_extrema

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import statistics
import math
import matplotlib
import random
import sys

import os
import scipy.io
import scipy.signal

from scipy import stats
import seaborn as sns
#from ggplot import ggplot, geom_line, aes

plt.rcParams.update({'font.size': 5})

f_theta = (10,22)
#filename = 'full_trials_allsubjs_troughdata_det_power_signals'
filename = 'full_trials_allsubjs_troughdata'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)

subj = 9 #Python Index
trial_number = 74 #Python Index
trough_index = 264 #MATLAB Value
myDelta = 0.005 #window to make sure CBC marked minima is within some time value of find method marked beta event
n_oscilations = 2 #use for windowing dataframes (number of oscilations before and after beta event)

#srate = data['Fs']
Fs = 600 # sampling rate
all_signals=data['all_data']
trough_data=data['all_trough_data']
trough_data = trough_data[0]
print(len(trough_data))
trough_data = trough_data[subj]
trough_data_indexes = trough_data[:,1]
trough_data_hits = trough_data[:,3]
trough_data_trial_numbers = trough_data[:,0]
print(trough_data_indexes)
print(trough_data_trial_numbers)

trough_params_ind = trough_data[subj] #row (1,2,3,4) of trough quantifications

signalsInd = all_signals[subj,:]
signalInd_avg = signalsInd.mean(axis=0)
signal_raw = signalInd_avg #use when plotting fit on avg signal
left_edge =60 #change to alter added edge length
right_edge = len(signal_raw) - left_edge

print(len(signal_raw))

ind_signal = signalsInd[trial_number] #change to determine indivual trial nunmber

#signal=signal.transpose() #transpose if time vector and signal are not same dimensions
t = data['tVec']

t_raw= t[0]

f_lowpass = 60
N_seconds = 0.40


signalInd_lowpass = lowpass_filter(ind_signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

plt.figure(figsize=(8,6))
plt.plot(t_raw,signalInd_avg)
plt.title('subj '+str(subj)+' average signal')
plt.savefig('average_signal_subj' + str(subj))
plt.show()

plt.figure(figsize=(8,6))
plt.plot(t_raw,ind_signal,'k')
plt.plot(t_raw, signalInd_lowpass,'b')
plt.title('Ind signal raw')
plt.show()
plt.plot()

neg = 0

def signal_addEdges(signal_raw):  #function to add reflected signal to edges to reduce edge artifacts
    right_edge = len(signal_raw) - left_edge
    if neg == 1:
        sig_left_edge = (-1*signal_raw[left_edge::-1])
        my_signal = np.concatenate([sig_left_edge, signal_raw])
        signal = np.concatenate([my_signal, (-1*signal_raw[:right_edge:-1])])
    elif neg == 0:
        sig_left_edge = (signal_raw[left_edge::-1])
        my_signal = np.concatenate([sig_left_edge, signal_raw])
        signal = np.concatenate([my_signal, (signal_raw[:right_edge:-1])])

    return signal

signal = signal_addEdges(signal_raw)
sig_time = (len(signal_raw)/Fs)

#adjust time vector to account for edge artifacts
t1=t_raw
t2=t1[right_edge:]-sig_time #subtract time length of signal
t3=t1[:left_edge]+sig_time
t = np.concatenate([t2,t1])
t = np.concatenate([t,t3])
t = t+(sig_time/2)+sig_time
tlim = (0, 3*sig_time) #time interval in seconds
tidx = np.logical_and(t>=tlim[0], t<tlim[1])

#from CBC code: marking peaks, troughs, and midpoints in waveform
N_seconds = 0.59

signal_low= lowpass_filter(signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

N_seconds_theta = 0.02
signal_narrow = bandpass_filter(signal, Fs, f_theta, remove_edge_artifacts=False, N_seconds=N_seconds_theta)

Ps, Ts = find_extrema(signal, Fs, f_theta, filter_kwargs={'N_cycles': 3})

tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]

from bycycle.cyclepoints import find_zerox
zeroxR, zeroxD = find_zerox(signal_low, Ps, Ts)

tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)] #trough time points
tidxDs = zeroxD[np.logical_and(zeroxD>tlim[0]*Fs, zeroxD<tlim[1]*Fs)]
tidxRs = zeroxR[np.logical_and(zeroxR>tlim[0]*Fs, zeroxR<tlim[1]*Fs)]

t_raw_scale = t - (1.5*sig_time)

#example plot of one waveform
plt.figure(figsize=(8, 6))
plt.plot(t_raw_scale[tidx], signal_low[tidx], 'k')
plt.plot(t_raw_scale[tidx],signal_narrow[tidx],'g')
plt.plot(t_raw_scale[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t_raw_scale[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.plot(t_raw_scale[tidxDs], signal_low[tidxDs], 'm.', ms=10)
plt.plot(t_raw_scale[tidxRs], signal_low[tidxRs], 'g.', ms=10)
plt.plot(t_raw_scale[trough_index-1+left_edge], signal_low[trough_index-1+left_edge], 'y.',ms=20)
x_coords = [(sig_time/2)+sig_time+t_raw[0]-1.5, sig_time/2+(2*sig_time)+t_raw[0]-1.5]
for xc in x_coords:
    plt.axvline(x=xc)
#plt.xlim((tlim))
plt.title('Cycle-by-cycle on Reflected Signal')
plt.show()

#compute cycle features and create matrix of CBC values for each oscilation
from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params

burst_kwargs = {'amplitude_fraction_threshold': .18,
                'amplitude_consistency_threshold': .09,
                'period_consistency_threshold': .09,
                'monotonicity_threshold': .1 ,
                'N_cycles_min': 1}

#burst detect individual trials
#create a dataframe of CBC feature values for each oscilation
df_ind = compute_features(signal, Fs, f_theta,
center_extrema='T', burst_detection_kwargs=burst_kwargs)
plot_burst_detect_params(signal, Fs, df_ind, burst_kwargs, figsize=(12,1,5))
#df_ind = df_ind[df_ind['is_burst']]
print("dfIND VALUES")
print(len(signal_raw))
print(len(signal_raw)-left_edge)
print(len(df_ind))

#you can index specific features from the dataframe
print(df_ind['sample_trough'])
print(df_ind['sample_trough'][0])
print(df_ind['amp_fraction'])
print(tidxTs)
print(Ts)

dropBeta = True #set true if you don't want beta event oscilation in windowed dataframe
#
# def minima_quality_check(Troughs, t_raw, trough_index_def, delta):
#     for t in Troughs:
#         if abs(t_raw[t] - t_raw[trough_index_def-1+left_edge]) < delta:
#             return("CBC min within MATLAB min passed!")
#     return("CBC min -> MATLAB min NOT passed")

#look at specific number of oscilations before and after the beta event: create function that windows a trial around the beta event
def windowDf(input_df, input_trough, delta, n_oscilations):
    for cycle in range(len(input_df)):
        if cycle>=n_oscilations+1 and (cycle+n_oscilations+1)<=len(input_df):
            #NOTE: abs(input_df['sample_trough'][cycle]-input_trough-left_edge)<delta*Fs was removed as another condition for the above if statement
            trough_cycle = cycle
            temp_df = input_df.drop(np.arange(0,trough_cycle-n_oscilations), axis = 0)
            return_df = temp_df.drop(np.arange(trough_cycle+n_oscilations+1,len(input_df)), axis = 0)
            if dropBeta == True:
                return_df = return_df.drop([cycle], axis = 0)
                return(return_df)
            else:
                return(return_df)
#
# print(df_ind)
# df_ind_windowed = windowDf(df_ind, trough_index, myDelta, 2)
# print(df_ind_windowed)
# print(df_ind_windowed['sample_trough'])


 # matrix with all relevant values
#features_mat.to_csv(r'C:\Users\rjayaram\Desktop\Research\cycle-by-cycle\matrix_s10.csv',index = True, header = True)
#df.to_excel('300_300_GrandAvg_peak2peak_HighPower_subj'+''.xlsx')

#
# plot_burst_detect_params(signal, Fs, df, burst_kwargs, figsize=(12,1.5))
# my_xcoords = [left_edge, t[-1]-left_edge]
# for xc in x_coords:
#     plt.axvline(x=xc)

#burst detect indivual subject average
# df_ind_avg = compute_features(signal_addEdges(signalInd_avg), Fs, f_theta,
# center_extrema='T', burst_detection_kwargs=burst_kwargs)
# plt.title('avg signal '+str(subj))
#plot_burst_detect_params(signal_addEdges(signalInd_avg), Fs, df_ind_avg, burst_kwargs, figsize=(12,1.5))

#plot trial with oscilation markings
plt.figure(figsize=(8, 6))
# plt.plot(t_raw, sig0, color = '0.75')
# plt.plot(t_raw, sig1, color = '0.75')
# plt.plot(t_raw, sig2, color = '0.75')
# plt.plot(t_raw, sig3, color = '0.75')
# plt.plot(t_raw, sig4, color = '0.75')
# plt.plot(t_raw, sig5, color = '0.75')
# plt.plot(t_raw, sig6, color = '0.75')
# plt.plot(t_raw, sig7, color = '0.75')
# plt.plot(t_raw, sig8, color = '0.75')
# plt.plot(t_raw, sig9, color = '0.75')
plt.plot(t_raw_scale[tidx], signal_low[tidx], 'k')
plt.plot(t_raw_scale[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t_raw_scale[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.plot(t_raw_scale[tidxDs], signal_low[tidxDs], 'm.', ms=10)
plt.plot(t_raw_scale[tidxRs], signal_low[tidxRs], 'g.', ms=10)
plt.plot(t_raw_scale[trough_index-1+left_edge], signal_low[trough_index-1+left_edge], 'y.',ms=20)
#subtract one since data is indexed in MATLAB
#subtract left edge to align matlab index with edged tine vector
plt.xlim((t_raw[0], t_raw[-1]))
plt.title('Cycle-by-cycle fitting subj '+str(subj)+' trial'+str(trial_number))
#plt.savefig('Cycle-by-cycle fitting pretr subj' + str(subj))
plt.show()

print('trough time values')
print(t_raw_scale[tidxTs])

trough_data_time = None #try to work in this variable when building larger loop across trials
 #time in seconds

#check whether CBC marked minima is within some time value of find method marked beta event
def minima_quality_check(Troughs, t_raw, trough_index_def, delta):
    for t in Troughs:
        if abs(t_raw[t] - t_raw[trough_index_def-1+left_edge]) < delta:
            return("CBC min within MATLAB min passed!")
    return("CBC min -> MATLAB min NOT passed")


checks = 0

# windowed_dfs=[]
# for event in range(len(trough_data_trial_numbers)):
#     trial_num = int(trough_data[event,0]-1)
#     trough_index_temp = int(trough_data_indexes[event]) #index is MATLAB VALUES
#     signal_temp = signal_addEdges(signalsInd[trial_num])
#     non_windowed_df = compute_features(signal_addEdges(signalsInd[trial_num]), Fs, f_theta,
#     center_extrema='T', burst_detection_kwargs=burst_kwargs)
#     windowed_df = windowDf(non_windowed_df, trough_index_temp, myDelta, 2)
#     # print(non_windowed_df)
#     # print(windowed_df)
#     windowed_dfs.append(windowed_df)
# df_all_cycles_windowed = pd.concat(windowed_dfs)
#
#
# #df_subjects = df_cycles_windowed.groupby(['group','trial_number']).mean()[features_keep].reset_index(
#
# features_keep = ['period', 'amp_fraction','time_rdsym', 'time_ptsym']
# for feature in features_keep:
#     h = sns.histplot(x=feature, data = df_cycles_windowed)
#     plt.xlabel(feature,fontsize=15)
#     plt.ylabel('count',fontsize=15)
#     plt.xticks(size=12)
#     plt.yticks(size=12)
#     plt.title("subj "+str(subj)+" "+feature, fontsize=18)
#     plt.savefig("subj_"+str(subj)+"_"+feature+"_distribution")
#     plt.show()

#looping trough all trials within a single subject
windowed_dfs = []
preBeta_dfs = []
postBeta_dfs = []
for event in range(len(trough_data_trial_numbers)):
    trial_num = int(trough_data[event,0]-1)
    trough_index_temp = int(trough_data_indexes[event]) #index is MATLAB VALUES
    signal_temp = signal_addEdges(signalsInd[trial_num])
    non_windowed_df = compute_features(signal_addEdges(signalsInd[trial_num]), Fs, f_theta,
    center_extrema='T', burst_detection_kwargs=burst_kwargs)
    windowed_df = windowDf(non_windowed_df, trough_index_temp, myDelta, n_oscilations)
    if str(type(windowed_df)) == "<class 'pandas.core.frame.DataFrame'>":
        windowed_dfs.append(windowed_df)
        preBeta_df = windowed_df.iloc[np.arange(n_oscilations)]
        preBeta_dfs.append(preBeta_df)
        postBeta_df = windowed_df.iloc[np.arange(n_oscilations, n_oscilations*2)]
        postBeta_dfs.append(postBeta_df)
df_all_cycles_windowed = pd.concat(windowed_dfs) #all dfs from all trials in a given subject
df_preBeta_windowed = pd.concat(preBeta_dfs) #pre beta dfs '''
df_postBeta_windowed = pd.concat(postBeta_dfs)#post beta dfs '''

print("DF LENGTHS")
print(len(df_preBeta_windowed))
print(len(df_postBeta_windowed))

#create histograms comparing pre vs post in a given subject
features_keep = ['period', 'amp_fraction','time_rdsym', 'time_ptsym', 'monotonicity', 'band_amp'] #features of interest
for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_preBeta_windowed)
    h_post = sns.histplot(x=feature, data = df_postBeta_windowed, color = 'orange')
    plt.xlabel(feature,fontsize=15)
    plt.xlabel(stats.wilcoxon(df_preBeta_windowed[feature],df_postBeta_windowed[feature]), fontsize = 10) #computer SR wilcoxon p value
    plt.ylabel('count',fontsize=15)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("subj "+str(subj)+" "+feature, fontsize=18)
    plt.savefig("subj_"+str(subj)+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()

#printing p value to terminal for each feature
print('subj ' + str(subj)+' Wilcoxon values')
stat_period, p_val_period = stats.wilcoxon(df_preBeta_windowed['period'], df_postBeta_windowed['period'])
print('period:')
print(stat_period, p_val_period)
print('')

stat_ampF, p_val_ampF = stats.wilcoxon(df_preBeta_windowed['amp_fraction'], df_postBeta_windowed['amp_fraction'])
print('amp_fraction:')
print(stat_ampF, p_val_ampF)
print('')

stat_rdsym, p_val_rsdym = stats.wilcoxon(df_preBeta_windowed['time_rdsym'], df_postBeta_windowed['time_rdsym'])
print('time_rdsym:')
print(stat_rdsym, p_val_rsdym)
print('')

stat_ptsym, p_val_ptsym = stats.wilcoxon(df_preBeta_windowed['time_ptsym'], df_postBeta_windowed['time_ptsym'])
print('time_ptsym:')
print(stat_ptsym, p_val_ptsym)
print('')

stat_monot, p_val_monot = stats.wilcoxon(df_preBeta_windowed['monotonicity'], df_postBeta_windowed['monotonicity'])
print('monotonicity:')
print(stat_monot, p_val_monot)
print('')

stat_bandAmp, p_val_bandAmp = stats.wilcoxon(df_preBeta_windowed['band_amp'], df_postBeta_windowed['band_amp'])
print('band_amp:')
print(stat_bandAmp, p_val_bandAmp)
print('')

windowed_dfs_all = []
preBeta_dfs_all = []
postBeta_dfs_all = []

#median values for pre vs post
pre_median_bandAmps = []
post_median_bandAmps = []

pre_median_rdsym = []
post_median_rdsym = []

pre_median_ptsym =[]
post_median_ptsym = []

pre_median_mono=[]
post_median_mono=[]

pre_median_periods=[]
post_median_periods =[]

pre_median_ampConst=[]
post_median_ampConst=[]

pre_median_ampFrac = []
post_median_ampFrac = []

pre_median_periodConst=[]
post_median_periodConst=[]


#separating hits and misses; initialize empty arrays
#median values for hits vs misses
hits_dfs_all = []
miss_dfs_all = []

hit_median_bandAmps = []
miss_median_bandAmps = []

hit_median_rdsym = []
miss_median_rdsym = []

hit_median_ptsym =[]
miss_median_ptsym = []

hit_median_mono=[]
miss_median_mono=[]

hit_median_periods=[]
miss_median_periods =[]


pre_hits_dfs_all = []
post_hits_dfs_all = []

pre_miss_dfs_all = []
post_miss_dfs_all = []

#hits vs miss WITH pre vs post
#initialize median value arrays
pre_hits_median_bandAmp=[]
post_hits_median_bandAmp=[]
pre_miss_median_bandAmp=[]
post_miss_median_bandAmp=[]

pre_hits_median_rdsym=[]
post_hits_median_rdsym=[]
pre_miss_median_rdsym=[]
post_miss_median_rdsym=[]

pre_hits_median_ptsym=[]
post_hits_median_ptsym=[]
pre_miss_median_ptsym=[]
post_miss_median_ptsym=[]


pre_hits_median_periods=[]
post_hits_median_periods=[]
pre_miss_median_periods=[]
post_miss_median_periods=[]

pre_hits_median_mono=[]
post_hits_median_mono=[]
pre_miss_median_mono=[]
post_miss_median_mono=[]


subj_trough_data = data['all_trough_data'][0]
subj_trough_data = subj_trough_data[subj]
#separating hits from misses in a given subject
where_zeros = np.where(subj_trough_data[:,3]==0)
where_ones = np.where(subj_trough_data[:,3]==1)
print(where_ones[0])
trough_zeros = subj_trough_data[:,3][where_zeros[0]]
trough_ones = subj_trough_data[:,3][where_ones[0]]
matrix_zeros = subj_trough_data[:len(trough_ones),:]
matrix_ones = subj_trough_data[len(trough_ones):,:]

print('matrix 0 and 1')
print(subj_trough_data[:,3])
print(matrix_zeros[:,3])
print(matrix_ones[:,3])


#sanity check to ensure hits and misses are being split correctly (system exits if error)
if len(subj_trough_data[:,3]) != len(matrix_zeros[:,3])+len(matrix_ones[:,3]):
    print('error splittig 1s and 0s')
    sys.exit()
#create new normalized trough data matrix such that number of hits and misses are the same
if len(matrix_ones[:,3])<=len(matrix_zeros[:,3]):
    min_value = len(matrix_ones[:,3])
    index_array = np.random.randint(len(matrix_zeros[:,3]), size = min_value) #index vector of random ints from range 0 to min_value used to randomly select trials from zeros
    matrix_zeros = matrix_zeros[index_array]
    print('more 0 than 1')
else: #same thing as above but if zeros<ones
    min_value = len(matrix_zeros[:,3])
    index_array = np.random.randint(len(matrix_ones[:,3]), size = min_value)
    matrix_ones = matrix_ones[index_array]
    print('more 1 than 0')

if len(matrix_ones)!=len(matrix_zeros):
    print('error selecting random trials')
    sys.exit()

#create new normalized matrix w/ randomly selected trials
norm_trough_data = np.concatenate((matrix_zeros, matrix_ones), axis =0)

#loop through all trials in a given subject (use new normalized trough data)
subj_trial_numbers = norm_trough_data[:,0]
subj_trough_indexes = norm_trough_data[:,1]
subj_hits = norm_trough_data[:,3]
subj_windowed_dfs = []
subj_preBeta_dfs = []
subj_postBeta_dfs = []
count = 0
for event in range(len(norm_trough_data)):
    trial_num = int(subj_trial_numbers[event]-1)
    trough_index = int(subj_trough_indexes[event]) #index is MATLAB VALUES
    signal_temp = signal_addEdges(signalsInd[trial_num])
    subj_non_windowed_df = compute_features(signal_temp, Fs, f_theta,
    center_extrema='T', burst_detection_kwargs=burst_kwargs)
    subj_windowed_df = windowDf(subj_non_windowed_df, trough_index, myDelta, n_oscilations)
    if str(type(subj_windowed_df)) == "<class 'pandas.core.frame.DataFrame'>":
        subj_windowed_dfs.append(subj_windowed_df)
        preBeta_df = subj_windowed_df.iloc[np.arange(n_oscilations)]
        subj_preBeta_dfs.append(preBeta_df)
        postBeta_df = subj_windowed_df.iloc[np.arange(n_oscilations, n_oscilations*2)]
        subj_postBeta_dfs.append(postBeta_df)
        if int(norm_trough_data[:,3][event])==1:
            hitsPre = preBeta_df
            hitsPost = postBeta_df
            pre_hits_dfs_all.append(hitsPre)
            post_hits_dfs_all.append(hitsPost)
        elif int(norm_trough_data[:,3][event])==0:
            missPre = preBeta_df
            missPost = postBeta_df
            pre_miss_dfs_all.append(missPre)
            post_miss_dfs_all.append(missPost)

    else:
        print(event)
        print(norm_trough_data[event])
        print(subj_windowed_df)
        print(type(subj_windowed_df))
        print('error: not df type') #bug still there where if we use myDelta sanity check, dfs saved as none type
        count+=1
        print(count)
        sys.exit()

print(norm_trough_data[:,3])

#concatenate all hit and miss oscilation vectors into a single matrix
concat_pre_hits_dfs_all = pd.concat(pre_hits_dfs_all)
concat_post_hits_dfs_all = pd.concat(post_hits_dfs_all)

concat_pre_miss_dfs_all = pd.concat(pre_miss_dfs_all)
concat_post_miss_dfs_all = pd.concat(post_miss_dfs_all)

#histogram comparisons of hits vs misses (separate pre and post)
#pre (hits vs misses)
binsize = 50
plt.hist(concat_pre_hits_dfs_all['band_amp'], bins = binsize, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['band_amp'], bins = binsize, alpha=0.5, label = 'Pre Misses')
print(len(concat_pre_hits_dfs_all['band_amp']))
print(len(concat_pre_miss_dfs_all['band_amp']))
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['band_amp'],concat_pre_miss_dfs_all['band_amp'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn BandAmp pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_pre_hitsVSmiss_extrema_hits"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['time_rdsym'], bins = binsize, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['time_rdsym'], bins = binsize, alpha=0.5, label = 'Pre Misses')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['time_rdsym'],concat_pre_miss_dfs_all['time_rdsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_RDSYM pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_pre_hitsVSmiss_extrema_hits"+"_RDSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['time_ptsym'], bins = binsize, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['time_ptsym'], bins = binsize, alpha=0.5, label = 'Pre Misses')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['time_ptsym'],concat_pre_miss_dfs_all['time_ptsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_PTSYM pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_pre_hitsVSmiss_extrema_hits"+"_PTSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['monotonicity'], bins = binsize, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['monotonicity'], bins = binsize, alpha=0.5, label = 'Pre Misses')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['monotonicity'],concat_pre_miss_dfs_all['monotonicity'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_Monotonicity pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_pre_hitsVSmiss_extrema_hits"+"_Monotonicity"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['period'], bins = binsize, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['period'], bins = binsize, alpha=0.5, label = 'Pre Misses')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['period'],concat_pre_miss_dfs_all['period'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_Period pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_pre_hitsVSmiss_extrema_hits"+"_Periods"+"_Osc"+str(n_oscilations))
plt.show()


#post (hits vs misses)
plt.hist(concat_post_hits_dfs_all['band_amp'], bins = binsize, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['band_amp'], bins = binsize, alpha=0.5, label = 'Post Misses')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['band_amp'],concat_post_miss_dfs_all['band_amp'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_BandAmp post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_post_hitsVSmiss_extrema_hits"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['time_rdsym'], bins = binsize, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['time_rdsym'], bins = binsize, alpha=0.5, label = 'Post Misses')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['time_rdsym'],concat_post_miss_dfs_all['time_rdsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_RDSYM post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_post_hitsVSmiss_extrema_hits"+"_RDSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['time_ptsym'], bins = binsize, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['time_ptsym'], bins = binsize, alpha=0.5, label = 'Post Misses')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['time_ptsym'],concat_post_miss_dfs_all['time_ptsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_PTSYM post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_post_hitsVSmiss_extrema_hits"+"_PTSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['monotonicity'], bins = binsize, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['monotonicity'], bins = binsize, alpha=0.5, label = 'Post Misses')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['monotonicity'],concat_post_miss_dfs_all['monotonicity'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_Monotonicity post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_post_hitsVSmiss_extrema_hits"+"_Monotonicity"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['period'], bins = binsize, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['period'], bins = binsize, alpha=0.5, label = 'Post Misses')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['period'],concat_post_miss_dfs_all['period'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('subj '+str(subj)+' HP attn_Period post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('subj_'+str(subj)+"hp_attn_post_hitsVSmiss_extrema_hits"+"_Periods"+"_Osc"+str(n_oscilations))
plt.show()




sys.exit()











for sub in range(10):
    trough_data_all=data['all_trough_data']
    trough_data_loop_all = trough_data_all[0]
    trough_data_loop_all = trough_data_loop_all[sub] #matrix of trough values for a single subject

    where_zeros = np.where(trough_data_loop_all==0)
    where_ones = np.where(trough_data_loop_all==1)
    trough_zeros = trough_data_loop_all[:,3][where_zeros[0]]
    trough_ones = trough_data_loop_all[:,3][where_ones[0]]
    matrix_zeros = trough_data_loop_all[:len(trough_zeros),:]
    matrix_ones = trough_data_loop_all[len(trough_zeros):,:]
    print('red')


    sorted_matrix_zeros = matrix_zeros[np.argsort(matrix_zeros[:,4])]
    sorted_matrix_ones = matrix_ones[np.argsort(matrix_ones[:,4])]

    print(sorted_matrix_zeros[:,3])
    print(sorted_matrix_ones[:,3])

    #dont sort here but normalize by randomly selecting size smaller from larger of 0s and 1s
    if len(sorted_matrix_ones[:,3]) <= len(sorted_matrix_zeros[:,3]):
        min_value = len(matrix_ones[:,3])
        max_value = len(matrix_zeros[:,3])
        sorted_matrix_zeros = sorted_matrix_zeros[:min_value,:]
        #index trials that are used when cutting down size of larger index_array
        index_array = np.random.randint(len(matrix_ones), size = min_value)
        matrix_zeros = matrix_zeros[index_array]

    elif len(sorted_matrix_ones[:,3]) > len(sorted_matrix_zeros[:,3]):
        min_value = len(matrix_zeros[:,3])
        max_value = len(matrix_ones[:,3])
        sorted_matrix_ones = sorted_matrix_ones[:min_value,:]
        index_array = np.random.randint(len(matrix_zeros), size = min_value)
        matrix_ones = matrix_ones[index_array]

    print(len(sorted_matrix_zeros[:,3])==len(sorted_matrix_ones[:,3]))
    hp_matrix = np.concatenate((matrix_zeros, matrix_ones), axis =0)

    #reassign trough_data_loop matrix variable to hp_matrix values
    trough_data_loop = hp_matrix
    print(hp_matrix[:,3])

    trough_data_trial_numbers_loop = trough_data_loop[:,0]
    trough_data_indexes_loop = trough_data_loop[:,1]
    trough_data_hits_loop = trough_data_loop[:,3]
    print('grape')
    print(trough_data_hits_loop)
    print(len(trough_data_trial_numbers_loop))

    signalsInd_loop = all_signals[sub,:]
    ##index highest power events here
    #find minimum number of 0s and 1s in hits array
    # where_zeros = np.where(trough_data_loop==0)
    # where_ones = np.where(trough_data_loop==1)
    # trough_zeros = trough_data_hits_loop[where_zeros[0]]
    # trough_ones = trough_data_hits_loop[where_ones[0]]
    # matrix_zeros = trough_data_loop[:len(trough_zeros),:]
    # matrix_ones = trough_data_loop[len(trough_zeros):,:]
    #
    # sorted_matrix_zeros = matrix_zeros[np.argsort(matrix_zeros[:,4])]
    # sorted_matrix_ones = matrix_ones[np.argsort(matrix_ones[:,4])]
    #
    # if len(matrix_ones[:,3]) <= len(matrix_zeros[:,3]):
    #     min_value = len(matrix_ones[:,3])
    #     max_value = len(matrix_zeros[:,3])
    #     sorted_matrix_zeros = sorted_matrix_zeros[:min_value,:]
    # else:
    #     min_value = len(matrix_zeros[:,3])
    #     max_value = len(matrix_ones[:,3])
    #     sorted_matrix_ones = sorted_matrix_ones[:min_value,:]
    #
    # hp_matrix = np.concatenate((sorted_matrix_zeros, sorted_matrix_ones), axis =0)

    #want to take min(len of zeros, len of ones), use to index hits and misses for each subject

    counter_h = 0
    counter_m = 0
    for event in range(len(trough_data_trial_numbers_loop)):
        trial_num = int(trough_data_loop[event,0]-1)
        trough_index_temp = int(trough_data_indexes_loop[event]) #index is MATLAB VALUES
        signal_temp = signal_addEdges(signalsInd_loop[trial_num])
        non_windowed_df = compute_features(signal_addEdges(signalsInd_loop[trial_num]), Fs, f_theta,
        center_extrema='T', burst_detection_kwargs=burst_kwargs)
        windowed_df = windowDf(non_windowed_df, trough_index_temp, myDelta, n_oscilations)
        if str(type(windowed_df)) == "<class 'pandas.core.frame.DataFrame'>":
            windowed_dfs_all.append(windowed_df)
            preBeta_df = windowed_df.iloc[np.arange(n_oscilations)]
            preBeta_dfs_all.append(preBeta_df) #all pre beta cycles across all subjects
            postBeta_df = windowed_df.iloc[np.arange(n_oscilations, n_oscilations*2)]
            postBeta_dfs_all.append(postBeta_df) #all post beta cycles across all subjects

            no_beta_df = preBeta_df.append(postBeta_df)

            ###compare dfs from hits vs misses rather than pre vs post
            print(trough_data_hits_loop[event])
            if int(hp_matrix[:,3][event])==1:
                hit_no_beta_df = no_beta_df
                hitsPre = preBeta_df
                hitsPost = postBeta_df
                pre_hits_dfs_all.append(hitsPre)
                post_hits_dfs_all.append(hitsPost)
                counter_h+=1
            elif int(hp_matrix[:,3][event])==0:
                miss_no_beta_df = no_beta_df
                missPre = preBeta_df
                missPost = postBeta_df
                pre_miss_dfs_all.append(missPre)
                post_miss_dfs_all.append(missPost)
                counter_m+=1


    #hits vs misses
    concat_pre_hits_dfs_all = pd.concat(pre_hits_dfs_all)
    print('plum')
    print(len(concat_pre_hits_dfs_all))
    concat_post_hits_dfs_all = pd.concat(post_hits_dfs_all)

    concat_pre_miss_dfs_all = pd.concat(pre_miss_dfs_all)
    print(len(concat_pre_miss_dfs_all))
    concat_post_miss_dfs_all = pd.concat(post_miss_dfs_all)

    pre_hits_median_bandAmp.append(np.nanmedian(concat_pre_hits_dfs_all['band_amp']))
    post_hits_median_bandAmp.append(np.nanmedian(concat_post_hits_dfs_all['band_amp']))
    pre_miss_median_bandAmp.append(np.nanmedian(concat_pre_miss_dfs_all['band_amp']))
    post_miss_median_bandAmp.append(np.nanmedian(concat_post_miss_dfs_all['band_amp']))

    pre_hits_median_rdsym.append(np.nanmedian(concat_pre_hits_dfs_all['time_rdsym']))
    post_hits_median_rdsym.append(np.nanmedian(concat_post_hits_dfs_all['time_rdsym']))
    pre_miss_median_rdsym.append(np.nanmedian(concat_pre_miss_dfs_all['time_rdsym']))
    post_miss_median_rdsym.append(np.nanmedian(concat_post_miss_dfs_all['time_rdsym']))

    pre_hits_median_ptsym.append(np.nanmedian(concat_pre_hits_dfs_all['time_ptsym']))
    post_hits_median_ptsym.append(np.nanmedian(concat_post_hits_dfs_all['time_ptsym']))
    pre_miss_median_ptsym.append(np.nanmedian(concat_pre_miss_dfs_all['time_ptsym']))
    post_miss_median_ptsym.append(np.nanmedian(concat_post_miss_dfs_all['time_ptsym']))

    pre_hits_median_mono.append(np.nanmedian(concat_pre_hits_dfs_all['monotonicity']))
    post_hits_median_mono.append(np.nanmedian(concat_post_hits_dfs_all['monotonicity']))
    pre_miss_median_mono.append(np.nanmedian(concat_pre_miss_dfs_all['monotonicity']))
    post_miss_median_mono.append(np.nanmedian(concat_post_miss_dfs_all['monotonicity']))

    pre_hits_median_periods.append(np.nanmedian(concat_pre_hits_dfs_all['period']))
    post_hits_median_periods.append(np.nanmedian(concat_post_hits_dfs_all['period']))
    pre_miss_median_periods.append(np.nanmedian(concat_pre_miss_dfs_all['period']))
    post_miss_median_periods.append(np.nanmedian(concat_post_miss_dfs_all['period']))

#dataframes sorted data by power separted by hits and misses
pre_sort_hits_BandAmp = np.sort(concat_pre_hits_dfs_all['band_amp'])
pre_extremes_hits_BandAmp = np.concatenate((pre_sort_hits_BandAmp[:int(len(pre_sort_hits_BandAmp)/3)],pre_sort_hits_BandAmp[int(len(pre_sort_hits_BandAmp)*2/3):]))
post_sort_hits_BandAmp = np.sort(concat_post_hits_dfs_all['band_amp'])
post_extremes_hits_BandAmp = np.concatenate((post_sort_hits_BandAmp[:int(len(post_sort_hits_BandAmp)/3)],post_sort_hits_BandAmp[int(len(post_sort_hits_BandAmp)*2/3):]))
pre_sort_miss_BandAmp = np.sort(concat_pre_miss_dfs_all['band_amp'])
pre_extremes_miss_BandAmp = np.concatenate((pre_sort_miss_BandAmp[:int(len(pre_sort_miss_BandAmp)/3)],pre_sort_miss_BandAmp[int(len(pre_sort_miss_BandAmp)*2/3):]))
post_sort_miss_BandAmp = np.sort(concat_post_miss_dfs_all['band_amp'])
post_extremes_miss_BandAmp = np.concatenate((post_sort_miss_BandAmp[:int(len(post_sort_miss_BandAmp)/3)],post_sort_miss_BandAmp[int(len(post_sort_miss_BandAmp)*2/3):]))

pre_sort_hits_rdsym = np.sort(concat_pre_hits_dfs_all['time_rdsym'])
pre_extremes_hits_rdsym = np.concatenate((pre_sort_hits_rdsym[:int(len(pre_sort_hits_rdsym)/3)],pre_sort_hits_rdsym[int(len(pre_sort_hits_rdsym)*2/3):]))
post_sort_hits_rdsym = np.sort(concat_post_hits_dfs_all['time_rdsym'])
post_extremes_hits_rdsym = np.concatenate((post_sort_hits_rdsym[:int(len(post_sort_hits_rdsym)/3)],post_sort_hits_rdsym[int(len(post_sort_hits_rdsym)*2/3):]))
pre_sort_miss_rdsym = np.sort(concat_pre_miss_dfs_all['time_rdsym'])
pre_extremes_miss_rdsym = np.concatenate((pre_sort_miss_rdsym[:int(len(pre_sort_miss_BandAmp)/3)],pre_sort_miss_rdsym[int(len(pre_sort_miss_rdsym)*2/3):]))
post_sort_miss_rdsym = np.sort(concat_post_miss_dfs_all['time_rdsym'])
post_extremes_miss_rdsym = np.concatenate((post_sort_miss_rdsym[:int(len(post_sort_miss_rdsym)/3)],post_sort_miss_rdsym[int(len(post_sort_miss_rdsym)*2/3):]))

pre_sort_hits_ptsym = np.sort(concat_pre_hits_dfs_all['time_ptsym'])
pre_extremes_hits_ptsym = np.concatenate((pre_sort_hits_ptsym[:int(len(pre_sort_hits_ptsym)/3)],pre_sort_hits_ptsym[int(len(pre_sort_hits_ptsym)*2/3):]))
post_sort_hits_ptsym = np.sort(concat_post_hits_dfs_all['time_ptsym'])
post_extremes_hits_ptsym = np.concatenate((post_sort_hits_ptsym[:int(len(post_sort_hits_ptsym)/3)],post_sort_hits_ptsym[int(len(post_sort_hits_ptsym)*2/3):]))
pre_sort_miss_ptsym = np.sort(concat_pre_miss_dfs_all['time_ptsym'])
pre_extremes_miss_ptsym = np.concatenate((pre_sort_miss_ptsym[:int(len(pre_sort_miss_ptsym)/3)],pre_sort_miss_ptsym[int(len(pre_sort_miss_ptsym)*2/3):]))
post_sort_miss_ptsym = np.sort(concat_post_miss_dfs_all['time_ptsym'])
post_extremes_miss_ptsym = np.concatenate((post_sort_miss_ptsym[:int(len(post_sort_miss_ptsym)/3)],post_sort_miss_ptsym[int(len(post_sort_miss_ptsym)*2/3):]))


pre_sort_hits_mono = np.sort(concat_pre_hits_dfs_all['monotonicity'])
pre_extremes_hits_mono = np.concatenate((pre_sort_hits_mono[:int(len(pre_sort_hits_mono)/3)],pre_sort_hits_mono[int(len(pre_sort_hits_mono)*2/3):]))
post_sort_hits_mono = np.sort(concat_post_hits_dfs_all['monotonicity'])
post_extremes_hits_mono = np.concatenate((post_sort_hits_mono[:int(len(post_sort_hits_mono)/3)],post_sort_hits_mono[int(len(post_sort_hits_mono)*2/3):]))
pre_sort_miss_mono = np.sort(concat_pre_miss_dfs_all['monotonicity'])
pre_extremes_miss_mono = np.concatenate((pre_sort_miss_mono[:int(len(pre_sort_miss_mono)/3)],pre_sort_miss_mono[int(len(pre_sort_miss_mono)*2/3):]))
post_sort_miss_mono = np.sort(concat_post_miss_dfs_all['monotonicity'])
post_extremes_miss_mono = np.concatenate((post_sort_miss_mono[:int(len(post_sort_miss_mono)/3)],post_sort_miss_mono[int(len(post_sort_miss_mono)*2/3):]))

pre_sort_hits_periods = np.sort(concat_pre_hits_dfs_all['period'])
pre_extremes_hits_periods = np.concatenate((pre_sort_hits_periods[:int(len(pre_sort_hits_periods)/3)],pre_sort_hits_periods[int(len(pre_sort_hits_periods)*2/3):]))
post_sort_hits_periods = np.sort(concat_post_hits_dfs_all['period'])
post_extremes_hits_periods = np.concatenate((post_sort_hits_periods[:int(len(post_sort_hits_periods)/3)],post_sort_hits_periods[int(len(post_sort_hits_periods)*2/3):]))
pre_sort_miss_periods = np.sort(concat_pre_miss_dfs_all['period'])
pre_extremes_miss_periods = np.concatenate((pre_sort_miss_periods[:int(len(pre_sort_miss_periods)/3)],pre_sort_miss_periods[int(len(pre_sort_miss_periods)*2/3):]))
post_sort_miss_periods = np.sort(concat_post_miss_dfs_all['period'])
post_extremes_miss_periods = np.concatenate((post_sort_miss_periods[:int(len(post_sort_miss_periods)/3)],post_sort_miss_periods[int(len(post_sort_miss_periods)*2/3):]))

##pre (hits vs misses)
plt.hist(concat_pre_hits_dfs_all['band_amp'], bins = 100, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['band_amp'], bins = 100, alpha=0.5, label = 'Pre Misses')
print(len(concat_pre_hits_dfs_all['band_amp']))
print(len(concat_pre_miss_dfs_all['band_amp']))
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['band_amp'],concat_pre_miss_dfs_all['band_amp'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn BandAmp pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_pre_hitsVSmiss_extrema_hits"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['time_rdsym'], bins = 100, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['time_rdsym'], bins = 100, alpha=0.5, label = 'Pre Misses')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['time_rdsym'],concat_pre_miss_dfs_all['time_rdsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_RDSYM pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_pre_hitsVSmiss_extrema_hits"+"_RDSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['time_ptsym'], bins = 100, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['time_ptsym'], bins = 100, alpha=0.5, label = 'Pre Misses')
plt.hist(pre_sort_hits_ptsym, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['time_ptsym'],concat_pre_miss_dfs_all['time_ptsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_PTSYM pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_pre_hitsVSmiss_extrema_hits"+"_PTSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['monotonicity'], bins = 100, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['monotonicity'], bins = 100, alpha=0.5, label = 'Pre Misses')
plt.hist(pre_sort_hits_mono, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['monotonicity'],concat_pre_miss_dfs_all['monotonicity'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_Monotonicity pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_pre_hitsVSmiss_extrema_hits"+"_Monotonicity"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_pre_hits_dfs_all['period'], bins = 100, alpha=0.5, label = 'Pre Hits')
plt.hist(concat_pre_miss_dfs_all['period'], bins = 100, alpha=0.5, label = 'Pre Misses')
plt.hist(pre_sort_hits_periods, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_pre_hits_dfs_all['period'],concat_pre_miss_dfs_all['period'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_Period pre hits vs pre misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_pre_hitsVSmiss_extrema_hits"+"_Periods"+"_Osc"+str(n_oscilations))
plt.show()


##post (hits vs misses)
plt.hist(concat_post_hits_dfs_all['band_amp'], bins = 100, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['band_amp'], bins = 100, alpha=0.5, label = 'Post Misses')
plt.hist(post_sort_hits_BandAmp, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['band_amp'],concat_post_miss_dfs_all['band_amp'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_BandAmp post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_post_hitsVSmiss_extrema_hits"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['time_rdsym'], bins = 100, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['time_rdsym'], bins = 100, alpha=0.5, label = 'Post Misses')
plt.hist(post_sort_hits_rdsym, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['time_rdsym'],concat_post_miss_dfs_all['time_rdsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_RDSYM post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_post_hitsVSmiss_extrema_hits"+"_RDSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['time_ptsym'], bins = 100, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['time_ptsym'], bins = 100, alpha=0.5, label = 'Post Misses')
plt.hist(post_sort_hits_ptsym, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['time_ptsym'],concat_post_miss_dfs_all['time_ptsym'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_PTSYM post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_post_hitsVSmiss_extrema_hits"+"_PTSYM"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['monotonicity'], bins = 100, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['monotonicity'], bins = 100, alpha=0.5, label = 'Post Misses')
plt.hist(post_sort_hits_mono, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['monotonicity'],concat_post_miss_dfs_all['monotonicity'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_Monotonicity post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_post_hitsVSmiss_extrema_hits"+"_Monotonicity"+"_Osc"+str(n_oscilations))
plt.show()

plt.hist(concat_post_hits_dfs_all['period'], bins = 100, alpha=0.5, label = 'Post Hits')
plt.hist(concat_post_miss_dfs_all['period'], bins = 100, alpha=0.5, label = 'Post Misses')
plt.hist(post_sort_hits_periods, bins = 100, alpha=0.15, label='All Data')
plt.xlabel(str(stats.wilcoxon(concat_post_hits_dfs_all['period'],concat_post_miss_dfs_all['period'])), size =11)
plt.ylabel('Count',size = 12)
plt.title('HP attn_Period post hits vs post misses (extremes) Osc'+str(n_oscilations), size =12)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("hp_attn_post_hitsVSmiss_extrema_hits"+"_Periods"+"_Osc"+str(n_oscilations))
plt.show()

print('apple')

sys.exit()


## all data
hits_median_bandAmpVals = [pre_hits_median_bandAmp, post_hits_median_bandAmp]
#pre vs post
plt.boxplot(hits_median_bandAmpVals)
plt.plot([1,2], hits_median_bandAmpVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Hits Median Band Amplitude Across Subjects', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_bandAmp, post_hits_median_bandAmp), fontsize = 12)
plt.savefig("Hits_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_bandAmpVals = [pre_miss_median_bandAmp, post_miss_median_bandAmp]
plt.boxplot(miss_median_bandAmpVals)
plt.plot([1,2], miss_median_bandAmpVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Misses Median Band Amplitude Across Subjects', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_bandAmp, post_miss_median_bandAmp), fontsize = 12)
plt.savefig("Miss_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_bandAmpVals = [pre_hits_median_bandAmp, pre_miss_median_bandAmp]
plt.boxplot(pre_median_bandAmpVals)
plt.plot([1,2], pre_median_bandAmpVals, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Pre Median Band Amplitude Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_bandAmp, pre_miss_median_bandAmp), fontsize = 12)
plt.savefig("Pre_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
post_median_bandAmpVals = [post_hits_median_bandAmp, post_miss_median_bandAmp]
plt.boxplot(post_median_bandAmpVals)
plt.plot([1,2], post_median_bandAmpVals, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Post Median Band Amplitude Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_bandAmp, post_miss_median_bandAmp), fontsize = 12)
plt.savefig("Post_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()


import sys
sys.exit()


#pre vs post
hits_median_rdsym = [pre_hits_median_rdsym, post_hits_median_rdsym]
plt.boxplot(hits_median_rdsym)
plt.plot([1,2], hits_median_rdsym, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Attn Hits Median RDSYM Across Subjects', fontsize = 12)
plt.ylabel('RDSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_rdsym, post_hits_median_rdsym), fontsize = 12)
plt.savefig("attn_Hits_median"+"_rdsym"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_rdsym = [pre_miss_median_rdsym, post_miss_median_rdsym]
plt.boxplot(miss_median_rdsym)
plt.plot([1,2], miss_median_rdsym, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Attn Misses Median RDSYM Across Subjects', fontsize = 12)
plt.ylabel('RDSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_rdsym, post_miss_median_rdsym), fontsize = 12)
plt.savefig("attn_Miss_median"+"_rdsym"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_rdsym = [pre_hits_median_rdsym, pre_miss_median_rdsym]
plt.boxplot(pre_median_rdsym)
plt.plot([1,2], pre_median_rdsym, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Attn Pre Median RDSYM Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('RDSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_rdsym, pre_miss_median_rdsym), fontsize = 12)
plt.savefig("attn_Pre_median"+"_rdsym"+"_Osc"+str(n_oscilations))
plt.show()
post_median_rdsym = [post_hits_median_rdsym, post_miss_median_rdsym]
plt.boxplot(post_median_rdsym)
plt.plot([1,2], post_median_rdsym, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Attn Post Median RDSYM Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('RDSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_rdsym, post_miss_median_rdsym), fontsize = 12)
plt.savefig("amplitude_fraction_thresholdPost_median"+"_rdysm"+"_Osc"+str(n_oscilations))
plt.show()

#pre vs post
hits_median_ptsym = [pre_hits_median_ptsym, post_hits_median_ptsym]
plt.boxplot(hits_median_ptsym)
plt.plot([1,2], hits_median_ptsym, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Attn_Hits Median PTSYM Across Subjects', fontsize = 12)
plt.ylabel('PTSYM', fontsize = 12)
print(hits_median_ptsym)
#plt.xlabel(stats.wilcoxon(pre_hits_median_ptsym, post_hits_median_ptsym), fontsize = 12)
plt.savefig("attn_Hits_median"+"_ptsym"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_ptsym = [pre_miss_median_ptsym, post_miss_median_ptsym]
plt.boxplot(miss_median_ptsym)
plt.plot([1,2], miss_median_ptsym, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Misses Median PTSYM Across Subjects', fontsize = 12)
plt.ylabel('PTSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_ptsym, post_miss_median_ptsym), fontsize = 12)
plt.savefig("Miss_median"+"_ptsym"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_ptsym = [pre_hits_median_ptsym, pre_miss_median_ptsym]
plt.boxplot(pre_median_ptsym)
plt.plot([1,2], pre_median_ptsym, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Pre Median PTSYM Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('PTSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_ptsym, pre_miss_median_ptsym), fontsize = 12)
plt.savefig("Pre_median"+"_ptsym"+"_Osc"+str(n_oscilations))
plt.show()
post_median_ptsym = [post_hits_median_ptsym, post_miss_median_ptsym]
plt.boxplot(post_median_ptsym)
plt.plot([1,2], post_median_ptsym, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Post Median PTSYM Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('PTSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_ptsym, post_miss_median_ptsym), fontsize = 12)
plt.savefig("Post_median"+"_ptsym"+"_Osc"+str(n_oscilations))
plt.show()

#pre vs post
hits_median_mono = [pre_hits_median_mono, post_hits_median_mono]
plt.boxplot(hits_median_mono)
plt.plot([1,2], hits_median_mono, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Hits Median Monotonicity Across Subjects', fontsize = 12)
plt.ylabel('Monotonicity', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_mono, post_hits_median_mono), fontsize = 12)
plt.savefig("Hits_median"+"_mono"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_mono = [pre_miss_median_mono, post_miss_median_mono]
plt.boxplot(miss_median_mono)
plt.plot([1,2], miss_median_mono, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Misses Median Monotonicity Across Subjects', fontsize = 12)
plt.ylabel('Monotonicity', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_mono, post_miss_median_mono), fontsize = 12)
plt.savefig("Miss_median"+"_mono"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_mono = [pre_hits_median_mono, pre_miss_median_mono]
plt.boxplot(pre_median_mono)
plt.plot([1,2], pre_median_mono, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Pre Median Monotonicity Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Monotonicity', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_mono, pre_miss_median_mono), fontsize = 12)
plt.savefig("Pre_median"+"_monotonicity"+"_Osc"+str(n_oscilations))
plt.show()
post_median_mono = [post_hits_median_mono, post_miss_median_mono]
plt.boxplot(post_median_mono)
plt.plot([1,2], post_median_mono, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Post Median Monotonicity Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Monotonicity', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_mono, post_miss_median_mono), fontsize = 12)
plt.savefig("Post_median"+"_monotonicity"+"_Osc"+str(n_oscilations))
plt.show()

#pre vs post
hits_median_periods = [pre_hits_median_periods, post_hits_median_periods]
plt.boxplot(hits_median_periods)
plt.plot([1,2], hits_median_periods, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Hits Median Periods Across Subjects', fontsize = 12)
plt.ylabel('Period', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_periods, post_hits_median_periods), fontsize = 12)
plt.savefig("Hits_median"+"_period"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_periods = [pre_miss_median_periods, post_miss_median_periods]
plt.boxplot(miss_median_periods)
plt.plot([1,2], miss_median_periods, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Misses Median Periods Across Subjects', fontsize = 12)
plt.ylabel('Period', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_periods, post_miss_median_periods), fontsize = 12)
plt.savefig("Miss_median"+"_period"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_periods = [pre_hits_median_periods, pre_miss_median_periods]
plt.boxplot(pre_median_periods)
plt.plot([1,2], pre_median_periods, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Pre Median Perids Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Period', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_periods, pre_miss_median_periods), fontsize = 12)
plt.savefig("Pre_median"+"_period"+"_Osc"+str(n_oscilations))
plt.show()
post_median_periods = [post_hits_median_periods, post_miss_median_periods]
plt.boxplot(post_median_periods)
plt.plot([1,2], post_median_periods, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Post Median Periods Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Period', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_mono, post_miss_median_mono), fontsize = 12)
plt.savefig("Post_median"+"_period"+"_Osc"+str(n_oscilations))
plt.show()


##extreme data (bottom and top third values)
hits_median_extreme_bandAmpVals = [pre_hits_median_bandAmp, post_hits_median_bandAmp]
#pre vs post
plt.boxplot(hits_median_bandAmpVals)
plt.plot([1,2], hits_median_bandAmpVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Hits Median Band Amplitude Across Subjects', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_bandAmp, post_hits_median_bandAmp), fontsize = 12)
plt.savefig("Hits_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
miss_median_bandAmpVals = [pre_miss_median_bandAmp, post_miss_median_bandAmp]
plt.boxplot(miss_median_bandAmpVals)
plt.plot([1,2], miss_median_bandAmpVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Misses Median Band Amplitude Across Subjects', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_miss_median_bandAmp, post_miss_median_bandAmp), fontsize = 12)
plt.savefig("Miss_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
#hits vs misses
pre_median_bandAmpVals = [pre_hits_median_bandAmp, pre_miss_median_bandAmp]
plt.boxplot(pre_median_bandAmpVals)
plt.plot([1,2], pre_median_bandAmpVals, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Pre Median Band Amplitude Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_hits_median_bandAmp, pre_miss_median_bandAmp), fontsize = 12)
plt.savefig("Pre_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()
post_median_bandAmpVals = [post_hits_median_bandAmp, post_miss_median_bandAmp]
plt.boxplot(post_median_bandAmpVals)
plt.plot([1,2], post_median_bandAmpVals, c='b')
plt.xticks([1,2],['hits','misses'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Post Median Band Amplitude Across Subjects (Hits vs Misses)', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(post_hits_median_bandAmp, post_miss_median_bandAmp), fontsize = 12)
plt.savefig("Post_median"+"_BandAmp"+"_Osc"+str(n_oscilations))
plt.show()


# import sys
# sys.exit()


df_all_cycles_windowed_all = pd.concat(windowed_dfs_all)
#df_all_cycles_windowed_all=df_all_cycles_windowed_all['is_burst']
#toggle this to change burst detection setting

#concatenate all dfs to a single df
df_preBeta_windowed_all = pd.concat(preBeta_dfs_all)
df_postBeta_windowed_all = pd.concat(postBeta_dfs_all)

#df_hits = pd.concat(hits_dfs_all)
df_hits_pre = pd.concat(pre_hits_dfs_all)
df_hits_post = pd.concat(post_hits_dfs_all)

#df_misses = pd.concat(miss_dfs_all)
df_misses_pre = pd.concat(pre_miss_dfs_all)
df_misses_post = pd.concat(post_miss_dfs_all)



#debugging stuff if some median values are not saved as a number type
print("ALL DF LENGTHS")
countPre = 0
for i in df_preBeta_windowed_all['period']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['period']:
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

countPre = 0
for i in df_preBeta_windowed_all['amp_fraction']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['amp_fraction']:
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

countPre = 0
for i in df_preBeta_windowed_all['time_rdsym']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['time_rdsym']:
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

countPre = 0
for i in df_preBeta_windowed_all['time_ptsym']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['time_ptsym']:
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

countPre = 0
for i in df_preBeta_windowed_all['monotonicity']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['monotonicity']:
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

countPre = 0
for i in df_preBeta_windowed_all['band_amp']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed_all['band_amp']:
    if math.isnan(i):
        countPost+=1



#pre vs post
for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_misses_pre)
    h_post = sns.histplot(x=feature, data = df_misses_post, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.wilcoxon(df_misses_pre[feature],df_misses_post[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("misses all subjects pre (blue) vs post (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend Misses Pre vs Post all subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()
for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_hits_pre)
    h_post = sns.histplot(x=feature, data = df_hits_post, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.wilcoxon(df_hits_pre[feature],df_hits_post[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("hits all subjects pre (blue) vs post (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend Hits Pre vs Post all subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()

for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_hits_pre)
    h_post = sns.histplot(x=feature, data = df_hits_post, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.wilcoxon(df_hits_pre[feature],df_hits_post[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("hits all subjects pre (blue) vs post (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend Hits Pre vs Post all subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()

#hits vs misses
for feature in features_keep:
    h_hits = sns.histplot(x=feature, data = df_hits_pre)
    h_miss = sns.histplot(x=feature, data = df_misses_pre, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.mannwhitneyu(df_hits_pre[feature],df_misses_pre[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("Pre Beta all subjects Hits (blue) vs Misses (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend PreBeta_HitsvsMisses_all_subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()
for feature in features_keep:
    h_hits = sns.histplot(x=feature, data = df_hits_post)
    h_miss = sns.histplot(x=feature, data = df_misses_post, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.mannwhitneyu(df_hits_post[feature],df_misses_post[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("Post Beta all subjects Hits (blue) vs Misses (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend PostBeta_HitsvsMisses_all_subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()

##extreme data histograms
#pre vs post
for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_misses_pre)
    h_post = sns.histplot(x=feature, data = df_misses_post, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.wilcoxon(df_misses_pre[feature],df_misses_post[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("misses all subjects pre (blue) vs post (orange)"+" "+feature, fontsize=18)
    plt.savefig("Attend Misses Pre vs Post all subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()


import sys
sys.exit()

# for feature in features_keep:
#     h_hits = sns.histplot(x=feature, data = df_hits)
#     h_misses = sns.histplot(x=feature, data = df_misses, color = 'orange')
#
#     plt.xlabel(feature,fontsize=12)
#     plt.xlabel(stats.mannwhitneyu(df_hits[feature],df_misses[feature]), fontsize = 12)
#     plt.ylabel('count',fontsize=12)
#     plt.xticks(size=12)
#     plt.yticks(size=12)
#     plt.title("all subjects hits (blue) vs misses (orange)"+" "+feature, fontsize=18)
#     plt.savefig("all subjects"+"_"+feature+"_distribution"+"hits (blue) vs misses (orange)")
#     plt.show()



from scipy.stats import mannwhitneyu
df_bandAmpPre_nanDrop = df_preBeta_windowed_all['band_amp'].dropna()
df_bandAmpPost_nanDrop = df_postBeta_windowed_all['band_amp'].dropna()

hist_bandAmp_pre = sns.histplot(data = df_bandAmpPre_nanDrop)
hist_bandAmp_post = sns.histplot(data = df_bandAmpPost_nanDrop, color = 'orange')
plt.xlabel(feature,fontsize=12)
plt.xlabel(stats.mannwhitneyu(df_bandAmpPre_nanDrop,df_bandAmpPost_nanDrop), fontsize = 12)
plt.ylabel('count',fontsize=12)
plt.xticks(size=12)
plt.yticks(size=12)
plt.title('NaN droped BandAmp', fontsize=12)
#plt.savefig("all subj"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
plt.show()



# for feature in features_keep:
#     b_pre = sns.boxplot(x=feature, data = df_preBeta_windowed_all)
#     b_post = sns.boxplot(x=feature, data = df_postBeta_windowed_all, color = 'orange')
#     plt.xlabel(feature,fontsize=15)
#     plt.ylabel('count',fontsize=15)
#     plt.xticks(size=12)
#     plt.yticks(size=12)
#     plt.title("subj "+str(subj)+" "+feature, fontsize=18)
#     #plt.savefig("subj_"+str(subj)+"_"+feature+"_distribution")
#     plt.show()








#plt.title('pre-beta: lowpass - p/t and r/d midpoints - subject 10')
#plt.title('subject 1 avg p/t and r/d signal')


#plot_burst_detect_params(signal_low, Fs, df, burst_kwargs, tlims=None, figsize=(6,6))

# for feat, feat_name in feature_names.items():
#     x_treatment = df_subjects[df_subjects['group']=='hits'][feat]
#     x_control = df_subjects[df_subjects['group']=='misses'][feat]
#     U, p = stats.mannwhitneyu(x_treatment, x_control)
#     #print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))
