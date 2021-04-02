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

import os
import scipy.io
import scipy.signal

from scipy import stats
import seaborn as sns
#from ggplot import ggplot, geom_line, aes


plt.rcParams.update({'font.size': 5})


f_theta = (10,22)
filename = 'full_trials_allsubjs_troughdata'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)

subj = 0 #Python Index
trial_number = 74 #Python Index
trough_index = 264 #MATLAB Value
myDelta = 0.005
n_oscilations = 2 #use for windowing dataframes

#srate = data['Fs']
Fs = 600 # sampling rate
all_signals=data['all_data']
trough_data=data['all_trough_data']
trough_data = trough_data[0]
print(len(trough_data))
trough_data = trough_data[subj]
trough_data_indexes = trough_data[:,1]
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

signal_raw = ind_signal

# grand_avg = grand_avg/(2*len(all_signals_hits))
#
# for i in range(len(all_signals_hits)):
#     grand_avg = [all_signals_hits[i,:].mean(axis=0), all_signals_misses[i,:].mean(axis=0)])
#     grand_avg += grand_avg
#
# signal_raw = grand_avg

#remove beta event
# print (len(signalsInd))

# for i in range(len(all_signals)):
#     signal_ind = all_signals[i,:].mean(axis=0)
#     sig_avg+=signal_ind
#
# sig_avg_all= sig_avg/len(all_signals)
# signal_raw = sig_avg_all

#signal=signal.transpose() #transpose if time vector and signal are not same dimensions
t = data['tVec']

t_raw= t[0]

f_lowpass = 60
N_seconds = 0.40


signalInd_lowpass = lowpass_filter(ind_signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

plt.figure(figsize=(8,6))
plt.plot(t_raw,ind_signal,'k')
plt.plot(t_raw, signalInd_lowpass,'b')
plt.title('Ind signal raw')
plt.show()

neg = 0

def signal_addEdges(signal_raw):
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

t1=t_raw
t2=t1[right_edge:]-sig_time #subtract time length of signal
t3=t1[:left_edge]+sig_time
t = np.concatenate([t2,t1])
t = np.concatenate([t,t3])
t = t+(sig_time/2)+sig_time
tlim = (0, 3*sig_time) #time interval in seconds
tidx = np.logical_and(t>=tlim[0], t<tlim[1])


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

#
# plt.figure(figsize=(8, 6))
# plt.plot(t[tidx], signal_low[tidx], 'k') #change back to t
# plt.title('Reflected Signal')
# x_coords = [(sig_time/2)+sig_time+t_raw[0], sig_time/2+(2*sig_time)+t_raw[0]]
# for xc in x_coords:
#     plt.axvline(x=xc)
# plt.xlim((tlim))
# plt.show()


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

from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params

burst_kwargs = {'amplitude_fraction_threshold': .18,
                'amplitude_consistency_threshold': .09,
                'period_consistency_threshold': .09,
                'monotonicity_threshold': .1 ,
                'N_cycles_min': 1}

#burst detect individual trials
df_ind = compute_features(signal, Fs, f_theta,
center_extrema='T', burst_detection_kwargs=burst_kwargs)
plot_burst_detect_params(signal, Fs, df_ind, burst_kwargs, figsize=(12,1,5))
#df_ind = df_ind[df_ind['is_burst']]
print("dfIND VALUES")
print(len(signal_raw))
print(len(signal_raw)-left_edge)
print(len(df_ind))


print(df_ind['sample_trough'])
print(df_ind['sample_trough'][0])
print(df_ind['amp_fraction'])
print(tidxTs)
print(Ts)

dropBeta = True
#
# def minima_quality_check(Troughs, t_raw, trough_index_def, delta):
#     for t in Troughs:
#         if abs(t_raw[t] - t_raw[trough_index_def-1+left_edge]) < delta:
#             return("CBC min within MATLAB min passed!")
#     return("CBC min -> MATLAB min NOT passed")

def windowDf(input_df, input_trough, delta, n_oscilations):
    for cycle in range(len(input_df)):
        if abs(input_df['sample_trough'][cycle]-input_trough-left_edge)<delta*Fs and (cycle>=n_oscilations+1 and (cycle+n_oscilations+1)<=len(input_df)):
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
df_all_cycles_windowed = pd.concat(windowed_dfs)
df_preBeta_windowed = pd.concat(preBeta_dfs)
df_postBeta_windowed = pd.concat(postBeta_dfs)

print("DF LENGTHS")
print(len(df_preBeta_windowed))
print(len(df_postBeta_windowed))


countPre = 0
for i in df_preBeta_windowed['band_amp']:
     if math.isnan(i):
          countPre+=1
countPost = 0
for i in df_postBeta_windowed['band_amp']:
    print(i)
    if math.isnan(i):
        countPost+=1
print(countPre, countPost)

features_keep = ['period', 'amp_fraction','time_rdsym', 'time_ptsym', 'monotonicity', 'band_amp']
for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_preBeta_windowed)
    h_post = sns.histplot(x=feature, data = df_postBeta_windowed, color = 'orange')
    plt.xlabel(feature,fontsize=15)
    plt.xlabel(stats.wilcoxon(df_preBeta_windowed[feature],df_postBeta_windowed[feature]), fontsize = 10)
    plt.ylabel('count',fontsize=15)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("subj "+str(subj)+" "+feature, fontsize=18)
    plt.savefig("subj_"+str(subj)+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()


from scipy.stats import wilcoxon

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





for sub in range(10):
    trough_data_all=data['all_trough_data']
    trough_data_loop = trough_data_all[0]
    trough_data_loop = trough_data_loop[sub]
    trough_data_indexes_loop = trough_data_loop[:,1]
    trough_data_trial_numbers_loop = trough_data_loop[:,0]
    signalsInd_loop = all_signals[sub,:]

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
            preBeta_dfs_all.append(preBeta_df)
            postBeta_df = windowed_df.iloc[np.arange(n_oscilations, n_oscilations*2)]


            postBeta_dfs_all.append(postBeta_df)



    pre_median_bandAmps.append(np.nanmedian(preBeta_df['band_amp']))
    post_median_bandAmps.append(np.nanmedian(postBeta_df['band_amp']))

    pre_median_rdsym.append(np.nanmedian(preBeta_df['time_rdsym']))
    post_median_rdsym.append(np.nanmedian(postBeta_df['time_rdsym']))

    pre_median_ptsym.append(np.nanmedian(preBeta_df['time_ptsym']))
    post_median_ptsym.append(np.nanmedian(postBeta_df['time_ptsym']))

    pre_median_mono.append(np.nanmedian(preBeta_df['monotonicity']))
    post_median_mono.append(np.nanmedian(postBeta_df['monotonicity']))

    pre_median_periods.append(np.nanmedian(preBeta_df['period']))
    post_median_periods.append(np.nanmedian(postBeta_df['period']))

    pre_median_ampConst.append(np.nanmedian(preBeta_df['amp_consistency']))
    post_median_ampConst.append(np.nanmedian(postBeta_df['amp_consistency']))

    pre_median_ampFrac.append(np.nanmedian(preBeta_df['amp_fraction']))
    post_median_ampFrac.append(np.nanmedian(postBeta_df['amp_fraction']))

    pre_median_periodConst.append(np.nanmedian(preBeta_df['period_consistency']))
    post_median_periodConst.append(np.nanmedian(postBeta_df['period_consistency']))




median_bandAmpVals = [pre_median_bandAmps, post_median_bandAmps]
plt.boxplot(median_bandAmpVals)
plt.plot([1,2], median_bandAmpVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median Band Amplitude Across Subjects', fontsize = 12)
plt.ylabel('Band Amplitude', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_bandAmps, post_median_bandAmps), fontsize = 12)
plt.show()

median_rdsymVals = [pre_median_rdsym, post_median_rdsym]
plt.boxplot(median_rdsymVals)
plt.plot([1,2], median_rdsymVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median RDSYM Across Subjects', fontsize = 12)
plt.ylabel('RDSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_rdsym, post_median_rdsym), fontsize = 12)
plt.show()

median_ptsymVals = [pre_median_ptsym, post_median_ptsym]
plt.boxplot(median_ptsymVals)
plt.plot([1,2], median_ptsymVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median PTSYM Across Subjects', fontsize = 12)
plt.ylabel('PTSYM', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_ptsym, post_median_ptsym), fontsize = 12)
plt.show()

median_monoVals = [pre_median_mono, post_median_mono]
plt.boxplot(median_monoVals)
plt.plot([1,2], median_monoVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median Monotonicity Across Subjects', fontsize = 12)
plt.ylabel('Monotonicity', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_mono, post_median_mono), fontsize = 12)
plt.show()

median_periodVals = [pre_median_periods, post_median_periods]
plt.boxplot(median_periodVals)
plt.plot([1,2], median_periodVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median Periods Across Subjects', fontsize = 12)
plt.ylabel('Periods', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_periods, post_median_periods), fontsize = 12)
plt.show()

# import sys
# sys.exit()


median_ampFracVals = [pre_median_ampFrac, post_median_ampFrac]
plt.boxplot(median_ampFracVals)
plt.plot([1,2], median_ampFracVals, c='b')
plt.xticks([1,2],['pre','post'], fontsize = 12)
plt.yticks(fontsize=12)
plt.title('Median Amplitude Fraction Across Subjects', fontsize = 12)
plt.xlabel(stats.wilcoxon(pre_median_ampFrac, post_median_ampFrac), fontsize = 12)
plt.ylabel('Amplitude Fraction', fontsize = 12)

plt.show()


#PUT P VALS In Graphs





print(stats.wilcoxon(pre_median_bandAmps,post_median_bandAmps))
print(stats.wilcoxon(pre_median_rdsym,post_median_rdsym))


# sns.boxplot(y=pre_median_bandAmps)
# sns.boxplot(y=post_median_bandAmps)
# plt.show()


df_all_cycles_windowed_all = pd.concat(windowed_dfs_all)
#df_all_cycles_windowed_all=df_all_cycles_windowed_all['is_burst']
#toggle this to change burst detection setting

df_preBeta_windowed_all = pd.concat(preBeta_dfs_all)
df_postBeta_windowed_all = pd.concat(postBeta_dfs_all)

df_preBeta_windowed_all
df_postBeta_windowed_all



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
print(countPre, countPost)

print(df_preBeta_windowed_all['band_amp'])


print(countPre, countPost)









for feature in features_keep:
    h_pre = sns.histplot(x=feature, data = df_preBeta_windowed_all)
    h_post = sns.histplot(x=feature, data = df_postBeta_windowed_all, color = 'orange')

    plt.xlabel(feature,fontsize=12)
    plt.xlabel(stats.wilcoxon(df_preBeta_windowed_all[feature],df_postBeta_windowed_all[feature]), fontsize = 12)
    plt.ylabel('count',fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("all subjects"+" "+feature, fontsize=18)
    plt.savefig("all subjects"+"_"+feature+"_distribution"+"_Osc"+str(n_oscilations))
    plt.show()

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
