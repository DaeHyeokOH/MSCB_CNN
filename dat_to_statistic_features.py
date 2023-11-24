import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import time
import math
import statistics
from neurodsp.spectral import compute_spectrum, compute_absolute_power
from fooof import FOOOF

#상수
VIDEO = 40
PARTICIPANT = 32
CHANNEL = 32
SESSION = 15

SESSION_TIME = 4

F_SAMPLING = 128
F_THETA = 4
F_ALPHA = 8
F_BETA = 13
F_GAMMA = 30
F_MAX = 45

AVOID_INF = 1e-6 

def create_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

##Band Pass Filter
def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')

    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order = 5):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)

    return y

def Band_Division(eeg):
    eeg_theta = bandpass_filter(eeg, F_THETA, F_ALPHA, F_SAMPLING, order = 5)
    eeg_alpha = bandpass_filter(eeg, F_ALPHA, F_BETA, F_SAMPLING, order = 5)
    eeg_beta = bandpass_filter(eeg, F_BETA, F_GAMMA, F_SAMPLING, order = 5)
    eeg_gamma = bandpass_filter(eeg, F_GAMMA, F_MAX, F_SAMPLING, order = 5)

    return eeg_theta, eeg_alpha, eeg_beta, eeg_gamma

##Differential Entropy
#save : par, vid, cha, session -> features!
def differential_entropy(signal):
    return np.log10(2*math.pi*math.e*statistics.variance(signal))/2

#eeg : original eeg signal 
#return differetnial entropy of one session, npy array
def session_differential_entropy(eeg):
    eeg_theta, eeg_alpha, eeg_beta, eeg_gamma = Band_Division(eeg)
    
    return np.array([differential_entropy(eeg_theta), differential_entropy(eeg_alpha), differential_entropy(eeg_beta), differential_entropy(eeg_gamma)])

##NPS
def make_frequency_spectral(eeg):
    freqs, spectrum = compute_spectrum(eeg, fs=F_SAMPLING, method='welch')

    return freqs, spectrum

def NPS_features(freqs, spectrum, freq_range):
    fm = FOOOF(verbose = False)
    fm.fit(freqs, spectrum, freq_range)
    power = compute_absolute_power(freqs, spectrum, freq_range)
    aperiodic = fm.aperiodic_params_
    aperiodic = np.append(aperiodic, power)
    periodic = fm.peak_params_.ravel()

    # print(f"range : {freq_range} aperiodic lentgh : {len(aperiodic)} periodic length : {len(periodic)}")
    all_params = np.concatenate([aperiodic, periodic])

    return all_params

def session_NPS_features(freqs, spectrum):
    theta_params = NPS_features(freqs, spectrum, [F_THETA, F_ALPHA])
    alpha_params = NPS_features(freqs, spectrum, [F_ALPHA, F_BETA])
    beta_params = NPS_features(freqs, spectrum, [F_BETA, F_GAMMA])
    gamma_params = NPS_features(freqs, spectrum, [F_GAMMA, F_MAX])

    return np.concatenate([theta_params,alpha_params,beta_params,gamma_params])

create_dir("./Data")

standard_time = time.time()
count = 0
# standard_time = time.time()

# print(f"folder_making : {time.time()-standard_time}")
Statistic_Features = np.empty((PARTICIPANT,VIDEO,CHANNEL,SESSION), dtype = object)
#Start making
for participant_id in tqdm(range(1, PARTICIPANT+1)):

    filepath = "./Data/raw_data/s" + format(participant_id, '02') +".dat"

    with open(filepath, 'rb') as f:
        x_dict = pickle.load(f, encoding='latin1')

    # # # print(f"data loading : {time.time()-standard_time}")
    #analyze unknown dictionary
    # print(x_dict.keys())

    #(40,40,8064) (video_#, channel#, EEG)
    video_data = x_dict["data"]

    #fix video_num
    for video_num in range(1,VIDEO+1):
        ####PART1.#####  DE Calculation
        for session_num in range(1,SESSION+1):
            #numpy array, (32,128*4) : (channel, EEG)
            session_data = video_data[video_num-1 , 0:CHANNEL, 128*(session_num-1)*SESSION_TIME + 128*3 : 128*session_num*SESSION_TIME + 128*3]
            #Fix Channel
            for channel_num in range(1,CHANNEL+1):
                channel_session_data = session_data[channel_num-1]

                session_DE = session_differential_entropy(channel_session_data)

                

                freqs, spectrum = make_frequency_spectral(channel_session_data)
                session_NPS = session_NPS_features(freqs, spectrum)

                #index 0 : 3 -> DEs, 4:5 -> aperiodics, last : periodics
                session_statistic_features = np.concatenate([session_DE, session_NPS])
                
                #eliminate nan
                mask = np.isnan(session_statistic_features)

                # Remove elements with NaN values in place
                session_statistic_features = session_statistic_features[~mask]


                #Save Statistic features
                Statistic_Features[participant_id - 1 , video_num - 1 , channel_num - 1 , session_num - 1] = session_statistic_features

                count+=1

                if count%200 == 0:
                    print(f"percentage {round(count/(PARTICIPANT*VIDEO*CHANNEL*SESSION)*100,2)} last time : {(time.time()-standard_time)/count*PARTICIPANT*VIDEO*CHANNEL*SESSION}")

#dump numpy array (par, vid, cha, session) (32,40,32,15)
pickle.dump(Statistic_Features, open("./Data/Statistic_Features_np.pkl", 'wb'))








