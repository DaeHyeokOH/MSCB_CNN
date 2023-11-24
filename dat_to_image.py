import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt
import time
import sys
#본 파일은 .dat파일을 읽고 이를 par#, video#, session#, channel#, features, vlence, arousal이 column인 DF를 만들도록 한다.
#fix participant

#상수
VIDEO = 40
PARTICIPANT = 32
CHANNEL = 32
SESSION = 15
TOTAL_IMAGE = 2457600

SESSION_TIME = 4

F_SAMPLING = 128
F_THETA = 4
F_ALPHA = 8
F_BETA = 13
F_GAMMA = 30
F_MAX = 45

AVOID_INF = 1e-6 

def status_loading(path):
    if path in os.listdir("./"):
        status_np = np.load(path)
    else:
        status_np = np.array([0,1,1,1,1,1])

    return status_np

##quit & restart
def save_quit_restart(count, particiapnt_id, video_num, channel_num, session_num, read_flag):
    status_np = np.array([count, particiapnt_id, video_num, channel_num, session_num, 1])
    np.save("status.npy", status_np)

    sys.exit()

def update_parameters(participant_id, video_num, channel_num, session_num, read_flag):
    read_flag = 0
    channel_num += 1
    if channel_num > CHANNEL:
        channel_num = 1
        session_num += 1
        if session_num > SESSION:
            session_num = 1
            video_num += 1
            if video_num > VIDEO:
                video_num = 1
                participant_id += 1
                read_flag = 1
    return participant_id, video_num, channel_num, session_num, read_flag

##creting files
def create_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

#Last time interpreter
def seconds_to_days(seconds):
    days = seconds // (24 * 3600)
    hours = (seconds % (24 * 3600)) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    last_time = f"{int(days)} Days {int(hours)}h {int(minutes)}m {round(seconds,2)}s"
    if int(days) == 0:
        last_time = f"{int(hours)}h {int(minutes)}m {round(seconds,2)}s"

        if int(hours) == 0:
            last_time = f"{int(minutes)}m {round(seconds,2)}s"

            if int(minutes) == 0:
                last_time = f"{round(seconds,2)}s"
        
    return last_time

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

#Make spectogram
def create_spectrogram(signal_data, fs, filename):
    # time1 = time.time()
    f, t, Sxx = spectrogram(signal_data, fs)
    # time2 = time.time()
    # Plot the spectrogram
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='nearest')
    del f, t, Sxx
    # time3 = time.time()
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    # time4 = time.time()
    # temp_df = pd.DataFrame({"scipy_spectogram": [
    #                        time2-time1], "plotting": [time3-time2], "save_fig": [time4-time3]})

    # return temp_df

##percentage Output
prev_time = time.time()
def print_percentage(process_count, prev_time):
    if process_count % 30== 0:
        percentage = round(process_count/TOTAL_IMAGE*100, 3)
        consumed_time = round(time.time()-standard_time, 2)
        last_time = seconds_to_days(
            ((time.time()-prev_time)/30)*(TOTAL_IMAGE-process_count))
        print(f"percentage : {percentage} 처리된 이미지 : {process_count}/{TOTAL_IMAGE} 소모 시간 : {consumed_time} 예상 소요 시간 : {last_time} 구간 속도 : {round(30/(time.time()-prev_time),2)}")
        prev_time = time.time()
        
    return prev_time



standard_time = time.time()

spectrogram_detail_time_df = pd.DataFrame()


#세팅 불러오기
status_path = "status.npy"
status = status_loading(status_path)

process_count = status[0]
participant_id = status[1] 
video_num = status[2]
channel_num = status[3]
session_num = status[4]
read_flag = status[5]
###########

##처리할 이미지를 저장하기 위한 폴더 생성
create_dir("./Data/Test_Data/")
create_dir("./Data/Train_Data")
create_dir("./Data/Validation_Data/")

#Start making
while not((participant_id,video_num, channel_num, session_num) == (PARTICIPANT,VIDEO,CHANNEL,SESSION)):
    
    if read_flag: 
        filepath = "./Data/raw_data/s" + format(participant_id, '02') +".dat"

        with open(filepath, 'rb') as f:
            x_dict = pickle.load(f, encoding='latin1')

    #analyze unknown dictionary
    # print(x_dict.keys())

    #(40,4) (video_#, label_#) : (valence, arousal, dominance, liking)
    emotion_data = x_dict["labels"]

    #(40,40,8064) (video_#, channel#, EEG)
    video_data = x_dict["data"]

    # video_start_time = time.time()
    ####PART1.#####  Valence Arousal Data
    Val_Aro_value_ndarray = emotion_data[video_num-1,[0,1]]
    Val_Aro_value_list = Val_Aro_value_ndarray.tolist()

    valence = Val_Aro_value_list[0]
    arousal = Val_Aro_value_list[1]

    if arousal >=5:
        if valence >= 5:
            picture_name = 'HAHV'
        else:
            picture_name = 'HALV'
    else:
        if valence >= 5:
            picture_name = 'LAHV'
        else:
            picture_name = 'LALV'

    #numpy array, (32,128*4) : (channel, EEG)
    session_data = video_data[video_num-1 , 0:CHANNEL, 128*(session_num-1)*SESSION_TIME + 128*3 : 128*session_num*SESSION_TIME + 128*3]

    channel_session_data = session_data[channel_num-1]
    save_path = ""
    if video_num <= 16:
        save_path = "./Data/Train_Data/"
    elif video_num <= 24:
        save_path = "./Data/Validation_Data/"
    else:
        save_path = "./Data/Test_Data/"
    
    band_type_list = ['theta', 'alpha', 'beta', 'gamma']

    eeg_theta, eeg_alpha, eeg_beta, eeg_gamma = Band_Division(channel_session_data)
    

    wave_dic = {"theta" : eeg_theta, "alpha" : eeg_alpha, "beta" : eeg_beta, "gamma" : eeg_gamma}
    file_name = str(participant_id) + "_" + str(video_num) + "_" + str(channel_num) + "_" + str(session_num)

    # channel_start_time =time.time()
    for band_type in band_type_list:
        file_path = save_path + file_name + "_" + band_type + "_" + picture_name + ".png"  
        
        create_spectrogram(wave_dic[band_type], F_SAMPLING, file_path)
        
        process_count+=1
        prev_time = print_percentage(process_count, prev_time)
    
    if process_count%120 == 0:
        save_quit_restart(process_count, participant_id,video_num, channel_num, session_num, read_flag)
    
    participant_id, video_num, channel_num, session_num, read_flag = update_parameters(participant_id, video_num, channel_num, session_num, read_flag)
    

print("Congratulation!!! Finished!!!!!")








