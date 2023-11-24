import numpy as np
import pickle
from tqdm import tqdm
import time

VIDEO = 40
PARTICIPANT = 32
CHANNEL = 32
SESSION = 15

standard_time = time.time()

def nan_inf_check(features):
    nan = np.isnan(features)
    #True, False -> 1, 0
    nan = np.dot(nan, 1)

    inf = np.isinf(features)
    # True, False -> 1, 0
    inf = np.dot(inf, 1)

    return nan, inf
    
with open('./Data/keep/Statistic_Features_np.pkl', 'rb') as f:
    statistic_features = pickle.load(f)

per_nan_num = 0
per_inf_num = 0
nan_list = np.array([0,0,0,0,0,0])
inf_list = np.array([0,0,0,0,0,0])
count = 0 

max_len = 0
for par in tqdm(range(PARTICIPANT)):
    for vid in range(VIDEO):
        for cha in range(CHANNEL):
            for ses in range(SESSION):
                ses_features = statistic_features[par][vid][cha][ses]

                if len(ses_features) > max_len:
                    max_len = len(ses_features)

                features = ses_features[0:6]
                nan, inf = nan_inf_check(features)
                nan_list += nan
                inf_list += inf

                aperiodic_features = ses_features[6:]
                per_nan, per_inf = nan_inf_check(aperiodic_features)
                
                per_nan_num += np.sum(per_nan)
                per_inf_num += np.sum(per_inf)

                count+=1
                # if count%300 == 0:
                    # print(f"percentage {round(count/(PARTICIPANT*VIDEO*CHANNEL*SESSION)*100,2)} last time : {(time.time()-standard_time)/count*PARTICIPANT*VIDEO*CHANNEL*SESSION}")

print("nan status")
print(f"theta DE : {nan_list[0]} alpha DE : {nan_list[1]} beta DE : {nan_list[2]} gamma DE : {nan_list[3]}")
print(f"aperiodic para, 1st : {nan_list[4]} 2nd {nan_list[5]}")
print(f"periodic para, number is {per_nan_num}")
print("--------------------------------------------------------------------------------------------------")
print("inf status")
print(f"theta DE : {inf_list[0]} alpha DE : {inf_list[1]} beta DE : {inf_list[2]} gamma DE : {inf_list[3]}")
print(f"aperiodic para, 1st : {inf_list[4]} 2nd {inf_list[5]}")
print(f"periodic para, number is {per_inf_num}")
print(f"max len is {max_len}")

