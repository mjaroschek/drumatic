# Notebook for visualizing onset detection.

import os
os.chdir("/home/maximilian/drummidi/")

import pickle

from Pipeline import Pipeline
from Signal import *
from drum_environment import*


###
# Define mode and sound sample
###

#mode = "model"
mode = "spec"

#s1=read_wavefile(root_folder + "mixed/k0TLFbpsL7M.wav")
# s1=read_wavefile(root_folder + "drums/k0TLFbpsL7M.wav")
s1=read_wavefile(root_folder + "drum_short.wav");s1=downsample(s1,5)
#s1=read_wavefile(root_folder + "test_mixed.wav");

###
# Using Model
###

if mode=="model":
    f=open(root_folder + "model.obj","rb")
    m=pickle.load(f)
    f.close()
    p=Pipeline([lambda x:perturb_zeroes(x)])
    p.add_function(lambda x:onset_detection_with_model(x,m,sleep_after_onset))
    (sigs,_)=p.apply(s1)
    visualize([s1,sigs[-1]])


###
# Using spectrogram
###

if mode=="spec":
    p=Pipeline([lambda x:onset_detection_spec(x,win_buf_len,percentage,onset_delta_spec,look_back,sleep_after_onset)])
    (sigs,_)=p.apply(s1)
    visualize([s1,sigs[-1]])

