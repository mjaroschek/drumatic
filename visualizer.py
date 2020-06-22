#!/usr/bin/env python3
import sys
import math
from random import randint

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import numpy as np
import sounddevice as sd
import soundfile as sf


process_all=False
path="/home/maximilian/drummidi/drums/"
hitspath="/home/maximilian/drummidi/hits/"
imagepath="/home/maximilian/drummidi/plots/"
cutoff=30000

downsample=5
win_size=200
step_size=100

ids=[i.split(".wav")[0] for i in os.listdir(path)]

def checkInclusion(i):
    global hits
    
    i=int(i)
    j=0
    while j<len(hits) and int(hits[j])<=i:
        if i==int(hits[j]):
            hits=hits[j+1:]
            return True
        j=j+1
        
j=0
for i in ids:
    j=j+1
    filename=path + i +".wav"
    hitsname=hitspath+i+".txt"
    
    with sf.SoundFile(filename) as f:
        data=[i[0] for i in f.read()]
        sr=f.samplerate

    data=data[::downsample]

    f=open(hitsname)
    hits=f.read()
    f.close()
    hits=hits[1:-1].split(",")
    data=[abs(i) for i in data][:cutoff]
    fig, ax = plt.subplots()
    fig.suptitle(i, fontsize=16)
    lines = ax.plot(data)
    data2=[1 if checkInclusion(i) else 0 for i in range(len(data))][:cutoff]
    lines = ax.plot(data2)

    plt.savefig(imagepath + i +".png")
    plt.close(fig)




