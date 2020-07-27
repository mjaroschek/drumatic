#!/usr/bin/env python3
import sounddevice as sd

import os
os.chdir("/home/maximilian/drummidi/")

import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Signal import *
from experimental import *
from drum_environment import *


columns = 80
low=100
high=2000
block_duration=50
gain=10

samplerate = 8000#sd.query_devices(None, 'input')['default_samplerate']
sd.default.samplerate = samplerate
signal_length=stft_window_length+stft_window_overlap*look_back

nfreqs=40
minAmp=0
maxAmp=150

visible_data=[0 for i in range(samplerate*4)]
visible_spec=[[(minAmp+maxAmp)/2 for j in range(int(samplerate*4/stft_window_overlap))] for i in range(nfreqs)]

fbank=init_mel_filter(samplerate,stft_window_length)

hitsig=0
d=[0 for i in range(signal_length)]

def callback(indata, frames, time, status):
    global visible_data
    global visible_spec
    global hitsig
    global d
    if any(indata):
        indata=sum(indata.tolist(),[])
        visible_data=visible_data[len(indata):] + indata
        sig=Signal(visible_data[-signal_length:],samplerate,computeAll=False)
        #spec=transpose(sig.spec_scaled())
        #sig=pre_emphasis_filter(sig)
        spec=mel_scale_filter(sig,fbank)
        spec=move_up(spec)
        #k=[[1,2,1],[0,0,0],[-1,-2,-1]] #sobel
        #k=[[1,1,1],[0,0,0],[-1,-1,-1]] #prewitt
        #spec=convolve(spec,k)
        #spec=[[10. * np.log10(i) for i in j] for j in spec]
        visible_spec=transpose(transpose(visible_spec)[len(spec[0]):]+transpose(spec))

        # spec=refine(spec)
        # for i,s in enumerate(transpose(spec)):
        #     if i<len(spec[0])-1:
        #         if s[0]==1:
        #             d[i*stft_window_overlap:(i+1)*stft_window_overlap]=[1]*stft_window_overlap
        #         else:
        #             d[i*stft_window_overlap:(i+1)*stft_window_overlap]=[0]*stft_window_overlap
        # visible_hits=visible_hits[signal_length:] + d

def stream():
    with sd.InputStream(None, channels=1, callback=callback,blocksize=signal_length):
        while True:
            if stop:
                break
    
def animate(i):
    lines[0].set_xdata(range(len(visible_data)))
    lines[0].set_ydata(visible_data)
    lines[1].set_array(np.flipud([i for i in visible_spec]))
    return lines

fig,axs=plt.subplots(nrows=2)
axs[0].set_xlim([0, len(visible_data)])
axs[0].set_ylim([-1, 1])
axs[1].set_xlim([0, len(visible_spec[0])])
axs[1].set_ylim([0, nfreqs])

#lines = [axs[0].plot(visible_data,animated=True)[0],axs[1].imshow(visible_spec,animated=True)]
lines = [axs[0].plot(visible_data)[0],axs[1].imshow(np.flipud([i for i in visible_spec]),vmin=minAmp, vmax=maxAmp,aspect='auto')]

#lines=lines+plt.plot(visible_hits,animated=True)
ani = FuncAnimation(fig, animate,blit=True,interval=2000)

thread = threading.Thread(target=stream)
thread.daemon = True
stop=False
thread.start()

plt.show()
stop=True
thread.join()
