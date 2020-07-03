from numpy import ceil,floor,sqrt,flipud,log10,gcd,round,float64
from matplotlib.widgets import Slider,Button,TextBox
from scipy import signal
from random import randint
import matplotlib.pyplot as plt
import soundfile as sf
from drum_environment import *

def create_datapoint(sig,start,end):
    data=Signal(sig.data[start:end],sig.sr).spec
    return sum([i.tolist() for i in data],[])

def perturb_zeroes(sig):
    m=min([abs(i) for i in sig.data if i!=0])
    return [1,Signal([i if i!=0 else m for i in sig.data],sig.sr)]

def save_wavefile(sig,filename):
    sf.write(filename,sig.data,sig.sr)

def read_wavefile(filename,start=0,length=0,unit="s"):
    # Read a wavefile using soundfile and store it as a signal object. Reads only one channel!
    if start==0 and length==0:
        with sf.SoundFile(filename) as f:
            if f.channels>1:
                data=[i[0] for i in f.read()]
            else:
                data=[i for i in f.read()]
            sr=f.samplerate
    else:
        info=sf.info(filename)
        total_length=info.frames
        sr=info.samplerate
        if unit=="s":
            start_in_hz=int(round(start*sr))
            length_in_hz=int(round(length*sr))
        else:
            start_in_hz=int(start)
            length_in_hz=int(length)
        if length_in_hz<=0:
            length_in_hz=total_length-start_in_hz
        if start<0:
            start_in_hz=total_length-length_in_hz
        else:
            length_in_hz=min(length_in_hz,total_length-start_in_hz)
        end_in_hz=start_in_hz+length_in_hz
        blocks=sf.blocks(filename, blocksize=length_in_hz,start=start_in_hz,stop=end_in_hz)
        for block in blocks:
            if info.channels>1:
                data=[i[0] for i in block]
            else:
                data=[i for i in block]
        blocks.close()
    return Signal(data,sr)

def downsample(sig,factor):
    # Reduce the samplerate of a Signal by a positive integer factor that divides the
    # original samplerate.
    factor=int(factor)
    if factor<=0:
        raise ValueError("Factor has to be greater than 0")
    if sig.sr%factor!=0:
        raise ValueError("Factor has to be a divisor of the sample rate")
    sr=int(sig.sr/factor)
    data=sig.data[::factor]
    return [1,Signal(data,sr)]

def onset_detection(sig,win_buf_len,win_size,step_size,mask,hit_factor,hit_delta,sleep_after_hit,normalize=True):
    # Detect onsets in a signal. All oarameters except for sig controll the
    # behavior of the detection algorithm. See drum_environment.py for details.
    # This method works online, i.e. to decide if an onset is present at the
    # current position, it only takes into account past unformation and not
    # upcoming datapoints.
    def form(window):
        return sum([mask[i]*abs(window[i]) for i in range(len(window))])/len(window)

    def decision(val,l):
        # print("v="+str(val)+", l="+str(l) + ", d=" +str(min(l)*hit_factor +hit_delta))
        # print()
        return val>min(l)*hit_factor +hit_delta

    window_buffer=[0]*win_buf_len
    ind=0
    all_window_values=[]
    hit_signal_data=[]
    sleep_timer=0
    ones=[1]*step_size
    zeroes=[0]*step_size
    bound=win_buf_len*step_size
    while ind<len(sig.data):        
        current_window=sig.data[ind:ind+win_size]
        if len(current_window)==win_size:
            s=form(current_window)
            all_window_values.append(((ind,ind+win_size),s))
            if ind>=bound:
                if decision(s,window_buffer) and sleep_timer==0:
                    hit_signal_data=hit_signal_data + ones
                    sleep_timer=max(sleep_after_hit,win_buf_len+1)
                else:
                    hit_signal_data=hit_signal_data + zeroes
            else:
                hit_signal_data=hit_signal_data + zeroes
            window_buffer.append(s)
            window_buffer=window_buffer[1:win_buf_len+1]
        ind=ind+step_size
        if sleep_timer > 0:
           sleep_timer=sleep_timer-1

    hit_signal_data=hit_signal_data + [0]*(len(sig.data)-len(hit_signal_data))
    m=max([i[1] for i in all_window_values])
    all_window_values=[(i[0],i[1]/m) for i in all_window_values]
    return [1,Signal(hit_signal_data,sig.sr),all_window_values]

def onset_detection_spec(sig,win_buf_len,percentage,hit_delta,look_back,sleep_after_hit):
    # Detect onsets via the stft.
    def decision(current_window,past_window):
        s=0
        old=[max([abs(j) for j in i]) for i in zip(*past_window[:2])]
        new=[max([abs(j) for j in i]) for i in zip(*(past_window[-look_back-1:-1]+[current_window]))]
        for i,j in zip(new,old):
            if i>=j+hit_delta:
                s=s+1
        return s>percentage*len(current_window)
        
    hit_signal_data=[0]*len(sig.data)
    spec=[[i[j] for i in sig.spec] for j in range(len(sig.spec[0]))]
    sleep_timer=0
    window_buffer=spec[:win_buf_len]
    window_step_differenhce=sig.stft_window_length-sig.stft_window_overlap
    
    for j in range(len(spec)):
        s=spec[j]
        if j%10000==0:
            print("        " + str(j) + "/" + str(len(spec)))
        hit_detected=False
        if sleep_timer > 0:
            sleep_timer=sleep_timer-1
        if sleep_timer==0 and decision(s,window_buffer):
            hit_signal_data[j*window_step_difference:(j+1)*window_step_difference]=[1]*window_step_difference
            sleep_timer=sleep_after_hit
        window_buffer=window_buffer[1:win_buf_len]+[s]
        j=j+1
                            
    hit_signal_data=hit_signal_data + [0]*(len(sig.data)-len(hit_signal_data))
    return [1,Signal(hit_signal_data,sig.sr)]

def onset_detection_with_model(sig,m,sleep_after_hit):
    # Detect onsets via a trained model that takes the stft data of the signal.
    def decision(start,end):
        p=create_datapoint(sig,start,end)
        return m.predict([p])[0]==1
    
    hit_signal_data=[0]*len(sig.data)
    sleep_timer=0
    window_step_difference=sig.stft_window_length-sig.stft_window_overlap

    for j in range(look_back*window_step_difference,len(sig.spec)):
        if j%10000==0:
            print("        " + str(j) + "/" + str(len(sig.spec)))
        start=j-look_back*window_step_difference
        end=j+sig.stft_window_length
        hit_detected=False
        if sleep_timer > 0:
            sleep_timer=sleep_timer-1
        if sleep_timer==0 and decision(start,end):
            hit_signal_data[j*window_step_difference:(j+1)*window_step_difference]=[1]*window_step_difference
            sleep_timer=sleep_after_hit
        j=j+1
                            
    hit_signal_data=hit_signal_data + [0]* (len(sig.data)-len(hit_signal_data))
    return [1,Signal(hit_signal_data,sig.sr)]

def data_by_seconds(sig,start,end,enlarge=True):
    # Get the data of a signal in a certain time interval. The enlarge parameter
    # controls if the resulting signal should be slightlly smaller or slightly
    # bigger if rounding error occur.
    if enlarge:
        start=floor(start*sampleRate)
        end=ceil(end*sampleRate)
    else:
        start=ceil(start*sampleRate)
        end=floor(end*sampleRate)
    return sig.data[start:end]

def length_in_seconds(sig):
    # Get the duration of the signal in seconds.
    return len(sig.data)/sig.sr

def cut_from_middle(sig,length,random_start=True):
    # Return a signal of the specificed length and whose data is taken from the
    # middle of the provided signal.
    if length>len(sig.data):
        raise ValueError
    if random_start:
        s=randint(0,len(sig)-length)
    else:
        start=int((len(sig.data)-length)/2)
    end=start+length
    return Signal(sig.data[start:end],sig.sr)

def add(sig1,sig2,scale=1):
    # Add two signals together. The scale parameter sets the ratio of the max
    # amplitude of the first signal and the max amplitude of the second signal
    # in the mix
    if sig1.sr!=sig2.sr:
        raise ValueError
    d1=sig1.data
    d2=sig2.data
    if len(d1)<len(d2):
        d1=d1+[0]*(len(d2)-len(d1))
    else:
        d2=d2+[0]*(len(d1)-len(d2))
    if scale<=0:
       return Signal([i+j for i,j in zip(d1,d2)],sig1.sr)
    a1=max([abs(i) for i in d1])
    a2=max([abs(i) for i in d2])
    return Signal([scale*i+a1/a2*j for i,j in zip(d1,d2)],sig1.sr)

def save_signal(sig,filename,append=False):
    # Write signal object to .sig file. If append is set to true, the data is
    # added to the end to the file
    if append:
        f=open(filename,"a")
    else:
        f=open(filename,"w")
    f.write(str(sig.sr)+",")
    f.write(str(sig.data)[1:-1]+"\n")
    f.close()

def read_signals(filename):
    # Load signal object from .sig file
    f=open(filename,"r")
    signals=[i.split(",") for i in f.readlines()]
    signals=[Signal([float64(j) for j in i[1:]],int(i[0])) for i in signals]
    return signals
    
def visualize(sigs,windows=None):
    # Visualization of the spectrogram, the waveform and the hit detection for a
    # signal. The redrawing routine is not optimized, as this is just meant for
    # visual debugging.
    slider_stepsize=1000
    
    fig, axs = plt.subplots(nrows=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)

    Fs=sigs[0].sr

    def update_axis_0(ival_start,interval_in_seconds):
        ival_start_in_seconds=ival_start/sigs[0].sr
        ind_t0=max(int(floor(ival_start_in_seconds/sigs[0].t[1])),0)
        ind_t1=min(int(ceil((ival_start_in_seconds+interval_in_seconds)/sigs[0].t[1])),len(sigs[0].t)-1)
        
        extent = sigs[0].t[ind_t0], sigs[0].t[ind_t1], sigs[0].freqs[0], sigs[0].freqs[-1]
        axs[0].clear()
        axs[0].imshow(flipud([i[ind_t0:ind_t1] for i in sigs[0].spec_scaled]),extent=extent)
        axs[0].axis('auto')
        fig.canvas.draw_idle()

    def update_axis_1(ival_start,window_index,interval_in_seconds,update_labels=True):
        interval_in_hz=int(interval_in_seconds*sigs[0].sr)
        ival_end=ival_start+interval_in_hz
        window_index=window_index
        axs[1].clear()
        for i in sigs:
            d=i.data[ival_start:ival_end]
            d=d+[0 for i in range(interval_in_hz-len(d))]
            waveform,=axs[1].plot(d)

        if windows!=None and windows[window_index][0][0] >= ival_start and windows[window_index][0][0]<=ival_end:
            v=windows[window_index][1]
            axs[1].scatter(x=range(windows[window_index][0][0]-ival_start,windows[window_index][0][1]-ival_start),y=[v for i in range(windows[window_index][0][1]-windows[window_index][0][0])],color="red",zorder=10,marker="s",s=13)
            if v< 0.7:
                axs[1].text(x=windows[window_index][0][1]-ival_start,y=v+0.2,s=str(round(v,3)))
            else:
                axs[1].text(x=windows[window_index][0][1]-ival_start,y=v-0.3,s=str(round(v,3)))
        
        axs[1].set_xmargin(0)
        axs[1].set_ybound(lower=-1,upper=1)

        if update_labels:
            labels=axs[1].xaxis.get_ticklabels()
            locs=axs[1].xaxis.get_majorticklocs()
            for i in range(len(labels)):
                labels[i].set_text(str(int(ival_start+locs[i])))
            axs[1].xaxis.set_ticklabels(labels)
        fig.canvas.draw_idle()
    
    def slider_callback_on_changed(val):
        ival_start=int(val)
        window_index=int(window_text_box.text)
        ival_length=float(length_text_box.text)
        update_axis_0(ival_start,ival_length)
        update_axis_1(ival_start,window_index,ival_length)

    def ival_button_fwd_callback_on_clicked(val):
        v=slider.val+slider_stepsize
        if v<=len(sigs[0].data):
            slider.set_val(v)

    def ival_button_bwd_callback_on_clicked(val):
        v=slider.val-slider_stepsize
        if v>=0:
            slider.set_val(v)

    def window_button_fwd_callback_on_clicked(val):
        v=int(window_text_box.text)+1
        if v<=len(windows)-1:
            window_text_box.set_val(str(v))
        
    def window_button_bwd_callback_on_clicked(val):
        v=int(window_text_box.text)-1
        if v>=0:
            window_text_box.set_val(str(int(window_text_box.text)-1))

    def length_text_box_callback_on_submit(val):
        window_index=int(window_text_box.text)
        ival_start=int(slider.val)
        ival_length=float(val)
        update_axis_0(ival_start,ival_length)
        update_axis_1(ival_start,window_index,ival_length)
        
    def window_text_box_callback_on_submit(val):
        window_index=int(val)
        ival_start=int(slider.val)
        ival_length=float(length_text_box.text)
        update_axis_1(ival_start,window_index,ival_length)
        
    update_axis_0(0,4)
    update_axis_1(0,0,4,update_labels=False)
        
    ival_btn_bwd_axis = plt.axes([0.25, 0.05, 0.15, 0.075])
    ival_btn_fwd_axis = plt.axes([0.55, 0.05, 0.15, 0.075])
    ival_btn_fwd = Button(ival_btn_fwd_axis, 'Forward')
    ival_btn_fwd.on_clicked(ival_button_fwd_callback_on_clicked)
    ival_btn_bwd = Button(ival_btn_bwd_axis, 'Backward')
    ival_btn_bwd.on_clicked(ival_button_bwd_callback_on_clicked)

    slider_axis  = fig.add_axes([0.125, 0.15, 0.7, 0.03])
    slider = Slider(slider_axis, "", 0, len(sigs[0].data), valstep=1000,valinit=0)
    slider.on_changed(slider_callback_on_changed)

    length_text_box_axis = fig.add_axes([0.125, 0.2, 0.1, 0.07])
    length_text_box = TextBox(length_text_box_axis, 'Length  ', initial="4")
    length_text_box.on_submit(length_text_box_callback_on_submit)

    window_text_box_axis = fig.add_axes([0.425, 0.2, 0.1, 0.07])
    window_text_box = TextBox(window_text_box_axis, 'Window  ', initial="0")
    if windows!=None:
        window_text_box.on_submit(window_text_box_callback_on_submit)

    window_btn_fwd_axis = plt.axes([0.555, 0.24, 0.07, 0.04])
    window_btn_bwd_axis = plt.axes([0.555, 0.19, 0.07, 0.04])
    window_btn_fwd = Button(window_btn_fwd_axis, 'Up')
    if windows!=None:
        window_btn_fwd.on_clicked(window_button_fwd_callback_on_clicked)
    window_btn_bwd = Button(window_btn_bwd_axis, 'Down')
    if windows!=None:
        window_btn_bwd.on_clicked(window_button_bwd_callback_on_clicked)

    plt.show()
    
class Signal:
    # Signal class. Contains the data, the sampling rate, and the short time
    # fourier transform as frequencies, time steps, and amplitudes. Length and
    # overlap of the FFT segments are fixed.

    def __init__(self,data,samplerate):
        self.data=data
        self.sr=samplerate
        self.stft_window_length=256
        self.stft_window_overlap=128
        (self.freqs,self.t,self.spec)=signal.stft(self.data,fs=self.sr,nperseg=self.stft_window_length,noverlap=self.stft_window_overlap)
        self.spec=[sqrt(i.real**2+i.imag**2) for i in self.spec]
        self.spec_scaled=[10. * log10(i) for i in self.spec]
