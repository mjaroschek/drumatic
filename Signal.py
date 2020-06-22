from numpy import ceil, floor,sqrt,flipud,log10
from matplotlib.widgets import Slider,Button,TextBox
from scipy import signal
import matplotlib.pyplot as plt

def downsample(s,factor):
    if s.sr%factor!=0:
        raise ValueError("Factor has to be a divisor of the sample rate")
    sr=int(s.sr/factor)
    data=s.data[::factor]
    return [1,Signal(data,sr)]

def detect_hits(sig,win_buf_len,win_size,step_size,mask,hit_delta,sleep_after_hit):
    def form(window):
        return sum([mask[i]*abs(window[i]) for i in range(len(window))])/len(window)

    def decision(val,l):
        return val>sum(l)/len(l)*hit_delta

    window_buffer=[0 for i in range(win_buf_len)]
    ind=0
    all_window_values=[]
    hit_signal_data=[]
    sleep_timer=0
    ones=[1 for i in range(step_size)]
    zeroes=[0 for i in range(step_size)]
    
    while ind<len(sig.data):        
        current_window=sig.data[ind:ind+win_size]

        if len(current_window)==win_size:
            s=form(current_window)
            all_window_values.append(((ind,ind+win_size),s))
            if ind!=0:
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

    hit_signal_data=hit_signal_data + [0 for i in range(len(sig.data)-len(hit_signal_data))]
    m=max([i[1] for i in all_window_values])
    all_window_values=[(i[0],i[1]/m) for i in all_window_values]
    return [1,Signal(hit_signal_data,sig.sr),all_window_values]

def data_by_seconds(sig,start, end, enlarge=True):
    if enlarge:
        start=floor(start*sampleRate)
        end=ceil(end*sampleRate)
    else:
        start=ceil(start*sampleRate)
        end=floor(end*sampleRate)
    return sig.data[start:end]

def length_in_seconds(sig):
    return len(sig.data)/sig.sr

def visualize(sigs,windows):
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

    def update_axis_1(ival_start,window_index,interval_in_seconds,updateLabels=True):
        interval_in_hz=interval_in_seconds*sigs[0].sr
        ival_end=ival_start+interval_in_hz
        window_index=window_index
        if updateLabels:
            labels=axs[1].xaxis.get_ticklabels()
        axs[1].clear()
        for i in sigs:
            d=i.data[ival_start:ival_end]
            d=d+[0 for i in range(interval_in_hz-len(d))]
            waveform,=axs[1].plot(d)

        if windows[window_index][0][0] >= ival_start and windows[window_index][0][0]<=ival_end:
            axs[1].scatter(x=range(windows[window_index][0][0]-ival_start,windows[window_index][0][1]-ival_start),y=[windows[window_index][1] for i in range(windows[window_index][0][1]-windows[window_index][0][0])],color="red",zorder=10,marker="s",s=13)

        axs[1].set_xmargin(0)
        axs[1].set_ybound(lower=-1,upper=1)


        if updateLabels:
            j=ival_start
            difference=int(labels[1].get_text())-int(labels[0].get_text())
            for i in labels:
                i.set_text(str(j))
                j=j+difference
            axs[1].xaxis.set_ticklabels(labels)
        fig.canvas.draw_idle()

    update_axis_0(0,4)
    update_axis_1(0,0,4,updateLabels=False)
    
    def slider_callback_on_changed(val):
        ival_start=int(val)
        window_index=int(window_text_box.text)
        ival_length=int(length_text_box.text)
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
        if v<=len(windows):
            window_text_box.set_val(str(v))
        
    def window_button_bwd_callback_on_clicked(val):
        v=int(window_text_box.text)-1
        if v>=0:
            window_text_box.set_val(str(int(window_text_box.text)-1))

    def length_text_box_callback_on_submit(val):
        window_index=int(window_text_box.text)
        ival_start=int(slider.val)
        ival_length=int(val)
        update_axis_0(ival_start,ival_length)
        update_axis_1(ival_start,window_index,ival_length)
        
    def window_text_box_callback_on_submit(val):
        window_index=int(val)
        ival_start=int(slider.val)
        ival_length=int(length_text_box.text)
        update_axis_1(ival_start,window_index,ival_length)
        
        
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
    window_text_box.on_submit(window_text_box_callback_on_submit)

    window_btn_fwd_axis = plt.axes([0.555, 0.24, 0.07, 0.04])
    window_btn_bwd_axis = plt.axes([0.555, 0.19, 0.07, 0.04])
    window_btn_fwd = Button(window_btn_fwd_axis, 'Up')
    window_btn_fwd.on_clicked(window_button_fwd_callback_on_clicked)
    window_btn_bwd = Button(window_btn_bwd_axis, 'Down')
    window_btn_bwd.on_clicked(window_button_bwd_callback_on_clicked)
    plt.show()

    
class Signal:

    def __init__(self,data,samplerate):
        self.data=data
        self.sr=samplerate
        (self.freqs,self.t,self.spec)=self.fft=signal.stft(self.data,fs=self.sr,nperseg=256,noverlap=128)
        self.spec=[sqrt(i.real**2+i.imag**2) for i in self.spec]
        self.spec_scaled=[10. * log10(i) for i in self.spec]
