from random import *
from pytube import YouTube
import moviepy.editor as mp
from Signal import *
import pickle
from drum_environment import *
from Pipeline import Pipeline
from numpy import savez
import os

def download():
    # Download the audio track of a youtube video and save it as a raw audio
    # file

    files = os.listdir(path_to_lists)

    for file in files:
        f = open(path_to_lists + file)
        url_list=f.read().splitlines()
        list_path=root_folder + file.split(".")[0] +"/"
        try:
            os.makedirs(list_path)
        except FileExistsError:
            pass
        processed_ids=[i[:-4] for i in os.listdir(list_path)]
        f.close()

        for j,url in enumerate(url_list,start=1):
            name=url.split("v=")[-1]
            if process_all or not name in processed_ids:
                print("Downloading: " + name + "    ("+str(j)+"/"+str(len(url_list))+")")
                yt = YouTube(url)
                yt_stream_audio = yt.streams.filter(only_audio=True)
                yt_stream_audio[0].download(output_path=list_path, filename=name)
                clip = mp.AudioFileClip(list_path + name +  ".mp4")
                clip.write_audiofile(list_path + name +".wav",fps=int(44100/samplefactor))
                clip.close()
                os.remove(list_path+name+".mp4")
    
def onset_detection():
    # Perform onset detection on the raw audio files and save the binary signal
    # as a .sig file.

    ids=[i.split(".wav")[0] for i in os.listdir(path_to_sound)]

    for j,i in enumerate(ids,start=1):
        filename=path_to_sound + i +".wav"

        if os.path.isfile(path_to_onset+i+".sig") and not process_all:
            continue

        print("Processing: " + i + "    ("+str(j)+"/"+str(len(ids))+")")

        s=read_wavefile(filename)
        print("    Finished reading")

        p=Pipeline([lambda x:onset_detection_spec(x,win_buf_len,percentage,onset_delta_spec,look_back,sleep_after_onset)])

        print("    Starting pipeline")
        (sigs,_)=p.apply(s)

        try:
            os.makedirs(path_to_onset)
        except FileExistsError:
            pass
        save_signal(sigs[-1],path_to_onset + i + ".sig")

def mix():
    # Mix ambient sound into the drum sounds
    seed(982365)
    
    ids=[i.split(".wav")[0] for i in os.listdir(path_to_sound)]
    noise_files=os.listdir(path_to_noise)

    for j,i in enumerate(ids,start=1):
        filename=path_to_sound + i +".wav"

        if os.path.isfile(path_to_mixed+i+".wav") and not process_all:
            continue

        print("Mixing: " + i + "    ("+str(j)+"/"+str(len(ids))+")")
        print("    Reading file") 
        s=read_wavefile(filename)
        n=randint(0,len(noise_files)-1)
        info=sf.info(path_to_noise+noise_files[n])
        start=randint(0,info.frames-len(s.data))
        print("    Reading noise") 
        noise=read_wavefile(path_to_noise+noise_files[n],start=start,length=sf.info(filename).frames,unit="hz")
        s=add(s,noise,randint(200,250)/100)
        print("    Saving mix") 
        save_wavefile(s,path_to_mixed+i+".wav")
        print("    Saving ambient track") 
        save_wavefile(noise,path_to_ambient+i+".wav")
    
def create_training_data():
        ids=[i.split(".wav")[0] for i in os.listdir(path_to_mixed)]
        datapoints=0

        onset_file=open(onset_filename,"w")
        no_onset_file=open(no_onset_filename,"w")

        f=open(onset_signals_filename,"w")
        f.close()
        f=open(no_onset_signals_filename,"w")
        f.close()
        
        for j,i in enumerate(ids,start=1):        
            print("Processing: " + i + "    ("+str(j)+"/"+str(len(ids))+")")
            onset_signal=read_signals(path_to_onset+i+".sig")[0]
            sound_signal=read_wavefile(path_to_mixed+i+".wav")
            ambient_signal=read_wavefile(path_to_ambient+i+".wav")
            window_step_difference=onset_signal.stft_window_length-onset_signal.stft_window_overlap
            for j in range(0,len(sound_signal.data),window_step_difference):
                if onset_signal.data[j]==1:
                    datapoints=datapoints+2
                    start=j-look_back*window_step_difference
                    end=j+onset_signal.stft_window_length
                    d=create_datapoint(sound_signal,start,end)
                    save_signal(Signal(sound_signal.data[start:end],sound_signal.sr),onset_signals_filename,True)
                    onset_file.write(str(d)[1:-1]+"\n")
                    d=create_datapoint(ambient_signal,start,end)
                    save_signal(Signal(ambient_signal.data[start:end],ambient_signal.sr),no_onset_signals_filename,True)
                    no_onset_file.write(str(d)[1:-1]+"\n")
        print("Created " + str(datapoints) + " datapoints")
        onset_file.close()
        no_onset_file.close()
