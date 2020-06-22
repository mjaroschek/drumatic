from pytube import YouTube
import moviepy.editor as mp
import re
import os

# Set path where the yt lists are stored
listpath="./yt_lists/"
# specify if you want to redownload already downloaded videos
dl_all=False

files = os.listdir(listpath)

for file in files:
    f = open(listpath + file)
    url_list=f.read().splitlines()
    path="./" + file.split(".")[0] +"/"
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    processedIDs=[i[:-4] for i in os.listdir(path)]
    f.close()

    for url in url_list:
        name=url.split("v=")[-1]
        if dl_all or not name in processedIDs:
            yt = YouTube(url)
            yt_stream_audio = yt.streams.filter(only_audio=True)
            yt_stream_audio[0].download(output_path=path, filename=name)
            clip = mp.AudioFileClip(path + name +  ".mp4")
            clip.write_audiofile(path + name +".wav")
            clip.close()
            os.remove(path+name+".mp4")
