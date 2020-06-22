#!/usr/bin/env python3
import sys
from readwav import readwavefile

process_all=True
path="/home/maximilian/drummidi/drums/"
writepath="/home/maximilian/drummidi/hits/"

ids=[i.split(".wav")[0] for i in os.listdir(path)]

j=0
for i in ids:
    print("hier")
    j=j+1
    filename=path + i +".wav"
    filewritepath=writepath

    if os.path.isfile(filewritepath+i+".txt") and not process_all:
        continue
    
    print("Processing: " + i + "    ("+str(j)+"/"+str(len(ids))+")")

    data,sr=readwavefile(filename)

    detect_hits(data)

    try:
        os.makedirs(filewritepath)
    except FileExistsError:
        pass
    f=open(filewritepath + i + ".txt","x")
    f.write(str(hits))
    f.close()
