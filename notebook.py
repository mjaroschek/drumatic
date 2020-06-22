import os
os.chdir("/home/maximilian/drummidi/")

from readwav import *
from Pipeline import Pipeline
from Signal import *
from drum_environment import*

s=readwavefile("./drum.wav")
p=Pipeline([lambda x:downsample(x,samplefactor)])
p.add_function(lambda x:detect_hits(x,win_buf_len,win_size,step_size,mask,hit_delta,sleep_after_hit))
((s2,s3),(_,wins))=p.apply(s)
wins=wins[0]
visualize([s2,s3],wins)

