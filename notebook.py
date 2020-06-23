import os
os.chdir("/home/maximilian/drummidi/")

from readwav import *
from Pipeline import Pipeline
from Signal import *
from drum_environment import*

#s=readwavefile("./drum.wav")
s=readwavefile("./drums/4VGLGCEKDs0.wav")
p=Pipeline([lambda x:downsample(x,samplefactor)])
# p.add_function(lambda x:[1,Signal(x.data[19500:21000],x.sr)])
# p.add_function(lambda x:onset_detection(x,win_buf_len,win_size,step_size,mask,hit_factor,hit_delta,sleep_after_hit))
# ((s1,s2,s3),(_,_,wins))=p.apply(s)
p.add_function(lambda x:onset_detection(x,win_buf_len,win_size,step_size,mask,hit_factor,hit_delta,sleep_after_hit))
((s2,s3),(_,wins))=p.apply(s)
wins=wins[0]
visualize([s2,s3],wins)

