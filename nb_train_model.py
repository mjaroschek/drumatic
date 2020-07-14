# Notebook for training a SVM to perform onset detection.

import os
os.chdir("/home/maximilian/drummidi/")

from drum_environment import *
import pickle
from model_trainer import *

(d1,y1,d2,y2,d3,y3)=read_data()
(m,score)=train_and_test(d1,y1,d2,y2,kernel="linear")
f=open(root_folder + "model.obj","wb")
pickle.dump(m,f)
f.close()
