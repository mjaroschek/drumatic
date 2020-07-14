# Notebook for creating training data for onset detection from youtube videos.

import os
os.chdir("/home/maximilian/drummidi/")

from youtube import *

download()
onset_detection()
mix()
create_training_data()
