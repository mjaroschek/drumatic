# the method onset_detection computes a mean value of the signal over a window
# and compares it to previous windows via a linear function. If the value of
# this function is greater than some threshold, the window is classified as
# onset. The next few windows are then automatically classified as "no onset".

# The method onset_detection_freq works on the spectrogram of the signal. Onsets
# are detected via a simple vertical edge detection algorithm.

# The method onset_detection_model detects onsets via a trained ML model.


# The number of previous windows to which the current window is compared to.
win_buf_len=5

# The number of data points in a window.
win_size=200

# This determines how many data points are between the current window and the
# next one.
step_size=100

# This determines how many windows are set to "no onset" after an onset is
# detected.
sleep_after_onset=5

# The factor used to compare the avg amplitude of the current window to that of
# the past ones.
onset_factor=1

# The additive constant used to compare the avg amplitude of the current window to that of
# the past ones.
onset_delta=0.02

# The Mask used to weigh the values in a window.
mask=[1 for i in range(win_size)]

# Not used in the onset detection, but beforehand to reduce the sampling rate of
# signals.
samplefactor=5

# Determines the percentage of the number of frequencies (compared to the total
# number of frequencies) whose change in amplitude in an stft frame comapred to
# the previous frames exceeds a certain threshold
percentage=0.75

# Controls the difference in amplitude such that a frequency is considered for a
# possible detected onset.
onset_delta_spec=0.0005

# Controls how many STFT frames are used to perform onset detection on the
# spectrogram.
look_back=2

# Paths to the different files
root_folder="/home/maximilian/drummidi/"
path_to_mixed=root_folder + "mixed/"
path_to_onset=root_folder + "onsets/"
path_to_ambient=root_folder + "ambient/"
path_to_sound=root_folder + "drums/"
path_to_noise=root_folder + "noise/"
path_to_data=root_folder+"data/"
path_to_lists=root_folder+"yt_lists/"
onset_filename=path_to_data + "onset.txt"
onset_signals_filename=path_to_data + "onset.sig"
no_onset_filename=path_to_data + "no_onset.txt"
no_onset_signals_filename=path_to_data + "no_onset.sig"


# This controls whether the whole pipeline -- downloading from yt, detecting
# onsets, creating training data -- is performed on all specified yt videos, even
# if they were already processed.
process_all=True

# Options for STFT computation
stft_window_length=256
stft_window_overlap=int(stft_window_length/2)
