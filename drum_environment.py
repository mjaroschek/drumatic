# the hit detection computes a mean value of the signal over a window and
# compares it to previous windows via a linear function. If the value of this
# function is greater than some threshold, the window is classified as
# onset. The next few windows are then automatically classified as "no onset".

# The number of previous windows to which the current window is compared to.
win_buf_len=5

# The number of data points in a window.
win_size=200

# This determines how many data points are between the current window and the
# next one.
step_size=100

# This determines how many windows are set to "no onset" after an onset is
# detected.
sleep_after_hit=5

# The factor used to compare the avg amplitude of the current window to that of
# the past ones.
hit_factor=1

# The additive constant used to compare the avg amplitude of the current window to that of
# the past ones.
hit_delta=0.02

# The Mask used to weigh the values in a window.
mask=[1 for i in range(win_size)]

# Not used in the onset detection, but beforehand to reduce the sampling rate of
# signals.
samplefactor=5

percentage=0.75
hit_delta_spec=0.0005
look_back=2
