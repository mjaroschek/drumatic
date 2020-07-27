
import os
os.chdir("/home/maximilian/drummidi/")

from experimental import *

s1=read_wavefile(root_folder + "drum_short.wav");
#s1=read_wavefile(root_folder + "drum.wav");
#s1=read_wavefile(root_folder + "drum_problem.wav");
#s1=read_wavefile(root_folder + "test_mixed.wav");
#s1=read_wavefile(root_folder + "test_mixed2.wav");
#s1=read_wavefile(root_folder + "test_noise.wav");
#s1=read_wavefile(root_folder + "test_speech_short.wav");
original=s1.spec_scaled()

s1=pre_emphasis_filter(s1)
fbank=init_mel_filter(s1.sr,s1.stft_window_length)
spec=mel_scale_filter(s1,fbank)
spec=move_up(spec)
filtered=spec

#spec=max_up(spec,11)

#k=[[1,2,1],[0,0,0],[-1,-2,-1]] #sobel
k=[[1,1,1],[0,0,0],[-1,-1,-1]] #prewitt

spec=convolve(spec,k)
convolved=spec
spec=[[10. * np.log10(i) for i in j] for j in spec]
logged=spec
#spec=move_up(spec)
refined=refine(spec)

detected=[0 for i in s1.data]
for i,s in enumerate(transpose(refined)):
    if s[0]==1:
        detected[i*stft_window_overlap:(i+1)*stft_window_overlap]=[1]*stft_window_overlap

fig, axs = plt.subplots(nrows=7)
extent = s1.t()[0], s1.t()[-1], s1.freqs()[0], s1.freqs()[-1]
axs[0].plot(s1.data)
axs[0].set_xticks([])
axs[0].set_xmargin(0)
axs[0].set_ybound(lower=-1,upper=1)
axs[0].axis('auto')
axs[1].imshow(np.flipud(original),extent=extent)
axs[1].set_xticks([])
axs[1].axis('auto')
axs[2].imshow(np.flipud(filtered),extent=extent)
axs[2].set_xticks([])
axs[2].axis('auto')
axs[3].imshow(np.flipud(convolved),extent=extent)
axs[3].set_xticks([])
axs[3].axis('auto')
axs[4].imshow(np.flipud(logged),extent=extent)
axs[4].set_xticks([])
axs[4].axis('auto')
axs[5].imshow(np.flipud(refined),extent=extent)
axs[5].set_xticks([])
axs[5].axis('auto')
axs[6].plot(detected)
axs[6].set_xticks([])
axs[6].set_xmargin(0)
axs[6].set_ybound(lower=-1,upper=1)
axs[6].axis('auto')
plt.show()
plt.close()
