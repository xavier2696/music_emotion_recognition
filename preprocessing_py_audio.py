#does not work!
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

[Fs, x] = audioBasicIO.readAudioFile("MEMD_audio/2.mp3")
print(x.shape)
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.500*Fs, 0.025*Fs)
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1])
plt.show()