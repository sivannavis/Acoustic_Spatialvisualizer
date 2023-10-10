import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load("violin_metu_2.wav", mono=False)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.show()