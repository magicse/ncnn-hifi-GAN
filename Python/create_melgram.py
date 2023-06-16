import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav

if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

n_fft = 1024
num_mels = 80
sampling_rate = 22050
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000
save_path = 'spectrogram.png'

def get_mel(x):
    return mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)

wav, sr = load_wav("Hi_How_are_You.wav")

wav = torch.FloatTensor(wav).to(device)
x = get_mel(wav.unsqueeze(0))

#torch.set_printoptions(linewidth=1000)
#torch.set_printoptions(threshold=float('inf'))
print ("x shape", x.shape, "data", x)
#torch.set_printoptions(linewidth=75)
print("Max value:", x.max())
print("Min value:", x.min())


# Convert the spectrogram to a NumPy array
spec_np = x.detach().cpu().numpy()
spec_np = spec_np[0]  # Remove the batch dimension if present

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(spec_np, origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Frame')
plt.ylabel('Mel Filter')
plt.title('Mel Spectrogram')

# Save the image if save_path is provided
if save_path is not None:
    plt.savefig(save_path)

# Show the plot
plt.show()

# Normalize the melgram values between 0 and 255
spec_np_normalized = (spec_np - np.min(spec_np)) / (np.max(spec_np) - np.min(spec_np))
spec_np_normalized *= 255

# Convert the normalized melgram to uint8
spec_np_uint8 = spec_np_normalized.astype(np.uint8)

# Flip the melgram horizontally
spec_np_flipped = cv2.flip(spec_np_uint8, 0)

# Save the flipped melgram as an image using OpenCV
cv2.imwrite('melgram_flipped.jpg', spec_np_flipped)
