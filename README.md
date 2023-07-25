# ncnn-hifi-GAN

![img](https://github.com/magicse/ncnn-hifi-GAN/assets/13585785/d788069e-769d-4b8d-b13f-4783d7c446aa)

VULKAN support in progress...

HiFi-GAN - GAN-based high-speed Neural Vocoder for Efficient and High Fidelity Speech Synthesis in TTS pipeline and Realistic Voice Conversion.

HiFi-GAN has improved the shortcomings of poor voice quality in previous GAN-based works.

The experimental results prove that HiFi-GAN can generate 22.05 kHz speech 13.4 times faster than autoregressive models.

In TTS based on deep learning, there are two stages to generate speech from text: 

1. generate mel-spec from text, typically such as Tacotron and FastSpeech , 
2. generate speech from mel-spec, such as WaveNet and WaveRNN .

The performance of WaveNet is almost the same as that of human speech, but the generation speed is too slow. Recently, GAN-based Vocoder, such as MelGAN, tries to further increase the speed of speech generation. However, this type of model sacrifices quality while improving efficiency. Therefore, researchers hope to have a Vocoder with both efficiency and quality, this is HiFi-GAN.
## How to use.
1. Download model [hifivoice](https://github.com/magicse/ncnn-hifi-GAN/releases/download/models/models.zip) and place it in /models folder. 
2. hifivoice.exe -i melgram_flipped.jpg
3. The input range of the mel-spectrogram for the vocoder is approximately from -11 to 2. 
For example, we take a mel-spectrogram saved in a regular jpg file with a magnitude range of 0..255. 
To use mel-spectrogram from a picture, the values need to be scaled. Mel_Image = Mel_Image * (1/255) * 13 - 11 = we get a range of values from -11 to 2.
4. Mel spectrogram paramters:
   - n_fft = 1024
   - num_mels = 80
   - sampling_rate = 22050
   - hop_size = 256
   - win_size = 1024
   - fmin = 0
   - fmax = 8000   

[NCNN](https://github.com/Tencent/ncnn) is a high-performance neural network.

[HiFi-GAN](https://github.com/jik876/hifi-gan) Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.


