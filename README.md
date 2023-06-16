# ncnn-hifi-GAN

![melgram_flipped](https://github.com/magicse/ncnn-hifi-GAN/assets/13585785/b1042b11-2efa-4442-9422-bf4a4d2b1d4e)

HiFi-GAN - GAN-based high-speed Neural Vocoder for Efficient and High Fidelity Speech Synthesis for TTS and Realistic Voice Conversion.

HiFi-GAN has improved the shortcomings of poor voice quality in previous GAN-based works.

The experimental results prove that HiFi-GAN can generate 22.05 kHz speech 13.4 times faster than autoregressive models.

In TTS based on deep learning, there are two stages to generate speech from text: 

(1) generate mel-spec from text, typically such as Tacotron and FastSpeech , 

(2) generate speech from mel-spec, such as WaveNet and WaveRNN .

The performance of WaveNet is almost the same as that of human speech, but the generation speed is too slow. Recently, GAN-based Vocoder, such as MelGAN, tries to further increase the speed of speech generation. However, this type of model sacrifices quality while improving efficiency. Therefore, researchers hope to have a Vocoder with both efficiency and quality, this is HiFi-GAN.



