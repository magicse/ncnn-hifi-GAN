#include <omp.h>
#include "net.h"
#include <iostream>
#include <iomanip>

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <sndfile.hh>

void pretty_printv(const ncnn::Mat& m)
{
    for (int q = 0; q < m.c; q++)
    {
        for (int i = 0; i < m.d; i++)
        {
            printf("Channel %d, Depth %d:\n", q+1, i+1);

            const float* ptr = m.channel(q).depth(i);
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%.4f ", ptr[x]);
                    //printf("%.6f ", ptr[x]);
                    //printf("%.3f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("------------------------\n");
        }
    }
}

static int HIFIVOICE(const cv::Mat& mel0)
{
    int MAX_WAV_VALUE = 32768.0;
    int n_fft = 1024;
    //Start
	ncnn::Net HIFIVOICE;
	HIFIVOICE.opt.num_threads = 4;
    HIFIVOICE.opt.use_vulkan_compute = true;

	if (HIFIVOICE.load_param("./models/hifivoice.ncnn.param"))
		exit(-1);
	if (HIFIVOICE.load_model("./models/hifivoice.ncnn.bin"))
		exit(-1);

    cv::Mat melscpectro;
    melscpectro = mel0.clone();
    cv::flip(melscpectro, melscpectro, 0);
    cv::imshow("Mel Spectro", melscpectro);
    cv::waitKey(0);
    // Convert to grayscale
    cv::cvtColor(melscpectro, melscpectro, cv::COLOR_BGR2GRAY);
    melscpectro.convertTo(melscpectro, CV_32F, 1.0/255);
    melscpectro = melscpectro * 13 - 11;
    double Min_mel_bank, Max_mel_bank;
    minMaxLoc(melscpectro, &Min_mel_bank, &Max_mel_bank);
    std::cout << "Max mel magnitude val: " << Max_mel_bank << std::endl;
    std::cout << "Min mel magnitude val: " << Min_mel_bank << std::endl;

    std::cout << melscpectro.size () << "; ch: " << melscpectro.channels () << std::endl;

    // cv::Mat a(h, w, CV_32FC1);
    ncnn::Mat MelIn(melscpectro.cols, melscpectro.rows, 1, (void*)melscpectro.data);
    MelIn = MelIn.clone();

    ncnn::Mat out;

    ncnn::Extractor ex = HIFIVOICE.create_extractor();
    ex.input("in0", MelIn);
    ex.extract("out0", out);
    std::cout << "Out matrix size W x H = " << out.w << " x "<< out.h <<" number of channels " << out.c <<std::endl;

    int numSamples = out.w * out.h;
    //max_magnitude = max_amplitude / (n_fft / 2)

    for (int i = 0; i < numSamples; i++)
    {
        out[i] = out[i]* MAX_WAV_VALUE;
    }
    //pretty_printv(out);

    const char* outputPath = "output.wav";
    SF_INFO sfInfoOut;
    sfInfoOut.samplerate = 22050;
    float* audioData = new float[numSamples];
    memcpy(audioData, out.data, sizeof(float) * numSamples);

    std::vector<int16_t> int16AudioData(numSamples);
    for (int i = 0; i < numSamples; i++)
    {
        int16AudioData[i] = static_cast<int16_t>(std::round(audioData[i]));
        if (int16AudioData[i] > std::numeric_limits<int16_t>::max())
        {
            int16AudioData[i] = std::numeric_limits<int16_t>::max();
        }
        else if (int16AudioData[i] < std::numeric_limits<int16_t>::min())
        {
            int16AudioData[i] = std::numeric_limits<int16_t>::min();
        }
    }


    sfInfoOut.channels = 1;
    sfInfoOut.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    sfInfoOut.frames = numSamples;

    SNDFILE* sndFileOut = sf_open(outputPath, SFM_WRITE, &sfInfoOut);
    if (!sndFileOut) {
        // Failed to open the output file
        // Handle the error accordingly
    }

    // Write the audio data to the output file
    sf_write_short(sndFileOut, int16AudioData.data(), numSamples);

    // Close the output file
    sf_close(sndFileOut);

    return 0;
}

int hifivoice_main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* melpath = argv[1];
    printf("imagepath0: %s\n", melpath);
    printf("argv[0]: %s\n", argv[0]);
    printf("argv[1]: %s\n", argv[1]);

    cv::Mat mel0 = cv::imread(melpath, 1);
    if (mel0.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", mel0);
        return -1;
    }

 	//opticalflow(fr0, fr1);
	HIFIVOICE(mel0);

    return 0;
}
