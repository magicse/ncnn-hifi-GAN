#include <omp.h>
#include <string>
#include <vector>
#include <ostream>
#include <random>
#include <chrono>

#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <hifivoice.h>

#define PBWIDTH 60
//#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

void printProgress(double percentage) {

    char PBSTR[] = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
    std::array<char, PBWIDTH> PBSTR_2;
    PBSTR_2.fill(254u);
    //PBSTR_2.fill(0xDB);

    int bar_size = PBWIDTH;
    int val = (int) (percentage );
    int lpad = (int) (percentage * bar_size/100);
    int rpad = bar_size - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR_2, rpad, "");
    fflush(stdout);
}


int main(int argc, char** argv)
{
    int opt = 0;
    char *mel = NULL;
    char *out_fname = NULL;

    while ((opt = getopt(argc, argv, "i:0:1:o:")) != -1) {
            switch(opt){
                case 'i':
                    mel = optarg;
                    printf("\nInput option value=%s", mel);
                    break;
                case 'o':
                    out_fname = optarg;
                    printf("\nOutput option value=%s", out_fname);
                    break;
                case '?':
                    if (optopt == 'i'){
                        printf("\nMissing input file name");
                    } else if (optopt == 'o') {
                         printf("\nMissing output file name");
                    } else {
                        printf("\nInvalid option received");
                        printf("\nUsage: hifivoice -i input file name -o output file name");
                    }
                    break;
            }

    }

    //printProgress(50);
    //char *argv_my[] = {"face", "./img/test.jpg", NULL };
    //char *argv_my[] = {"face", "./img/3.jpg", NULL };
    char *argv_my[] = {"mel", mel, NULL };
    printf("\npath = %s", mel);
    hifivoice_main(2, argv_my);
    printf("\nFinal");

    #pragma omp parallel
    {
        //printf("thread num %d: first hello world!\n",omp_get_thread_num());
        //printf("thread num %d; second hello world!\n",omp_get_thread_num());
    }
return 0;
}



