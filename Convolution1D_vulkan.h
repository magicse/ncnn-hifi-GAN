// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTION1D_VULKAN_H
#define LAYER_CONVOLUTION1D_VULKAN_H

#include "layer.h"
//#include "convolution1d.h"

using namespace ncnn;

class Convolution1D_vulkan : virtual public ncnn::Layer
{
public:
    Convolution1D_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);
	virtual int create_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
	virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

protected:
    void make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, int kernel_w, const Option& opt) const;


public:
    ncnn::Layer* padding;
	mutable Mat weight_data_packed;
    Mat bias_data_packed;

	VkMat weight_data_gpu;
    VkMat bias_data_gpu;

	VkImageMat weight_data_gpu_image;
    VkImageMat bias_data_gpu_image;

    Pipeline* pipeline_convolution1d;
	Pipeline* pipeline_convolution1d_pack4;
	Pipeline* pipeline_convolution1d_pack1to4;
    Pipeline* pipeline_convolution1d_1x1s1d1;

    // param
    int num_output;
    int kernel_w;
    int dilation_w;
    int stride_w;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    float pad_value;
    int bias_term;

    int weight_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;

    // convolution as fc
    ncnn::Layer* reshape_1x1xw;
    ncnn::Layer* reshape_w;
};

#endif // LAYER_CONVOLUTION1D_VULKAN_H



