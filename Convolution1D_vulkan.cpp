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
#include <vulkan/vulkan.h> // Include the Vulkan-Hpp library
#include <gpu.h>

#include "Convolution1D_vulkan.h"
//DEFINE_LAYER_CREATOR(Convolution1D_vulkan)
#include "fused_activation.h"
#include "layer_shader_type.h"
#include "mlayer_shader_type.h"
#include "layer_type.h"
#include <iostream>
#include <iomanip>

#include "convolution1d.comp.hex.h"
#include "convolution1d_pack4.comp.hex.h"
#include "convolution1d_pack1to4.comp.hex.h"


void pretty_printv2(const ncnn::Mat& m)
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
                    printf("%.6f ", ptr[x]);
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

Convolution1D_vulkan::Convolution1D_vulkan()
{
    one_blob_only = true;
	support_vulkan = true;
    support_image_storage = true;

    //support_packing = false;

    padding = 0;
	pipeline_convolution1d = 0;
	pipeline_convolution1d_pack4 = 0;
	pipeline_convolution1d_1x1s1d1 = 0;
    reshape_1x1xw = 0;
    reshape_w = 0;

}

int Convolution1D_vulkan::create_pipeline(const Option& _opt)
{
    std::cout << "=== Create Pipeline: ===" << std::endl;
    if (dynamic_weight)
    {
        support_vulkan = false;
        support_image_storage = false;
        return 0;
    }

   std::cout << "=== Create Pipeline: Layer Name & Index === " << type << " " << typeindex << std::endl;

    // Create a convolution pipeline using Vulkan
    Option opt = _opt;
    std::cout << "Bottoms Size: " << bottoms.size() << std::endl;
    std::cout << "Top Size: " << tops.size() << std::endl;
    std::cout << "Bottom_shapes Size: " << bottom_shapes.size() << std::endl;
    const Mat& Test = bottoms[0];
    const Mat& Test2 = bottom_shapes[0];

    std::cout << "Test: WxHxC x elempack " << Test.w << " " << Test.h  << " " << Test.c << " " << Test.elempack << std::endl;
    std::cout << "Test2: WxHxC x elempack " << Test2.w << " " << Test2.h  << " " << Test2.c << " " << Test2.elempack << std::endl;

	const Mat& shape =  bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::cout << "=== Create Pipeline: New shape from bottom_shapes WxHxC x Dims ===" << shape.w << " " << shape.h << " " << shape.c << " " << shape.d <<std::endl;
    std::cout << "=== Create Pipeline: New out_shape from top_shapes WxHxC x Dims ===" << out_shape.w << " " << out_shape.h << " " << out_shape.c << " " << out_shape.d <<std::endl;

    const int maxk = kernel_w;
    int num_input = weight_data_size / kernel_w / num_output;

    static std::vector<uint32_t> spirv;

    static ncnn::Mutex lock;
    {
        ncnn::MutexLockGuard guard(lock);
        if (spirv.empty())
        {
            compile_spirv_module(convolution1d_comp_data, sizeof(convolution1d_comp_data), opt, spirv);
        }
    }

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h, shape.c, (void*)0);
        }
        else if ((pad_left == -233 && pad_right == -233)
                 || (pad_left == -234 && pad_right == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            if (wpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h, shape.c, (void*)0);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;


    size_t elemsize;
    size_t out_elemsize;

    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_bordered_packed;
    //if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h, num_input / elempack, (void*)0, elemsize, elempack);
    if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, num_input / elempack, shape_bordered.c, (void*)0, elemsize, elempack);
    if (shape_bordered.dims == 2) shape_bordered_packed = Mat(shape_bordered.w, num_input / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    //if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w,  num_output / out_elempack, out_shape.c, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::cout << "=== Create Pipeline: New shape_bordered WxHxC x Dims ===" << shape_bordered.w << " " << shape_bordered.h << " " << shape_bordered.c << " " << shape_bordered.d <<std::endl;
    std::cout << "=== Create Pipeline: New shape_bordered_packed WxHxC x Dims ===" << shape_bordered_packed.w << " " << shape_bordered_packed.h << " " << shape_bordered_packed.c << " " << shape_bordered_packed.d <<std::endl;

    // fc
    if (kernel_w == 1)
    {
        {
            reshape_1x1xw = ncnn::create_layer(ncnn::LayerType::Reshape);
            reshape_1x1xw->vkdev = vkdev;

            reshape_1x1xw->bottom_shapes.resize(1);
            reshape_1x1xw->bottom_shapes[0] = Mat(num_input, (void*)0);
            reshape_1x1xw->top_shapes.resize(1);
            reshape_1x1xw->top_shapes[0] = Mat(1, num_input, (void*)0);

            ncnn::ParamDict pd;
            pd.set(0, num_input); // w

            reshape_1x1xw->load_param(pd);

            reshape_1x1xw->create_pipeline(opt);
        }

        {
            reshape_w = ncnn::create_layer(ncnn::LayerType::Reshape);
            reshape_w->vkdev = vkdev;

            reshape_w->bottom_shapes.resize(1);
            reshape_w->bottom_shapes[0] = Mat(1, num_output, (void*)0);
            reshape_w->top_shapes.resize(1);
            reshape_w->top_shapes[0] = Mat(num_output, (void*)0);

            ncnn::ParamDict pd;
            pd.set(0, num_output); // w

            reshape_w->load_param(pd);

            reshape_w->create_pipeline(opt);
        }
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1); // count shapes
        padding->bottom_shapes[0] = shape;
        std::cout << "=== Create Pipeline: Shape in padding WxHxC x Dims " << shape.w << " " << shape.h << " " << shape.c << " " << shape.d << std::endl;
        padding->top_shapes.resize(1); // count shapes
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        //pd.set(0, 0);
        //pd.set(1, 0);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);
        std::cout << "=== Create Pipeline: padding pipeline pad_left pad_right : ===" << pad_left <<" "<< pad_right << std::endl;
        padding->load_param(pd);

        padding->create_pipeline(opt);
    }
    
    std::cout << "=== Create Pipeline: data_packed.create maxk : ===" << maxk << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create num_input : ===" << num_input << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create elempack : ===" << elempack << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create num_input / elempack : ===" << num_input / elempack << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create num_output : ===" << num_output << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create out_elempack : ===" << out_elempack << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create num_output / out_elempack : ===" << num_output / out_elempack << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create (size_t)4 * elempack * out_elempack : ===" << (size_t)4 * elempack * out_elempack << std::endl;
    std::cout << "=== Create Pipeline: data_packed.create elempack * out_elempack : ===" << elempack * out_elempack << std::endl;
    std::cout << "=== Create Pipeline: weight_data WxHxC : === " << weight_data.w << " x " << weight_data.h << " x " << weight_data.c << std::endl;
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);
        convert_packing(weight_data_r2, weight_data_packed, out_elempack, opt);
        std::cout << "=== Create Pipeline: weight_data WxHxC reshaped : === " << weight_data_r2.w << " x " << weight_data_r2.h << " x " << weight_data_r2.c << std::endl;

/*
        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        const Mat k0 = weight_data_r2.channel(q + i);

                        for (int j = 0; j < elempack; j++)
                        {
                            const float* k00 = k0.row(p + j);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }
*/

    std::cout << "=== Create Pipeline: weight_data WxHxC reshaped packed : === " << weight_data_packed.w << " x " << weight_data_packed.h << " x " << weight_data_packed.c << std::endl;
    }
    std::cout << "=== Create Pipeline: weight data_packed.create finish : ===" << std::endl;

    if (bias_term)
    {
        pretty_printv2(bias_data);
        convert_packing(bias_data, bias_data_packed, out_elempack, opt);
    }


    if (is_conv1x1s1d1)
    {
        bool use_cooperative_matrix = vkdev->info.support_cooperative_matrix_16_8_8() && opt.use_cooperative_matrix && !opt.use_image_storage && !opt.use_shader_pack8 && opt.use_fp16_storage && num_input % 8 == 0 && num_output % 8 == 0;

        std::vector<vk_specialization_type> specializations(4 + 8);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape_bordered_packed.w;
        specializations[4 + 1].i = shape_bordered_packed.h;
        specializations[4 + 2].i = shape_bordered_packed.c;
        specializations[4 + 3].i = shape_bordered_packed.cstep;
        specializations[4 + 4].i = out_shape_packed.w;
        specializations[4 + 5].i = out_shape_packed.h;
        specializations[4 + 6].i = out_shape_packed.c;
        specializations[4 + 7].i = out_shape_packed.cstep;

        int shader_type_index = -1;
        if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_1x1s1d1;
        if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_1x1s1d1;
        if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_1x1s1d1;
        if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_1x1s1d1;
        if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_1x1s1d1;
        if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_1x1s1d1;
        if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_1x1s1d1;
        if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_1x1s1d1;
        if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_1x1s1d1;

        if (use_cooperative_matrix)
        {
            shader_type_index = LayerShaderType::convolution_pack4_1x1s1d1_cm_16_8_8;
        }

        pipeline_convolution1d_1x1s1d1 = new Pipeline(vkdev);
        if (use_cooperative_matrix)
        {
            // TODO proper unroll y
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(32, 4, 1); // 16_8_8 ly*4
        }
        else if (opt.use_shader_local_memory)
        {
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(8, 8, 1);
        }
        else
        {
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(8, std::min(8, num_output / out_elempack), 1);
        }
        pipeline_convolution1d_1x1s1d1->create(shader_type_index, opt, specializations);
    }
    else
    {
        std::vector<vk_specialization_type> specializations(7 + 10);
        specializations[0].i = kernel_w;
        specializations[1].i = dilation_w;
        specializations[2].i = stride_w;
        specializations[3].i = bias_term;
        specializations[4].i = activation_type;
        specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[7 + 0].i = shape_bordered_packed.dims;
        specializations[7 + 1].i = shape_bordered_packed.w;
        specializations[7 + 2].i = shape_bordered_packed.h;
        specializations[7 + 3].i = shape_bordered_packed.c;
        specializations[7 + 4].i = shape_bordered_packed.cstep;
        specializations[7 + 5].i = out_shape_packed.dims;
        specializations[7 + 6].i = out_shape_packed.w;
        specializations[7 + 7].i = out_shape_packed.h;
        specializations[7 + 8].i = out_shape_packed.c;
        specializations[7 + 9].i = out_shape_packed.cstep;

        std::cout << "=== Create Pipeline: Specializations: ===" << std::endl;
        std::cout << "=== Create Pipeline: kernel_w: " << specializations[0].i << std::endl;
        std::cout << "=== Create Pipeline: dilation_w: " << specializations[1].i << std::endl;
        std::cout << "=== Create Pipeline: stride_w: " << specializations[2].i << std::endl;
        std::cout << "=== Create Pipeline: bias_term: " << specializations[3].i << std::endl;
        std::cout << "=== Create Pipeline: activation_type: " << specializations[4].i << std::endl;
        std::cout << "=== Create Pipeline: activation_param_0: " << specializations[5].f << std::endl;
        std::cout << "=== Create Pipeline: activation_param_1: " << specializations[6].f << std::endl;
        std::cout << "=== Create Pipeline: dims: " << specializations[7 + 0].i << std::endl;
        std::cout << "=== Create Pipeline: w: " << specializations[7 + 1].i << std::endl;
        std::cout << "=== Create Pipeline: c: " << specializations[7 + 2].i << std::endl;
        std::cout << "=== Create Pipeline: cstep: " << specializations[7 + 3].i << std::endl;
        std::cout << "=== Create Pipeline: outdims: " << specializations[7 + 4].i << std::endl;
        std::cout << "=== Create Pipeline: outw: " << specializations[7 + 5].i << std::endl;
        std::cout << "=== Create Pipeline: outc: " << specializations[7 + 6].i << std::endl;
        std::cout << "=== Create Pipeline: outcstep: " << specializations[7 + 7].i << std::endl;

        Mat local_size_xyz(8, 8, std::min(4, (num_output / out_elempack + 1) / 2), (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(1, out_shape_packed.w);
            local_size_xyz.h = std::min(1, out_shape_packed.h);
            local_size_xyz.c = std::min(1, (out_shape_packed.c + 1) / 2);
        }

        int shader_type_index = -1;
        if (elempack == 1 && out_elempack == 1) {
                shader_type_index = LayerShaderType::convolution;
                std::cout << "=== Shader_type : === convolution" << std::endl;
        }
        if (elempack == 4 && out_elempack == 4){
                shader_type_index = LayerShaderType::convolution_pack4;
                std::cout << "=== Shader_type : === convolution_pack4" << std::endl;
        }
        if (elempack == 1 && out_elempack == 4) {
                shader_type_index = LayerShaderType::convolution_pack1to4;
                std::cout << "=== Shader_type : === convolution_pack1to4" << std::endl;
        }
        if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1;
        if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8;
        if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8;
        if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1;
        if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8;
        if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4;

        std::cout << "=== Shader_type_index : === " << shader_type_index << std::endl;

        pipeline_convolution1d = new Pipeline(vkdev);
        pipeline_convolution1d->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution1d->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack4
    {
        std::vector<vk_specialization_type> specializations(7 + 10);
        specializations[0].i = kernel_w;
        specializations[1].i = dilation_w;
        specializations[2].i = stride_w;
        specializations[3].i = bias_term;
        specializations[4].i = activation_type;
        specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[7 + 0].i = shape_bordered_packed.dims;
        specializations[7 + 1].i = shape_bordered_packed.w;
        specializations[7 + 2].i = shape_bordered_packed.h;
        specializations[7 + 3].i = shape_bordered_packed.c;
        specializations[7 + 4].i = shape_bordered_packed.cstep;
        specializations[7 + 5].i = out_shape_packed.dims;
        specializations[7 + 6].i = out_shape_packed.w;
        specializations[7 + 7].i = out_shape_packed.h;
        specializations[7 + 8].i = out_shape_packed.c;
        specializations[7 + 9].i = out_shape_packed.cstep;

        std::cout << "=== Create Pipeline: Pack4 Specializations: ===" << std::endl;
        std::cout << "=== Create Pipeline: Pack4 kernel_w: " << specializations[0].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 dilation_w: " << specializations[1].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 stride_w: " << specializations[2].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 bias_term: " << specializations[3].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 activation_type: " << specializations[4].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 activation_param_0: " << specializations[5].f << std::endl;
        std::cout << "=== Create Pipeline: Pack4 activation_param_1: " << specializations[6].f << std::endl;
        std::cout << "=== Create Pipeline: Pack4 dims: " << specializations[7 + 0].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 w: " << specializations[7 + 1].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 c: " << specializations[7 + 2].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 cstep: " << specializations[7 + 3].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 outdims: " << specializations[7 + 4].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 outw: " << specializations[7 + 5].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 outc: " << specializations[7 + 6].i << std::endl;
        std::cout << "=== Create Pipeline: Pack4 outcstep: " << specializations[7 + 7].i << std::endl;


        Mat local_size_xyz(8, 8, std::min(4, (num_output / out_elempack + 1) / 2), (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(1, out_shape_packed.w);
            local_size_xyz.h = std::min(1, out_shape_packed.h);
            local_size_xyz.c = std::min(1, (out_shape_packed.c + 1) / 2);
        }

        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(convolution1d_pack4_comp_data, sizeof(convolution1d_pack4_comp_data), opt, spirv);
            }
        }

        pipeline_convolution1d_pack4 = new Pipeline(vkdev);
        pipeline_convolution1d_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution1d_pack4->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack1to4
    {
        std::vector<vk_specialization_type> specializations(7 + 10);
        specializations[0].i = kernel_w;
        specializations[1].i = dilation_w;
        specializations[2].i = stride_w;
        specializations[3].i = bias_term;
        specializations[4].i = activation_type;
        specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[7 + 0].i = shape_bordered_packed.dims;
        specializations[7 + 1].i = shape_bordered_packed.w;
        specializations[7 + 2].i = shape_bordered_packed.h;
        specializations[7 + 3].i = shape_bordered_packed.c;
        specializations[7 + 4].i = shape_bordered_packed.cstep;
        specializations[7 + 5].i = out_shape_packed.dims;
        specializations[7 + 6].i = out_shape_packed.w;
        specializations[7 + 7].i = out_shape_packed.h;
        specializations[7 + 8].i = out_shape_packed.c;
        specializations[7 + 9].i = out_shape_packed.cstep;

        std::cout << "=== Create Pipeline: Pack1to4 Specializations: ===" << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 kernel_w: " << specializations[0].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 dilation_w: " << specializations[1].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 stride_w: " << specializations[2].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 bias_term: " << specializations[3].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 activation_type: " << specializations[4].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 activation_param_0: " << specializations[5].f << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 activation_param_1: " << specializations[6].f << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 dims: " << specializations[7 + 0].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 w: " << specializations[7 + 1].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 c: " << specializations[7 + 2].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 cstep: " << specializations[7 + 3].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 outdims: " << specializations[7 + 4].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 outw: " << specializations[7 + 5].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 outc: " << specializations[7 + 6].i << std::endl;
        std::cout << "=== Create Pipeline: Pack1to4 outcstep: " << specializations[7 + 7].i << std::endl;


        Mat local_size_xyz(8, 8, std::min(4, (num_output / out_elempack + 1) / 2), (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(1, out_shape_packed.w);
            local_size_xyz.h = std::min(1, out_shape_packed.h);
            local_size_xyz.c = std::min(1, (out_shape_packed.c + 1) / 2);
        }

        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(convolution1d_pack1to4_comp_data, sizeof(convolution1d_pack1to4_comp_data), opt, spirv);
            }
        }

        pipeline_convolution1d_pack1to4 = new Pipeline(vkdev);
        pipeline_convolution1d_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution1d_pack1to4->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int Convolution1D_vulkan::load_param(const ParamDict& pd)
{
    std::cout << "=== Load Param: ===" << std::endl;
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    return 0;
}

int Convolution1D_vulkan::load_model(const ModelBin& mb)
{
    std::cout << "=== Load Model Weight&Bias: ===" << std::endl;
    std::cout << "=== Load Model Weight Size: === " << weight_data_size << std::endl;
    if (dynamic_weight)
        return 0;

    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Convolution1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    std::cout << "=== Upload Model: ===" << std::endl;
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    std::cout << "=== Upload Model Weight_data_size: ===" << weight_data_size << std::endl;

    // Calculate the number of input channels
    int num_input = weight_data_size / kernel_w / num_output;
    std::cout << "=== Upload Model Num_input: === " << num_input << std::endl;

    // Upload weight data to Vulkan-specific resources
    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);
        std::cout << "=== Upload Model Converted to weight_data_gpu_image: === " << std::endl;
        weight_data_packed.release();
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
        std::cout << "=== Upload Model Converted to weight_data_gpu: WxHxC x Dims x Elempack x Elemsize === " << weight_data_gpu.w << " " << weight_data_gpu.h << " " << weight_data_gpu.c << " " << weight_data_gpu.dims << " " << weight_data_gpu.elempack << " " << weight_data_gpu.elemsize << std::endl;
        weight_data_packed.release();
    }

    // Load bias data if bias_term is true
    if (bias_term)
    {
        // Upload bias data to Vulkan-specific resources
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
            bias_data.release();
        }
        else
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
            bias_data.release();
        }
    }

    return 0;
}

int Convolution1D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution1d;
    pipeline_convolution1d = 0;

    delete pipeline_convolution1d_pack4;
    pipeline_convolution1d_pack4 = 0;


    delete pipeline_convolution1d_1x1s1d1;
    pipeline_convolution1d_1x1s1d1 = 0;

	    // fc
    if (reshape_1x1xw)
    {
        reshape_1x1xw->destroy_pipeline(opt);
        delete reshape_1x1xw;
        reshape_1x1xw = 0;
    }

    if (reshape_w)
    {
        reshape_w->destroy_pipeline(opt);
        delete reshape_w;
        reshape_w = 0;
    }
    return 0;
}


static int convolution1d_vulkan(const VkMat& bottom_blob, VkMat& top_blob, const VkMat& weight_data, const VkMat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

   /* #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float* sptr = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val = *sptr;
                    float wt = kptr[k];
                    sum += val * wt;

                    sptr += dilation_w;
                }

                kptr += kernel_w;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[j] = sum;
        }
    }*/

    return 0;
}

int Convolution1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::cout << "=== Forward 1 Evalute: ===" << std::endl;
    std::cout << "=== Forward 1 bottom_blob Elempack Elemsize: === " << bottom_blob.elempack << " " << bottom_blob.elemsize <<std::endl;

    int w = bottom_blob.w;
    int h = bottom_blob.h;

    std::cout << "=== Forward 1 bottom_blob shape W: === " << bottom_shapes[0].w << std::endl;
    int channels = bottom_blob.c;

    std::cout << "=== Forward 1 bottom_blob WxHxC x Dims: === " << bottom_blob.w <<" "<< bottom_blob.h << " " << bottom_blob.c <<" "<<bottom_blob.dims << std::endl;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            VkMat bottom_blob_1x1xw;
            {
                Option opt_reshape = opt;
                opt_reshape.blob_vkallocator = opt.workspace_vkallocator;

                reshape_1x1xw->forward(bottom_blob, bottom_blob_1x1xw, cmd, opt_reshape);
            }

            if (bottom_blob_1x1xw.empty())
                return -100;

            VkMat top_blob_1x1xw;
            int ret = forward(bottom_blob_1x1xw, top_blob_1x1xw, cmd, opt);
            if (ret != 0)
                return ret;

            return reshape_w->forward(top_blob_1x1xw, top_blob, cmd, opt);
        }
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;

    std::cout << "=== Forward 1 bottom_blob_bordered WxHxC x Dims : === " << bottom_blob_bordered.w <<" "<< bottom_blob_bordered.h <<" "<< bottom_blob_bordered.c <<" "<< bottom_blob_bordered.dims << std::endl;
    std::cout << "=== Forward 1 kernel_w : === " << kernel_w << std::endl;
    std::cout << "=== Forward 1 Pad_left Pad_right : === " << pad_left <<" "<< pad_right << std::endl;

    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;
            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    std::cout << "=== Forward 1 bottom_blob_bordered padded WxHxC x Dims : ===" << bottom_blob_bordered.w <<" "<< bottom_blob_bordered.h <<" "<< bottom_blob_bordered.c << " " << bottom_blob_bordered.d << std::endl;
    std::cout << "=== Forward 1 bottom_blob_bordered padded Elempack x Elemsize : ===" << bottom_blob_bordered.elempack <<" "<< bottom_blob_bordered.elemsize << std::endl;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = h;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    if (is_conv1x1s1d1)
    {
        bool use_cooperative_matrix = vkdev->info.support_cooperative_matrix_16_8_8() && opt.use_cooperative_matrix && !opt.use_image_storage && !opt.use_shader_pack8 && opt.use_fp16_storage && channels * elempack % 8 == 0 && num_output % 8 == 0;

        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = bottom_blob_bordered.w;
        constants[1].i = 1;
        constants[2].i = bottom_blob_bordered.c;
        constants[3].i = bottom_blob_bordered.cstep;
        constants[4].i = top_blob.w;
        constants[5].i = 1;
        constants[6].i = top_blob.c;
        constants[7].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
        dispatcher.h = top_blob.c;
        dispatcher.c = 1;

        if (use_cooperative_matrix)
        {
            dispatcher.w = ((top_blob.w * top_blob.h + 15) / 16 + 3) / 4 * 32;
            dispatcher.h = (top_blob.c + 1) / 2;
            dispatcher.c = 1;
        }

        cmd.record_pipeline(pipeline_convolution1d_1x1s1d1, bindings, constants, dispatcher);

        return 0;
    }


    top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    std::cout << "=== Forward 1 Num_output Out_elempack Out_elemsize: === " << num_output << " " << out_elempack << " " << out_elemsize <<std::endl;
    std::cout << "=== Forward 1 top_blob WxHxC x OutCstep: === " << top_blob.w << " " << top_blob.h << " " << top_blob.c << " " << top_blob.cstep << std::endl;
    std::cout << "=== Forward 1 bottom_blob_bordered Cstep : === " << bottom_blob_bordered.cstep << std::endl;


    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    VkMat dispatcher;
    //dispatcher.w = (top_blob.w + 1) / 2;
    //dispatcher.h = (top_blob.h + 1) / 2;
    //dispatcher.c = (top_blob.c + 1) / 2;

    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (top_blob.h + 1) / 2;
    dispatcher.c = 1;
    std::cout << "=== Forward 1 Dispatcher WxHxC : === " << dispatcher.w << " " << dispatcher.h << " " << dispatcher.c << std::endl;

    //dispatcher.w = 1;
    //dispatcher.h = 1;
    //dispatcher.c = 1;

    if (elempack == 4)
    {
        //const int maxk = kernel_w;
        //int num_input = weight_data_size / kernel_w / num_output;
        //Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);
        //std::cout << "=== Forward 1 : weight_data WxHxC reshaped : === " << weight_data_r2.w << " x " << weight_data_r2.h << " x " << weight_data_r2.c << std::endl;
        //convert_packing(weight_data_r2, weight_data_packed, out_elempack, opt);
        //std::cout << "=== Forward 1 : weight_data WxHxC reshaped packed : === " << weight_data_packed.w << " x " << weight_data_packed.h << " x " << weight_data_packed.c << std::endl;

        cmd.record_pipeline(pipeline_convolution1d_pack4, bindings, constants, dispatcher);
    }
    else // if (elempack == 1)
    {
        cmd.record_pipeline(pipeline_convolution1d_pack1to4, bindings, constants, dispatcher);
    }

    return 0;
}

//int Convolution1D_vulkan::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
int Convolution1D_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    std::cout << "=== Evalute forward 2: ===" << std::endl;
    return 0;
}

//void Convolution1D_vulkan::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
void Convolution1D_vulkan::make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, opt);
}

void Convolution1D_vulkan::make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, int _kernel_w, const Option& opt) const
{
    return 0;
}
