// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"

namespace ncnn {

class Convolution : public Layer
{
public:
    Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int create_requantize_op(void);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt);

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;// -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;

    Mat weight_data_int8_scales;
    float bottom_blob_int8_scale;
    float top_blob_int8_scale;

    bool use_int8_inference;
    bool use_int8_requantize;

    ncnn::Layer* quantize;
    std::vector<ncnn::Layer*> dequantize_ops;
    std::vector<ncnn::Layer*> requantize_ops;

    // merge de/requantize op into convolution op
    std::vector<float> dequantize_scales;
    std::vector<float> requantize_scales;

    // implementation type, 0 means do not use auto pack model
    int impl_type;

    #if BISONAI_KILL_THE_BITS
    std::vector<std::vector<int>> assignments;
    int original_input_channels;
    int reduced_input_channels;
    int input_feature_size;
    #endif
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_H
