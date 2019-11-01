// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <float.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
// initialize vulkan runtime before main()
GlobalGpuInstance g_global_gpu_instance;
#endif // NCNN_VULKAN

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const { return 0; }
    virtual int read(void* buf, int size) const { memset(buf, 0, size); return size; }
};

static int g_warmup_loop_count = 800; // BISONAI
static int g_loop_count = 4;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    DataReaderFromEmpty dr;
    net.load_model(dr);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
    // sleep(10); // BISONAI
#endif

    ncnn::Mat out;

    // warm up
    for (int i=0; i<g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ex.extract("output", out);
    }

    std::vector<double> times;

    for (int i=0; i<g_loop_count; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("data", in);
            ex.extract("output", out);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time = end-start;

        times.push_back(double(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()));
    }

    printf("%20s ", comment);
    for(const auto & t : times)
        printf("%f ", t);
    printf("\n");
}

int main(int argc, char** argv)
{
    int loop_count = 200;
    int num_threads = 1;
    int powersave = 0;
    int gpu_device = -1;
    int experiment_type = 7;

    if (argc >= 2)
    {
        experiment_type = atoi(argv[1]);
        if (experiment_type != 7 && experiment_type != 14 && experiment_type != 28)
        {
            printf("The only available experiments are for 7x7, 14x14, or 28x28 input sizes.\n");
            printf("Please select one of those: 7, 14, or 28.");
            exit(1);
        }
    }
    if (argc >= 3)
    {
        loop_count = atoi(argv[2]);
    }

    bool use_vulkan_compute = gpu_device != -1;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobBufferAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingBufferAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    // BISONAI: Convolution using packing on arm64 seems to be significantly slower.
    opt.use_packing_layout = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);

    if (experiment_type == 7)
    {
        // 7x7
        benchmark("conv3x3/conv2x32x3x3_2x32x7x7", ncnn::Mat(7, 7, 512), opt);
        benchmark("conv3x3/conv2x64x3x3_2x64x7x7", ncnn::Mat(7, 7, 512), opt);
        benchmark("conv3x3/conv2x128x3x3_2x128x7x7", ncnn::Mat(7, 7, 512), opt);
        benchmark("conv3x3/conv2x256x3x3_2x256x7x7", ncnn::Mat(7, 7, 512), opt);
    }
    else if (experiment_type == 14)
    {
        // 14x14
        benchmark("conv3x3/conv2x16x3x3_2x16x14x14", ncnn::Mat(14, 14, 256), opt);
        benchmark("conv3x3/conv2x32x3x3_2x32x14x14", ncnn::Mat(14, 14, 256), opt);
        benchmark("conv3x3/conv2x64x3x3_2x64x14x14", ncnn::Mat(14, 14, 256), opt);
        benchmark("conv3x3/conv2x128x3x3_2x128x14x14", ncnn::Mat(14, 14, 256), opt);
    }
    else if (experiment_type == 28)
    {
        // 28x28
        benchmark("conv3x3/conv2x8x3x3_2x8x28x28", ncnn::Mat(28, 28, 128), opt);
        benchmark("conv3x3/conv2x16x3x3_2x16x28x28", ncnn::Mat(28, 28, 128), opt);
        benchmark("conv3x3/conv2x32x3x3_2x32x28x28", ncnn::Mat(28, 28, 128), opt);
        benchmark("conv3x3/conv2x64x3x3_2x64x28x28", ncnn::Mat(28, 28, 128), opt);
    }

    return 0;
}
