// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cast_arm.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Cast_arm)

Cast_arm::Cast_arm()
{
#if __ARM_NEON && (__ARM_FP & 2)
    support_packing = cpu_support_arm_vfpv4();
#endif // __ARM_NEON && (__ARM_FP & 2)
}

int Cast_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON && (__ARM_FP & 2)
    if (opt.use_packing_layout)
    {

    if (elempack == 4)
    {
        size_t out_elemsize = elemsize;
        if (type_to == 1)
        {
            // float32
            out_elemsize = 4 * elempack;
        }
        else if (type_to == 2)
        {
            // float16
            out_elemsize = 2 * elempack;
        }
        else if (type_to == 3)
        {
            // int8
            out_elemsize = elempack;
        }

        if (dims == 1)
        {
            top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        }
        else if (dims == 2)
        {
            top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        }
        else if (dims == 3)
        {
            top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = w * h * elempack;

        if (type_from == 1 && type_to == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int nn = size;

#if __aarch64__
                asm volatile(
                    "0:                             \n"
                    "ld1    {v0.4s}, [%1], #16      \n"
                    "fcvtn  v1.4h, v0.4s            \n"
                    "subs   %w0, %w0, #1            \n"
                    "st1    {v1.4h}, [%2], #8       \n"
                    "bne    0b                      \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
#else
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vcvt.f16.f32 d2, q0            \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d2}, [%2 :64]!     \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
#endif // __aarch64__
            }
        }

        if (type_from == 2 && type_to == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                int nn = size;

#if __aarch64__
                asm volatile(
                    "0:                             \n"
                    "ld1    {v0.4h}, [%1], #8       \n"
                    "fcvtl  v1.4s, v0.4h            \n"
                    "subs   %w0, %w0, #1            \n"
                    "st1    {v1.4s}, [%2], #16      \n"
                    "bne    0b                      \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
#else
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #64]           \n"
                    "vld1.s16   {d0}, [%1 :64]!     \n"
                    "vcvt.f32.f16 q1, d0            \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d2-d3}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
#endif // __aarch64__
            }
        }

        // TODO more cast type

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON && (__ARM_FP & 2)

    return Cast::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
