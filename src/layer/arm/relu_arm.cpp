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

#include "relu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU_arm)

ReLU_arm::ReLU_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int ReLU_arm::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt)
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            signed char* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 4;
            int remain = size - (nn << 4);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            int8x16_t _zero = vdupq_n_s8(0);
            for (; nn>0; nn--)
            {
                int8x16_t _p = vld1q_s8(ptr);
                _p = vmaxq_s8(_p, _zero);
                vst1q_s8(ptr, _p);

                ptr += 16;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "veor       q1, q0, q0          \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.s8    {d0-d1}, [%1 :128]  \n"
                "vmax.s8    q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.s8    {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                if (*ptr < 0)
                    *ptr = 0;

                ptr++;
            }
        }
    }
    else
    {
        // TODO
        // #pragma omp parallel for num_threads(opt.num_threads)
        // for (int q=0; q<channels; q++)
        // {
        //     float* ptr = bottom_top_blob.channel(q);

        //     for (int i=0; i<size; i++)
        //     {
        //         if (ptr[i] < 0)
        //             ptr[i] *= slope;
        //     }
        // }
    }

    return 0;
}

int ReLU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt)
{
    if (bottom_top_blob.elemsize == 1u)
        return ReLU_arm::forward_inplace_int8(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    if (elempack == 4)
    {
        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

#if __aarch64__
                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b \n"

                    "lsr    w4, %w2, #3             \n"// w4 = nn = size >> 3
                    "cmp    w4, #0                  \n"
                    "beq    1f                      \n"

                    "0:                             \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "fmax   v1.4s, v1.4s, v16.4s    \n"
                    "fmax   v2.4s, v2.4s, v16.4s    \n"
                    "fmax   v3.4s, v3.4s, v16.4s    \n"
                    "sub    %0, %0, #64             \n"
                    "fmax   v4.4s, v4.4s, v16.4s    \n"
                    "fmax   v5.4s, v5.4s, v16.4s    \n"
                    "fmax   v6.4s, v6.4s, v16.4s    \n"
                    "fmax   v7.4s, v7.4s, v16.4s    \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "subs   w4, w4, #1              \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "bne    0b                      \n"

                    "1:                             \n"

                    "and    w4, %w2, #7             \n"// w4 = remain = size & 7

                    "cmp    w4, #4                  \n"// w4 >= 4
                    "blt    2f                      \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "fmax   v1.4s, v1.4s, v16.4s    \n"
                    "fmax   v2.4s, v2.4s, v16.4s    \n"
                    "fmax   v3.4s, v3.4s, v16.4s    \n"
                    "sub    w4, w4, #4              \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "2:                             \n"

                    "cmp    w4, #2                  \n"// w4 >= 2
                    "blt    3f                      \n"
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld1    {v0.4s, v1.4s}, [%0]    \n"
                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "fmax   v1.4s, v1.4s, v16.4s    \n"
                    "sub    w4, w4, #2              \n"
                    "st1    {v0.4s, v1.4s}, [%0], #32 \n"
                    "3:                             \n"

                    "cmp    w4, #0                  \n"// w4 > 0
                    "beq    4f                      \n"
                    "prfm   pldl1keep, [%0, #128]   \n"
                    "ld1    {v0.4s}, [%0]           \n"
                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "st1    {v0.4s}, [%0], #16      \n"
                    "4:                             \n"

                    : "=r"(ptr)     // %0
                    : "0"(ptr),
                      "r"(size)     // %2
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16"
                );
#else // __aarch64__
                asm volatile(
                    "veor       q12, q12            \n"

                    "lsr        r4, %2, #3          \n"// r4 = nn = size >> 3
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "0:                             \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0!, {d0-d7}        \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d16-d23}       \n"
                    "vmax.f32   q0, q0, q12         \n"
                    "vmax.f32   q1, q1, q12         \n"
                    "vmax.f32   q2, q2, q12         \n"
                    "vmax.f32   q3, q3, q12         \n"
                    "sub        %0, %0, #64         \n"
                    "vmax.f32   q8, q8, q12         \n"
                    "vmax.f32   q9, q9, q12         \n"
                    "vmax.f32   q10, q10, q12       \n"
                    "vmax.f32   q11, q11, q12       \n"
                    "vstm       %0!, {d0-d7}        \n"
                    "subs       r4, r4, #1          \n"
                    "vstm       %0!, {d16-d23}      \n"
                    "bne        0b                  \n"

                    "1:                             \n"

                    "and        r4, %2, #7          \n"// r4 = remain = size & 7

                    "cmp        r4, #4              \n"// r4 >= 4
                    "blt        2f                  \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vmax.f32   q0, q0, q12         \n"
                    "vmax.f32   q1, q1, q12         \n"
                    "vmax.f32   q2, q2, q12         \n"
                    "vmax.f32   q3, q3, q12         \n"
                    "sub        r4, r4, #4          \n"
                    "vstm       %0!, {d0-d7}        \n"
                    "2:                             \n"

                    "cmp        r4, #2              \n"// r4 >= 2
                    "blt        3f                  \n"
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]  \n"
                    "vmax.f32   q0, q0, q12         \n"
                    "vmax.f32   q1, q1, q12         \n"
                    "sub        r4, r4, #2          \n"
                    "vst1.f32   {d0-d3}, [%0 :128]! \n"
                    "3:                             \n"

                    "cmp        r4, #0              \n"// r4 > 0
                    "beq        4f                  \n"
                    "pld        [%0, #128]          \n"
                    "vld1.f32   {d0-d1}, [%0 :128]  \n"
                    "vmax.f32   q0, q0, q12         \n"
                    "vst1.f32   {d0-d1}, [%0 :128]! \n"
                    "4:                             \n"

                    : "=r"(ptr)     // %0
                    : "0"(ptr),
                      "r"(size)     // %2
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
                );
#endif // __aarch64__
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(slope);
                for (int i=0; i<size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _zero = vdupq_n_f32(0.f);
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmaxq_f32(_p, _zero);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "veor       q1, q0, q0          \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmax.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr = std::max(*ptr, 0.f);

                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "veor       q1, q0, q0          \n"
                "vdup.f32   q2, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vcle.f32   q3, q0, q1          \n"
                "vmul.f32   q4, q0, q2          \n"
                "vbit.32    q0, q4, q3          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(slope)    // %4
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
