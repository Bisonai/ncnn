This repository is accompanying a blog post [Kill the bits and gain the speed?](https://martinkersner.com/2019/11/28/kill-the-bits/).

In the root directory, you can find two compilation scripts `build_kill_the_bits_aarch64.sh` and `build_kill_the_bits_armv7.sh` for `arm64` and `armv7`, respectively.
Experimental layer definitions are defined in `experiments/conv3x3` directory.

## Experiments
1. Make a directory at `/data/local/tmp/kill-the-bits` on your mobile device.

```bash
adb shell
cd /data/local/tmp
mkdir kill-the-bits
```

2. Compile inference engine.

```bash
./build_kill_the_bits_aarch64.sh
# ./build_kill_the_bits_armv7.sh
```
3. Copy `benchncnn` to your device.

```bash
adb push build-android-aarch64/benchmark/benchncnn /data/local/tmp/kill-the-bits
#adb push build-android-armv7/benchmark/benchncnn /data/local/tmp/kill-the-bits
```

4. Copy a directory `experiments/conv3x3` to `/data/local/tmp`

```bash
adb push experiments/conv3x3 /data/local/tmp/kill-the-bits
```

5. Launch experiments.

```
./benchncnn 7
# ./benchncnn 14
# ./benchncnn 28
```

![](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)
# ncnn

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/Tencent/ncnn/master/LICENSE.txt)
[![Build Status](https://travis-ci.org/Tencent/ncnn.svg?branch=master)](https://travis-ci.org/Tencent/ncnn)
[![Coverage Status](https://coveralls.io/repos/github/Tencent/ncnn/badge.svg?branch=master)](https://coveralls.io/github/Tencent/ncnn?branch=master)


ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn 从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能 APP，将 AI 带到你的指尖。ncnn 目前已在腾讯多款应用中使用，如 QQ，Qzone，微信，天天P图等。

---

### Current building status matrix

| System | CPU (32bit) | CPU (64bit) | GPU (32bit) | GPU (64bit) |
| :---: | :---: | :---: | :--: | :--: |
| Linux (GCC) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=linux-gcc&label=build)](https://travis-ci.org/Tencent/ncnn) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=linux-gcc-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| Linux (Clang) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=linux-clang&label=build)](https://travis-ci.org/Tencent/ncnn) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=linux-clang-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| Windows (VS2017) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=windows-vs2017&label=build)](https://travis-ci.org/Tencent/ncnn) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=windows-vs2017-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| MacOS | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=macos-clang&label=build)](https://travis-ci.org/Tencent/ncnn) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=macos-clang-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| Android | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=android-armv7&label=build)](https://travis-ci.org/Tencent/ncnn) | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=android-armv7-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=android-aarch64&label=build)](https://travis-ci.org/Tencent/ncnn) | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=android-aarch64-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| iOS | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=ios-iphoneos&label=build)](https://travis-ci.org/Tencent/ncnn) | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=ios-iphoneos&label=build)](https://travis-ci.org/Tencent/ncnn) | — | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=ios-iphoneos-gpu&label=build)](https://travis-ci.org/Tencent/ncnn) |
| iOS Simulator | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=ios-iphonesimulator&label=build)](https://travis-ci.org/Tencent/ncnn) | [![Build Status](https://badges.herokuapp.com/travis/Tencent/ncnn?env=NAME=ios-iphonesimulator&label=build)](https://travis-ci.org/Tencent/ncnn) | — | — |

---

### Support most commonly used CNN network
### 支持大部分常用的 CNN 网络

* Classical CNN: VGG AlexNet GoogleNet Inception ...
* Practical CNN: ResNet DenseNet SENet FPN ...
* Light-weight CNN: SqueezeNet MobileNetV1/V2/V3 ShuffleNetV1/V2 MNasNet ...
* Detection: MTCNN facedetection ...
* Detection: VGG-SSD MobileNet-SSD SqueezeNet-SSD MobileNetV2-SSDLite ...
* Detection: Faster-RCNN R-FCN ...
* Detection: YOLOV2 YOLOV3 MobileNet-YOLOV3 ...
* Segmentation: FCN PSPNet UNet ...

---

### HowTo

**[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build) on Linux / Windows / Raspberry Pi3 / Android / NVIDIA Jetson / iOS**

* [Build for NVIDIA Jetson](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-nvidia-jetson)
* [Build for Linux x86](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86)
* [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)
* [Build for MacOSX](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macosx)
* [Build for Raspberry Pi 3](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-raspberry-pi-3)
* [Build for ARM Cortex-A family with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
* [Build for iOS on MacOSX with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macosx-with-xcode)
* [Build for iOS on Linux with cctools-port](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-linux-with-cctools-port)
* [Build for Hisilicon platform with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-hisilicon-platform-with-cross-compiling)

**[download prebuild binary package for android and ios](https://github.com/Tencent/ncnn/releases)**

**[use ncnn with alexnet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet) with detailed steps, recommended for beginners :)**

**[ncnn 组件使用指北 alexnet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh) 附带详细步骤，新人强烈推荐 :)**

**[use netron for ncnn model visualization](https://lutzroeder.github.io/netron/)**

[ncnn low-level operation api](https://github.com/Tencent/ncnn/wiki/low-level-operation-api)

[ncnn param and model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)

[ncnn operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)

[how to implement custom layer step by step](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)

---

### FAQ

**[ncnn throw error](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-throw-error)**

**[ncnn produce wrong result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)**

**[ncnn vulkan](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan)**

---

### Features

* Supports convolutional neural networks, supports multiple input and multi-branch structure, can calculate part of the branch
* No third-party library dependencies, does not rely on BLAS / NNPACK or any other computing framework
* Pure C ++ implementation, cross-platform, supports android, ios and so on
* ARM NEON assembly level of careful optimization, calculation speed is extremely high
* Sophisticated memory management and data structure design, very low memory footprint
* Supports multi-core parallel computing acceleration, ARM big.LITTLE cpu scheduling optimization
* Supports GPU acceleration via the next-generation low-overhead vulkan api
* The overall library size is less than 700K, and can be easily reduced to less than 300K
* Extensible model design, supports 8bit quantization and half-precision floating point storage, can import caffe/pytorch/mxnet/onnx models
* Support direct memory zero copy reference load network model
* Can be registered with custom layer implementation and extended
* Well, it is strong, not afraid of being stuffed with 卷   QvQ

### 功能概述

* 支持卷积神经网络，支持多输入和多分支结构，可计算部分分支
* 无任何第三方库依赖，不依赖 BLAS/NNPACK 等计算框架
* 纯 C++ 实现，跨平台，支持 android ios 等
* ARM NEON 汇编级良心优化，计算速度极快
* 精细的内存管理和数据结构设计，内存占用极低
* 支持多核并行计算加速，ARM big.LITTLE cpu 调度优化
* 支持基于全新低消耗的 vulkan api GPU 加速
* 整体库体积小于 700K，并可轻松精简到小于 300K
* 可扩展的模型设计，支持 8bit 量化和半精度浮点存储，可导入 caffe/pytorch/mxnet/onnx 模型
* 支持直接内存零拷贝引用加载网络模型
* 可注册自定义层实现并扩展
* 恩，很强就是了，不怕被塞卷 QvQ

---
### supported platform matrix

* ✅ = known work and runs fast with good optimization
* ✔️ = known work, but speed may not be fast enough
* ❔ = shall work, not confirmed
* / = not applied

|    |Windows|Linux|Android|MacOS|iOS|
|---|---|---|---|---|---|
|intel-cpu|✔️|✔️|❔|✔️|/|
|intel-gpu|✔️|✔️|❔|❔|/|
|amd-cpu|✔️|✔️|❔|✔️|/|
|amd-gpu|✔️|✔️|❔|❔|/|
|nvidia-gpu|✔️|✔️|❔|❔|/|
|qcom-cpu|❔|✔️|✅|/|/|
|qcom-gpu|❔|✔️|✔️|/|/|
|arm-cpu|❔|❔|✅|/|/|
|arm-gpu|❔|❔|✔️|/|/|
|apple-cpu|/|/|/|/|✅|
|apple-gpu|/|/|/|/|✔️|


---

### Example project

* https://github.com/nihui/ncnn-android-squeezenet
* https://github.com/chehongshu/ncnnforandroid_objectiondetection_Mobilenetssd
* https://github.com/moli232777144/mtcnn_ncnn

![](https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-2.jpg)
![](https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-23.jpg)
![](https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-m.png)

### 技术交流QQ群：637093648(超多大佬)  答案：卷卷卷卷卷

---

### License

BSD 3 Clause
