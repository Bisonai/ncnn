#!/usr/bin/env bash

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="armeabi-v7a" \
    -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-19 \
    -DBISONAI_KILL_THE_BITS=ON \
    ..
make -j4
make install
popd
