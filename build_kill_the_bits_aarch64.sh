#!/usr/bin/env bash

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 \
    -DBISONAI_KILL_THE_BITS=ON \
    ..
make -j4
make install
popd
