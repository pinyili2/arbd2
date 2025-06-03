rm -rf build
mkdir build && cd build
/opt/homebrew/bin/cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DUSE_SYCL_ACPP=ON -DCMAKE_CXX_COMPILER=acpp ..
make -j 8
