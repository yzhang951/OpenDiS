#!/bin/sh
# Example: 
#   rm -rf build/; ./configure.sh -DCMAKE_BUILD_TYPE:STRING=Debug

mkdir -p build
cd build
echo "cd build ; cmake $@ -S .."
echo ""
cmake $@ -S ..

echo ""
echo "to build and install:"
echo "  cmake --build build -j 4 ; cmake --build build --target install"
echo ""
