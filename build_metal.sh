#!/bin/bash

# Build script for ARBD with Metal support
# This script demonstrates how to properly configure CMake for Metal backend

set -e  # Exit on any error

echo "=== ARBD Metal Build Script ==="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: Metal is only supported on macOS"
    exit 1
fi

# Check if Xcode command line tools are installed
if ! xcode-select -p &> /dev/null; then
    echo "Error: Xcode command line tools are required for Metal support"
    echo "Please run: xcode-select --install"
    exit 1
fi

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build-metal"
CLEAN_BUILD=false
RUN_TESTS=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug       Build in Debug mode (default: Release)"
            echo "  --clean       Clean build directory before building"
            echo "  --test        Run Metal tests after building"
            echo "  --verbose     Enable verbose output"
            echo "  --build-dir   Specify build directory (default: build-metal)"
            echo "  --help        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --clean --test --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Build directory: $BUILD_DIR"
echo "  Clean build: $CLEAN_BUILD"
echo "  Run tests: $RUN_TESTS"
echo "  Verbose: $VERBOSE"

# Clean build directory if requested
if [[ "$CLEAN_BUILD" == "true" ]]; then
    echo ""
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
echo ""
echo "Configuring CMake with Metal support..."

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DUSE_METAL=ON
    -DUSE_CUDA=OFF          # Disable CUDA for Metal-only build
    -DUSE_SYCL_ACPP=OFF     # Disable SYCL for Metal-only build
    -DUSE_SYCL_ICPX=OFF     # Disable SYCL for Metal-only build
    -DUSE_PYBIND=OFF        # Disable Python bindings for simplicity
)

if [[ "$VERBOSE" == "true" ]]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build the project
echo ""
echo "Building project..."
if [[ "$VERBOSE" == "true" ]]; then
    make VERBOSE=1
else
    make -j$(sysctl -n hw.ncpu)  # Use all CPU cores
fi

echo ""
echo "Build completed successfully!"

# Show what was built
echo ""
echo "Built targets:"
ls -la arbd_tests 2>/dev/null && echo "  ✓ arbd_tests (test executable)"
ls -la arbd 2>/dev/null && echo "  ✓ arbd (main executable)"
ls -la src/libarbd.a 2>/dev/null && echo "  ✓ libarbd.a (static library)"

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    echo ""
    echo "Running Metal tests..."
    if [ -f "build-metal/arbd_tests" ]; then
        cd build-metal
        ./arbd_tests "[metal]"
        exit_code=$?
        cd ..
        if [ $exit_code -ne 0 ]; then
            echo "Error: Metal tests failed."
            exit 1
        fi
        echo "Metal tests passed."
    else
        echo "Error: Test executable not found"
        exit 1
    fi
fi

echo ""
echo "=== Build Summary ==="
echo "Metal backend build completed successfully!"
echo "Build directory: $BUILD_DIR"
echo ""
echo "To run tests manually:"
echo "  cd $BUILD_DIR"
echo "  ./arbd_tests \"[metal]\""
echo ""
echo "To run specific test categories:"
echo "  ./arbd_tests \"[metal][memory]\"     # Memory tests"
echo "  ./arbd_tests \"[metal][manager]\"    # Manager tests"
echo "  ./arbd_tests \"[metal][compute]\"    # Compute tests"
echo ""
echo "To run with verbose output:"
echo "  ./arbd_tests \"[metal]\" -s" 