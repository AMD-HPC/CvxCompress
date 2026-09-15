CC ?= gcc
CXX ?= g++

CFLAGS=-fopenmp -O3 -fPIC -g -Wno-unused-result
# Check if we're on an x86 architecture and if CC supports -mavx
ifeq ($(shell uname -m), x86_64)
    CFLAGS += -mavx
endif

LDFLAGS=-fopenmp -lm
TFLAG=-std=c++11

# Detect platform and set library extension accordingly
ifeq ($(shell uname), Darwin)
    LIB_EXT = dylib
	rflags = -install_name @rpath/libcvxcompress.dylib
else
    LIB_EXT = so
	rflags =
endif

BUILDDIR ?= build

OBJECTS=CvxCompress.o Wavelet_Transform_Slow.o Wavelet_Transform_Fast.o Run_Length_Encode_Slow.o Block_Copy.o Read_Raw_Volume.o

HIPCC ?= hipcc
# HIP_ARCH may be a space-separated list to build a fat binary that runs on
# several GPUs, e.g. HIP_ARCH="gfx942 gfx950" for an MI300x + MI355x binary.
HIP_ARCH ?= gfx90a
HIP_OFFLOAD = $(foreach a,$(HIP_ARCH),--offload-arch=$(a))
HIPCFLAGS = -O2 -std=c++17 -fopenmp
HIPLDFLAGS = -lm

# Only `test_wavelet_buffer_hip` requires `rocrand` ie: `-lrocrand `

# Could drop `hip/` directory and use `$(foreach OBJ,$(HIP_OBJECTS), hip/$(OBJ))`
HIP_OBJECTS=hip/hipCompress.o
HIP_API_HEADERS=hip/hipCompress.h hip/hipCompact.h hip/hipBlockCopy.h \
	hip/hipPlaneIO.h hip/hipWaveletBitmap.h hip/hipWaveletOctree.h \
	hip/hipWaveletQuadtree2D.h hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc

all: CvxCompress_Test CvxCompress_Test_Dyn Test_Compression Compress_SEAM_Basin Test_With_Generated_Input

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

lib: $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) -o libcvxcompress.$(LIB_EXT) $(OBJECTS) $(rflags)

libcvxcompress.$(LIB_EXT) : $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) -o libcvxcompress.$(LIB_EXT) $(OBJECTS)

Wavelet_Transform_Fast.o: Wavelet_Transform_Fast.cpp Ds79_Base.cpp Us79_Base.cpp
	$(CXX) -c $(CFLAGS) $<

CvxCompress_Test: CvxCompress_Test.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(TFLAG) $(OBJECTS)  CvxCompress_Test.o -o CvxCompress_Test

CvxCompress_Test_Dyn: CvxCompress_Test.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) CvxCompress_Test.o  -L. -lcvxcompress -o $@

CvxCompress_GenCode: CvxCompress_GenCode.o CvxCompress.hxx Wavelet_Transform_Slow.o
	$(CXX) -O2 Wavelet_Transform_Slow.o  CvxCompress_GenCode.o -o CvxCompress_GenCode

Test_Compression: Test_Compression.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(TFLAG) $(OBJECTS)  Test_Compression.o -o Test_Compression

Ds79_Base.cpp Us79_Base.cpp: CvxCompress.hxx CvxCompress_GenCode
	./CvxCompress_GenCode

Compress_SEAM_Basin: Compress_SEAM_Basin.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) $(TFLAG) $<  -L. -lcvxcompress  -o $@

Test_With_Generated_Input: Test_With_Generated_Input.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) $(TFLAG) $<  -L. -lcvxcompress  -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(TFLAG) $*.cpp

# ---------------------------------------------------------------------------
# HIP GPU tests
# ---------------------------------------------------------------------------

libhipcvxcompress.$(LIB_EXT) : $(HIP_OBJECTS)
	$(HIPCC) -shared $(HIPLDFLAGS) -o libhipcvxcompress.$(LIB_EXT) $(HIP_OBJECTS)

hip/hipCompress.o: hip/hipCompress.cpp $(HIP_API_HEADERS)
	$(HIPCC) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -fPIC -c $(HIPCFLAGS) $< -o $@

hip/%.o: hip/%.cpp
	$(HIPCC) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -fPIC -c $(HIPCFLAGS) hip/$*.cpp -o $@

# Buffer-instruction wavelet kernel test
test_wavelet_buffer_hip: tests/test_wavelet_buffer_hip.cpp hip/hipWaveletTransformBuffer.cpp | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -save-temps=obj -DBUILDDIR=\"$(BUILDDIR)\" -I. -Ihip -Itests -lrocrand -fopenmp tests/test_wavelet_buffer_hip.cpp hip/hipWaveletTransformBuffer.cpp $(HIPLDFLAGS) -o $(BUILDDIR)/test_wavelet_buffer_hip

# Inverse wavelet transform unit test
test_inverse_wavelet_hip: tests/test_inverse_wavelet_hip.cpp hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_inverse_wavelet_hip.cpp -lm -o $(BUILDDIR)/test_inverse_wavelet_hip

# hipCompress public API test
test_compress_api_hip: tests/test_compress_api_hip.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_compress_api_hip.cpp hip/hipCompress.cpp -L. -lcvxcompress '-Wl,-rpath,$$ORIGIN/..' -lm -o $(BUILDDIR)/test_compress_api_hip

# 2D compression test
test_compress_2d_hip: tests/test_compress_2d_hip.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_compress_2d_hip.cpp hip/hipCompress.cpp -lm -o $(BUILDDIR)/test_compress_2d_hip

# 2D quadtree (GPU) vs CvxCompress (CPU) rate-distortion comparison on real data
bench_quadtree_vs_cvx_2d: tests/bench_quadtree_vs_cvx_2d.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) CvxCompress.hxx libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/bench_quadtree_vs_cvx_2d.cpp hip/hipCompress.cpp -L. -lcvxcompress '-Wl,-rpath,$$ORIGIN/..' -lm -o $(BUILDDIR)/bench_quadtree_vs_cvx_2d

# Async / aux-stream behaviour on 3D data: split-vs-serial equivalence, input
# lifetime, and measured overlap of the aux tail with caller work.
test_async_overlap_3d: tests/test_async_overlap_3d.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_async_overlap_3d.cpp hip/hipCompress.cpp -L. -lcvxcompress '-Wl,-rpath,$$ORIGIN/..' -lm -o $(BUILDDIR)/test_async_overlap_3d

# Full-encode throughput, one codec at a time (argv[5] selects it). No
# -lcvxcompress: the codec kernels live in the headers, so the binary must
# compile its own copy or an A/B on a header change compares two identical .so's.
bench_encode_full: tests/bench_encode_full.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/bench_encode_full.cpp hip/hipCompress.cpp -lm -o $(BUILDDIR)/bench_encode_full

# Async pipeline example (for profiling)
example_async_pipeline: tests/example_async_pipeline.cpp hip/hipCompress.cpp $(HIP_API_HEADERS) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) $(HIP_OFFLOAD) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/example_async_pipeline.cpp hip/hipCompress.cpp -lm -o $(BUILDDIR)/example_async_pipeline

scan_raw_scales: tests/scan_raw_scales.cpp CvxCompress.hxx libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(CXX) $(CFLAGS) $(TFLAG) -I. tests/scan_raw_scales.cpp -L. -lcvxcompress '-Wl,-rpath,$$ORIGIN/..' -o $(BUILDDIR)/scan_raw_scales

clean:
	rm -f *.o
	rm -f libcvxcompress.$(LIB_EXT) CvxCompress_Test CvxCompress_Test_Dyn CvxCompress_GenCode Test_Compression Compress_SEAM_Basin Test_With_Generated_Input
	rm -f Ds79_Base.cpp Us79_Base.cpp
	rm -f hip/*.o
	rm -f libhipcvxcompress.$(LIB_EXT)
	rm -rf $(BUILDDIR)
