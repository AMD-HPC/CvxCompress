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
HIP_ARCH ?= gfx90a
HIPCFLAGS = -O2 -std=c++17 -fopenmp
HIPLDFLAGS = -lm -lrocrand

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

# Buffer-instruction wavelet kernel test
test_wavelet_buffer_hip: tests/test_wavelet_buffer_hip.cpp hip/hipWaveletTransformBuffer.cpp | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -save-temps=obj -DBUILDDIR=\"$(BUILDDIR)\" -I. -Ihip -Itests -fopenmp tests/test_wavelet_buffer_hip.cpp hip/hipWaveletTransformBuffer.cpp $(HIPLDFLAGS) -o $(BUILDDIR)/test_wavelet_buffer_hip

# Quantize + RLE z-line unit test (CPU-only, no HIP)
test_quantize_rle: tests/test_quantize_rle.cpp hip/quantize_rle_ref.h Run_Length_Escape_Codes.hxx | $(BUILDDIR)
	$(CXX) -O2 $(TFLAG) -I. -Ihip -Itests tests/test_quantize_rle.cpp -o $(BUILDDIR)/test_quantize_rle

# Z-line vs full-block CR benchmark (CPU-only, links libcvxcompress)
test_zline_cr_benchmark: tests/test_zline_cr_benchmark.cpp hip/quantize_rle_ref.h libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(CXX) $(CFLAGS) $(TFLAG) -I. -Ihip -Itests tests/test_zline_cr_benchmark.cpp -L. -lcvxcompress -o $(BUILDDIR)/test_zline_cr_benchmark

# GPU quantize+RLE encode test (validates against CPU reference)
test_quantize_rle_hip: tests/test_quantize_rle_hip.cpp hip/quantize_rle_ref.h Run_Length_Escape_Codes.hxx hip/hipQuantizeRLE.h | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -I. -Ihip -Itests tests/test_quantize_rle_hip.cpp -lm -o $(BUILDDIR)/test_quantize_rle_hip

# GPU quantize+RLE encode performance benchmark
test_quantize_rle_perf_hip: tests/test_quantize_rle_perf_hip.cpp Run_Length_Escape_Codes.hxx | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -save-temps=obj -I. -Ihip -Itests tests/test_quantize_rle_perf_hip.cpp -lm -o $(BUILDDIR)/test_quantize_rle_perf_hip

# Fused inverse (decode + inverse wavelet) test
test_inverse_fused_hip: tests/test_inverse_fused_hip.cpp hip/hipWaveletRLEInverse.h hip/hipWaveletRLE.h hip/hipRLEDecode.h hip/hipWaveletTransformBuffer.cpp hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_inverse_fused_hip.cpp hip/hipWaveletTransformBuffer.cpp -lm -o $(BUILDDIR)/test_inverse_fused_hip

# Z-line RLE decoder unit test
test_rle_decode_hip: tests/test_rle_decode_hip.cpp hip/hipRLEDecode.h hip/quantize_rle_ref.h Run_Length_Escape_Codes.hxx | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -I. -Ihip -Itests tests/test_rle_decode_hip.cpp -lm -o $(BUILDDIR)/test_rle_decode_hip

# Inverse wavelet transform unit test
test_inverse_wavelet_hip: tests/test_inverse_wavelet_hip.cpp hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_inverse_wavelet_hip.cpp -lm -o $(BUILDDIR)/test_inverse_wavelet_hip

# Fused wavelet+RLE kernel test and benchmark
test_wavelet_rle_fused_hip: tests/test_wavelet_rle_fused_hip.cpp hip/hipWaveletRLE.h hip/hipWaveletRLEInverse.h hip/hipRLEDecode.h hip/hipQuantizeRLE.h hip/hipWaveletTransformBuffer.cpp libcvxcompress.$(LIB_EXT) hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -mllvm -unroll-threshold=10000 -save-temps=obj -I. -Ihip -Itests tests/test_wavelet_rle_fused_hip.cpp hip/hipWaveletTransformBuffer.cpp -L. -lcvxcompress -lm -o $(BUILDDIR)/test_wavelet_rle_fused_hip

# hipCompress public API test
test_compress_api_hip: tests/test_compress_api_hip.cpp hip/hipCompress.cpp hip/hipCompress.h hip/hipBlockCopy.h hip/hipWaveletRLE.h hip/hipWaveletRLEInverse.h hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc libcvxcompress.$(LIB_EXT) | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/test_compress_api_hip.cpp hip/hipCompress.cpp -L. -lcvxcompress -lm -o $(BUILDDIR)/test_compress_api_hip

# Async pipeline example (for profiling)
example_async_pipeline: tests/example_async_pipeline.cpp hip/hipCompress.cpp hip/hipCompress.h hip/hipBlockCopy.h hip/hipWaveletRLE.h hip/hipWaveletRLEInverse.h hip/ds79.h hip/us79_reg32.inc hip/ds79_reg32.inc | $(BUILDDIR)
	$(HIPCC) $(HIPCFLAGS) --offload-arch=$(HIP_ARCH) -mllvm -unroll-threshold=10000 -I. -Ihip -Itests tests/example_async_pipeline.cpp hip/hipCompress.cpp -lm -o $(BUILDDIR)/example_async_pipeline

clean:
	rm -f *.o
	rm -f libcvxcompress.$(LIB_EXT) CvxCompress_Test CvxCompress_Test_Dyn CvxCompress_GenCode Test_Compression Compress_SEAM_Basin Test_With_Generated_Input
	rm -f Ds79_Base.cpp Us79_Base.cpp
	rm -rf $(BUILDDIR)
