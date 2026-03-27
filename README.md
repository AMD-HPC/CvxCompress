# CvxCompres library
This CvxCompress was originally authored by Thor Johnsen, based on an implementation by Ray Ergas. 
For more information here are references to that earlier work. 

```
Ergas, R.A., "Wavelet Transform Techniques for Seismic Data Compression and Denoising," Society of Exploration Geophysicists Annual Meeting, 1995.

Ergas, R.A., Donoho, P.L., and Villasenor, J.D., "Multiresolution Analysis for Seismic Data Compression Using Wavelets," IEEE Transactions on Signal Processing, 1995.

Ergas, R.A., "Seismic data compression--A key technology for the future", Society of Exploration Geophysicists Annual Meeting, 1996.
https://library.seg.org/doi/pdf/10.1190/1.1826518
```

# HIP GPU Port

> **Status: Experimental / Proof of Concept**
> - APIs are unstable and may change without notice
> - Optimizations are ongoing
> - Not tested in an integrated production setting
> - No backward compatibility guarantee for the compressed bitstream format
> - AMD GPUs only (ROCm 7.x, gfx90a/gfx942)
> - Compression ratios differ from the CPU reference due to different block tiling strategies

See [HIP_API.md](HIP_API.md) for the AMD GPU port of this library targeting
MI200/MI300X via HIP. It implements a fused wavelet + quantization + RLE pipeline
in single GPU kernels, achieving 14–27x speedup over the fully-parallelized CPU
reference (128-core EPYC 9554, AVX, best thread count) with matching error norms
(to floating-point rounding). Includes API reference, usage examples, and async
pipeline integration.

# Building

## GCC

It is believed that this will work any reasonably modern gcc version. 

## Tests

There are two test sets included, below is the command line to run and the output.   

```
Prompt> ( setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${cwd} ; ./CvxCompress_Test )
*
* CvxCompress module tests (GCC13.2, AVX).
*

2. Verify correctness of forward wavelet transform...[Passed!]

3. Verify correctness of inverse wavelet transform...[Passed!]

4. Test throughput of wavelet transform (forward + inverse)...
 ->   8 x   8 x   8 ( L1 ) ::  5.009 secs - 25725 MCells/s - 3106 GF/s
 ->  16 x  16 x  16 ( L1 ) ::  3.770 secs - 34178 MCells/s - 4422 GF/s
 ->  32 x  32 x  32 ( L2 ) ::  5.681 secs - 22680 MCells/s - 3032 GF/s
 ->  64 x  64 x  64 ( L3 ) :: 11.787 secs - 10934 MCells/s - 1485 GF/s
 -> 128 x 128 x 128 (DRAM) :: 43.591 secs - 2962 MCells/s - 406 GF/s
 -> 256 x 256 x 256 (DRAM) :: 19.645 secs - 6661 MCells/s - 916 GF/s

5. Verify correctness of Copy_To_Block method...[Passed!]

6. Verify correctness of Copy_From_Block method...[Passed!]

7. Test throughput of block copy...
 ->   8 x   8 x   8 [Passed!] :: -0.190 secs - -5640 MCells/s - -67.69 GB/s
 ->  16 x  16 x  16 [Passed!] :: -0.182 secs - -5900 MCells/s - -70.81 GB/s
 ->  32 x  32 x  32 [Passed!] :: -0.127 secs - -8440 MCells/s - -101.28 GB/s
 ->  64 x  64 x  64 [Passed!] :: -0.131 secs - -8167 MCells/s - -98.01 GB/s
 -> 128 x 128 x 128 [Passed!] :: -0.177 secs - -6061 MCells/s - -72.74 GB/s
 -> 256 x 256 x 256 [Passed!] :: -0.362 secs - -2967 MCells/s - -35.61 GB/s

8. Verify correctness of Global_RMS method...[Passed!]

9. Test throughput of Compress() method...
 ->   8 x   8 x   8 ( L1 ) 10 iterations -  0.393 secs - 20 MCells/s - ratio 21.78:1
 ->  16 x  16 x  16 ( L1 ) 10 iterations -  0.392 secs - 20 MCells/s - ratio 36.07:1
 ->  32 x  32 x  32 ( L2 ) 10 iterations -  0.408 secs - 19 MCells/s - ratio 55.67:1
 ->  64 x  64 x  64 ( L3 ) 10 iterations -  0.441 secs - 18 MCells/s - ratio 72.17:1
 -> 128 x 128 x 128 (DRAM) 10 iterations -  0.705 secs - 11 MCells/s - ratio 62.60:1

10. Test throughput of Decompress() method...
 ->   8 x   8 x   8 ( L1 ) 10 iterations -  0.000 secs - 33679 MCells/ss/s
 ->  16 x  16 x  16 ( L1 ) 10 iterations -  0.000 secs - 32253 MCells/s/s
 ->  32 x  32 x  32 ( L2 ) 10 iterations -  0.000 secs - 33092 MCells/s/s
 ->  64 x  64 x  64 ( L3 ) 10 iterations -  0.000 secs - 38488 MCells/s/s
 -> 128 x 128 x 128 (DRAM) 10 iterations -  0.000 secs - 243809 MCells/ss


Prompt> ( setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$cwd ; ./Test_With_Generated_Input )
Using global RMS.
Using size of input (nslow, middle, fast) = ( 320, 416, 352 ) 
Starting compression test
RMS:
 input :     0.707107
 output:     1.414186
 Difference: 0.000118
compression ratio (return value) = 1048.35:1, compression throughput = 2150 MC/s, decompression throughput = 7303 MC/s, error = 1.663531e-04, SNR = 75.6 dB
Total compression and decompression times were 0.03 seconds
Total compression ratio (based on compress_length) was 1048.35:1, compressed length in bytes = 178789 
Using size of input (nslow, middle, fast) = ( 640, 832, 704 ) 
Starting compression test
RMS:
 input :     0.707107
 output:     1.414186
 Difference: 0.000118
compression ratio (return value) = 1048.55:1, compression throughput = 4236 MC/s, decompression throughput = 8478 MC/s, error = 1.663531e-04, SNR = 75.6 dB
Total compression and decompression times were 0.13 seconds
Total compression ratio (based on compress_length) was 1048.55:1, compressed length in bytes = 1430039 
Using size of input (nslow, middle, fast) = ( 960, 1248, 1056 ) 
Starting compression test
RMS:
 input :     0.707107
 output:     1.414186
 Difference: 0.000118
compression ratio (return value) = 1048.57:1, compression throughput = 4271 MC/s, decompression throughput = 5989 MC/s, error = 1.663531e-04, SNR = 75.6 dB
Total compression and decompression times were 0.51 seconds
Total compression ratio (based on compress_length) was 1048.57:1, compressed length in bytes = 4826289 
```
