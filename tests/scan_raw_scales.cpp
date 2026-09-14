// Sweep CvxCompress (CPU) over a raw float32 volume and print CR + rel-L2 per
// scale, in exactly the format the GPU solver's -scan-scales path prints.
//
// This exists because Test_Compression cannot do it: Read_Raw_Volume was
// hijacked in 2024 to synthesize a sinusoid volume and ignores its filename
// argument, so the only CPU numbers available were on synthetic input. To put
// the incumbent on the same panel as the GPU codecs, the driver has to read a
// real dumped wavefield.
//
// The volume is headerless -- dimensions come from the command line, because a
// frame dumped by rtm_solver's -dump-field is a bare nx*ny*nz float32 blob in
// the same x-fast/z-slow order CvxCompress wants.
//
// rel-L2 here is RMS(x_hat - x) / RMS(x), which for a fixed element count is
// identical to ||x_hat - x||_2 / ||x||_2 -- the same quantity the solver
// reports, so the two tables can be read against each other.
//
// build:
//   g++ -O3 -fopenmp -I. tests/scan_raw_scales.cpp -L. -lcvxcompress -o scan_raw_scales
// run:
//   ./scan_raw_scales frame.u.raw 1024 1024 256 32 32 32 "0.02 0.035 0.05"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "CvxCompress.hxx"

int main(int argc, char **argv) {
    if (argc != 9) {
        printf("usage: %s <raw-f32> <nx> <ny> <nz> <bx> <by> <bz> \"<scales>\"\n"
               "  nx fast, nz slow; scales are space-separated\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int nx = atoi(argv[2]), ny = atoi(argv[3]), nz = atoi(argv[4]);
    int bx = atoi(argv[5]), by = atoi(argv[6]), bz = atoi(argv[7]);
    long nn = (long)nx * ny * nz;

    float *vol = 0;
    if (posix_memalign((void **)&vol, 64, sizeof(float) * nn)) { perror("alloc"); return 1; }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    if (fread(vol, sizeof(float), nn, f) != (size_t)nn) {
        fprintf(stderr, "short read: %s wants %ld floats\n", path, nn); return 1;
    }
    fclose(f);

    double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
    for (long i = 0; i < nn; ++i) acc += (double)vol[i] * vol[i];
    double rms_orig = sqrt(acc / (double)nn);

    int nthr = 0;
#pragma omp parallel
    { nthr = omp_get_num_threads(); }
    printf("cvxcompress-cpu: field=%s n=%ld dims=%dx%dx%d block=%dx%dx%d rms=%g threads=%d\n",
           path, nn, nx, ny, nz, bx, by, bz, rms_orig, nthr);

    // 5x the input is what Test_Compression reserves: the transform is lossy
    // but the RLE has no guaranteed bound, so the output buffer is oversized
    // rather than sized from an assumed ratio.
    unsigned int *comp = 0;
    if (posix_memalign((void **)&comp, 64, (sizeof(float) * nn * 5L))) { perror("alloc"); return 1; }

    // Compress() consumes vol in place (the wavelet transform is destructive),
    // so every scale gets its own pristine copy.
    float *work = 0;
    if (posix_memalign((void **)&work, 64, sizeof(float) * nn)) { perror("alloc"); return 1; }

    char *saveptr = 0;
    for (char *tok = strtok_r(argv[8], " ", &saveptr); tok; tok = strtok_r(0, " ", &saveptr)) {
        float s = atof(tok);
        memcpy(work, vol, sizeof(float) * nn);

        CvxCompress cvx;
        long clen = 0;
        double t0 = omp_get_wtime();
        float ratio = cvx.Compress(s, work, nx, ny, nz, bx, by, bz, comp, clen);
        double t1 = omp_get_wtime();

        int nx2, ny2, nz2;
        double t2 = omp_get_wtime();
        float *rec = cvx.Decompress(nx2, ny2, nz2, comp, clen);
        double t3 = omp_get_wtime();

        double acc2 = 0.0;
#pragma omp parallel for reduction(+ : acc2)
        for (long i = 0; i < nn; ++i) {
            double d = (double)rec[i] - (double)vol[i];
            acc2 += d * d;
        }
        double rel = sqrt(acc2 / (double)nn) / rms_orig;

        printf("cvxcompress scale=%g CR=%.2f relL2=%.5f enc_ms=%.1f dec_ms=%.1f "
               "enc_MCs=%.0f dec_MCs=%.0f\n",
               s, ratio, rel, 1e3 * (t1 - t0), 1e3 * (t3 - t2),
               nn / ((t1 - t0) * 1e6), nn / ((t3 - t2) * 1e6));
        fflush(stdout);
        free(rec);
    }
    return 0;
}
