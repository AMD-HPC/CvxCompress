// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Rate-distortion comparison of the GPU 2D quadtree codec vs the CPU CvxCompress
// reference codec, on REAL 2D seismic data (a z-slice of a 512^3 RTM snapshot).
//
// CvxCompress is a 3D codec (block sizes >= 8 in z), so to obtain its behaviour
// on genuine 2D data we compress a z-constant volume: NZ copies of the same
// slice.  Its z-wavelet detail bands are then identically zero, so the stream
// encodes only the 2D content, and the fair 2D ratio is CR_2D = ratio / NZ
// (equivalently raw_one_slice / compressed_bytes).  Both codecs share the same
// Antonini 7/9 wavelet and truncation quantizer, but their scale->threshold maps
// differ, so we sweep each independently and compare CR at matched distortion
//   vol_rel_l2 = sqrt( sum (x-xhat)^2 / sum x^2 )   (spatial domain, 2D slice).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <hip/hip_runtime.h>

#include "hipCompress.h"
#include "CvxCompress.hxx"
#include "ds79.h"

#define HIPCHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Read a single z-slice (ny x nx, x fastest) from a raw x-fastest fp32 volume
// of grid (gnz,gny,gnx) at z index zc.
static bool load_zslice(const std::string& path, int gnz, int gny, int gnx,
                        int zc, int nx, int ny, std::vector<float>& out)
{
    if (zc < 0 || zc >= gnz || nx > gnx || ny > gny) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open fail %s\n", path.c_str()); return false; }
    out.resize((size_t)nx * ny);
    // center-crop nx x ny out of the gny x gnx slice
    int ox = (gnx - nx) / 2, oy = (gny - ny) / 2;
    long base = (long)zc * gny * gnx;
    bool ok = true;
    for (int j = 0; j < ny && ok; ++j) {
        long off = (base + (long)(oy + j) * gnx + ox) * (long)sizeof(float);
        if (std::fseek(f, off, SEEK_SET) != 0) { ok = false; break; }
        if (std::fread(&out[(size_t)j * nx], sizeof(float), nx, f) != (size_t)nx) ok = false;
    }
    std::fclose(f);
    return ok;
}

// Read a 3D crop (cz,cy,cx; x fastest) from a raw x-fastest fp32 volume
// (gnz,gny,gnx) at origin (z0,y0,x0), into out (z,y,x order, x fastest).
static bool load_crop3d(const std::string& path, int gnz, int gny, int gnx,
                        int z0, int y0, int x0, int cz, int cy, int cx,
                        std::vector<float>& out)
{
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0+cz > gnz || y0+cy > gny || x0+cx > gnx) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open fail %s\n", path.c_str()); return false; }
    out.resize((size_t)cz * cy * cx);
    bool ok = true;
    for (int k = 0; k < cz && ok; ++k)
        for (int j = 0; j < cy && ok; ++j) {
            long off = ((long)(z0+k) * gny * gnx + (long)(y0+j) * gnx + x0) * (long)sizeof(float);
            if (std::fseek(f, off, SEEK_SET) != 0) { ok = false; break; }
            float* dst = &out[((size_t)k * cy + j) * cx];
            if (std::fread(dst, sizeof(float), cx, f) != (size_t)cx) ok = false;
        }
    std::fclose(f);
    return ok;
}

static double rel_l2(const float* a, const float* b, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)a[i] * (double)a[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// ---- GPU 2D quadtree: compress+decompress one slice, return CR and rel_l2 ----
static void gpu_quadtree_rd(const std::vector<float>& slice, int nx, int ny,
                            float scale, double* cr_out, double* err_out)
{
    const size_t total = (size_t)nx * ny;
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, nx, ny, 1, 0, HIP_COMPRESS_KERNEL_QUADTREE));

    float* d_in = nullptr; float* d_out = nullptr; unsigned char* d_comp = nullptr;
    size_t comp_cap = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_cap));
    HIPCHECK(hipMalloc(&d_in, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_comp, comp_cap));
    HIPCHECK(hipMemcpy(d_in, slice.data(), total * sizeof(float), hipMemcpyHostToDevice));

    HIPCHECK(hipComputeRMS(d_in, nx, nx, 0, 0, 0, nx, ny, 1, plan->d_rms, plan, 0));
    long clen = 0; float cr = 0.0f;
    HIPCHECK(hipCompress(scale, plan->d_rms, d_in, d_comp, plan, 0));
    HIPCHECK(hipCompressSynchronize(plan, &clen, &cr));
    HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> rec(total);
    HIPCHECK(hipMemcpy(rec.data(), d_out, total * sizeof(float), hipMemcpyDeviceToHost));

    *cr_out = (double)(total * sizeof(float)) / (double)clen;
    *err_out = rel_l2(slice.data(), rec.data(), total);

    hipFree(d_in); hipFree(d_out); hipFree(d_comp);
    hipCompressDestroyPlan(plan);
}

// ---- CPU CvxCompress on a z-constant NZ-deep volume, isolating 2D behaviour ---
static void cpu_cvx_rd(const std::vector<float>& slice, int nx, int ny, int NZ,
                       float scale, double* cr_out, double* err_out)
{
    const size_t plane = (size_t)nx * ny;
    const size_t nelem = plane * NZ;
    std::vector<float> vol(nelem);
    for (int z = 0; z < NZ; ++z)
        std::memcpy(&vol[(size_t)z * plane], slice.data(), plane * sizeof(float));

    std::vector<unsigned int> comp(nelem);  // generous: raw-sized
    CvxCompress cvx;
    long clen = 0;
    float ratio = cvx.Compress(scale, vol.data(), nx, ny, NZ, 32, 32, 32,
                               /*use_local_RMS=*/false, comp.data(), clen);

    int nx2 = 0, ny2 = 0, nz2 = 0;
    float* rec = cvx.Decompress(nx2, ny2, nz2, comp.data(), clen);

    // 2D ratio: NZ identical slices compress to ~one slice's cost.
    *cr_out = (double)ratio / (double)NZ;
    *err_out = rec ? rel_l2(slice.data(), rec, plane) : -1.0;   // z=0 plane
    if (rec) free(rec);
}

// ---- GPU 3D octree: compress+decompress a crop, return CR and rel_l2 --------
static void gpu_octree_rd(const std::vector<float>& vol, int nx, int ny, int nz,
                          float scale, double* cr_out, double* err_out)
{
    const size_t total = (size_t)nx * ny * nz;
    hipCompressPlan* plan = nullptr;
    HIPCHECK(hipCompressCreatePlan(&plan, nx, ny, nz, 0, HIP_COMPRESS_KERNEL_OCTREE));

    float* d_in = nullptr; float* d_out = nullptr; unsigned char* d_comp = nullptr;
    size_t comp_cap = 0;
    HIPCHECK(hipCompressMaxOutputSize(plan, &comp_cap));
    HIPCHECK(hipMalloc(&d_in, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_out, total * sizeof(float)));
    HIPCHECK(hipMalloc(&d_comp, comp_cap));
    HIPCHECK(hipMemcpy(d_in, vol.data(), total * sizeof(float), hipMemcpyHostToDevice));

    HIPCHECK(hipComputeRMS(d_in, nx, nx*ny, 0, 0, 0, nx, ny, nz, plan->d_rms, plan, 0));
    long clen = 0; float cr = 0.0f;
    HIPCHECK(hipCompress(scale, plan->d_rms, d_in, d_comp, plan, 0));
    HIPCHECK(hipCompressSynchronize(plan, &clen, &cr));
    HIPCHECK(hipDecompress(d_comp, d_out, plan, 0));
    HIPCHECK(hipDeviceSynchronize());

    std::vector<float> rec(total);
    HIPCHECK(hipMemcpy(rec.data(), d_out, total * sizeof(float), hipMemcpyDeviceToHost));
    *cr_out = (double)(total * sizeof(float)) / (double)clen;
    *err_out = rel_l2(vol.data(), rec.data(), total);

    hipFree(d_in); hipFree(d_out); hipFree(d_comp);
    hipCompressDestroyPlan(plan);
}

// ---- CPU CvxCompress on a real 3D crop, return CR and rel_l2 ----------------
static void cpu_cvx_rd_3d(const std::vector<float>& vol, int nx, int ny, int nz,
                          float scale, double* cr_out, double* err_out)
{
    const size_t nelem = (size_t)nx * ny * nz;
    std::vector<float> work(vol);                 // Compress may modify in place
    std::vector<unsigned int> comp(nelem + 1024);
    CvxCompress cvx;
    long clen = 0;
    float ratio = cvx.Compress(scale, work.data(), nx, ny, nz, 32, 32, 32,
                               /*use_local_RMS=*/false, comp.data(), clen);
    int nx2 = 0, ny2 = 0, nz2 = 0;
    float* rec = cvx.Decompress(nx2, ny2, nz2, comp.data(), clen);
    *cr_out = (double)ratio;
    *err_out = rec ? rel_l2(vol.data(), rec, nelem) : -1.0;
    if (rec) free(rec);
}

// Local host copies of the quadtree serial helpers (the kernel header is only
// compiled in hipCompress.cpp's TU; including it here would duplicate kernels).
static inline int qt_width(unsigned a) {
    if (a <= 0x7fu) return 1; if (a <= 0x7fffu) return 2;
    if (a <= 0x7fffffu) return 3; return 4;
}
static inline bool qt_sub_any(const uint32_t* m, int ox, int oy, int cs) {
    uint32_t xmask = (cs>=32)?0xFFFFFFFFu:(uint32_t)((((uint32_t)1<<cs)-1)<<ox);
    for (int iy=oy; iy<oy+cs; ++iy) if (m[iy]&xmask) return true; return false;
}
static inline int qt_child_occ(const uint32_t* m, int px, int py, int side) {
    int cs=side>>1,b=0; for(int c=0;c<4;++c){int cx=c&1,cy=(c>>1)&1;
        if(qt_sub_any(m,px+cx*cs,py+cy*cs,cs)) b|=(1<<c);} return b;
}
// byte-packed DFS node count (== wqt2d_encode length); 0 for empty.
static inline int qt_nodes(const uint32_t* m) {
    int rootb=qt_child_occ(m,0,0,32); if(!rootb) return 0;
    int px[6],py[6],side[6],bb[6],ch[6],sp=0,pos=0;
    px[0]=0;py[0]=0;side[0]=32;bb[0]=rootb;ch[0]=0;sp=1;++pos;
    while(sp>0){int i=sp-1,cs=side[i]>>1,found=-1;
        if(cs>1) for(int c=ch[i];c<4;++c) if(bb[i]&(1<<c)){found=c;break;}
        if(found>=0){ch[i]=found+1;int cx=found&1,cy=(found>>1)&1;
            int ox=px[i]+cx*cs,oy=py[i]+cy*cs,b=qt_child_occ(m,ox,oy,cs);++pos;
            px[sp]=ox;py[sp]=oy;side[sp]=cs;bb[sp]=b;ch[sp]=0;++sp;
        } else --sp;
    } return pos;
}

// Host 2D DS 7/9 forward on a 32x32 block (row-major q[iy*32+ix]); separable and
// linear, so row-then-col matches the GPU's col-then-row result.
static void host_block_forward(float* b /*32x32*/) {
    float line[32];
    for (int iy = 0; iy < 32; ++iy) {                 // rows (x)
        for (int ix = 0; ix < 32; ++ix) line[ix] = b[iy*32+ix];
        ds79_forward(line, 32);
        for (int ix = 0; ix < 32; ++ix) b[iy*32+ix] = line[ix];
    }
    for (int ix = 0; ix < 32; ++ix) {                 // cols (y)
        for (int iy = 0; iy < 32; ++iy) line[iy] = b[iy*32+ix];
        ds79_forward(line, 32);
        for (int iy = 0; iy < 32; ++iy) b[iy*32+ix] = line[iy];
    }
}

// CPU byte-breakdown of the 2D quadtree stream on the real slice, at matched
// fidelity vs CvxCompress (same mulfac => same quantized ints => same rel_l2).
static void breakdown_2d(const std::vector<float>& slice, int nx, int ny,
                         const std::vector<float>& scales)
{
    double ss = 0.0; for (float v : slice) ss += (double)v * v;
    double rms = std::sqrt(ss / slice.size());
    const int nbx = nx/32, nby = ny/32, nb = nbx*nby;
    const long hdr = 8 + 12L*nb + 4;                   // octree-style header
    const double raw = (double)nx*ny*4.0;

    // All schemes share the SAME quantization (fixed mulfac) => matched fidelity;
    // only the value-region byte cost differs.  Significance is nibble-packed (as
    // shipped).  Each scheme's own width-table / exception overhead is included.
    //   row  : one byte-width per nonempty 32-cell row      (current codec)
    //   g8/g4: one byte-width per nonempty 8- / 4-cell group (finer fixed width)
    //   pfor : bimodal patched fixed width - base W_lo for all + high bytes for the
    //          few exceptions + 1 exception-bit per nonzero + 1 byte (W_lo,W_hi)
    //   pv   : ideal per-value width (lower bound; no table)
    std::printf("\n2D value-coder model (per-block; all schemes matched fidelity; sig=nibble):\n");
    std::printf("%-7s %-6s | %-7s %-7s %-7s %-7s %-7s | %-6s %-6s %-6s %-6s\n",
        "scale","nnz%","CR_row","CR_g8","CR_g4","CR_pfor","CR_pv",
        "g8/row","g4/row","pf/row","pv/row");

    for (float sc : scales) {
        float mulfac = (rms>0)? (float)(1.0/(rms*sc)) : 1.0f;
        long tnnz=0, t_sigN=0;
        long t_vrow=0, t_vg8=0, t_vg4=0, t_vpfor=0, t_vpv=0;
        for (int by=0; by<nby; ++by)
        for (int bx=0; bx<nbx; ++bx) {
            float blk[1024];
            for (int iy=0; iy<32; ++iy)
                for (int ix=0; ix<32; ++ix)
                    blk[iy*32+ix] = slice[(size_t)(by*32+iy)*nx + (bx*32+ix)];
            host_block_forward(blk);
            int q[1024]; uint32_t m[32];
            for (int iy=0; iy<32; ++iy){ uint32_t mm=0;
                for (int ix=0; ix<32; ++ix){ int iv=(int)(mulfac*blk[iy*32+ix]); q[iy*32+ix]=iv; if(iv) mm|=(1u<<ix);} 
                m[iy]=mm; }
            int nodes = qt_nodes(m);
            long sigN = 128; { long nb2=(nodes+1)/2; if(nodes>0 && nb2<128) sigN=nb2; }

            long vrow=0, vg8=0, vg4=0, vpv=0;
            int ne_row=0, ne_g8=0, ne_g4=0;
            int wcnt[5]={0,0,0,0,0}; int nnz_blk=0, Whi=0;   // for PFOR
            for (int iy=0; iy<32; ++iy){ uint32_t mm=m[iy]; if(!mm) continue; ++ne_row;
                unsigned mxa=0; int cnt=0;
                for (int ix=0; ix<32; ++ix) if(mm&(1u<<ix)){ int v=q[iy*32+ix]; unsigned a=v<0?-v:v;
                    int w=qt_width(a); if(a>mxa)mxa=a; ++cnt; ++tnnz; ++nnz_blk;
                    vpv += w; ++wcnt[w]; if(w>Whi)Whi=w; }
                vrow += (long)cnt * qt_width(mxa);
                for (int g=0; g<4; ++g){ unsigned gm=0; int gc=0;
                    for (int ix=g*8; ix<g*8+8; ++ix) if(mm&(1u<<ix)){ int v=q[iy*32+ix]; unsigned a=v<0?-v:v; if(a>gm)gm=a; ++gc; }
                    if(gc){ ++ne_g8; vg8 += (long)gc*qt_width(gm); } }
                for (int g=0; g<8; ++g){ unsigned gm=0; int gc=0;
                    for (int ix=g*4; ix<g*4+4; ++ix) if(mm&(1u<<ix)){ int v=q[iy*32+ix]; unsigned a=v<0?-v:v; if(a>gm)gm=a; ++gc; }
                    if(gc){ ++ne_g4; vg4 += (long)gc*qt_width(gm); } }
            }
            // PFOR: minimise nnz*W_lo + n_exc(W_lo)*(W_hi-W_lo) over W_lo in [1,W_hi]
            long vpfor=0;
            if (nnz_blk){ long best=-1;
                for (int wlo=1; wlo<=Whi; ++wlo){ int nexc=0; for(int w=wlo+1; w<=4; ++w) nexc+=wcnt[w];
                    long b=(long)nnz_blk*wlo + (long)nexc*(Whi-wlo); if(best<0||b<best) best=b; }
                vpfor = best + (nnz_blk+7)/8 + 1;      // + exception mask + (W_lo,W_hi) byte
            }
            long trow=vrow + (2L*ne_row+7)/8;
            long tg8 =vg8  + (2L*ne_g8 +7)/8;
            long tg4 =vg4  + (2L*ne_g4 +7)/8;

            t_sigN+=sigN; t_vrow+=trow; t_vg8+=tg8; t_vg4+=tg4; t_vpfor+=vpfor; t_vpv+=vpv;
        }
        auto CR=[&](long vtot){ return raw/(double)(hdr + t_sigN + vtot); };
        double c_row=CR(t_vrow), c_g8=CR(t_vg8), c_g4=CR(t_vg4), c_pf=CR(t_vpfor), c_pv=CR(t_vpv);
        double nnz_pct = 100.0*tnnz/((double)nb*1024.0);
        std::printf("%-7.4g %-6.2f | %-7.2f %-7.2f %-7.2f %-7.2f %-7.2f | %-6.3f %-6.3f %-6.3f %-6.3f\n",
            sc, nnz_pct, c_row, c_g8, c_g4, c_pf, c_pv,
            c_g8/c_row, c_g4/c_row, c_pf/c_row, c_pv/c_row);
    }
}

int main(int argc, char** argv)
{
    std::string panel, mode = "2d";
    int gnz = 512, gny = 512, gnx = 512;
    int nx = 512, ny = 512, NZ = 32, zc = -1;
    int x0 = 0, y0 = 0, z0 = 0, cx = 512, cy = 512, cz = 128;   // 3D crop
    std::vector<float> gpu_scales = {1e-2f, 2e-2f, 5e-2f, 1e-1f, 2e-1f, 4e-1f};
    std::vector<float> cpu_scales = {1e-3f, 2e-3f, 5e-3f, 1e-2f, 2e-2f, 5e-2f};

    auto parse_list = [](const char* s, std::vector<float>& v) {
        v.clear(); char b[4096]; std::strncpy(b, s, 4095); b[4095] = '\0';
        for (char* t = std::strtok(b, ","); t; t = std::strtok(nullptr, ",")) v.push_back((float)atof(t));
    };
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--panel") && i+1 < argc) panel = argv[++i];
        else if (!std::strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!std::strcmp(argv[i], "--pdims") && i+3 < argc) { gnz=atoi(argv[++i]); gny=atoi(argv[++i]); gnx=atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--nx") && i+1 < argc) nx = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ny") && i+1 < argc) ny = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--z")  && i+1 < argc) zc = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nz") && i+1 < argc) NZ = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--crop") && i+6 < argc) {
            x0=atoi(argv[++i]); y0=atoi(argv[++i]); z0=atoi(argv[++i]);
            cx=atoi(argv[++i]); cy=atoi(argv[++i]); cz=atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--gpu-scales") && i+1 < argc) parse_list(argv[++i], gpu_scales);
        else if (!std::strcmp(argv[i], "--cpu-scales") && i+1 < argc) parse_list(argv[++i], cpu_scales);
    }
    if (panel.empty()) {
        std::fprintf(stderr, "Usage: %s --panel FILE.raw [--mode 2d|3d] [--pdims NZ NY NX]\n"
                             "  2d: [--nx N --ny N] [--z ZC] [--nz NZstack]\n"
                             "  3d: [--crop X0 Y0 Z0 CX CY CZ]\n", argv[0]);
        return 1;
    }

    if (mode == "3d") {
        if (cx % 32 || cy % 32 || cz % 32) { std::fprintf(stderr, "3D crop dims must be multiples of 32\n"); return 1; }
        std::vector<float> vol;
        if (!load_crop3d(panel, gnz, gny, gnx, z0, y0, x0, cz, cy, cx, vol)) {
            std::fprintf(stderr, "failed to load 3D crop\n"); return 1;
        }
        double ss = 0.0; for (float v : vol) ss += (double)v * v;
        std::printf("panel=%s 3D crop origin=(%d,%d,%d) dims=%dx%dx%d rms=%.3e\n",
                    panel.c_str(), x0, y0, z0, cx, cy, cz, std::sqrt(ss / vol.size()));

        std::printf("\nGPU 3D octree:\n%-10s %-12s %-10s\n", "scale", "vol_rel_l2", "CR");
        std::vector<std::pair<double,double>> gpu_rd;
        for (float sc : gpu_scales) {
            double cr, err; gpu_octree_rd(vol, cx, cy, cz, sc, &cr, &err);
            std::printf("%-10.4g %-12.4e %-10.3f\n", sc, err, cr);
            gpu_rd.push_back({err, cr});
        }
        std::printf("\nCPU CvxCompress (3D):\n%-10s %-12s %-10s\n", "scale", "vol_rel_l2", "CR");
        std::vector<std::pair<double,double>> cpu_rd;
        for (float sc : cpu_scales) {
            double cr, err; cpu_cvx_rd_3d(vol, cx, cy, cz, sc, &cr, &err);
            std::printf("%-10.4g %-12.4e %-10.3f\n", sc, err, cr);
            cpu_rd.push_back({err, cr});
        }
        auto interp = [](std::vector<std::pair<double,double>> rd, double t)->double {
            std::sort(rd.begin(), rd.end());
            if (t < rd.front().first || t > rd.back().first) return -1.0;
            for (size_t i = 1; i < rd.size(); ++i) if (t <= rd[i].first) {
                double e0=rd[i-1].first,e1=rd[i].first,c0=rd[i-1].second,c1=rd[i].second;
                double u=(std::log(t)-std::log(e0))/(std::log(e1)-std::log(e0));
                return std::exp(std::log(c0)+u*(std::log(c1)-std::log(c0)));
            }
            return -1.0;
        };
        std::printf("\nCR at matched fidelity (log-log interpolated):\n%-12s %-13s %-13s %-10s\n",
                    "vol_rel_l2", "GPU_oct_CR", "CPU_cvx_CR", "oct/cvx");
        for (double tgt : {2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}) {
            double g = interp(gpu_rd, tgt), c = interp(cpu_rd, tgt);
            if (g > 0 && c > 0) std::printf("%-12.3e %-13.3f %-13.3f %-10.3f\n", tgt, g, c, g / c);
            else std::printf("%-12.3e %-13s %-13s %-10s\n", tgt, g>0?"-":"oob", c>0?"-":"oob", "-");
        }
        return 0;
    }

    if (nx % 32 || ny % 32) { std::fprintf(stderr, "nx,ny must be multiples of 32\n"); return 1; }
    if (zc < 0) zc = gnz / 2;

    std::vector<float> slice;
    if (!load_zslice(panel, gnz, gny, gnx, zc, nx, ny, slice)) {
        std::fprintf(stderr, "failed to load slice\n"); return 1;
    }

    // slice RMS (for context)
    double ss = 0.0; for (float v : slice) ss += (double)v * v;
    double srms = std::sqrt(ss / slice.size());
    std::printf("panel=%s slice z=%d crop=%dx%d rms=%.3e  (CPU codec: z-constant NZ=%d, 32^3 blocks)\n",
                panel.c_str(), zc, nx, ny, srms, NZ);

    if (mode == "bd2d") { breakdown_2d(slice, nx, ny, gpu_scales); return 0; }

    std::printf("\nGPU 2D quadtree:\n%-10s %-12s %-10s\n", "scale", "vol_rel_l2", "CR");
    std::vector<std::pair<double,double>> gpu_rd;  // (err, cr)
    for (float sc : gpu_scales) {
        double cr, err; gpu_quadtree_rd(slice, nx, ny, sc, &cr, &err);
        std::printf("%-10.4g %-12.4e %-10.3f\n", sc, err, cr);
        gpu_rd.push_back({err, cr});
    }

    std::printf("\nCPU CvxCompress (2D-isolated):\n%-10s %-12s %-10s\n", "scale", "vol_rel_l2", "CR");
    std::vector<std::pair<double,double>> cpu_rd;
    for (float sc : cpu_scales) {
        double cr, err; cpu_cvx_rd(slice, nx, ny, NZ, sc, &cr, &err);
        std::printf("%-10.4g %-12.4e %-10.3f\n", sc, err, cr);
        cpu_rd.push_back({err, cr});
    }

    // CR at matched distortion: interpolate each codec's CR(err) in log-log at a
    // few shared rel_l2 targets that both curves bracket.
    auto interp_cr = [](std::vector<std::pair<double,double>> rd, double target)->double {
        std::sort(rd.begin(), rd.end());
        if (target < rd.front().first || target > rd.back().first) return -1.0;
        for (size_t i = 1; i < rd.size(); ++i) {
            if (target <= rd[i].first) {
                double e0 = rd[i-1].first, e1 = rd[i].first;
                double c0 = rd[i-1].second, c1 = rd[i].second;
                double t = (std::log(target) - std::log(e0)) / (std::log(e1) - std::log(e0));
                return std::exp(std::log(c0) + t * (std::log(c1) - std::log(c0)));
            }
        }
        return -1.0;
    };
    std::printf("\nCR at matched fidelity (log-log interpolated):\n%-12s %-12s %-12s %-10s\n",
                "vol_rel_l2", "GPU_qt_CR", "CPU_cvx_CR", "qt/cvx");
    for (double tgt : {1e-3, 2e-3, 5e-3, 1e-2, 2e-2}) {
        double g = interp_cr(gpu_rd, tgt), c = interp_cr(cpu_rd, tgt);
        if (g > 0 && c > 0)
            std::printf("%-12.3e %-12.3f %-12.3f %-10.3f\n", tgt, g, c, g / c);
        else
            std::printf("%-12.3e %-12s %-12s %-10s\n", tgt,
                        g > 0 ? "-" : "oob", c > 0 ? "-" : "oob", "-");
    }
    return 0;
}
