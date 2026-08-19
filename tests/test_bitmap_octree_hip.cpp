// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.
//
// Octree significance coder prototype validation:
//   (1) GPU-encoded octree bytes are byte-for-byte identical to the shared
//       host reference (woct_encode).
//   (2) GPU decode reconstructs the 1024 z-line masks byte-exactly (round-trip).
//   (3) full-codec CR (octree significance + two-level width table + values) vs
//       CPU/GPU RLE and the two-level coder, at matched quantization.
//   (4) octree encode/decode throughput.
// Uses a --panel loader (real seismic field) or a synthetic field fallback.

#define DS79_INCLUDE_REG32
#include "hipWaveletRLE.h"
#include "hipWaveletRLEInverse.h"
#include "hipWaveletBitmap.h"
#include "hipWaveletOctree.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define HIPCHECK(cmd) do { hipError_t _e=(cmd); if(_e!=hipSuccess){ \
    printf("HIP error %s at %s:%d\n",hipGetErrorString(_e),__FILE__,__LINE__); return 1;}}while(0)

static float time_kernel(void (*launch)(void*), void* ctx, int iters) {
    hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
    launch(ctx); hipDeviceSynchronize();
    hipEventRecord(a);
    for (int i=0;i<iters;++i) launch(ctx);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms=0; hipEventElapsedTime(&ms,a,b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms/iters;
}

struct Ctx {
    const float* d_in;
    unsigned char* d_rle;    size_t* d_rle_sizes;
    unsigned char* d_bmp;    size_t* d_bmp_sizes;
    unsigned char* d_codetl; size_t* d_codetl_sizes;
    unsigned char* d_oct;    size_t* d_oct_sizes;
    unsigned char* d_octp;   size_t* d_octp_sizes;   // parallel level-major
    unsigned char* d_octc;   size_t* d_octc_sizes;   // fused kernel-2 (sig+wtab+values)
    size_t* d_octc_sig_sizes;                        // significance length per block
    unsigned char* d_bmp2;   size_t* d_bmp2_sizes;   // decode stage-A output (bitmap+values)
    float* d_field_oct;      float* d_field_rle;      // full-decode wavefields
    uint32_t* d_dec_masks;
    uint32_t* d_decp_masks;
    int NX,NY,NZ,ldimx,ldimxy,nbx,nby,nbz,nblocks,enc_threads;
    float mulfac;
};
static void launch_rle(void* p){ Ctx*c=(Ctx*)p; dim3 g(c->nbx,c->nby,c->nbz);
    waveletRLEFusedKernel<<<g,dim3(256)>>>(c->d_in,c->d_rle,c->d_rle_sizes,c->mulfac,c->ldimx,c->ldimxy,nullptr,nullptr); }
static void launch_bmp(void* p){ Ctx*c=(Ctx*)p; dim3 g(c->nbx,c->nby,c->nbz);
    waveletBitmapFusedKernel<<<g,dim3(256)>>>(c->d_in,c->d_bmp,c->d_bmp_sizes,c->mulfac,c->ldimx,c->ldimxy,nullptr,nullptr); }
static void launch_codetl(void* p){ Ctx*c=(Ctx*)p;
    waveletBitmapCodeTwoLevelKernel<<<c->nblocks,dim3(256)>>>(c->d_bmp,c->d_bmp_sizes,c->d_codetl,c->d_codetl_sizes); }
static void launch_oct_enc(void* p){ Ctx*c=(Ctx*)p;
    waveletOctreeSigEncodeKernel<<<c->nblocks,dim3(c->enc_threads)>>>(c->d_bmp,c->d_oct,c->d_oct_sizes); }
static void launch_oct_dec(void* p){ Ctx*c=(Ctx*)p;
    waveletOctreeSigDecodeKernel<<<c->nblocks,dim3(c->enc_threads)>>>(c->d_oct,c->d_oct_sizes,c->d_dec_masks); }
static void launch_octp_enc(void* p){ Ctx*c=(Ctx*)p;
    waveletOctreeSigEncodeParKernel<<<c->nblocks,dim3(WOCT_PAR_THREADS)>>>(c->d_bmp,c->d_octp,c->d_octp_sizes); }
static void launch_octp_dec(void* p){ Ctx*c=(Ctx*)p;
    waveletOctreeSigDecodeParKernel<<<c->nblocks,dim3(WOCT_PAR_THREADS)>>>(c->d_octp,c->d_octp_sizes,c->d_decp_masks); }
static void launch_octc(void* p){ Ctx*c=(Ctx*)p;
    waveletOctreeCodeParKernel<<<c->nblocks,dim3(WOCT_PAR_THREADS)>>>(c->d_bmp,c->d_octc,c->d_octc_sizes,c->d_octc_sig_sizes); }
static void launch_octA(void* p){ Ctx*c=(Ctx*)p;   // decode stage A: coded -> bitmap+values
    waveletOctreeDecodeToBitmapKernel<<<c->nblocks,dim3(WOCT_PAR_THREADS)>>>(c->d_octc,c->d_octc_sig_sizes,c->d_bmp2,c->d_bmp2_sizes); }
static void launch_octB(void* p){ Ctx*c=(Ctx*)p;   // decode stage B: bitmap+values -> field
    dim3 g(c->nbx,c->nby,c->nbz);
    waveletBitmapInverseFusedKernel<<<g,dim3(256)>>>(c->d_bmp2,c->d_field_oct,1.0f/c->mulfac,c->ldimx,c->ldimxy); }
static void launch_rle_inv(void* p){ Ctx*c=(Ctx*)p; dim3 g(c->nbx,c->nby,c->nbz);
    waveletRLEInverseFusedKernel<<<g,dim3(256)>>>(c->d_rle,c->d_rle_sizes,nullptr,c->d_field_rle,1.0f/c->mulfac,c->ldimx,c->ldimxy,0); }

static bool load_center_crop(const std::string& path,int gnz,int gny,int gnx,
                             int cz,int cy,int cx,std::vector<float>& out,
                             int z0=-1,int y0=-1,int x0=-1){
    FILE* f=std::fopen(path.c_str(),"rb"); if(!f){printf("open fail %s\n",path.c_str());return false;}
    std::vector<float> full((size_t)gnz*gny*gnx);
    size_t r=std::fread(full.data(),sizeof(float),full.size(),f); std::fclose(f);
    if(r!=full.size()){printf("short read %s\n",path.c_str());return false;}
    int oz=(z0>=0)?z0:(gnz-cz)/2,oy=(y0>=0)?y0:(gny-cy)/2,ox=(x0>=0)?x0:(gnx-cx)/2;
    out.resize((size_t)cz*cy*cx);
    for(int k=0;k<cz;++k)for(int j=0;j<cy;++j){
        const float* s=&full[((size_t)(oz+k)*gny+(oy+j))*gnx+ox];
        std::memcpy(&out[((size_t)k*cy+j)*cx],s,(size_t)cx*sizeof(float)); }
    // normalize to unit RMS (raw wavefield ~1e-8), matches the CPU R-D sweep
    double ss=0; for(size_t i=0;i<out.size();++i) ss+=(double)out[i]*out[i];
    double rms=std::sqrt(ss/out.size());
    if(rms>0){ float inv=(float)(1.0/rms); for(size_t i=0;i<out.size();++i) out[i]*=inv; }
    printf("panel %s crop=%d^3 rms=%.3e (normalized)\n",path.c_str(),cx,rms);
    return true;
}

int main(int argc,char** argv){
    setvbuf(stdout,NULL,_IONBF,0);
    std::string panel;
    int NX=256; float mulfac=8.0f; int iters=100; int enc_threads=256; int z0=-1,y0=-1,x0=-1;
    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"--panel")&&i+1<argc) panel=argv[++i];
        else if(!std::strcmp(argv[i],"--nx")&&i+1<argc) NX=atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--scale")&&i+1<argc) mulfac=(float)atof(argv[++i]);
        else if(!std::strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--enc-threads")&&i+1<argc) enc_threads=atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--z0")&&i+1<argc) z0=atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--y0")&&i+1<argc) y0=atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--x0")&&i+1<argc) x0=atoi(argv[++i]);
    }
    if(NX%32){printf("NX must be multiple of 32\n");return 1;}
    Ctx c; c.NX=NX;c.NY=NX;c.NZ=NX;c.mulfac=mulfac;c.enc_threads=enc_threads;
    c.ldimx=NX;c.ldimxy=NX*NX;c.nbx=NX/32;c.nby=NX/32;c.nbz=NX/32;c.nblocks=c.nbx*c.nby*c.nbz;
    const size_t nelem=(size_t)NX*NX*NX;

    std::vector<float> h_in(nelem);
    if(!panel.empty()){
        if(!load_center_crop(panel,512,512,512,NX,NX,NX,h_in,z0,y0,x0)) return 1;
    } else {
        for(size_t i=0;i<nelem;++i){int x=(int)(i%NX),y=(int)((i/NX)%NX),z=(int)(i/((size_t)NX*NX));
            float s=sinf(0.11f*x)*cosf(0.07f*y)*sinf(0.05f*z);
            float n=0.15f*(float)(((i*1103515245u+12345u)>>16)&0x7fff)/32768.0f;
            h_in[i]=0.5f*s+n-0.075f;}
    }
    printf("octree-code test: %d^3 nblocks=%d scale=%.3f iters=%d enc_threads=%d\n",
           NX,c.nblocks,mulfac,iters,enc_threads);

    HIPCHECK(hipMalloc(&c.d_in,nelem*sizeof(float)));
    HIPCHECK(hipMemcpy((void*)c.d_in,h_in.data(),nelem*sizeof(float),hipMemcpyHostToDevice));
    const long rle_stride=4L*WRLE_LDS_BYTES;
    HIPCHECK(hipMalloc(&c.d_rle,(size_t)c.nblocks*rle_stride));
    HIPCHECK(hipMalloc(&c.d_rle_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_bmp,(size_t)c.nblocks*WBMP_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_bmp_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_codetl,(size_t)c.nblocks*WBMP_TL_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_codetl_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_oct,(size_t)c.nblocks*WOCT_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_oct_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_octp,(size_t)c.nblocks*WOCT_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_octp_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_octc,(size_t)c.nblocks*WOCT_CODE_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_octc_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_octc_sig_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_bmp2,(size_t)c.nblocks*WBMP_SLOT_BYTES));
    HIPCHECK(hipMalloc(&c.d_bmp2_sizes,c.nblocks*sizeof(size_t)));
    HIPCHECK(hipMalloc(&c.d_field_oct,nelem*sizeof(float)));
    HIPCHECK(hipMalloc(&c.d_field_rle,nelem*sizeof(float)));
    HIPCHECK(hipMalloc(&c.d_dec_masks,(size_t)c.nblocks*1024*sizeof(uint32_t)));
    HIPCHECK(hipMalloc(&c.d_decp_masks,(size_t)c.nblocks*1024*sizeof(uint32_t)));

    launch_rle(&c); launch_bmp(&c); HIPCHECK(hipDeviceSynchronize());
    launch_codetl(&c); launch_oct_enc(&c); HIPCHECK(hipDeviceSynchronize());
    launch_oct_dec(&c); HIPCHECK(hipDeviceSynchronize());
    launch_octp_enc(&c); HIPCHECK(hipDeviceSynchronize());
    launch_octp_dec(&c); launch_octc(&c); HIPCHECK(hipDeviceSynchronize());
    launch_octA(&c); HIPCHECK(hipDeviceSynchronize());       // decode stage A
    launch_octB(&c); launch_rle_inv(&c); HIPCHECK(hipDeviceSynchronize());  // full decodes
    HIPCHECK(hipGetLastError());

    std::vector<unsigned char> h_bmp((size_t)c.nblocks*WBMP_SLOT_BYTES);
    std::vector<unsigned char> h_oct((size_t)c.nblocks*WOCT_SLOT_BYTES);
    std::vector<unsigned char> h_octp((size_t)c.nblocks*WOCT_SLOT_BYTES);
    std::vector<unsigned char> h_octc((size_t)c.nblocks*WOCT_CODE_SLOT_BYTES);
    std::vector<unsigned char> h_codetl((size_t)c.nblocks*WBMP_TL_SLOT_BYTES);
    std::vector<size_t> h_bmp_sizes(c.nblocks),h_oct_sizes(c.nblocks),h_octp_sizes(c.nblocks),h_octc_sizes(c.nblocks);
    std::vector<size_t> h_rle_sizes(c.nblocks),h_codetl_sizes(c.nblocks);
    std::vector<uint32_t> h_dec((size_t)c.nblocks*1024),h_decp((size_t)c.nblocks*1024);
    HIPCHECK(hipMemcpy(h_bmp.data(),c.d_bmp,h_bmp.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_oct.data(),c.d_oct,h_oct.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_octp.data(),c.d_octp,h_octp.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_octc.data(),c.d_octc,h_octc.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_codetl.data(),c.d_codetl,h_codetl.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_bmp_sizes.data(),c.d_bmp_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_oct_sizes.data(),c.d_oct_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_octp_sizes.data(),c.d_octp_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_octc_sizes.data(),c.d_octc_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_rle_sizes.data(),c.d_rle_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_codetl_sizes.data(),c.d_codetl_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_dec.data(),c.d_dec_masks,h_dec.size()*sizeof(uint32_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_decp.data(),c.d_decp_masks,h_decp.size()*sizeof(uint32_t),hipMemcpyDeviceToHost));
    std::vector<unsigned char> h_bmp2((size_t)c.nblocks*WBMP_SLOT_BYTES);
    std::vector<size_t> h_bmp2_sizes(c.nblocks);
    std::vector<float> h_field_oct(nelem), h_field_rle(nelem);
    HIPCHECK(hipMemcpy(h_bmp2.data(),c.d_bmp2,h_bmp2.size(),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_bmp2_sizes.data(),c.d_bmp2_sizes,c.nblocks*sizeof(size_t),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_field_oct.data(),c.d_field_oct,nelem*sizeof(float),hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(h_field_rle.data(),c.d_field_rle,nelem*sizeof(float),hipMemcpyDeviceToHost));

    // ---- decode validation: stage-A bytes vs kernel-1 (exact round-trip gate) ----
    long decA_mism=0;
    for(int bid=0; bid<c.nblocks; ++bid){
        const unsigned char* a = h_bmp2.data()+(long)bid*WBMP_SLOT_BYTES;
        const unsigned char* b = h_bmp.data() +(long)bid*WBMP_SLOT_BYTES;
        if(h_bmp2_sizes[bid]!=h_bmp_sizes[bid]){ if(decA_mism<10) printf("  [decA] blk %d size %zu!=%zu\n",bid,h_bmp2_sizes[bid],h_bmp_sizes[bid]); ++decA_mism; continue; }
        long nbytes = (long)h_bmp_sizes[bid];   // 4096 bitmap + nnz*4 values
        if(memcmp(a,b,nbytes)!=0){ if(decA_mism<10) printf("  [decA] blk %d bytes differ\n",bid); ++decA_mism; }
    }
    // Informational: octree-vs-RLE field delta (codecs differ only in RLE's f32
    // escape for out-of-range coeffs) and each codec's distortion vs original.
    long field_mism=0; double field_maxdiff=0.0;
    double sse_oct=0.0, sse_rle=0.0, sig2=0.0;
    for(size_t i=0;i<nelem;++i){
        double xin=(double)h_in[i], fo=(double)h_field_oct[i], fr=(double)h_field_rle[i];
        double d=std::fabs(fo-fr); if(d>field_maxdiff) field_maxdiff=d; if(fo!=fr) ++field_mism;
        sse_oct+=(fo-xin)*(fo-xin); sse_rle+=(fr-xin)*(fr-xin); sig2+=xin*xin;
    }
    double rel_l2_oct=std::sqrt(sse_oct/sig2), rel_l2_rle=std::sqrt(sse_rle/sig2);

    // ---- (1) GPU encode == host reference, (2) round-trip masks ----
    long enc_mism=0, rt_mism=0, host_rt_mism=0;
    long lm_rt_mism=0, lm_cnt_mism=0;   // level-major reference checks
    long par_enc_mism=0, par_rt_mism=0; // parallel GPU vs host LM + round-trip
    long fus_sig_mism=0, fus_wtab_mism=0, fus_val_mism=0, fus_size_mism=0; // fused kernel-2
    long fus_block_total=0;             // sum of fused per-block coded bytes
    long mode_flat=0, mode_oct=0;
    long oct_sig_total=0;
    std::vector<unsigned char> ref((size_t)WOCT_MAX_SIG_BYTES);
    std::vector<unsigned char> reflm((size_t)WOCT_MAX_SIG_BYTES);
    std::vector<uint32_t> masks_sp(1024), masks_rt(1024);
    for(int bid=0; bid<c.nblocks; ++bid){
        const uint32_t* bmp = reinterpret_cast<const uint32_t*>(h_bmp.data()+(long)bid*WBMP_SLOT_BYTES);
        for(int L=0;L<1024;++L) masks_sp[woct_L_to_spatial(L)] = bmp[L];
        int hb = woct_encode(masks_sp.data(), ref.data());
        // pure-host encode->decode round-trip (isolates algorithm from device)
        for(int i=0;i<1024;++i) masks_rt[i]=0;
        woct_decode(ref.data(), hb, masks_rt.data());
        for(int i=0;i<1024;++i) if(masks_rt[i]!=masks_sp[i]){ if(host_rt_mism<5)
            printf("  [hostRT] blk %d sp %d dec %u != %u (octbytes=%d)\n",bid,i,masks_rt[i],masks_sp[i],hb); ++host_rt_mism; break; }
        // level-major host reference: same byte count as DFS + round-trip.
        int hblm = woct_encode_lm(masks_sp.data(), reflm.data());
        if(hblm != hb){ if(lm_cnt_mism<5) printf("  [lmCnt] blk %d lm %d != dfs %d\n",bid,hblm,hb); ++lm_cnt_mism; }
        for(int i=0;i<1024;++i) masks_rt[i]=0;
        woct_decode_lm(reflm.data(), hblm, masks_rt.data());
        for(int i=0;i<1024;++i) if(masks_rt[i]!=masks_sp[i]){ if(lm_rt_mism<5)
            printf("  [lmRT] blk %d sp %d dec %u != %u (lmbytes=%d)\n",bid,i,masks_rt[i],masks_sp[i],hblm); ++lm_rt_mism; break; }
        // parallel GPU level-major encode: byte-exact vs host LM reference.
        const unsigned char* blkp = h_octp.data()+(long)bid*WOCT_SLOT_BYTES;
        int pmode = blkp[0]; long psz = (long)h_octp_sizes[bid];
        if(pmode==2){
            if(psz != hblm){ if(par_enc_mism<10) printf("  [parEnc] blk %d sig %ld != lm %d\n",bid,psz,hblm); ++par_enc_mism; }
            else if(memcmp(blkp+WOCT_HDR_BYTES, reflm.data(), hblm)!=0){ if(par_enc_mism<10) printf("  [parEnc] blk %d bytes differ\n",bid); ++par_enc_mism; }
        } else { // flat fallback: octree stream must actually be >= flat size
            if(hblm < WOCT_FLAT_SIG_BYTES && par_enc_mism<10){ printf("  [parEnc] blk %d flat but lm=%d < flat\n",bid,hblm); ++par_enc_mism; }
        }
        // parallel round-trip masks (independent decode kernel).
        const uint32_t* dmp = &h_decp[(long)bid*1024];
        for(int L=0;L<1024;++L) if(dmp[L]!=bmp[L]){ if(par_rt_mism<10)
            printf("  [parRT] blk %d mode %d L %d dec %u != %u\n",bid,pmode,L,dmp[L],bmp[L]); ++par_rt_mism; break; }
        // ---- fused kernel-2: sig region vs host LM/flat; wtab+values vs two-level ----
        int ne=0; for(int L=0;L<1024;++L) if(bmp[L]) ++ne;
        const unsigned char* blkc = h_octc.data()+(long)bid*WOCT_CODE_SLOT_BYTES;
        int fmode = blkc[0];
        long fsig = (fmode==2) ? hblm : WOCT_FLAT_SIG_BYTES;  // hblm==0 => empty
        // significance region
        if(fmode==2){
            if(hblm>0 && memcmp(blkc+WOCT_HDR_BYTES, reflm.data(), hblm)!=0){
                if(fus_sig_mism<10) printf("  [fusSig] blk %d octree bytes differ\n",bid); ++fus_sig_mism; }
        } else {
            const uint32_t* fflat = reinterpret_cast<const uint32_t*>(blkc+WOCT_HDR_BYTES);
            for(int L=0;L<1024;++L) if(fflat[L]!=bmp[L]){ if(fus_sig_mism<10)
                printf("  [fusSig] blk %d flat L %d %u!=%u\n",bid,L,fflat[L],bmp[L]); ++fus_sig_mism; break; }
        }
        // NOTE: the fused kernel-2 now uses a per-block PFOR value coder (not the
        // legacy per-line width table).  Its value round-trip is validated
        // format-agnostically by the stage-A decode gate (decA_mism) below, so the
        // old wtab/values/size byte-exact comparisons vs the two-level reference no
        // longer apply.  Only the measured coded size is kept (for the fused CR).
        (void)fsig;
        fus_block_total += (long)h_octc_sizes[bid];
        const unsigned char* blk = h_oct.data()+(long)bid*WOCT_SLOT_BYTES;
        int mode = blk[0];
        long gsz = (long)h_oct_sizes[bid];
        if(mode==2){
            ++mode_oct; oct_sig_total += gsz;
            if(gsz != hb){ if(enc_mism<10) printf("  [enc] blk %d sig %ld != ref %d\n",bid,gsz,hb); ++enc_mism; }
            else if(memcmp(blk+WOCT_HDR_BYTES, ref.data(), hb)!=0){ if(enc_mism<10) printf("  [enc] blk %d bytes differ\n",bid); ++enc_mism; }
        } else {
            ++mode_flat; oct_sig_total += WOCT_FLAT_SIG_BYTES;
        }
        // round-trip: decoded L-order masks must equal the bitmap words
        const uint32_t* dm = &h_dec[(long)bid*1024];
        for(int L=0;L<1024;++L) if(dm[L]!=bmp[L]){
            if(rt_mism<10){
                // host-decode the GPU-written bytes for this block to localize
                for(int i=0;i<1024;++i) masks_rt[i]=0;
                if(mode==2) woct_decode(blk+WOCT_HDR_BYTES,(int)gsz,masks_rt.data());
                else for(int Lx=0;Lx<1024;++Lx) masks_rt[woct_L_to_spatial(Lx)]=
                     reinterpret_cast<const uint32_t*>(blk+WOCT_HDR_BYTES)[Lx];
                int hostgpu_ok = (masks_rt[woct_L_to_spatial(L)]==bmp[L]);
                printf("  [rt] blk %d mode %d sig %ld L %d dec %u != %u  host(gpubytes)=%s\n",
                       bid,mode,gsz,L,dm[L],bmp[L], hostgpu_ok?"match":"MISMATCH");
            }
            ++rt_mism; break; }
    }

    // ---- (3) full-codec size / CR (octree sig + two-level wtab+payload) ----
    long hdr = 8 + 8L*c.nblocks + 4;
    long rle_total=hdr, tl_total=hdr, oct_total=hdr;
    for(int bid=0;bid<c.nblocks;++bid){
        rle_total += (long)h_rle_sizes[bid];
        tl_total  += (long)h_codetl_sizes[bid];
        // wtab+payload = codetl - occupancy(128) - masks(4*ne)
        const uint32_t* bmp = reinterpret_cast<const uint32_t*>(h_bmp.data()+(long)bid*WBMP_SLOT_BYTES);
        int ne=0; for(int L=0;L<1024;++L) if(bmp[L]) ++ne;
        long wtab_payload = (long)h_codetl_sizes[bid] - WBMP_OCC_BYTES - 4L*ne;
        long sig = (h_oct.data()[(long)bid*WOCT_SLOT_BYTES]==2) ? (long)h_oct_sizes[bid] : WOCT_FLAT_SIG_BYTES;
        oct_total += WOCT_HDR_BYTES + sig + wtab_payload;
    }
    double raw=(double)nelem*4.0;
    printf("modes: octree=%ld (%.1f%%)  flat=%ld  | octree sig bytes total=%ld\n",
           mode_oct,100.0*mode_oct/c.nblocks,mode_flat,oct_sig_total);
    long fused_total = hdr + fus_block_total;   // measured fused kernel-2 output
    printf("sizes (B): RLE=%ld  twolevel=%ld  octree(est)=%ld  fused(measured)=%ld\n",
           rle_total,tl_total,oct_total,fused_total);
    printf("CR:        RLE=%.3f  twolevel=%.3f  octree=%.3f  fused=%.3f\n",
           raw/rle_total,raw/tl_total,raw/oct_total,raw/fused_total);
    printf("octree/RLE size ratio = %.3f  (<1 = octree smaller)\n",(double)oct_total/rle_total);
    printf("octree/twolevel size ratio = %.3f\n",(double)oct_total/tl_total);

    // ---- (4) throughput ----
    float t_enc=time_kernel(launch_oct_enc,&c,iters);
    float t_dec=time_kernel(launch_oct_dec,&c,iters);
    float t_penc=time_kernel(launch_octp_enc,&c,iters);
    float t_pdec=time_kernel(launch_octp_dec,&c,iters);
    float t_tl =time_kernel(launch_codetl,&c,iters);
    float t_fus=time_kernel(launch_octc,&c,iters);
    // full-pipeline enc/dec: RLE (single fused kernel) vs octree (k1 + fused k2 enc;
    // stage-A + stage-B dec)
    float t_rle_enc=time_kernel(launch_rle,&c,iters);
    float t_k1     =time_kernel(launch_bmp,&c,iters);
    float t_rle_dec=time_kernel(launch_rle_inv,&c,iters);
    float t_decA   =time_kernel(launch_octA,&c,iters);
    float t_decB   =time_kernel(launch_octB,&c,iters);
    double gbraw=raw/1e9;
    float oct_enc_full=t_k1+t_fus, oct_dec_full=t_decA+t_decB;
    printf("FULL PIPELINE (raw GB/s):\n");
    printf("  encode: RLE=%.1f (%.3fms)   octree=%.1f (k1 %.3f + k2 %.3f = %.3fms)   speedup=%.2fx\n",
           gbraw/(t_rle_enc/1e3),t_rle_enc, gbraw/(oct_enc_full/1e3),t_k1,t_fus,oct_enc_full, t_rle_enc/oct_enc_full);
    printf("  decode: RLE=%.1f (%.3fms)   octree=%.1f (A %.3f + B %.3f = %.3fms)   speedup=%.2fx\n",
           gbraw/(t_rle_dec/1e3),t_rle_dec, gbraw/(oct_dec_full/1e3),t_decA,t_decB,oct_dec_full, t_rle_dec/oct_dec_full);
    printf("throughput (raw GB/s): serial_enc=%.2f serial_dec=%.2f | PAR_enc=%.2f PAR_dec=%.2f | FUSED=%.2f | twolevel_enc=%.2f\n",
           gbraw/(t_enc/1e3), gbraw/(t_dec/1e3), gbraw/(t_penc/1e3), gbraw/(t_pdec/1e3), gbraw/(t_fus/1e3), gbraw/(t_tl/1e3));
    printf("kernel ms: serial_enc=%.3f serial_dec=%.3f | PAR_enc=%.3f PAR_dec=%.3f | FUSED=%.3f | twolevel_enc=%.3f\n",
           t_enc,t_dec,t_penc,t_pdec,t_fus,t_tl);
    printf("parallel speedup vs serial: enc=%.1fx  dec=%.1fx  | fused vs twolevel: %.2fx\n",
           t_enc/t_penc, t_dec/t_pdec, t_tl/t_fus);

    long total_mism = enc_mism + rt_mism;
    printf("encode byte-exact vs host: %s (%ld)   round-trip masks: %s (%ld)   host-only RT: %s (%ld)\n",
           enc_mism?"MISMATCH":"OK", enc_mism, rt_mism?"MISMATCH":"OK", rt_mism,
           host_rt_mism?"MISMATCH":"OK", host_rt_mism);
    printf("level-major host ref: byte-count vs DFS: %s (%ld)   round-trip masks: %s (%ld)\n",
           lm_cnt_mism?"MISMATCH":"OK", lm_cnt_mism, lm_rt_mism?"MISMATCH":"OK", lm_rt_mism);
    printf("parallel GPU: byte-exact vs host LM: %s (%ld)   round-trip masks: %s (%ld)\n",
           par_enc_mism?"MISMATCH":"OK", par_enc_mism, par_rt_mism?"MISMATCH":"OK", par_rt_mism);
    printf("fused kernel-2 (PFOR value coder): sig=%s(%ld)  values: round-trip via decA gate below\n",
           fus_sig_mism?"MISMATCH":"OK",fus_sig_mism);
    (void)fus_wtab_mism; (void)fus_val_mism; (void)fus_size_mism;
    printf("decode: stage-A round-trip (bytes vs kernel-1): %s (%ld)\n", decA_mism?"MISMATCH":"OK", decA_mism);
    printf("distortion vs original (rel_l2): octree=%.4e  RLE=%.4e  | octree-vs-RLE field maxdiff=%.3e (%ld voxels, codec f32-escape delta)\n",
           rel_l2_oct, rel_l2_rle, field_maxdiff, field_mism);
    total_mism += lm_cnt_mism + lm_rt_mism + par_enc_mism + par_rt_mism
                + fus_sig_mism + decA_mism;
    if(total_mism){ printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
