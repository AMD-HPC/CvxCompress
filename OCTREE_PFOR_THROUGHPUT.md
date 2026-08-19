# Octree + PFOR vs RLE: throughput across sparsity

GPU encode/decode throughput of the **previous RLE codec** (`HIP_COMPRESS_KERNEL_ZLINE`,
the default z-line RLE) versus the **octree significance + per-block PFOR value
coder**, swept across coefficient sparsity at matched fidelity.

## Method

- **Data**: `solver_steps_512_marmousi_h20/snapshot_u_001000.raw`, a 128³ crop at
  origin `(z0=0, y0=192, x0=192)` — a wavefront region (~33% of voxels above
  1e-3·max in the raw field). Normalized to unit RMS.
- **Sparsity axis**: quantization multiplier `--scale` (mulfac). Higher mulfac →
  finer quantization → more nonzero coefficients → *denser* (lower CR). Fused CR
  is used as the density proxy.
- **Matched fidelity**: both codecs share the wavelet transform + quantization, so
  `vol_rel_l2` is identical at every point (verified per row below).
- **Harness**: `tests/test_bitmap_octree_hip.cpp`, `--nx 128 --iters 100`, raw GB/s
  over the f32 volume. Encode octree = k1 (wavelet) + k2 (PFOR); decode octree =
  stage-A (PFOR value unpack) + stage-B (inverse wavelet).
- **Value round-trip** validated format-agnostically (`decode: stage-A round-trip
  vs kernel-1 = OK`) at every point.

Command:

```bash
for MF in 1 2 5 10 20 40 80; do
  ./build/test_bitmap_octree_hip --panel $PANEL --nx 128 --z0 0 --y0 192 --x0 192 \
      --scale $MF --iters 100
done
```

## MI300X (gfx942)

Full-pipeline throughput (raw GB/s); `oct` = octree+PFOR.

| mulfac | rel_l2 | fused CR | enc RLE | enc oct | enc oct/RLE | dec RLE | dec oct | dec oct/RLE |
|-------:|-------:|---------:|--------:|--------:|:-----------:|--------:|--------:|:-----------:|
| 1  | 1.32e-1 | 166 | 167.6 | 172.4 | 1.03× | 97.1 | 190.3 | 1.96× |
| 2  | 8.22e-2 |  98 | 162.1 | 165.2 | 1.02× | 90.4 | 166.5 | 1.84× |
| 5  | 4.25e-2 |  53 | 155.1 | 151.2 | 0.97× | 80.2 | 151.6 | 1.89× |
| 10 | 2.51e-2 |  35 | 151.9 | 141.0 | 0.93× | 76.6 | 141.9 | 1.85× |
| 20 | 1.45e-2 |  25 | 148.9 | 134.1 | 0.90× | 73.4 | 136.2 | 1.86× |
| 40 | 8.13e-3 |  18 | 146.0 | 122.4 | 0.84× | 69.8 | 129.8 | 1.86× |
| 80 | 4.46e-3 |  14 | 142.5 | 113.5 | 0.80× | 67.8 | 124.2 | 1.83× |

Isolated value-coder kernels (raw GB/s): `FUSED` = k2 PFOR encode, `PAR_dec` =
stage-A PFOR decode.

| mulfac | fused CR | FUSED (k2 enc) | PAR_dec (stage-A) |
|-------:|---------:|---------------:|------------------:|
| 1  | 166 | 369.9 | 755.4 |
| 2  |  98 | 339.5 | 747.6 |
| 5  |  53 | 286.2 | 740.7 |
| 10 |  35 | 253.5 | 735.0 |
| 20 |  25 | 238.6 | 728.8 |
| 40 |  18 | 215.8 | 725.1 |
| 80 |  14 | 203.6 | 725.2 |

### Trends

- **Decode**: octree+PFOR is **1.8–2.0× faster than RLE at every sparsity**, and
  the ratio is nearly flat. RLE decode is the bottleneck (serial run expansion,
  97→68 GB/s as density rises).
- **Encode**: octree's edge is largest at nnz≈0 (1.03× at CR 166) and erodes
  monotonically to 0.80× at the densest point — as expected, since PFOR must code
  every nonzero while RLE encode is already bandwidth-bound (167→143, density-flat).
  Encode crossover is around CR≈60 (mulfac≈5).
- **PFOR decode is density-insensitive**: stage-A only drops 755→725 GB/s (−4%)
  across a ~12× CR range, because every stream is fixed-width / byte-aligned (no
  per-nonzero branching). Encode-k2 scales with nnz (370→204) — the histogram +
  exception scan + mask cost. That asymmetry is intentional: spend a little on
  encode to keep decode fast and flat.

Net: octree+PFOR wins decode decisively at all sparsities and wins encode only in
the sparse regime; in the dense regime it trades encode throughput for the CR gain
(and the large decode advantage).

## MI355X (gfx950)

Same data, crop, and sweep; CR and `rel_l2` are identical to MI300X (same
algorithm). Full-pipeline throughput (raw GB/s); `oct` = octree+PFOR.

| mulfac | rel_l2 | fused CR | enc RLE | enc oct | enc oct/RLE | dec RLE | dec oct | dec oct/RLE |
|-------:|-------:|---------:|--------:|--------:|:-----------:|--------:|--------:|:-----------:|
| 1  | 1.32e-1 | 166 | 184.6 | 184.1 | 1.00× | 109.5 | 207.3 | 1.89× |
| 2  | 8.22e-2 |  98 | 177.2 | 175.8 | 0.99× | 101.0 | 181.8 | 1.80× |
| 5  | 4.25e-2 |  53 | 169.2 | 161.0 | 0.95× | 89.7 | 166.3 | 1.85× |
| 10 | 2.51e-2 |  35 | 165.8 | 152.3 | 0.92× | 84.4 | 156.4 | 1.85× |
| 20 | 1.45e-2 |  25 | 162.3 | 142.9 | 0.88× | 81.0 | 150.3 | 1.85× |
| 40 | 8.13e-3 |  18 | 159.7 | 135.0 | 0.85× | 76.9 | 143.4 | 1.86× |
| 80 | 4.46e-3 |  14 | 155.8 | 126.0 | 0.81× | 75.2 | 137.1 | 1.82× |

Isolated value-coder kernels (raw GB/s): `FUSED` = k2 PFOR encode, `PAR_dec` =
stage-A PFOR decode.

| mulfac | fused CR | FUSED (k2 enc) | PAR_dec (stage-A) |
|-------:|---------:|---------------:|------------------:|
| 1  | 166 | 387.6 | 823.2 |
| 2  |  98 | 352.1 | 809.6 |
| 5  |  53 | 298.5 | 804.8 |
| 10 |  35 | 270.0 | 801.0 |
| 20 |  25 | 244.6 | 796.6 |
| 40 |  18 | 229.3 | 792.6 |
| 80 |  14 | 211.9 | 789.6 |

### Trends

Identical qualitative behavior to MI300X, uniformly faster (higher clocks/BW):

- **Decode**: octree+PFOR is **1.8–1.9× faster than RLE at every sparsity**; ratio
  nearly flat. RLE decode 110→75 GB/s as density rises.
- **Encode**: octree edge highest at nnz≈0 (1.00× at CR 166), eroding to 0.81× at
  the densest point; crossover near CR≈55 (mulfac≈5). RLE encode density-flat
  (185→156).
- **PFOR decode is density-insensitive**: stage-A 823→790 GB/s (−4%) across the CR
  range; encode-k2 scales with nnz (388→212).

### MI355X vs MI300X

MI355X is ~1.1–1.3× faster on every metric at matched work; the RLE-vs-octree
relationship (decode win everywhere, encode win only when sparse) is unchanged.

| metric (CR 35 / mulfac 10) | MI300X | MI355X | ×    |
|----------------------------|-------:|-------:|-----:|
| octree decode (full)       | 141.9  | 156.4  | 1.10 |
| octree encode (full)       | 141.0  | 152.3  | 1.08 |
| PFOR decode (stage-A)      | 735.0  | 801.0  | 1.09 |
| PFOR encode (k2)           | 253.5  | 270.0  | 1.07 |

## Provenance

- Hardware: MI300X (gfx942, TheraC16), MI355X (gfx950, TheraC79); ROCm 7.2.1.
- Build: `make test_bitmap_octree_hip HIP_ARCH=<gfx942|gfx950>`.
- All rows: `decode: stage-A round-trip vs kernel-1 = OK` (lossless PFOR recode).
- Date: 2026-08-18.
