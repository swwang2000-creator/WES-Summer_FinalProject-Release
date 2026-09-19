/* M2 baseline: aligned square INT32 GEMM with a fixed 16x16 tile. */
__kernel void gemm_int32_tiled(__global const int *a,
                               __global const int *b,
                               __global int *c,
                               const int n)
{
    const int row = (int)get_global_id(0);
    const int col = (int)get_global_id(1);
    const int local_row = (int)get_local_id(0);
    const int local_col = (int)get_local_id(1);
    __local int tile_a[16][16];
    __local int tile_b[16][16];
    int sum = 0;
    int tile;

    for (tile = 0; tile < n / 16; ++tile) {
        tile_a[local_row][local_col] =
            a[row * n + tile * 16 + local_col];
        tile_b[local_row][local_col] =
            b[(tile * 16 + local_row) * n + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; ++k) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[row * n + col] = sum;
}
