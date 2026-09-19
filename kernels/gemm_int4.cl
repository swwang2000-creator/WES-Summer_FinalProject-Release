/* M3: packed signed INT4 input, same 16x16 tile and INT32 accumulation. */
inline int decode_int4(uchar packed, int lane)
{
    const int nibble = ((int)packed >> (lane * 4)) & 0xF;
    return (nibble < 8) ? nibble : nibble - 16;
}

__kernel void gemm_int4_tiled(__global const uchar *a,
                              __global const uchar *b,
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
        const int a_index = row * n + tile * 16 + local_col;
        const int b_index = (tile * 16 + local_row) * n + col;
        tile_a[local_row][local_col] =
            decode_int4(a[a_index >> 1], a_index & 1);
        tile_b[local_row][local_col] =
            decode_int4(b[b_index >> 1], b_index & 1);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; ++k) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[row * n + col] = sum;
}
