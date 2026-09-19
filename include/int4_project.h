#ifndef INT4_PROJECT_H
#define INT4_PROJECT_H

#include <stddef.h>
#include <stdint.h>

/* Explicit generator: all representations must share identical values. */
int int4_generate_inputs(int32_t *a, int32_t *b, size_t n, uint32_t seed);

int int4_convert_i8(const int32_t *src, int8_t *dst, size_t count);
int int4_pack_signed(const int32_t *src, uint8_t *dst, size_t count);
int int4_unpack_signed(uint8_t packed, unsigned lane, int32_t *value);

/* Independent scalar reference: C = A x B, row-major, INT32 arithmetic. */
int int4_gemm_reference(const int32_t *a, const int32_t *b, int32_t *c,
                        size_t n);

#endif
