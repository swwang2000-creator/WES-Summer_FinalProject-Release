#include "int4_project.h"

#include <stddef.h>

static uint32_t xorshift32(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int matrix_elements(size_t n, size_t *elements)
{
    if (elements == NULL || n == 0 || n > SIZE_MAX / n) {
        return -1;
    }
    *elements = n * n;
    return 0;
}

int int4_generate_inputs(int32_t *a, int32_t *b, size_t n, uint32_t seed)
{
    size_t elements;
    size_t i;
    uint32_t state;

    if (a == NULL || b == NULL || seed == 0 ||
        matrix_elements(n, &elements) != 0) {
        return -1;
    }

    state = seed;
    for (i = 0; i < elements; ++i) {
        a[i] = (int32_t)(xorshift32(&state) & 15u) - 8;
    }
    for (i = 0; i < elements; ++i) {
        b[i] = (int32_t)(xorshift32(&state) & 15u) - 8;
    }
    return 0;
}

int int4_convert_i8(const int32_t *src, int8_t *dst, size_t count)
{
    size_t i;

    if (src == NULL || dst == NULL) {
        return -1;
    }
    for (i = 0; i < count; ++i) {
        if (src[i] < -128 || src[i] > 127) {
            return -1;
        }
        dst[i] = (int8_t)src[i];
    }
    return 0;
}

int int4_pack_signed(const int32_t *src, uint8_t *dst, size_t count)
{
    size_t byte_count;
    size_t i;

    if (src == NULL || dst == NULL || count > SIZE_MAX - 1u) {
        return -1;
    }
    byte_count = (count + 1u) / 2u;
    for (i = 0; i < byte_count; ++i) {
        size_t first = 2u * i;
        int32_t lo;
        int32_t hi = 0;

        if (first >= count || src[first] < -8 || src[first] > 7) {
            return -1;
        }
        lo = src[first];
        if (first + 1u < count) {
            if (src[first + 1u] < -8 || src[first + 1u] > 7) {
                return -1;
            }
            hi = src[first + 1u];
        }
        dst[i] = (uint8_t)(((uint32_t)lo & 0xFu) |
                           (((uint32_t)hi & 0xFu) << 4));
    }
    return 0;
}

int int4_unpack_signed(uint8_t packed, unsigned lane, int32_t *value)
{
    uint32_t nibble;

    if (value == NULL || lane > 1u) {
        return -1;
    }
    nibble = ((uint32_t)packed >> (lane * 4u)) & 0xFu;
    *value = (nibble < 8u) ? (int32_t)nibble : (int32_t)nibble - 16;
    return 0;
}
