#include "int4_project.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", (message)); \
        return EXIT_FAILURE; \
    } \
} while (0)

static int test_known_packing(void)
{
    const int32_t values[] = {-8, 7, -1, 0};
    const uint8_t expected[] = {0x78u, 0x0Fu};
    uint8_t packed[2] = {0u, 0u};
    int32_t decoded;
    size_t i;

    CHECK(int4_pack_signed(values, packed, 4) == 0, "pack known values");
    for (i = 0; i < 2; ++i) {
        CHECK(packed[i] == expected[i], "low/high nibble order");
    }
    for (i = 0; i < 4; ++i) {
        CHECK(int4_unpack_signed(packed[i / 2], (unsigned)(i % 2),
                                 &decoded) == 0,
              "unpack known value");
        CHECK(decoded == values[i], "round-trip signed value");
    }
    return 0;
}

static int test_generation_and_conversion(void)
{
    int32_t a[16];
    int32_t b[16];
    int32_t a_again[16];
    int32_t b_again[16];
    int8_t converted[16];
    uint8_t packed[8];
    size_t i;

    CHECK(int4_generate_inputs(a, b, 4, 237u) == 0, "generate inputs");
    CHECK(int4_generate_inputs(a_again, b_again, 4, 237u) == 0,
          "regenerate inputs");
    for (i = 0; i < 16; ++i) {
        CHECK(a[i] >= -8 && a[i] <= 7, "A range");
        CHECK(b[i] >= -8 && b[i] <= 7, "B range");
        CHECK(a[i] == a_again[i] && b[i] == b_again[i],
              "deterministic seed");
    }
    CHECK(int4_convert_i8(a, converted, 16) == 0, "INT8 conversion");
    CHECK(int4_pack_signed(a, packed, 16) == 0, "INT4 packing");
    for (i = 0; i < 16; ++i) {
        int32_t decoded;
        CHECK(int4_unpack_signed(packed[i / 2], (unsigned)(i % 2),
                                 &decoded) == 0 && decoded == a[i],
              "generated round-trip");
        CHECK(converted[i] == a[i], "shared INT8 values");
    }
    return 0;
}

static int test_reference(void)
{
    const int32_t a[] = {1, 2, 3, 4};
    const int32_t b[] = {5, 6, 7, 8};
    const int32_t expected[] = {19, 22, 43, 50};
    int32_t c[4] = {0, 0, 0, 0};
    size_t i;

    CHECK(int4_gemm_reference(a, b, c, 2) == 0, "CPU reference GEMM");
    for (i = 0; i < 4; ++i) {
        CHECK(c[i] == expected[i], "reference result");
    }
    return 0;
}

int main(void)
{
    CHECK(test_known_packing() == 0, "known packing test");
    CHECK(test_generation_and_conversion() == 0, "generation test");
    CHECK(test_reference() == 0, "reference test");
    puts("CPU tests passed: packing, deterministic inputs, conversion, reference GEMM");
    return EXIT_SUCCESS;
}
