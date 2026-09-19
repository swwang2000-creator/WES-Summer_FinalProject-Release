#include "int4_project.h"

#include <stddef.h>
#include <stdint.h>

int int4_gemm_reference(const int32_t *a, const int32_t *b, int32_t *c,
                        size_t n)
{
    size_t i;
    size_t j;
    size_t k;

    if (a == NULL || b == NULL || c == NULL || n == 0 || n > SIZE_MAX / n) {
        return -1;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            int32_t sum = 0;
            for (k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    return 0;
}
