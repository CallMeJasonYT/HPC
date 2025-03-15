#include <immintrin.h>
#include "weno.h"

#pragma once

// WENO5 AVX implementation
void weno_avx_core(const float *a, const float *b, const float *c,
                    const float *d, const float *e, float *out,
                    const int NENTRIES)
{
    const int vec_size = 8; // AVX processes 8 floats at a time
    int i = 0;

    // Vectorized computation using AVX
    for (; i <= NENTRIES - vec_size; i += vec_size)
    {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 vd = _mm256_loadu_ps(&d[i]);
        __m256 ve = _mm256_loadu_ps(&e[i]);

        // Compute IS0, IS1, IS2
        __m256 is0 = va * (va * 4.0f / 3.0f - vb * 19.0f / 3.0f + vc * 11.0f / 3.0f) +
                     vb * (vb * 25.0f / 3.0f - vc * 31.0f / 3.0f) +
                     vc * vc * 10.0f / 3.0f;

        __m256 is1 = vb * (vb * 4.0f / 3.0f - vc * 13.0f / 3.0f + vd * 5.0f / 3.0f) +
                     vc * (vc * 13.0f / 3.0f - vd * 13.0f / 3.0f) +
                     vd * vd * 4.0f / 3.0f;

        __m256 is2 = vc * (vc * 10.0f / 3.0f - vd * 31.0f / 3.0f + ve * 11.0f / 3.0f) +
                     vd * (vd * 25.0f / 3.0f - ve * 19.0f / 3.0f) +
                     ve * ve * 4.0f / 3.0f;

        // Add small constant (WENOEPS) to denominators
        const __m256 epsilon = _mm256_set1_ps(WENOEPS);
        is0 = _mm256_add_ps(is0, epsilon);
        is1 = _mm256_add_ps(is1, epsilon);
        is2 = _mm256_add_ps(is2, epsilon);

        // Compute alpha values
        __m256 alpha0 = _mm256_div_ps(_mm256_set1_ps(0.1f), _mm256_mul_ps(is0, is0));
        __m256 alpha1 = _mm256_div_ps(_mm256_set1_ps(0.6f), _mm256_mul_ps(is1, is1));
        __m256 alpha2 = _mm256_div_ps(_mm256_set1_ps(0.3f), _mm256_mul_ps(is2, is2));

        __m256 alpha_sum = _mm256_add_ps(_mm256_add_ps(alpha0, alpha1), alpha2);

        // Compute omega values
        __m256 inv_alpha = _mm256_div_ps(_mm256_set1_ps(1.0f), alpha_sum);
        __m256 omega0 = _mm256_mul_ps(alpha0, inv_alpha);
        __m256 omega1 = _mm256_mul_ps(alpha1, inv_alpha);
        __m256 omega2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(omega0, omega1));

        // Compute result
        __m256 result = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(omega0, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.0f / 3.0f), va),
                                                   _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-7.0f / 6.0f), vb),
                                                                 _mm256_mul_ps(_mm256_set1_ps(11.0f / 6.0f), vc)))),

                _mm256_mul_ps(omega1, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f / 6.0f), vb),
                                                   _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(5.0f / 6.0f), vc),
                                                                 _mm256_mul_ps(_mm256_set1_ps(1.0f / 3.0f), vd))))),

            _mm256_mul_ps(omega2, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.0f / 3.0f), vc),
                                                _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(5.0f / 6.0f), vd),
                                                              _mm256_mul_ps(_mm256_set1_ps(-1.0f / 6.0f), ve)))));

        _mm256_storeu_ps(&out[i], result);
    }

    // Scalar fallback for remaining elements
    for (; i < NENTRIES; ++i)
    {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}

// Reference implementation using AVX and scalar fallback
void weno_avx_reference(const float * const a, const float * const b, const float * const c,
                           const float * const d, const float * const e, float * const out,
                           const int NENTRIES)
{
    weno_avx_core(a, b, c, d, e, out, NENTRIES);
}
