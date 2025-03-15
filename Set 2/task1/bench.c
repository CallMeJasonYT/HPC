#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"
#include "weno_avx.h"
#include "weno_omp.h"

float * myalloc(const int NENTRIES, const int verbose )
{
    const int initialize = 1;
    enum { alignment_bytes = 32 };
    float * tmp = NULL;

    const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
    assert(result == 0);

    if (initialize)
    {
        for(int i = 0; i < NENTRIES; ++i)
            tmp[i] = drand48();

        if (verbose)
        {
            for(int i = 0; i < NENTRIES; ++i)
                printf("tmp[%d] = %f\n", i, tmp[i]);
            printf("==============\n");
        }
    }
    return tmp;
}

double get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
    static const int verbose = 0;

    for(int i = 0; i < N; ++i)
    {
        assert(!isnan(ref[i]));
        assert(!isnan(val[i]));

        const double err = ref[i] - val[i];
        const double relerr = err / fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

        if (fabs(relerr) >= tol || fabs(err) >= tol)
        {
            // Print the error for debugging
            printf("Error at index %d: ref = %f, val = %f, abs_err = %e, rel_err = %e\n", i, ref[i], val[i], err, relerr);
        }

        assert(fabs(relerr) < tol || fabs(err) < tol);
    }

    if (verbose) printf("\t");
}

void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name)
{
    const int NENTRIES = 4 * (NENTRIES_ / 4);

    float * const a = myalloc(NENTRIES, verbose);
    float * const b = myalloc(NENTRIES, verbose);
    float * const c = myalloc(NENTRIES, verbose);
    float * const d = myalloc(NENTRIES, verbose);
    float * const e = myalloc(NENTRIES, verbose);
    float * const f = myalloc(NENTRIES, verbose);
    float * const gold = myalloc(NENTRIES, verbose);
    float * const result = myalloc(NENTRIES, verbose);

    // Time and run the reference WENO implementation
    double start_time = get_wtime();
    weno_minus_reference(a, b, c, d, e, gold, NENTRIES);
    weno_minus_reference(a, b, c, d, e, result, NENTRIES);
    double end_time = get_wtime();
    printf("WENO (reference) execution time: %f seconds\n", end_time - start_time);

    // Verify accuracy of WENO (reference) implementation
    const double tol = 1e-5;
    printf("Verifying accuracy of WENO (reference) with tolerance %.5e...\n", tol);
    check_error(tol, gold, result, NENTRIES);
    printf("WENO (reference) verification passed!\n");

    // Time and run the AVX WENO implementation
    start_time = get_wtime();
    weno_avx_reference(a, b, c, d, e, gold, NENTRIES);
    weno_avx_reference(a, b, c, d, e, result, NENTRIES);
    end_time = get_wtime();
    printf("WENO (AVX) execution time: %f seconds\n", end_time - start_time);

    // Verify accuracy of WENO (AVX) implementation
    printf("Verifying accuracy of WENO (AVX) with tolerance %.5e...\n", tol);
    check_error(tol, gold, result, NENTRIES);
    printf("WENO (AVX) verification passed!\n");

	// Time and run the OMP WENO implementation
    start_time = get_wtime();
    weno_omp_reference(a, b, c, d, e, gold, NENTRIES);
    weno_omp_reference(a, b, c, d, e, result, NENTRIES);
    end_time = get_wtime();
    printf("WENO (OMP) execution time: %f seconds\n", end_time - start_time);

    // Verify accuracy of WENO (OMP) implementation
    printf("Verifying accuracy of WENO (OMP) with tolerance %.5e...\n", tol);
    check_error(tol, gold, result, NENTRIES);
    printf("WENO (OMP) verification passed!\n");

    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(gold);
    free(result);
}

int main (int argc, char *argv[])
{
    const double desired_kb = atoi(argv[1]) * 0.5;
    const double desired_mb = atoi(argv[2]);

    /* performance on cache hits */
    {
        const int nentries = floor(desired_kb * 1024. / 7 / sizeof(float));
        const int ntimes = (int)floor(2. / (1e-7 * nentries));
        printf("nentries: %d and L1 Cache Size Available: %e KB\n", nentries, desired_kb);
        for(int i = 0; i < 4; ++i)
        {
            printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache");
			printf("\n");
        }
    }

    /* performance on data streams */
    {
        const int nentries = (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));
        printf("nentries: %d and Main Memory Size Available: %e MB\n", nentries, desired_mb);
        for(int i = 0; i < 4; ++i)
        {
            printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, 1, 0, "stream");
			printf("\n");
        }
    }

    return 0;
}
