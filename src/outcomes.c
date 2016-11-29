#include <stdlib.h>
#include <math.h>
#include "dtr.h"




void get_regrets_from_outcomes(const double *restrict outcomes,
    const unsigned int n, const unsigned int m,
    double *restrict regrets)
{
    /*
    compute regret, which is the difference between the best outcome
    and the current outcome:
    regret_{i, a} = max_{a'} outcome_{i, a'} - outcome_{i, a}

    outcomes: matrix, n by m
    regrets: matrix, n by m
    m: number of treatments
    */

    double *restrict best = (double *)malloc(n * sizeof(double));
    for (unsigned int i = 0; i < n; ++i)
        best[i] = -INFINITY;

    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int i = 0; i < n; ++i) {
            if (outcomes[i + n * a] > best[i])
                best[i] = outcomes[i + n * a];
        }
    }

    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int i = 0; i < n; ++i)
            regrets[i + n * a] = best[i] - outcomes[i + n * a];
    }

    free(best);
}



