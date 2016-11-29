#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dtr.h"




void find_clause(const unsigned int *restrict asc_iz,
    const unsigned int *restrict asc_rz,
    const double *restrict asc_sz,
    const unsigned int *restrict desc_iz,
    const unsigned int *restrict desc_rz,
    const double *restrict desc_sz,
    const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    statement *restrict clause)
{
    /*
    loop over covariate pair and treatment to find the best covariate(s)
    and treatment to use and call minimize_loss to find the best cutoff values

    asc_z, asc_iz, asc_rz: matrix of predictors, of size n by p
    desc_z, desc_iz, desc_rz: matrix of predictors, of size n by p
    regrets: matrix of potential improvements, of size n by m
    next: vector of indices, of length (n + 1),
          specifying the observations to use
    clause: scalar
    */

    double c1, c2, loss, opt_loss = INFINITY;
    int redundant1 = 0;

    /* loop over combinations of covariates */
    for (unsigned int j1 = 0; j1 < p; ++j1) {
        const unsigned int off1 = n * j1;

        #define DELTA 1e-8
        if (asc_sz[n - 1 + off1] - asc_sz[off1] < DELTA) {
            /* z_{j1} is constant */
            if (redundant1) continue;
            redundant1 = 1;
        }
        int redundant2 = 0;

        for (unsigned int j2 = j1 + 1; j2 < p; ++j2) {
            const unsigned int off2 = n * j2;

            if (asc_sz[n - 1 + off2] - asc_sz[off2] < DELTA) {
                /* z_{j2} is constant */
                if (redundant2) continue;
                redundant2 = 1;
            }

            for (unsigned int a = 0; a < m; ++a) {
                /* loop over action options */

                #define EPSILON 1e-10
                #define SEARCH(_type, _iz1, _rz1, _sz1, _iz2, _rz2, _sz2) \
                { \
                    minimize_loss( \
                        _iz1 + off1, _rz1 + off1, _sz1 + off1, \
                        _iz2 + off2, _rz2 + off2, _sz2 + off2, \
                        regrets + n * a, next, \
                        n, zeta, eta, \
                        &c1, &c2, &loss); \
                    if (loss < opt_loss - EPSILON) { \
                        /* use EPSILON to ignore numerical rounding errors */ \
                        opt_loss = loss; \
                        clause->a = a; \
                        clause->type = _type; \
                        clause->j1 = j1; \
                        clause->j2 = j2; \
                        clause->c1 = c1; \
                        clause->c2 = c2; \
                    } \
                }

                SEARCH(CONDITION_TYPE_LL,
                    asc_iz, asc_rz, asc_sz, asc_iz, asc_rz, asc_sz);
                SEARCH(CONDITION_TYPE_LR,
                    asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz);
                SEARCH(CONDITION_TYPE_RL,
                    desc_iz, desc_rz, desc_sz, asc_iz, asc_rz, asc_sz);
                SEARCH(CONDITION_TYPE_RR,
                    desc_iz, desc_rz, desc_sz, desc_iz, desc_rz, desc_sz);

                #undef EPSILON
                #undef SEARCH
            }  /* loop over a */
        }  /* loop over j2 */
        #undef DELTA
    }  /* loop over j1 */
}




void find_last_clause(const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int m,
    statement *restrict clause)
{
    /*
    for the last clause, no covariate can be used
    only the best treatment needs to be determined

    clause: scalar
    */

    double opt_loss = INFINITY;
    unsigned int opt_a = 0;

    for (unsigned int a = 0; a < m; ++a) {
        double loss = 0.0;
        unsigned int i = next[0], off = n * a;
        while (i < n) {
            loss += regrets[i + off];
            i = next[i + 1];
        }

        if (loss < opt_loss) {
            opt_loss = loss;
            opt_a = a;
        }
    }

    clause->a = opt_a;
    clause->type = CONDITION_TYPE_LL;
    clause->j1 = 0;
    clause->j2 = 1;
    clause->c1 = INFINITY;
    clause->c2 = INFINITY;
}




void apply_clause(const double *restrict z,
    unsigned int *restrict next, const unsigned int n,
    const statement *restrict clause, int *restrict action)
{
    /*
    assign the recommended action dictated by the given clause

    z: matrix of predictors, of size n by p
    next: vector of indices, of length (n + 1),
          specifying the observations to use
    clause: scalar
    action: vector of actions, of length n
            for those who failed the if-condition in the clause,
            their values are not changed
    */

    const double *restrict z1 = z + n * clause->j1;
    const double *restrict z2 = z + n * clause->j2;
    const double c1 = clause->c1, c2 = clause->c2;
    const unsigned int a = clause->a;

    /* update action */
    unsigned int i = next[0];
    switch (clause->type) {
    case CONDITION_TYPE_LL:
        while (i < n) {
            if (z1[i] <= c1 && z2[i] <= c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_LR:
        while (i < n) {
            if (z1[i] <= c1 && z2[i] > c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_RL:
        while (i < n) {
            if (z1[i] > c1 && z2[i] <= c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_RR:
        while (i < n) {
            if (z1[i] > c1 && z2[i] > c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;
    }

    /* update next */
    i = n;
    do {
        --i;
        if (action[i] >= 0 || next[i] > i)
            next[i] = next[i + 1];
    } while (i);
}



