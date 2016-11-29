#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dtr.h"




void find_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    const unsigned int max_length,
    statement *restrict rule, unsigned int *restrict rule_length,
    int *restrict action)
{
    /*
    build clauses one by one to form a rule

    z: matrix of predictors, of size n by p
    regrets: matrix of potential improvements, of size n by m
    zeta, eta: tuning parameters
    action: vector, of length n
    rule: vector, of length at least max_length
    rule_length: scalar, the actual length of the rule returned
    */

    /* sort z */
    unsigned int size = n * p * sizeof(double);
    double *restrict asc_sz = (double *)malloc(size);
    double *restrict desc_sz = (double *)malloc(size);

    size = n * p * sizeof(unsigned int);
    unsigned int *restrict asc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict asc_rz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_rz = (unsigned int *)malloc(size);

    sort_matrix(z, asc_iz, asc_rz, asc_sz, n, p);
    reverse_sort(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz, n, p);

    /* initialize next and action */
    unsigned int *restrict next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        next[i] = i;
        action[i] = -1;
    }
    next[n] = n;

    /* build clauses */
    for (unsigned int k = 0; (next[0] < n) && (k + 1 < max_length); ++k) {
        find_clause(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz,
            regrets, next, n, p, m, zeta, eta, rule + k);
        apply_clause(z, next, n, rule + k, action);
        *rule_length = k + 1;
    }

    if (next[0] < n) {
        /* for the rest, defaults to the overall best treatment */
        unsigned int k = max_length - 1;
        find_last_clause(regrets, next, n, m, rule + k);
        apply_clause(z, next, n, rule + k, action);
        *rule_length = max_length;
    }

    /* clean up */
    free(asc_sz);
    free(asc_iz);
    free(asc_rz);
    free(desc_sz);
    free(desc_iz);
    free(desc_rz);

    free(next);
}




void apply_rule(const double *restrict z,
    const unsigned int n,
    const statement *rule, const unsigned int rule_length,
    int *restrict action)
{
    /*
    apply the given rule to each row of z to get an action

    action: vector, of length n
    */

    /* initialize next and action */
    unsigned int *restrict next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        next[i] = i;
        action[i] = -1;
    }
    next[n] = n;    /* next is of length (n + 1) */

    /* apply clauses */
    for (unsigned int k = 0; (next[0] < n) && (k < rule_length); ++k) {
        apply_clause(z, next, n, rule + k, action);
    }

    /* clean up */
    free(next);
}




void cv_tune_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const int *restrict fold, const unsigned int num_folds,
    double *restrict cv_regret)
{
    /*
    use cross validation to choose zeta and eta

    fold: vector of length n with values in {0, ..., num_folds - 1}
    zeta_choices, eta_choices: vector of length num_choices
    cv_regret: vector of cross validated mean regret, of length num_choices
    */

    /* sort z */
    unsigned int size = n * p * sizeof(double);
    double *restrict asc_sz = (double *)malloc(size);
    double *restrict desc_sz = (double *)malloc(size);

    size = n * p * sizeof(unsigned int);
    unsigned int *restrict asc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict asc_rz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_rz = (unsigned int *)malloc(size);

    sort_matrix(z, asc_iz, asc_rz, asc_sz, n, p);
    reverse_sort(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz, n, p);

    /* allocate memory */
    unsigned int *restrict train_next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    unsigned int *restrict test_next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    int *restrict train_action = (int *)malloc(n * sizeof(int));
    int *restrict test_action = (int *)malloc(n * sizeof(int));

    statement *restrict rule = (statement *)
        malloc(max_length * sizeof(statement));
    unsigned int rule_length = 0;

    for (unsigned int index_choice = 0; index_choice < num_choices;
        ++index_choice) {
        const double zeta = zeta_choices[index_choice];
        const double eta = eta_choices[index_choice];

        /* initialize test_action */
        for (unsigned int i = 0; i < n; ++i)
            test_action[i] = -1;

        for (unsigned int index_fold = 0; index_fold < num_folds;
            ++index_fold) {
            /* initialize train_next, test_next and train_action */
            train_next[n] = test_next[n] = n;
            unsigned int i = n;
            do {
                --i;
                if (fold[i] == index_fold) {
                    /* this observation goes to the test set */
                    train_next[i] = train_next[i + 1];
                    test_next[i] = i;
                } else {
                    /* this observation goes to the train set */
                    train_next[i] = i;
                    test_next[i] = test_next[i + 1];
                }
            } while (i);

            for (unsigned int i = 0; i < n; ++i)
                train_action[i] = -1;

            /* build clauses */
            for (unsigned int k = 0; (train_next[0] < n)
                && (k + 1 < max_length); ++k) {
                find_clause(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz,
                    regrets, train_next, n, p, m, zeta, eta, rule + k);
                apply_clause(z, train_next, n, rule + k, train_action);
                rule_length = k + 1;
            }

            if (train_next[0] < n) {
                /* for the rest, defaults to the overall best treatment */
                /* and no need to call apply_clause for updating train_next */
                unsigned int k = max_length - 1;
                find_last_clause(regrets, train_next, n, m, rule + k);
                rule_length = max_length;
            }

            /* apply clauses */
            for (unsigned int k = 0; (test_next[0] < n) && (k < rule_length);
                ++k) {
                apply_clause(z, test_next, n, rule + k, test_action);
            }

        } /* loop over index_fold */

        /* evaluate rule using test_action */
        double temp = 0.0;
        for (unsigned int i = 0; i < n; ++i)
            temp += regrets[i + test_action[i] * n];

        cv_regret[index_choice] = temp / n;

    } /* loop over index_choice */

    /* clean up */
    free(asc_sz);
    free(asc_iz);
    free(asc_rz);
    free(desc_sz);
    free(desc_iz);
    free(desc_rz);

    free(train_next);
    free(test_next);
    free(train_action);
    free(test_action);

    free(rule);
}



