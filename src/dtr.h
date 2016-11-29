#ifndef DTR_H_INCLUDED
#define DTR_H_INCLUDED




/*
    Overview
    gaussian_kernel.c: RKHS regression with Gaussian kernel
    minimization.c: given covariates and action, find cutoff values
    clause.c: find covariates to use and action to take
    rule.c: find if-then list, which consists of several clauses
    outcomes.c, sort.c: utility functions
*/




/** gaussian_kernel.c **/

typedef struct {
    int n;                      /* number of samples in this group */
    double percent;             /* percent of samples in this group */
    double intercept;           /* mean of y */
    double *restrict x;         /* n by p */
    double *restrict npdx;      /* n * (n + 1) / 2 by p */
    double *restrict k;         /* n by n */
    double *restrict kpinv;     /* n by n */
    double *restrict alpha;     /* n */
    double *restrict y;         /* n */
    double *restrict work;      /* n by (4 + 2 * n) */
} training_data;

typedef struct {
    int p;                      /* number of predictors */
    training_data *data;        /* num_groups */
    int num_groups;
    double *restrict scaling;   /* p */
    double *restrict gamma;     /* p */
    double lambda;
    double *restrict work;      /* p + 1 */
} training_input;

void kernel_train(const int n, const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling,
    double *restrict node, double *restrict alpha,
    int *restrict offset, double *restrict gamma,
    double *restrict intercept);

void kernel_predict(const int p, const int num_groups, const int new_n,
    const double *restrict node, const double *restrict alpha,
    const int *restrict offset, const double *restrict gamma,
    const double *restrict intercept, const double *restrict new_x,
    double *restrict new_yhat);




/** outcomes.c **/

void get_regrets_from_outcomes(const double *restrict outcomes,
    const unsigned int n, const unsigned int m,
    double *restrict regrets);




/** sort.c **/

void sort_matrix(const double *restrict z,
    unsigned int *restrict iz,
    unsigned int *restrict rz,
    double *restrict sz,
    const unsigned int n, const unsigned int p);

void reverse_sort(const unsigned int *restrict iz,
    const unsigned int *restrict rz,
    const double *restrict sz,
    unsigned int *restrict rev_iz,
    unsigned int *restrict rev_rz,
    double *restrict rev_sz,
    const unsigned int n, const unsigned int p);




/** minimization.c **/

void minimize_loss(const unsigned int *restrict ix,
    const unsigned int *restrict rx,
    const double *restrict sx,
    const unsigned int *restrict iy,
    const unsigned int *restrict ry,
    const double *restrict sy,
    const double *restrict loss,
    const unsigned int *restrict next,
    const unsigned int n, const double zeta, const double eta,
    double *restrict opt_cx, double *restrict opt_cy,
    double *restrict opt_loss);




/** clause.c **/

#define CONDITION_TYPE_LL        1
#define CONDITION_TYPE_LR        2
#define CONDITION_TYPE_RL        3
#define CONDITION_TYPE_RR        4

typedef struct {
    unsigned int a;
    unsigned char type;
    unsigned int j1, j2;
    double c1, c2;
} statement;

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
    statement *restrict clause);

void find_last_clause(const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int m,
    statement *restrict clause);

void apply_clause(const double *restrict z,
    unsigned int *restrict next, const unsigned int n,
    const statement *restrict clause, int *restrict action);




/** rule.c **/

void find_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    const unsigned int max_length,
    statement *restrict rule, unsigned int *restrict rule_length,
    int *restrict action);

void apply_rule(const double *restrict z,
    const unsigned int n,
    const statement *rule, const unsigned int rule_length,
    int *restrict action);

void cv_tune_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const int *restrict fold, const unsigned int num_folds,
    double *restrict cv_regret);




#endif /* DTR_H_INCLUDED */



