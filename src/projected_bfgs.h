#ifndef PROJECTED_BFGS_H_INCLUDED
#define PROJECTED_BFGS_H_INCLUDED




#define MX_BFGS_ACCEPT      0.0001
#define MX_BFGS_SHRINKAGE   0.2
#define MX_BFGS_STRETCH     2
#define MX_BFGS_LONGEST     100
#define MX_BFGS_TINY        1e-20
#define MX_BFGS_NOT_UPDATE  0.25
#define MX_BFGS_RESTART     1.2




/* x, p, extra */
typedef double (*objective_function)(const double *, int, void *);
/* x, p, extra, gradient */
typedef void (*objective_gradient)(const double *, int, void *, double *);




void projected_bfgs(double *restrict x, const int p, void *extra,
    objective_function func, objective_gradient grad,
    const int max_iter, const double tol,
    const double *restrict lower, const double *restrict upper,
    const double step_limit, double *value);




#endif /* PROJECTED_BFGS_H_INCLUDED */



