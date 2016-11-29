#include <stdlib.h>
#include <math.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "projected_bfgs.h"




void projected_bfgs(double *restrict x, const int p, void *extra,
    objective_function func, objective_gradient grad,
    const int max_iter, const double tol,
    const double *restrict lower, const double *restrict upper,
    const double step_limit, double *value)
{
    /*
    BFGS algorithm for minimization with box constraints
    reference: Kim, D., Sra, S. & Dhillon, I. S. (2010).
    Tackling box-constrained optimization via a new
    projected quasi-Newton approach.
    SIAM Journal on Scientific Computing, 32, 3548–3563

    x: vector of length p, initial value
    extra: addition parameters needed by func
    func: objective function
    grad: gradient of objective function
    max_iter: number of maximum iterations in BFGS
    tol: relative tolerance for convergence in minimum value
    lower: vector of length p, lower bound constraints
    upper: vector of length p, upper bound constraints
    step_limit: largest step size allowed
    value: scalar, minimum value
    */

    const int q1 = p * sizeof(double);
    const int q2 = (p * (p + 1) / 2) * sizeof(double);
    int *restrict bind = (int *)malloc(p * sizeof(int));
    if (bind == 0) return;

    #define project_to_box                       \
        for (int k = 0; k < p; ++k) {            \
            if (x[k] <= lower[k]) {              \
                x[k] = lower[k]; bind[k] = 1;    \
            } else if (x[k] >= upper[k]) {       \
                x[k] = upper[k]; bind[k] = 2;    \
            } else {                             \
                bind[k] = 0;                     \
            }                                    \
        }
    project_to_box;

    double f = func(x, p, extra);
    *value = f;

    const int lwork = p * 4 + p * (p + 1) / 2;
    double *work = (double *)malloc(lwork * sizeof(double));
    if (work == 0) return;

    double *restrict g = work;          /* p-vector */
    double *restrict s = work + p;      /* p-vector */
    double *restrict y = work + 2 * p;  /* p-vector */
    double *restrict u = work + 3 * p;  /* p-vector */
    double *restrict H = work + 4 * p;  /* p by p symmetric matrix */
                                        /* packed form, upper part */
    grad(x, p, extra, g);

    const int inc_one = 1;
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const char upper_part = 'U';

    const int limit_not_updateH = (int)(p * MX_BFGS_NOT_UPDATE) + 1;
    const int limit_restartH = (int)(p * MX_BFGS_RESTART) + 1;
    int i = 1, j, not_updateH = 0, restartH = 0, fixed;
    int resetH = 1; /* whether to set H as a scaled identity matrix */
    int accept = 0; /* whether to accept a point in line search */
    double dotprod, f1, rho, temp;
    double gamma = 1.0;       /* scaling factor for H */
    double a, min_a;          /* step length in line search */
    double step_size;         /* l1 norm of s */

    while ((i <= max_iter) && (resetH != -1)) {

        /* periodical restart */
        if (restartH == limit_restartH) resetH = 1;

        /* search direction: s = - H_{free} g */
        memcpy(u, g, q1);
        for (int k = 0; k < p; ++k) {
            if (((bind[k] == 1) && (u[k] > 0.0))
                || ((bind[k] == 2) && (u[k] < 0.0))) {
                bind[k] = 3; u[k] = 0.0;
            }
        }

        if (resetH == 0) {
            ++restartH;
            F77_CALL(dspmv)(&upper_part, &p, &neg_one, H,
                u, &inc_one, &zero, s, &inc_one);
        } else {
            restartH = 0;
            /* s = -g */
            memcpy(s, u, q1);
            F77_CALL(dscal)(&p, &neg_one, s, &inc_one);
        }

        step_size = 0.0;
        for (int k = 0; k < p; ++k)
            step_size += fabs(s[k]);

        fixed = 0;
        for (int k = 0; k < p; ++k) {
            if ((bind[k] == 3)
                || ((bind[k] == 1) && (s[k] < 0.0))
                || ((bind[k] == 2) && (s[k] > 0.0))) {
                s[k] = 0.0; ++fixed;
            }
        }

        /* dotprod = s^T g */
        if (fixed < p)
            dotprod = F77_CALL(ddot)(&p, s, &inc_one, g, &inc_one);
        else
            dotprod = 1.0; /* in this case s is a zero vector */

        /* checks whether s is a downhill direction */
        if (dotprod < 0.0) {
            /* performs a line search along s */
            a = step_limit / step_size;
            if (a > 1.0) a = 1.0;

            min_a = (fabs(f) * MX_BFGS_ACCEPT * tol + MX_BFGS_TINY)
                / (-dotprod);
            if (min_a > 1.0) min_a = 1.0;

            accept = 0; memcpy(u, x, q1);
            while ((!accept) && (a >= min_a)) {
                /* x = u + a s */
                F77_CALL(daxpy)(&p, &a, s, &inc_one, x, &inc_one);
                project_to_box;
                f1 = func(x, p, extra);
                accept = (f1 <= f + MX_BFGS_ACCEPT * dotprod * a);
                if (!accept) {
                    a *= MX_BFGS_SHRINKAGE; memcpy(x, u, q1);
                }
            }

            /* checks convergence */
            if (accept) {
                temp = f - tol * (fabs(f) + MX_BFGS_TINY);
                f = f1;
                if (f >= temp) accept = 0;
            }

            /* checks sufficient descent */
            if (accept) {
                ++i;
                /* s = x0 - x1 */
                F77_CALL(daxpy)(&p, &neg_one, x, &inc_one, u, &inc_one);
                memcpy(s, u, q1);
                /* y = g0 - g1 */
                memcpy(y, g, q1);
                grad(x, p, extra, g);
                F77_CALL(daxpy)(&p, &neg_one, g, &inc_one, y, &inc_one);
                /* rho = s^T y */
                rho = F77_CALL(ddot)(&p, s, &inc_one, y, &inc_one);

                /* to keep positive definiteness, rho must > 0 */
                if (rho > 0.0) {
                    /* u = H y */
                    if (resetH == 0) {
                        F77_CALL(dspmv)(&upper_part, &p, &one, H,
                            y, &inc_one, &zero, u, &inc_one);
                    } else {
                        /* gamma = s^T y / y^T y */
                        temp = F77_CALL(dnrm2)(&p, y, &inc_one);
                        gamma = rho / (temp * temp);
                        /* u = gamma * y */
                        memcpy(u, y, q1);
                        F77_CALL(dscal)(&p, &gamma, u, &inc_one);

                        /* sets H to scaled identity matrix: H = gamma I */
                        memset(H, 0, q2);
                        for (j = 0; j < p; ++j)
                            H[j * (j + 3) / 2] = gamma;
                    }

                    /* performs the BFGS update of H */
                    temp = F77_CALL(ddot)(&p, y, &inc_one, u, &inc_one);
                    temp = (1.0 + temp / rho) / rho;
                    F77_CALL(dspr)(&upper_part, &p, &temp, s, &inc_one, H);
                    temp = -1.0 / rho;
                    F77_CALL(dspr2)(&upper_part, &p, &temp, s, &inc_one,
                        u, &inc_one, H);

                    resetH = 0;
                } else { /* i.e. rho <= 0 */
                    ++not_updateH;
                    if (not_updateH >= limit_not_updateH) {
                        resetH = 1; not_updateH = 0;
                    }
                }
            } else { /* i.e. !accept */
                if (resetH == 0) resetH = 1; else resetH = -1;
            }
        } else { /* i.e. dotprod >= 0 */
            if (resetH == 0) resetH = 1; else resetH = -1;
        }

    } /* while ((i <= max_iter) && (resetH != -1)) */

    *value = f;
    free(work);
    free(bind);
}



