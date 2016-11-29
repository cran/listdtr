#include <stdlib.h>
#include <math.h>
#include <R_ext/Lapack.h>
#include "projected_bfgs.h"
#include "dtr.h"




/********************************/
/* training in a single dataset */
/********************************/




/* index of the i-th diagonal element in packed format */
#define DIAG_ELEM(i) ((i * (i + 3)) / 2)




void transform_x(const int n, const int p,
    const double *restrict x,
    double *restrict npdx)
{
    /*
    for each j, compute (x_{ij} - x_{i'j})^2 for all pairs i <= i'

    x: matrix of size n by p, predictors, each row is an x_i
    npdx (negative pairwise distance in x): matrix of size n (n + 1) / 2 by p,
        in upper triangle packed format, thus npdx[i1, i2] is stored in
        the (i1 + i2 (i2 + 1) / 2)-th element for i1, i2 = 0, ..., n - 1
    */

    const int nrow_npdx = (n * (n + 1)) / 2;
    int offset_x, offset_npdx;
    double temp_x2, temp_d;

    for (int j = 0; j < p; ++j) {
        offset_x = n * j;
        for (int i2 = 0; i2 < n; ++i2) {
            offset_npdx = nrow_npdx * j + (i2 * (i2 + 1)) / 2;
            temp_x2 = x[i2 + offset_x];

            for (int i1 = 0; i1 < i2; ++i1) {
                temp_d = x[i1 + offset_x] - temp_x2;
                npdx[i1 + offset_npdx] = -temp_d * temp_d;
            }
            npdx[i2 + offset_npdx] = 0.0;
        }
    }
}




void center_y(const int n,
    double *restrict y, double *restrict intercept)
{
    /*
    center y to have mean zero

    y: vector of length n, responses
    intercept: scalar, sample mean of y
    */

    double mean = 0.0;
    for (int i = 0; i < n; ++i)
        mean += y[i];
    mean /= n;

    for (int i = 0; i < n; ++i)
        y[i] -= mean;
    *intercept = mean;
}




void compute_kernel(const int n, const int p,
    const double *restrict npdx,
    const double *restrict gamma,
    const double lambda,
    double *restrict k,
    double *restrict kpinv)
{
    /*
    compute the design matrix defined by gaussian kernel as well as
    its inverse

    npdx: matrix of size n (n + 1) / 2 by p
    gamma: vector of length p, scaling factors
    lambda: scalar
    k: matrix of size n by n, design matrix induced by gaussian kernel,
        in upper triangle packed format
    kpinv: inverse of (k + lambda * I_n)
    */

    const char normal = 'N';
    const int inc_one = 1;
    const double zero = 0.0, one = 1.0;
    const int nrow_npdx = (n * (n + 1)) / 2;

    /* DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY) */
    F77_NAME(dgemv)(&normal, &nrow_npdx, &p, &one, npdx, &nrow_npdx,
        gamma, &inc_one, &zero, k, &inc_one);

    for (int i = 0; i < nrow_npdx; ++i)
        k[i] = exp(k[i]);

    /* adding lambda to the diagonal */
    memcpy(kpinv, k, nrow_npdx * sizeof(double));
    for (int j = 0; j < n; ++j)
        kpinv[DIAG_ELEM(j)] += lambda;

    /* DPPTRF(UPLO,N,AP,INFO) */
    const char upper_part = 'U';
    int info = 0;
    F77_NAME(dpptrf)(&upper_part, &n, kpinv, &info);

    /* DPPTRI(UPLO,N,AP,INFO) */
    F77_NAME(dpptri)(&upper_part, &n, kpinv, &info);
}




void estimate_alpha(const int n,
    const double *restrict kpinv,
    const double *restrict y,
    double *restrict alpha)
{
    /*
    compute the regression coefficients (called alpha)
    let K be the kernel matrix indexed by gamma and lambda be a parameter
    alpha = argmin_{a} (y - K a)^T (y - K a) + lambda a^T K a
          = (K + lambda I)^{-1} y
    note that (K + lambda I)^{-1} is stored in kpinv

    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    y: vector of length n, responses
    alpha: vector of n, coefficients
    */

    /* DSPMV(UPLO,N,ALPHA,AP,X,INCX,BETA,Y,INCY) */
    const char upper_part = 'U';
    const int inc_one = 1;
    const double zero = 0.0, one = 1.0;
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, y, &inc_one,
        &zero, alpha, &inc_one);
}




void get_loocv_objective(const int n,
    const double *restrict kpinv,
    const double *restrict alpha,
    double *restrict objective)
{
    /*
    compute the value of leave-one-out cross validated error

    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    alpha: vector of n, coefficients
    objective: scalar, cross validated error
    */

    double obj = 0.0, temp;
    for (int i = 0; i < n; ++i) {
        temp = alpha[i] / kpinv[DIAG_ELEM(i)];
        obj += (temp > 0.0 ? temp : -temp);
    }
    *objective = obj / n;
}




void get_loocv_gradient(const int n, const int p,
    const double *restrict npdx,
    const double *restrict k,
    const double *restrict kpinv,
    const double *restrict alpha,
    const double *restrict y,
    double *work,
    double *restrict gradient)
{
    /*
    compute the gradient of LOOCV objective function

    npdx: matrix of size n (n + 1) / 2 by p, negative pairwise distance in x
    k: matrix of size n by n, design matrix induced by gaussian kernel
    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    alpha: vector of length n, coefficients
    y: vector of length n, responses
    work: matrix, of size n by (4 + 2 * n)
    gradient: vector of length (p + 1), the gradient of LOOCV value
        with respect to gamma (of length p) and lambda (scalar)
    */

    double temp, temp2;
    double *restrict vec1 = work + n * 0;
    double *restrict vec2 = work + n * 1;
    double *restrict vec3 = work + n * 2;
    double *restrict vec4 = work + n * 3;
    double *restrict mat1 = work + n * 4;
    double *restrict mat2 = work + n * (4 + n);

    /* compute mat1 = kpinv (diag(vec1) + y vec2^T) kpinv,
       where kpinv = (k + lambda I_n)^{-1} */
    for (int i = 0; i < n; ++i) {
        temp = kpinv[DIAG_ELEM(i)];
        temp2 = (alpha[i] > 0.0 ? 1.0 : (alpha[i] < 0.0 ? -1.0 : 0.0));
        vec1[i] = alpha[i] / (temp * temp) * temp2;
        vec2[i] = -1.0 / temp * temp2;
    }

    memset(mat1, 0, n * n * sizeof(double));

    /* DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA) */
    const double zero = 0.0, one = 1.0;
    const int inc_one = 1;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            vec3[i] = vec4[i] = (i <= j ? kpinv[i + (j * (j + 1)) / 2]
                : kpinv[j + (i * (i + 1)) / 2]);
        }
        F77_NAME(dger)(&n, &n, vec1 + j, vec3, &inc_one, vec4, &inc_one,
            mat1, &n);
    }

    /* DSPMV(UPLO,N,ALPHA,AP,X,INCX,BETA,Y,INCY) */
    const char upper_part = 'U';
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, y, &inc_one,
        &zero, vec3, &inc_one);
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, vec2, &inc_one,
        &zero, vec4, &inc_one);

    /* DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA) */
    F77_NAME(dger)(&n, &n, &one, vec3, &inc_one, vec4, &inc_one, mat1, &n);

    /* change mat1 to packed format consistent with k */
    for (int i2 = 0; i2 < n; ++i2) {
        for (int i1 = 0; i1 < i2; ++i1)
            mat2[i1 + (i2 * (i2 + 1)) / 2] =
                mat1[i1 + n * i2] + mat1[i2 + n * i1];
        mat2[DIAG_ELEM(i2)] = mat1[i2 + n * i2];
    }

    /* compute trace((\partial (k + lambda I_n) / \partial gamma_j) mat2) */
    const int nrow_npdx = (n * (n + 1)) / 2;
    int offset_npdx;
    double grd;
    for (int j = 0; j < p; ++j) {
        grd = 0.0;
        offset_npdx = nrow_npdx * j;
        for (int i = 0; i < nrow_npdx; ++i)
            grd += k[i] * npdx[i + offset_npdx] * mat2[i];
        gradient[j] = grd / n;
    }

    /* compute trace((\partial (k + lambda I_n) / \partial lambda) mat2) */
    grd = 0.0;
    for (int i = 0; i < n; ++i)
        grd += mat2[DIAG_ELEM(i)];
    gradient[p] = grd / n;
}




void predict_response(const int n, const int p, const int new_n,
    const double *restrict x,
    const double *restrict alpha,
    const double *restrict gamma,
    const double intercept,
    const double *restrict new_x,
    double *restrict new_yhat)
{
    /*
    predict the responses for new predictors

    x: matrix of size n by p, predictors
    alpha: vector of length n, coefficients
    gamma: vector of length p, scaling factors
    intercept: scalar, sample mean of y
    new_x: matrix of size new_n by p, new predictors
    new_yhat: vector of length new_n, predicted responses
    */

    double value, term;
    for (int k = 0; k < new_n; ++k) {
        value = intercept;
        for (int i = 0; i < n; ++i) {
            term = 0.0;
            for (int j = 0; j < p; ++j) {
                double temp = new_x[k + new_n * j] - x[i + n * j];
                term += gamma[j] * temp * temp;
            }
            value += alpha[i] * exp(-term);
        }
        new_yhat[k] = value;
    }
}




/*********************************/
/* training in multiple datasets */
/*********************************/




void allocate_training_data(const int n, const int p,
    const double percent, training_data *restrict data)
{
    data->n = n;
    data->percent = percent;
    data->intercept = 0.0;
    const int nrow_npdx = (n * (n + 1)) / 2;
    data->x     = (double *)malloc(n * p * sizeof(double));
    data->npdx  = (double *)malloc(nrow_npdx * p * sizeof(double));
    data->k     = (double *)malloc(n * n * sizeof(double));
    data->kpinv = (double *)malloc(n * n * sizeof(double));
    data->alpha = (double *)malloc(n * sizeof(double));
    data->y     = (double *)malloc(n * sizeof(double));
    data->work  = (double *)malloc(n * (4 + 2 * n) * sizeof(double));
}




void free_training_data(training_data *restrict data)
{
    free(data->x);
    free(data->npdx);
    free(data->k);
    free(data->kpinv);
    free(data->alpha);
    free(data->y);
    free(data->work);
}




training_input *allocate_training_input(const int n,
    const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling)
{
    /*
    group: vector, of length n, with values in 0, ... num_groups - 1
    scaling: vector, of length p, for gamma
    */

    /* cut x, y into blocks according to group */
    int *restrict count = (int *)malloc(num_groups * sizeof(int));
    for (int h = 0; h < num_groups; ++h)
        count[h] = 0;
    for (int i = 0; i < n; ++i)
        ++count[group[i]];

    training_data *restrict data = (training_data *)
        malloc(num_groups * sizeof(training_data));
    for (int h = 0; h < num_groups; ++h) {
        allocate_training_data(count[h], p,
            (double)(count[h]) / (double)n, data + h);
        count[h] = 0;
    }

    for (int i = 0; i < n; ++i) {
        const int h = group[i];
        const int i2 = count[h], n2 = data[h].n;
        for (int j = 0; j < p; ++j)
            data[h].x[i2 + n2 * j] = x[i + n * j];
        data[h].y[i2] = y[i];
        ++count[h];
    }

    for (int h = 0; h < num_groups; ++h) {
        transform_x(count[h], p, data[h].x, data[h].npdx);
        center_y(count[h], data[h].y, &(data[h].intercept));
    }

    free(count);

    /* initialize */
    training_input *restrict input = (training_input *)
        malloc(sizeof(training_input));

    input->p = p;
    input->data = data;
    input->num_groups = num_groups;

    input->scaling = (double *)malloc(p * sizeof(double));
    memcpy(input->scaling, scaling, p * sizeof(double));
    input->gamma = (double *)malloc(p * sizeof(double));
    input->lambda = 1.0;
    input->work = (double *)malloc((p + 1) * sizeof(double));

    return input;
}




void free_training_input(training_input *restrict input)
{
    for (int h = 0; h < input->num_groups; ++h)
        free_training_data(input->data + h);
    free(input->data);

    free(input->scaling);
    free(input->gamma);
    free(input->work);

    free(input);
}




double get_aggregate_loocv_objective(const double *param,
    int dim, void *extra)
{
    /*
    param contains scaled gamma and log lambda
    dim should be equal to p + 1
    */

    training_input *restrict input = (training_input *)extra;
    training_data *restrict data = input->data;

    /* maps param to gamma and lambda */
    for (int j = 0; j < input->p; ++j)
        input->gamma[j] = input->scaling[j] * param[j];
    input->lambda = exp(param[input->p]);

    double value = 0.0, temp = 0.0;
    for (int h = 0; h < input->num_groups; ++h) {
        compute_kernel(data[h].n, input->p,
            data[h].npdx, input->gamma, input->lambda,
            data[h].k, data[h].kpinv);
        estimate_alpha(data[h].n,
            data[h].kpinv, data[h].y, data[h].alpha);
        get_loocv_objective(data[h].n,
            data[h].kpinv, data[h].alpha, &temp);

        value += temp * data[h].percent;
    }

    return value;
}




void get_aggregate_loocv_gradient(const double *param,
    int dim, void *extra, double *value)
{
    /*
    param contains scaled gamma and log lambda
    dim should be equal to p + 1
    */

    training_input *restrict input = (training_input *)extra;
    training_data *restrict data = input->data;

    for (int j = 0; j < input->p + 1; ++j)
        value[j] = 0.0;

    for (int h = 0; h < input->num_groups; ++h) {
        /* no need to run compute_kernel and estimate_alpha
           since gradient(param) will only be called
           right after objective(param) */
        get_loocv_gradient(data[h].n, input->p,
            data[h].npdx, data[h].k, data[h].kpinv,
            data[h].alpha, data[h].y,
            data[h].work, input->work);

        for (int j = 0; j < input->p; ++j)
            value[j] += input->work[j] * input->scaling[j]
                * data[h].percent;
        value[input->p] += input->work[input->p] * input->lambda
            * data[h].percent;
    }
}




void train_model(training_input *restrict input)
{
    const int p = input->p;
    double *param = (double *)malloc((p + 1) * sizeof(double));

    /* initial values */

    int *restrict effect = (int *)malloc(p * sizeof(int));
    memset(effect, 0, p * sizeof(int));

    const double initial_param = 0.5;
    const double small_param = initial_param / 10.0 / p;
    for (int j = 0; j < p; ++j)
        param[j] = small_param;    /* for scaled gamma */
    param[p] = 0.0;                /* for log lambda */

    double value, old;
    double best_value = get_aggregate_loocv_objective(param, p + 1, input);
    int best_j;

    double unit = initial_param;
    const int num_rounds = 10;
    for (int round = 1; round <= num_rounds; ++round) {
        best_j = p;
        for (int j = 0; j < p; ++j) {
            old = param[j];
            param[j] = (effect[j] + 1) * unit;
            value = get_aggregate_loocv_objective(param, p + 1, input);
            if (value < best_value) {
                best_value = value;
                best_j = j;
            }
            param[j] = old;
        }

        /* currently unit == initial_param / round */
        if (best_j < p) {
            ++effect[best_j];
            if (round < num_rounds)
                unit = initial_param / (round + 1);
        } else {  /* best_j == p */
            unit = (round > 1 ? initial_param / (round - 1) : 0.0);
        }

        for (int j = 0; j < p; ++j)
            param[j] = (effect[j] ? effect[j] * unit : small_param);

        if (best_j == p) break;
    }

    /* optimization */

    double *lower = (double *)malloc((p + 1) * sizeof(double));
    double *upper = (double *)malloc((p + 1) * sizeof(double));
    for (int j = 0; j < p; ++j) {
        /* for scaled gamma */
        lower[j] = 0.0001 / p;
        upper[j] = 5.0;
    }
    /* for log lambda */
    lower[p] = -10.0;
    upper[p] = 10.0;

    projected_bfgs(param, p + 1, input, get_aggregate_loocv_objective,
        get_aggregate_loocv_gradient, 1000, 1e-8, lower, upper, 0.2,
        &value);
    /* compute again to ensure that gamma, lambda and value are correct */
    value = get_aggregate_loocv_objective(param, p + 1, input);

    free(param);
    free(effect);
    free(lower);
    free(upper);
}




void kernel_train(const int n, const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling,
    double *restrict node, double *restrict alpha,
    int *restrict offset, double *restrict gamma,
    double *restrict intercept)
{
    /*
    wrapper of train_model

    node: vector, of length (n * p)
    alpha: vector, n
    offset: vector, of length (num_groups + 1)
    intercept: vector, num_groups
    gamma: vector, p
    */

    training_input *input = allocate_training_input(n, p, num_groups,
        x, y, group, scaling);
    training_data *restrict data = input->data;

    train_model(input);

    /* save model */
    int cum_n = 0;
    for (int h = 0; h < num_groups; ++h) {
        memcpy(node + cum_n * p, data[h].x, data[h].n * p * sizeof(double));
        memcpy(alpha + cum_n, data[h].alpha, data[h].n * sizeof(double));
        offset[h] = cum_n;
        intercept[h] = data[h].intercept;
        cum_n += data[h].n;
    }
    offset[num_groups] = n;

    memcpy(gamma, input->gamma, p * sizeof(double));

    free_training_input(input);
}




void kernel_predict(const int p, const int num_groups, const int new_n,
    const double *restrict node, const double *restrict alpha,
    const int *restrict offset, const double *restrict gamma,
    const double *restrict intercept, const double *restrict new_x,
    double *restrict new_yhat)
{
    /*
    wrapper of predict_response
    */

    for (int h = 0; h < num_groups; ++h) {
        predict_response(offset[h + 1] - offset[h], p, new_n,
            node + offset[h] * p, alpha + offset[h], gamma, intercept[h],
            new_x, new_yhat + new_n * h);
    }
}



