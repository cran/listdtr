#include <stdlib.h>
#include <string.h>
#include <math.h>




typedef struct {
    unsigned int count; /* numer of observations within subtree */
    double total;       /* total sum within subtree */
    unsigned char cut;  /* 0 for left, 1 for right */
    double best;        /* best sum within subtree */
} node;




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
    double *restrict opt_loss)
{
    /*
    find a, b to minimize
        \sum_{i: x_i <= a, y_i <= b} (loss_i - zeta)
        - eta I(a = Inf) - eta I(b = Inf).
    return the optimal a, b as opt_cx and opt_cy
    and return the minimum in opt_loss.

    sx, ix, rx, sy, iy, ry, loss: vector, of length n
    next: vector, of length (n + 1)
    opt_cx, opt_cy, opt_loss: scalar
    */

    #define DELTA 1e-8

    if ((sx[n - 1] - sx[0] < DELTA) && (sy[n - 1] - sy[0] < DELTA)) {
        /* both x and y consist of constants */
        *opt_cx = sx[0]; *opt_cy = sy[0];
        *opt_loss = INFINITY;
        return;
    }

    #undef DELTA

    unsigned int tree_len = 1, k = 1, succ_k;
    while (k * 2 < n) {
        k *= 2;
        tree_len += k;
    }
    unsigned int array_len = k * 2;

    unsigned int entire_size = (tree_len + array_len) * sizeof(node);
    node *restrict tree = (node *)malloc(entire_size);

    unsigned int i, p, lchild, rchild;
    double left_best, right_best;
    unsigned int best_rx = 0, best_ry = 0;
    double optimum = INFINITY;

    /* set all fields to 0 */
    memset(tree, 0, entire_size);

    /* set the "penalty" for the infinite cutoff of x */
    p = tree_len + n - 1;
    i = 1;
    while (1) {
        tree[p].total = tree[p].best = -eta;
        tree[p].cut = i;
        if (p == 0) break;
        i = (p - 1) % 2;
        p = (p - 1) / 2;
    }

    k = 0;
    while (k < n && next[iy[k]] > iy[k])
        ++k;
    while (k < n) {
        succ_k = k + 1;
        while (succ_k < n && next[iy[succ_k]] > iy[succ_k])
            ++succ_k;

        i = iy[k];    /* definition of k ensures next[i] == i */

        #define EPSILON 1e-10
        #define UPDATE_TREE \
            do { \
                p = (p - 1) / 2; \
                lchild = 2 * p + 1; \
                rchild = lchild + 1; \
                \
                tree[p].count = tree[lchild].count + tree[rchild].count; \
                tree[p].total = tree[lchild].total + tree[rchild].total; \
                left_best = tree[lchild].best; \
                right_best = tree[lchild].total + tree[rchild].best; \
                if (tree[lchild].count == 0 \
                    || right_best < left_best - EPSILON) { \
                    tree[p].best = right_best; \
                    tree[p].cut = 1; \
                } else { \
                    tree[p].best = left_best; \
                    tree[p].cut = 0; \
                } \
            } while (p);

        p = tree_len + rx[i];  /* tied values in x have the same rx */
        tree[p].count += 1;
        tree[p].total += loss[i] - zeta;
        tree[p].best = tree[p].total;
        tree[p].cut = 1;       /* always inclusive at leaf level */
        UPDATE_TREE;           /* update the binary tree */

        if (succ_k >= n) {
            /* set the "penalty" for the infinite cutoff of y */
            p = tree_len;
            while (p < tree_len + n - 1 && tree[p].count == 0)
                ++p;           /* find the first observation */
            tree[p].total -= eta;
            tree[p].best = tree[p].total;
            UPDATE_TREE;
        }

        #undef EPSILON
        #undef UPDATE_TREE

        /* ties in y; note that i == iy[k] */
        if ((succ_k >= n) || (ry[iy[succ_k]] != ry[i])) {
            if (tree[0].best < optimum) {
                optimum = tree[0].best;

                best_rx = 0;
                while (best_rx < tree_len)
                    best_rx = 2 * best_rx + 1 + tree[best_rx].cut;
                best_rx -= tree_len;

                best_ry = k;
            }
        }

        k = succ_k;
    }

    /* just in case something goes wrong */
    if (best_rx >= n) best_rx = n - 1;
    if (best_ry >= n) best_ry = n - 1;

    /* adjust best_rx and best_ry for ties and get cutoff values */
    k = best_rx;
    succ_k = k + 1;
    while (succ_k < n &&
        (rx[ix[succ_k]] == best_rx || next[ix[succ_k]] > ix[succ_k]))
        ++succ_k;

    if (succ_k < n) {
        *opt_cx = 0.5 * (sx[k] + sx[succ_k]);
    } else {
        *opt_cx = (sx[0] <= sx[n - 1] ? INFINITY : -INFINITY);
    }

    k = best_ry;
    succ_k = k + 1;
    while (succ_k < n &&
        (ry[iy[succ_k]] == best_ry || next[iy[succ_k]] > iy[succ_k]))
        ++succ_k;

    if (succ_k < n) {
        *opt_cy = 0.5 * (sy[k] + sy[succ_k]);
    } else {
        *opt_cy = (sy[0] <= sy[n - 1] ? INFINITY : -INFINITY);
    }

    *opt_loss = optimum;

    free(tree);
}



