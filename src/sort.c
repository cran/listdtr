#include <stdlib.h>
#include <string.h>




#define SMALL_COUNT 4




void swap_sort(double *restrict vdat,
    unsigned int *restrict idat,
    const unsigned int n)
{
    #define SWAP(j, k) \
        if (vdat[j] > vdat[k]) { \
            double vtemp = vdat[j]; \
            vdat[j] = vdat[k]; vdat[k] = vtemp; \
            int itemp = idat[j]; \
            idat[j] = idat[k]; idat[k] = itemp; \
        }

    switch (n) {
    case 1:
        break;

    case 2:
        SWAP(0, 1);
        break;

    case 3:
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(0, 1);
        break;

    case 4:
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(2, 3);
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(0, 1);
        break;
    }

    #undef SWAP
}




void combine_arrays(double *restrict vsrc, double *restrict vdst,
    unsigned int *restrict isrc, unsigned int *restrict idst,
    const unsigned int mid, const unsigned int cnt)
{
    unsigned int jl = 0, jr = mid, k = 0;
    for (; k < cnt; ++k) {
        if ((jl < mid) && ((jr >= cnt) || (vsrc[jl] <= vsrc[jr]))) {
            vdst[k] = vsrc[jl];
            idst[k] = isrc[jl];
            ++jl;
        } else {
            vdst[k] = vsrc[jr];
            idst[k] = isrc[jr];
            ++jr;
        }
    }

}




void do_merge_sort(double *restrict vdat, unsigned int *restrict idat,
    double *restrict vbuf, unsigned int *restrict ibuf,
    const unsigned int n)
{
    const unsigned int off2 = n / 2;
    const unsigned int rest = n - off2;
    const unsigned int off1 = off2 / 2;
    const unsigned int off3 = off2 + rest / 2;
    const unsigned int len1 = off2 - off1;
    const unsigned int len2 = off3 - off2;
    const unsigned int len3 = n - off3;

    #define SORT(count, value, index) \
        if (count > SMALL_COUNT) \
            do_merge_sort(value, index, vbuf, ibuf, count); \
        else \
            swap_sort(value, index, count);

    /* sorts four subarrays */
    SORT(off1, vdat,        idat       );
    SORT(len1, vdat + off1, idat + off1);
    SORT(len2, vdat + off2, idat + off2);
    SORT(len3, vdat + off3, idat + off3);

    #undef SORT

    /* checks if array is already ordered */
    if ((vdat[off1 - 1] <= vdat[off1]) &&
        (vdat[off2 - 1] <= vdat[off2]) &&
        (vdat[off3 - 1] <= vdat[off3]))
        return;

    #define COPY(vsrc, vdst, isrc, idst, cnt) \
        memcpy(vdst, vsrc, (cnt) * sizeof(double)); \
        memcpy(idst, isrc, (cnt) * sizeof(unsigned int));

    /* merges array[0..off1) and array[off1..off2) */
    if (vdat[0] > vdat[off2 - 1]) {
        COPY(vdat + off1, vbuf, idat + off1, ibuf, len1);
        COPY(vdat, vbuf + len1, idat, ibuf + len1, off1);
    } else if (vdat[off1 - 1] <= vdat[off1]) {
        COPY(vdat, vbuf, idat, ibuf, off2);
    } else {
        combine_arrays(vdat, vbuf, idat, ibuf, off1, off2);
    }

    /* merges array[off2..off3) and array[off3..n) */
    if (vdat[off2] > vdat[n - 1]) {
        COPY(vdat + off3, vbuf + off2,
            idat + off3, ibuf + off2, len3);
        COPY(vdat + off2, vbuf + off2 + len3,
            idat + off2, ibuf + off2 + len3, len2);
    } else if (vdat[off3 - 1] <= vdat[off3]) {
        COPY(vdat + off2, vbuf + off2,
            idat + off2, ibuf + off2, rest);
    } else {
        combine_arrays(vdat + off2, vbuf + off2,
            idat + off2, ibuf + off2, len2, rest);
    }

    /* merges array[0..off2) and array[off2..n) */
    if (vbuf[0] > vbuf[n - 1]) {
        COPY(vbuf + off2, vdat, ibuf + off2, idat, rest);
        COPY(vbuf, vdat + rest, ibuf, idat + rest, off2);
    } else {
        combine_arrays(vbuf, vdat, ibuf, idat, off2, n);
    }

    #undef COPY
}




void *allocate_space_for_merge_sort(const unsigned int n)
{
    const unsigned int size = n * (sizeof(double) + sizeof(unsigned int));
    return malloc(size);
}




void free_space(void *work)
{
    free(work);
}




void merge_sort(double *restrict vdat,
    unsigned int *restrict idat, unsigned int *restrict rdat,
    const unsigned int n, void *restrict work)
{
    /* ordering indices */
    for (unsigned int j = 0; j < n; ++j)
        idat[j] = j;

    if (n > SMALL_COUNT) {
        do_merge_sort(vdat, idat,
            (double *)work, (unsigned int *)((double *)work + n), n);
    } else {
        swap_sort(vdat, idat, n);
    }

    /* ranks (zero-based) */
    #define DELTA 1e-8

    rdat[idat[0]] = 0;
    double last_value = vdat[0];
    unsigned int last_rank = 0;
    for (unsigned int j = 1; j < n; ++j) {
        if (vdat[j] - last_value < DELTA) {
            rdat[idat[j]] = last_rank;
        } else {
            rdat[idat[j]] = j;
            last_value = vdat[j];
            last_rank = j;
        }
    }

    #undef DELTA
}




#undef SMALL_COUNT




/*
    Sort matrix z column by column.
    Obtain ordering indices iz and ranks rz.
    The dimension of z is n by p.
*/

void sort_matrix(const double *restrict z,
    unsigned int *restrict iz,
    unsigned int *restrict rz,
    double *restrict sz,
    const unsigned int n, const unsigned int p)
{
    memcpy(sz, z, n * p * sizeof(double));
    void *work = allocate_space_for_merge_sort(n);

    for (unsigned int j = 0; j < p; ++j) {
        unsigned int offset = n * j;
        merge_sort(sz + offset, iz + offset, rz + offset, n, work);
    }

    free_space(work);
}




void reverse_sort(const unsigned int *restrict iz,
    const unsigned int *restrict rz,
    const double *restrict sz,
    unsigned int *restrict rev_iz,
    unsigned int *restrict rev_rz,
    double *restrict rev_sz,
    const unsigned int n, const unsigned int p)
{
    for (unsigned int j = 0; j < p; ++j) {
        unsigned int offset = n * j;
        for (unsigned int i = 0; i < n; ++i) {
            rev_iz[offset + i] = iz[offset + n - 1 - i];
            rev_rz[offset + i] = n - 1 - rz[offset + i];
            rev_sz[offset + i] = sz[offset + n - 1 - i];
        }
    }
}



