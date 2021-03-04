#include <R.h>
#include <Rinternals.h>
#include "dtr.h"




/*
    Summary
    RKHS regression: R_kernel_train, R_kernel_predict
    Regret transform: R_get_regrets_from_outcomes
    R <-> C conversion: R_save_rule_as_list, R_change_list_into_rule
    List-based rule: R_find_rule, R_apply_rule, R_cv_tune_rule

*/




SEXP R_kernel_train(SEXP R_x, SEXP R_y, SEXP R_group, SEXP R_num_groups,
    SEXP R_scaling)
{
    const int n = nrows(R_x), p = ncols(R_x);
    const int num_groups = INTEGER(R_num_groups)[0];

    SEXP R_list;
    PROTECT(R_list = allocVector(VECSXP, 5));

    SEXP R_node, R_alpha, R_offset, R_gamma, R_intercept;
    PROTECT(R_node = allocVector(REALSXP, n * p));
    PROTECT(R_alpha = allocVector(REALSXP, n));
    PROTECT(R_offset = allocVector(INTSXP, num_groups + 1));
    PROTECT(R_gamma = allocVector(REALSXP, p));
    PROTECT(R_intercept = allocVector(REALSXP, num_groups));
    SET_VECTOR_ELT(R_list, 0, R_node);
    SET_VECTOR_ELT(R_list, 1, R_alpha);
    SET_VECTOR_ELT(R_list, 2, R_offset);
    SET_VECTOR_ELT(R_list, 3, R_gamma);
    SET_VECTOR_ELT(R_list, 4, R_intercept);

    kernel_train(n, p, num_groups,
        REAL(R_x), REAL(R_y),
        INTEGER(R_group), REAL(R_scaling),
        REAL(R_node), REAL(R_alpha), INTEGER(R_offset),
        REAL(R_gamma), REAL(R_intercept));

    SEXP R_names;
    PROTECT(R_names = allocVector(STRSXP, 5));
    SET_STRING_ELT(R_names, 0, mkChar("node"));
    SET_STRING_ELT(R_names, 1, mkChar("alpha"));
    SET_STRING_ELT(R_names, 2, mkChar("offset"));
    SET_STRING_ELT(R_names, 3, mkChar("gamma"));
    SET_STRING_ELT(R_names, 4, mkChar("intercept"));
    namesgets(R_list, R_names);

    UNPROTECT(7);  /* 1 + 5 + 1 */
    return R_list;
}




SEXP R_kernel_predict(SEXP R_list, SEXP R_new_x)
{
    SEXP R_node, R_alpha, R_offset, R_gamma, R_intercept;

    R_node      = VECTOR_ELT(R_list, 0);
    R_alpha     = VECTOR_ELT(R_list, 1);
    R_offset    = VECTOR_ELT(R_list, 2);
    R_gamma     = VECTOR_ELT(R_list, 3);
    R_intercept = VECTOR_ELT(R_list, 4);

    const int p = length(R_gamma);
    const int num_groups = length(R_intercept);
    const int new_n = nrows(R_new_x);

    SEXP R_new_yhat;
    PROTECT(R_new_yhat = allocMatrix(REALSXP, new_n, num_groups));

    kernel_predict(p, num_groups, new_n,
        REAL(R_node), REAL(R_alpha), INTEGER(R_offset),
        REAL(R_gamma), REAL(R_intercept),
        REAL(R_new_x), REAL(R_new_yhat));

    UNPROTECT(1);
    return R_new_yhat;
}




SEXP R_get_regrets_from_outcomes(SEXP R_outcomes)
{
    const int n = nrows(R_outcomes);
    const int num_groups = ncols(R_outcomes);

    SEXP R_regrets;
    PROTECT(R_regrets = allocMatrix(REALSXP, n, num_groups));

    get_regrets_from_outcomes(REAL(R_outcomes), n, num_groups,
        REAL(R_regrets));

    UNPROTECT(1);
    return R_regrets;
}




static SEXP R_save_rule_as_list(const statement *restrict rule,
    const unsigned int rule_length)
{
    SEXP R_list;
    PROTECT(R_list = allocVector(VECSXP, 6));

    SEXP R_a, R_type, R_j1, R_j2, R_c1, R_c2;
    PROTECT(R_a = allocVector(INTSXP, rule_length));
    PROTECT(R_type = allocVector(STRSXP, rule_length));
    PROTECT(R_j1 = allocVector(INTSXP, rule_length));
    PROTECT(R_j2 = allocVector(INTSXP, rule_length));
    PROTECT(R_c1 = allocVector(REALSXP, rule_length));
    PROTECT(R_c2 = allocVector(REALSXP, rule_length));
    SET_VECTOR_ELT(R_list, 0, R_a);
    SET_VECTOR_ELT(R_list, 1, R_type);
    SET_VECTOR_ELT(R_list, 2, R_j1);
    SET_VECTOR_ELT(R_list, 3, R_j2);
    SET_VECTOR_ELT(R_list, 4, R_c1);
    SET_VECTOR_ELT(R_list, 5, R_c2);

    for (unsigned int i = 0; i < rule_length; ++i) {
        INTEGER(R_a)[i] = rule[i].a;
        switch (rule[i].type) {
        case CONDITION_TYPE_LL:
            SET_STRING_ELT(R_type, i, mkChar("LL"));
            break;
        case CONDITION_TYPE_LR:
            SET_STRING_ELT(R_type, i, mkChar("LR"));
            break;
        case CONDITION_TYPE_RL:
            SET_STRING_ELT(R_type, i, mkChar("RL"));
            break;
        case CONDITION_TYPE_RR:
            SET_STRING_ELT(R_type, i, mkChar("RR"));
            break;
        default:
            SET_STRING_ELT(R_type, i, mkChar("--"));
        }
        INTEGER(R_j1)[i] = rule[i].j1 + 1;
        INTEGER(R_j2)[i] = rule[i].j2 + 1;
        REAL(R_c1)[i] = rule[i].c1;
        REAL(R_c2)[i] = rule[i].c2;
    }

    SEXP R_names, R_rows, R_class;
    PROTECT(R_names = allocVector(STRSXP, 6));
    SET_STRING_ELT(R_names, 0, mkChar("a"));
    SET_STRING_ELT(R_names, 1, mkChar("type"));
    SET_STRING_ELT(R_names, 2, mkChar("j1"));
    SET_STRING_ELT(R_names, 3, mkChar("j2"));
    SET_STRING_ELT(R_names, 4, mkChar("c1"));
    SET_STRING_ELT(R_names, 5, mkChar("c2"));
    namesgets(R_list, R_names);

    char buffer[255];
    PROTECT(R_rows = allocVector(STRSXP, rule_length));
    for (unsigned int i = 0; i < rule_length; ++i) {
        snprintf(buffer, 255, "%d", i + 1);
        SET_STRING_ELT(R_rows, i, mkChar(buffer));
    }
    setAttrib(R_list, R_RowNamesSymbol, R_rows);

    PROTECT(R_class = allocVector(STRSXP, 1));
    SET_STRING_ELT(R_class, 0, mkChar("data.frame"));
    setAttrib(R_list, R_ClassSymbol, R_class);

    UNPROTECT(10);  /* 1 + 6 + 3 */
    return R_list;
}




static statement *R_change_list_into_rule(SEXP R_list, unsigned int *rule_length)
{
    SEXP R_a, R_type, R_j1, R_j2, R_c1, R_c2;
    R_a    = VECTOR_ELT(R_list, 0);
    R_type = VECTOR_ELT(R_list, 1);
    R_j1   = VECTOR_ELT(R_list, 2);
    R_j2   = VECTOR_ELT(R_list, 3);
    R_c1   = VECTOR_ELT(R_list, 4);
    R_c2   = VECTOR_ELT(R_list, 5);

    statement *rule = (statement *)malloc(length(R_a) * sizeof(statement));
    for (unsigned int i = 0; i < length(R_a); ++i) {
        rule[i].a = INTEGER(R_a)[i];
        const char *type = CHAR(STRING_ELT(R_type, i));
        if (type[0] == 'L') {
            rule[i].type = (type[1] == 'L' ? CONDITION_TYPE_LL :
                CONDITION_TYPE_LR);
        } else {
            rule[i].type = (type[1] == 'L' ? CONDITION_TYPE_RL :
                CONDITION_TYPE_RR);
        }
        rule[i].j1 = INTEGER(R_j1)[i] - 1;
        rule[i].j2 = INTEGER(R_j2)[i] - 1;
        rule[i].c1 = REAL(R_c1)[i];
        rule[i].c2 = REAL(R_c2)[i];
    }

    *rule_length = length(R_a);
    return rule;
}




SEXP R_find_rule(SEXP R_z, SEXP R_regrets, SEXP R_zeta, SEXP R_eta,
    SEXP R_max_length, SEXP R_action)
{
    const unsigned int max_length = INTEGER(R_max_length)[0];
    unsigned int rule_length = 0;
    statement *rule = (statement *)malloc(max_length * sizeof(statement));

    find_rule(REAL(R_z), REAL(R_regrets),
        nrows(R_z), ncols(R_z), ncols(R_regrets),
        REAL(R_zeta)[0], REAL(R_eta)[0], max_length,
        rule, &rule_length, INTEGER(R_action));
    SEXP R_list = R_save_rule_as_list(rule, rule_length);

    free(rule);
    return R_list;
}




SEXP R_apply_rule(SEXP R_list, SEXP R_z)
{
    const unsigned int n = nrows(R_z);
    SEXP R_action;
    PROTECT(R_action = allocVector(INTSXP, n));

    unsigned int rule_length = 0;
    statement *rule = R_change_list_into_rule(R_list, &rule_length);

    apply_rule(REAL(R_z), n, rule, rule_length, INTEGER(R_action));

    free(rule);

    UNPROTECT(1);
    return R_action;
}




SEXP R_cv_tune_rule(SEXP R_z, SEXP R_regrets,
    SEXP R_zeta_choices, SEXP R_eta_choices,
    SEXP R_max_length,
    SEXP R_fold, SEXP R_num_folds)
{
    const unsigned int num_choices = length(R_zeta_choices);

    SEXP R_cv_regret;
    PROTECT(R_cv_regret = allocVector(REALSXP, num_choices));

    cv_tune_rule(REAL(R_z), REAL(R_regrets),
        nrows(R_z), ncols(R_z), ncols(R_regrets),
        REAL(R_zeta_choices), REAL(R_eta_choices), num_choices,
        INTEGER(R_max_length)[0],
        INTEGER(R_fold), INTEGER(R_num_folds)[0],
        REAL(R_cv_regret));

    UNPROTECT(1);
    return R_cv_regret;
}




static const R_CallMethodDef callMethods[]  = {
    {"R_kernel_train", (DL_FUNC) & R_kernel_train, 5},
    {"R_kernel_predict", (DL_FUNC) & R_kernel_predict, 2},
    {"R_get_regrets_from_outcomes", (DL_FUNC) & R_get_regrets_from_outcomes, 1},
    {"R_find_rule", (DL_FUNC) & R_find_rule, 6},
    {"R_apply_rule", (DL_FUNC) & R_apply_rule, 2},
    {"R_cv_tune_rule", (DL_FUNC) & R_cv_tune_rule, 7},
    {NULL, NULL, 0}
};

void R_init_listdtr(DllInfo *dll)
{
   R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
   R_useDynamicSymbols(dll, FALSE);
   R_forceSymbols(dll, TRUE);
}
