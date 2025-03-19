#ifndef OPT_PROBLEM9_H
#define OPT_PROBLEM9_H

#include "rgc_controller/opt_problem.h"

class OptProblem9 : public OptProblem
{
public:
    OptProblem9(ModelMatrices *Robot);

    ~OptProblem9();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx, sum_f, sum_m;

    Eigen::Matrix<double, 3, 1> n1;
    Eigen::Matrix<double, 3, 1> t1;
};

#endif