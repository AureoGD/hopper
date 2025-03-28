#ifndef OPT_PROBLEM2_H
#define OPT_PROBLEM2_H

#include "rgc_controller/opt_problem.h"

class OptProblem2 : public OptProblem
{
public:
    OptProblem2(ModelMatrices *Robot);

    ~OptProblem2();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx;

    Eigen::Matrix<double, 2, 1> n1;
    Eigen::Matrix<double, 2, 1> t1;
};

#endif