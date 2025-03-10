#include "rgc_controller/op_wrapper.h"

Op_Wrapper::Op_Wrapper()
{

    this->solver.settings()->setVerbosity(false);

    this->_JumpRobot = new ModelMatrices();

    this->optP1 = new OptProblem1(_JumpRobot);
    this->op[0] = this->optP1;

    this->optP2 = new OptProblem2(_JumpRobot);
    this->op[1] = this->optP2;

    this->optP3 = new OptProblem3(_JumpRobot);
    this->op[2] = this->optP3;

    this->optP4 = new OptProblem4(_JumpRobot);
    this->op[3] = this->optP4;

    this->optP5 = new OptProblem5(_JumpRobot);
    this->op[4] = this->optP5;

    this->optP6 = new OptProblem6(_JumpRobot);
    this->op[5] = this->optP6;

    this->optP7 = new OptProblem7(_JumpRobot);
    this->op[6] = this->optP7;

    this->optP8 = new OptProblem8(_JumpRobot);
    this->op[7] = this->optP8;

    this->qhl.resize(3, 1);
}

Op_Wrapper::~Op_Wrapper()
{
}

void Op_Wrapper::RGCConfig(double _ts, double _Kp, double _Kd)
{

    // TODO - create a function that reads some loader file (YALM file)
    Eigen::MatrixXd Q, R, ref, Ub, Lb;

    Q.resize(3, 3);
    Q << 0.0025, 0, 0,
        0, 0.0025, 0,
        0, 0, 0.0025;

    R.resize(3, 3);
    R << 1.5, 0, 0,
        0, 1.5, 0,
        0, 0, 1.5;

    Ub.resize(10, 1);
    Ub << _JumpRobot->qU, 0, 0, OsqpEigen::INFTY, -this->g * 2.5 * _JumpRobot->m, 0, OsqpEigen::INFTY, -this->g * 2.5 * _JumpRobot->m;
    Lb.resize(10, 1);
    Lb << _JumpRobot->qL, 0, -OsqpEigen::INFTY, 0, -this->g * 0.5 * _JumpRobot->m, -OsqpEigen::INFTY, 0, -this->g * 0.5 * _JumpRobot->m;

    this->op[0]->SetInternalVariables();
    this->op[0]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[0]->UpdateModelConstants();

    this->ConfPO(0);
    this->op[0]->SetWeightMatrices(Q, R);
    this->op[0]->UpdateReferences();
    this->op[0]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[1]->SetInternalVariables();
    this->op[1]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[1]->UpdateModelConstants();

    this->ConfPO(1);

    this->op[1]->SetWeightMatrices(Q, R);
    this->op[1]->UpdateReferences();
    this->op[1]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[2]->SetInternalVariables();
    this->op[2]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[2]->UpdateModelConstants();

    this->ConfPO(2);

    this->op[2]->SetWeightMatrices(Q, R);
    this->op[2]->UpdateReferences();
    this->op[2]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[3]->SetInternalVariables();
    this->op[3]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[3]->UpdateModelConstants();

    this->ConfPO(3);

    Ub.resize(9, 1);
    Ub << _JumpRobot->qU, 0, OsqpEigen::INFTY, -this->g * 2.5 * _JumpRobot->m, 0, OsqpEigen::INFTY, -this->g * 2.5 * _JumpRobot->m;
    Lb.resize(9, 1);
    Lb << _JumpRobot->qL, -OsqpEigen::INFTY, 0, -this->g * 0.5 * _JumpRobot->m, -OsqpEigen::INFTY, 0, -this->g * 0.5 * _JumpRobot->m;

    this->op[3]->SetWeightMatrices(Q, R);
    this->op[3]->UpdateReferences();
    this->op[3]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[4]->SetInternalVariables();
    this->op[4]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[4]->UpdateModelConstants();

    this->ConfPO(4);

    this->op[4]->SetWeightMatrices(Q, R);
    this->op[4]->UpdateReferences();
    this->op[4]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[7]->SetInternalVariables();
    this->op[7]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[7]->UpdateModelConstants();

    this->ConfPO(7);

    this->op[7]->SetWeightMatrices(Q, R);
    this->op[7]->UpdateReferences();
    this->op[7]->SetConsBounds(Lb, Ub);

    // Flight phase POs

    Ub.resize(3, 1);
    Ub << _JumpRobot->qU;
    Lb.resize(3, 1);
    Lb << _JumpRobot->qL;

    this->ClearPO();
    this->ClearData();

    this->op[5]->SetInternalVariables();
    this->op[5]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[5]->UpdateModelConstants();

    this->ConfPO(5);

    this->op[5]->SetWeightMatrices(Q, R);
    this->op[5]->UpdateReferences();
    this->op[5]->SetConsBounds(Lb, Ub);

    this->ClearPO();
    this->ClearData();

    this->op[6]->SetInternalVariables();
    this->op[6]->SetConstants(_ts, 15, 10, _Kp, _Kd);
    this->op[6]->UpdateModelConstants();

    this->ConfPO(6);

    this->op[6]->SetWeightMatrices(Q, R);
    this->op[6]->UpdateReferences();
    this->op[6]->SetConsBounds(Lb, Ub);

    this->first_conf = 1;
}

void Op_Wrapper::UpdateSt(Eigen::Matrix<double, 3, 1> *_q,
                          Eigen::Matrix<double, 3, 1> *_qd,
                          Eigen::Matrix<double, 3, 1> *_qr,
                          Eigen::Matrix<double, 2, 1> *_dr,
                          Eigen::Matrix<double, 2, 1> *_r,
                          Eigen::Matrix<double, 2, 1> *_db,
                          Eigen::Matrix<double, 2, 1> *_b,
                          double _dth,
                          double _th)
{
    q = (*_q);
    qd = (*_qd);
    qr = (*_qr);
    r_vel = (*_dr);
    r_pos = (*_r);
    dth = _dth;
    th = _th;
    _JumpRobot->UpdateRobotStates(*_q, *_qd, th, *_b, *_db);
}

int Op_Wrapper::ChooseRGCPO(int npo)
{
    if (npo != this->last_op)
    {
        if (this->solver.isInitialized())
            this->ClearPO();

        if (this->solver.data()->isSet())
            this->ClearData();

        this->ConfPO(npo);
        this->last_op = npo;
        error_flag = false;
    }

    if (this->solver.isInitialized())
    {

        if (npo == 0 || npo == 1 || npo == 2 || npo == 3 || npo == 4 || npo == 7)
        {
            // update the states vector |dr, dth, q, g, qa|
            this->x << r_vel, dth, q, r_pos, th, g, qr;
        }

        if (npo == 5 || npo == 6)
        {
            // update the states vector |dq q dth th r qra|
            this->x << this->qd, this->q, dth, th, r_pos, this->qr;
        }

        this->qhl = this->op[npo]->qhl;
        this->op[npo]->UpdateStates(this->x);
        this->op[npo]->UpdateOptimizationProblem(this->H, this->F, this->Ain, this->Lb, this->Ub);
        int solve_status = this->SolvePO();
        if (solve_status == 1)
        {
            if (this->debug)
                std::cout << "solved" << std::endl;
            return 1;
        }
        else if (solve_status == 0)
        {
            if (this->debug)
                std::cout << "not solved" << std::endl;
            return 0;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        if (this->debug)
            std::cout << "RGC conf error" << std::endl;
        return -1;
    }
}

void Op_Wrapper::ResetPO()
{
    if (this->solver.isInitialized())
        this->ClearPO();
    if (this->solver.data()->isSet())
        this->ClearData();
    this->last_op = -1;
}

void Op_Wrapper::ClearPO()
{
    this->solver.clearSolverVariables();
    this->solver.clearSolver();
}

void Op_Wrapper::ClearData()
{
    this->solver.data()->clearLinearConstraintsMatrix();
    this->solver.data()->clearHessianMatrix();
}

void Op_Wrapper::ConfPO(int index)
{
    // first, resize the matrices

    this->x.resize(this->op[index]->nxa);
    this->x.setZero();

    this->H.resize(this->op[index]->nu * this->op[index]->M, this->op[index]->nu * this->op[index]->M);
    this->H.setZero();

    this->F.resize(1, this->op[index]->nu * this->op[index]->M);
    this->F.setZero();

    // std::cout << "Numer of constraints: " << this->op[index]->nc << std::endl;

    this->Ain.resize(this->op[index]->nc * this->op[index]->N, this->op[index]->nu * this->op[index]->M);
    this->Ain.setZero();

    this->Lb.resize(this->op[index]->nc * this->op[index]->N);
    this->Lb.setZero();

    this->Ub.resize(this->op[index]->nc * this->op[index]->N);
    this->Ub.setZero();

    // then, configure the solver

    this->solver.settings()->setVerbosity(0);

    this->solver.data()->setNumberOfVariables(this->op[index]->nu * this->op[index]->M);

    this->hessian_sparse = this->H.sparseView();
    this->solver.data()->clearHessianMatrix();
    this->solver.data()->setHessianMatrix(this->hessian_sparse);

    this->solver.data()->setGradient(F.transpose());

    this->solver.data()->setNumberOfConstraints(this->op[index]->nc * this->op[index]->N);
    this->linearMatrix = this->Ain.sparseView();
    this->solver.data()->setLinearConstraintsMatrix(this->linearMatrix);
    this->solver.data()->setLowerBound(this->Lb);
    this->solver.data()->setUpperBound(this->Ub);

    if (this->op[index]->nc != 0)
        this->constraints = 1;

    if (!this->first_conf)
    {
        if (!this->solver.initSolver())
            std::cout << "***************** PO " << index << " Inicialization Problem ***************** " << std::endl;
        else
            std::cout << "***************** PO " << index << " OK ***************** " << std::endl;
    }
    else
    {
        if (!this->solver.initSolver())
        {
            std::cout << "Error: " << index << std::endl;
        }
    }
}

int Op_Wrapper::SolvePO()
{

    this->hessian_sparse = this->H.sparseView();
    if (!this->solver.updateHessianMatrix(this->hessian_sparse))
        return -1;

    this->solver.updateGradient(this->F.transpose());

    if (this->constraints != 0)
    {
        this->linearMatrix = this->Ain.sparseView();
        this->solver.updateLinearConstraintsMatrix(this->linearMatrix);
        this->solver.updateBounds(this->Lb, this->Ub);
    }

    if (this->solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError)
    {
        if (this->solver.getStatus() != OsqpEigen::Status::Solved)
        {
            return 0;
        }

        this->QPSolution = this->solver.getSolution();
        this->delta_qr = this->QPSolution.block(0, 0, 3, 1);
        return 1;
    }
    else
    {
        if (this->debug)
            std::cout << "Not solved - error" << std::endl;
        return 0;
    }
}