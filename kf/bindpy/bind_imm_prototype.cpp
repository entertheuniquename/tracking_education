#include "bind_imm_prototype.h"

namespace py = pybind11;

class BindIMM_prot_10_KFCV_EKFCT_KFCA
{
private:
    Estimator::IMM_prototype<Eigen::MatrixXd,
    Estimator::KF<Eigen::MatrixXd,
                  Models10::FCV<Eigen::MatrixXd>,
                  Models10::H<Eigen::MatrixXd>,
                  Models10::G<Eigen::MatrixXd>>,
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCT<Eigen::MatrixXd>,
                   Models10::H<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCT_Jacobian<Eigen::MatrixXd>,
                   Models10::H_Jacobian<Eigen::MatrixXd>>,
    Estimator::KF<Eigen::MatrixXd,
                  Models10::FCA<Eigen::MatrixXd>,
                  Models10::H<Eigen::MatrixXd>,
                  Models10::G<Eigen::MatrixXd>>> imm;
public:

    BindIMM_prot_10_KFCV_EKFCT_KFCA(Eigen::MatrixXd in_state,
                               Eigen::MatrixXd in_covariance,
                               Eigen::MatrixXd in_process_noise,
                               Eigen::MatrixXd in_measurement_noise,
                               Eigen::MatrixXd in_mu,
                               Eigen::MatrixXd in_tp):
        imm(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise,
            in_mu,
            in_tp){}

    Eigen::MatrixXd predict(double dt){return imm.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return imm.correct(z).first;}
    Eigen::MatrixXd mu(){return imm.getMU();}
};

void bind_imm_prot(pybind11::module &m)
{
    py::class_<BindIMM_prot_10_KFCV_EKFCT_KFCA>(m, "BindIMM_prot_10_KFCV_EKFCT_KFCA")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindIMM_prot_10_KFCV_EKFCT_KFCA::predict)
        .def("correct",&BindIMM_prot_10_KFCV_EKFCT_KFCA::correct)
        .def("mu",&BindIMM_prot_10_KFCV_EKFCT_KFCA::mu);
}
