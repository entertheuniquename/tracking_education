#include "bind_imm.h"

namespace py = pybind11;

class BindEKF7_CT
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models7::FCT<Eigen::MatrixXd>,
                   Models7::H<Eigen::MatrixXd>,
                   Models7::G<Eigen::MatrixXd>> ekf7_ct;
public:
    BindEKF7_CT(Eigen::MatrixXd in_state,
                Eigen::MatrixXd in_covariance,
                Eigen::MatrixXd in_process_noise,
                Eigen::MatrixXd in_measurement_noise):
        ekf7_ct(in_state,
                in_covariance,
                in_process_noise,
                in_measurement_noise){}

    Eigen::MatrixXd predict(double dt){return ekf7_ct.predict(dt,Models7::FCT_Jacobian<Eigen::MatrixXd>()).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf7_ct.correct(z,z).first;}
};

class BindEKF27_CT
{
private:
    Estimator::EKF2<Eigen::MatrixXd,
                    Models7::FCT<Eigen::MatrixXd>,
                    Models7::H<Eigen::MatrixXd>,
                    Models7::G<Eigen::MatrixXd>,
                    Models7::FCT_Jacobian<Eigen::MatrixXd>,
                    Models7::H_Jacobian<Eigen::MatrixXd>> ekf27_ct;
public:
    BindEKF27_CT(Eigen::MatrixXd in_state,
                 Eigen::MatrixXd in_covariance,
                 Eigen::MatrixXd in_process_noise,
                 Eigen::MatrixXd in_measurement_noise):
        ekf27_ct(in_state,
                 in_covariance,
                 in_process_noise,
                 in_measurement_noise){}

    Eigen::MatrixXd predict(double dt){return ekf27_ct.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf27_ct.correct(z).first;}
};

class BindKF7_CV
{
private:
    Estimator::KF<Eigen::MatrixXd,
                  Models7::FCV<Eigen::MatrixXd>,
                  Models7::H<Eigen::MatrixXd>,
                  Models7::G<Eigen::MatrixXd>> kf7_cv;
public:

    BindKF7_CV(Eigen::MatrixXd in_state,
               Eigen::MatrixXd in_covariance,
               Eigen::MatrixXd in_process_noise,
               Eigen::MatrixXd in_measurement_noise):
        kf7_cv(in_state,
               in_covariance,
               in_process_noise,
               in_measurement_noise){}

    Eigen::MatrixXd predict(double dt){return kf7_cv.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return kf7_cv.correct(z).first;}
};

class BindIMM7
{
private:
    Estimator::IMM<Eigen::MatrixXd,
                   Estimator::KF<Eigen::MatrixXd,
                                 Models7::FCV<Eigen::MatrixXd>,
                                 Models7::H<Eigen::MatrixXd>,
                                 Models7::G<Eigen::MatrixXd>>,
                   Estimator::EKF<Eigen::MatrixXd,
                                  Models7::FCT<Eigen::MatrixXd>,
                                  Models7::H<Eigen::MatrixXd>,
                                  Models7::G<Eigen::MatrixXd>>> imm7;
public:

    BindIMM7(Eigen::MatrixXd in_state,
             Eigen::MatrixXd in_covariance,
             Eigen::MatrixXd in_process_noise,
             Eigen::MatrixXd in_measurement_noise,
             Eigen::MatrixXd in_mu,
             Eigen::MatrixXd in_tp):
        imm7(in_state,
             in_covariance,
             in_process_noise,
             in_measurement_noise,
             in_mu,
             in_tp){}

    Eigen::MatrixXd predict(double dt){return imm7.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return imm7.correct(z).first;}
    Eigen::MatrixXd mu(){return imm7.getMU();}
};

void bind_imm(pybind11::module &m)
{
    py::class_<BindEKF7_CT>(m, "BindEKF7_CT")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF7_CT::predict)
        .def("correct",&BindEKF7_CT::correct);
    py::class_<BindEKF27_CT>(m, "BindEKF27_CT")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF27_CT::predict)
        .def("correct",&BindEKF27_CT::correct);
    py::class_<BindKF7_CV>(m, "BindKF7_CV")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKF7_CV::predict)
        .def("correct",&BindKF7_CV::correct);
    py::class_<BindIMM7>(m, "BindIMM7")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindIMM7::predict)
        .def("correct",&BindIMM7::correct)
        .def("mu",&BindIMM7::mu);
}
