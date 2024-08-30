#include "bind_kf.h"

namespace py = pybind11;

class BindKF_10_CV_XX
{
private:
    Estimator::KF<Eigen::MatrixXd,
                  Models10::FCV<Eigen::MatrixXd>,
                  Models10::H<Eigen::MatrixXd>,
                  Models10::G<Eigen::MatrixXd>> kf;
public:

    BindKF_10_CV_XX(Eigen::MatrixXd in_state,
                    Eigen::MatrixXd in_covariance,
                    Eigen::MatrixXd in_process_noise,
                    Eigen::MatrixXd in_measurement_noise):
        kf(in_state,
           in_covariance,
           in_process_noise,
           in_measurement_noise)
    {}

    Eigen::MatrixXd predict(double dt){return kf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return kf.correct(z).first;}
};

class BindKF_10_CA_XX
{
private:
    Estimator::KF<Eigen::MatrixXd,
                  Models10::FCA<Eigen::MatrixXd>,
                  Models10::H<Eigen::MatrixXd>,
                  Models10::G<Eigen::MatrixXd>> kf;
public:

    BindKF_10_CA_XX(Eigen::MatrixXd in_state,
                    Eigen::MatrixXd in_covariance,
                    Eigen::MatrixXd in_process_noise,
                    Eigen::MatrixXd in_measurement_noise):
        kf(in_state,
           in_covariance,
           in_process_noise,
           in_measurement_noise){}

    Eigen::MatrixXd predict(double dt){return kf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return kf.correct(z).first;}
};

void bind_kf(pybind11::module &m)
{
    py::class_<BindKF_10_CV_XX>(m, "BindKF_10_CV_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKF_10_CV_XX::predict)
        .def("correct",&BindKF_10_CV_XX::correct);
    py::class_<BindKF_10_CA_XX>(m, "BindKF_10_CA_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKF_10_CA_XX::predict)
        .def("correct",&BindKF_10_CA_XX::correct);
}
