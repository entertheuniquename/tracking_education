#include "bind_ekf.h"

namespace py = pybind11;

class BindEKFE
{
private:
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModelZ<Eigen::MatrixXd>,Models::MeasureModelZ<Eigen::MatrixXd>> ekfe;
public:

    BindEKFE(Eigen::MatrixXd in_state,
             Eigen::MatrixXd in_covariance,
             Eigen::MatrixXd in_process_noise,
             Eigen::MatrixXd in_measurement_noise):
        ekfe(in_state,
             in_covariance,
             in_process_noise,
             in_measurement_noise){}

    Eigen::MatrixXd predEKFE(double dt)
    {
        return ekfe.predict(dt,0).first;//#! 0- без нуля не получается, хотя параметр должен быть необязательным
    }

    Eigen::MatrixXd corrEKFE(const Eigen::MatrixXd &z)
    {

        return ekfe.correct(z,z).first;
    }
};

void bind_ekf(pybind11::module &m)
{
    py::class_<BindEKFE>(m, "BindEKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKFE::predEKFE)
        .def("correct",&BindEKFE::corrEKFE);
}
