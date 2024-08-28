#include "bind_ekf_eigen3.h"

namespace py = pybind11;

class BindEKFE
{
private:
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>> ekfe;
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

class BindEKFE_xyz_ct
{
private:
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> ekfe;
public:

    BindEKFE_xyz_ct(Eigen::MatrixXd in_state,
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

void bind_ekf_eigen3(pybind11::module &m)
{
    py::class_<BindEKFE>(m, "BindEKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKFE::predEKFE)
        .def("correct",&BindEKFE::corrEKFE);
    py::class_<BindEKFE_xyz_ct>(m, "BindEKFE_xyz_ct")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKFE_xyz_ct::predEKFE)
        .def("correct",&BindEKFE_xyz_ct::corrEKFE);
}
