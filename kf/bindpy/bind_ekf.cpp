#include "bind_ekf.h"
#include "../source/models.h"

namespace py = pybind11;

class BindEKFE
{
private:
    Estimator::EKFE ekfe;
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
        return ekfe.predict(Models::stateModel_3Ax<Eigen::MatrixXd>, dt).first;
    }

    Eigen::MatrixXd corrEKFE(const Eigen::MatrixXd &z)
    {

        return ekfe.correct(z, Models::measureModel_3Bx<Eigen::MatrixXd>,z).first;
    }
};

void bind_ekf(pybind11::module &m)
{
    py::class_<BindEKFE>(m, "BindEKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKFE::predEKFE)
        .def("correct",&BindEKFE::corrEKFE);
}
