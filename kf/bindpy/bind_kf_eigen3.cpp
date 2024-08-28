#include "bind_kf_eigen3.h"

namespace py = pybind11;

class BindKFE
{
private:
    Estimator::KFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>> kfe;
public:

    BindKFE(Eigen::MatrixXd in_state,
            Eigen::MatrixXd in_covariance,
            Eigen::MatrixXd in_transition_state_model,
            Eigen::MatrixXd in_process_noise,
            Eigen::MatrixXd in_transition_process_noise_model,
            Eigen::MatrixXd in_transition_measurement_model,
            Eigen::MatrixXd in_measurement_noise):
        kfe(in_state,
            in_covariance,
            in_transition_state_model,
            in_process_noise,
            in_transition_process_noise_model,
            in_transition_measurement_model,
            in_measurement_noise){}

    Eigen::MatrixXd predKFE(double dt)
    {
        return kfe.predict(dt).first;
    }

    Eigen::MatrixXd corrKFE(const Eigen::MatrixXd &z)
    {
        return kfe.correct(z).first;
    }
};

void bind_kf(pybind11::module &m)
{
    py::class_<BindKFE>(m, "BindKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKFE::predKFE)
        .def("correct",&BindKFE::corrKFE);
}
