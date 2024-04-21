#include "bind_kf.h"

namespace py = pybind11;

class BindKFE
{
private:
    KFE kfe;
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

    Eigen::MatrixXd predKFE(const Eigen::MatrixXd &a,const Eigen::MatrixXd &g,const Eigen::MatrixXd &m)
    {
        return kfe.predict(a,g,m).first;
    }

    Eigen::MatrixXd corrKFE(const Eigen::MatrixXd &m,const Eigen::MatrixXd &z,const Eigen::MatrixXd &r)
    {
        return kfe.correct(m,z,r).first;
    }
};

void bind_kf(pybind11::module &m)
{
    py::class_<BindKFE>(m, "BindKFE")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindKFE::predKFE)
        .def("correct",&BindKFE::corrKFE);
}
