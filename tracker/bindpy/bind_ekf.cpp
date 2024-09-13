#include "bind_ekf.h"

namespace py = pybind11;

class BindEKF_10_CV_XX
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCV<Eigen::MatrixXd>,
                   Models10::H<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCV_Jacobian<Eigen::MatrixXd>,
                   Models10::H_Jacobian<Eigen::MatrixXd>> ekf;
public:

    BindEKF_10_CV_XX(Eigen::MatrixXd in_state,
                     Eigen::MatrixXd in_covariance,
                     Eigen::MatrixXd in_process_noise,
                     Eigen::MatrixXd in_measurement_noise):
        ekf(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise)
    {}

    Eigen::MatrixXd predict(double dt){return ekf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf.correct(z).first;}
    Eigen::MatrixXd getMeasPredict(){return ekf.getMeasurementPredict();}
    Eigen::MatrixXd getStatePredict(){return ekf.getStatePredict();}
};

class BindEKF_10_CV_POL
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCV<Eigen::MatrixXd>,
                   Models10::H_POL<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCV_Jacobian<Eigen::MatrixXd>,
                   Models10::H_POL_Jacobian<Eigen::MatrixXd>> ekf;
public:

    BindEKF_10_CV_POL(Eigen::MatrixXd in_state,
                      Eigen::MatrixXd in_covariance,
                      Eigen::MatrixXd in_process_noise,
                      Eigen::MatrixXd in_measurement_noise):
        ekf(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise)
    {}

    Eigen::MatrixXd predict(double dt){return ekf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf.correct(z).first;}
    Eigen::MatrixXd getMeasPredict(){return ekf.getMeasurementPredict();}
    Eigen::MatrixXd getStatePredict(){return ekf.getStatePredict();}
};

class BindEKF_10_CT_XX
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCT<Eigen::MatrixXd>,
                   Models10::H<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCT_Jacobian<Eigen::MatrixXd>,
                   Models10::H_Jacobian<Eigen::MatrixXd>> ekf;
public:

    BindEKF_10_CT_XX(Eigen::MatrixXd in_state,
                     Eigen::MatrixXd in_covariance,
                     Eigen::MatrixXd in_process_noise,
                     Eigen::MatrixXd in_measurement_noise):
        ekf(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise)
    {}

    Eigen::MatrixXd predict(double dt){return ekf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf.correct(z).first;}
};

class BindEKF_10_CT_deg_XX
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCT_deg<Eigen::MatrixXd>,
                   Models10::H<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCT_deg_Jacobian<Eigen::MatrixXd>,
                   Models10::H_Jacobian<Eigen::MatrixXd>> ekf;
public:

    BindEKF_10_CT_deg_XX(Eigen::MatrixXd in_state,
                         Eigen::MatrixXd in_covariance,
                         Eigen::MatrixXd in_process_noise,
                         Eigen::MatrixXd in_measurement_noise):
        ekf(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise)
    {}

    Eigen::MatrixXd predict(double dt){return ekf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf.correct(z).first;}
};

class BindEKF_10_CA_XX
{
private:
    Estimator::EKF<Eigen::MatrixXd,
                   Models10::FCA<Eigen::MatrixXd>,
                   Models10::H<Eigen::MatrixXd>,
                   Models10::G<Eigen::MatrixXd>,
                   Models10::FCA_Jacobian<Eigen::MatrixXd>,
                   Models10::H_Jacobian<Eigen::MatrixXd>> ekf;
public:

    BindEKF_10_CA_XX(Eigen::MatrixXd in_state,
                     Eigen::MatrixXd in_covariance,
                     Eigen::MatrixXd in_process_noise,
                     Eigen::MatrixXd in_measurement_noise):
        ekf(in_state,
            in_covariance,
            in_process_noise,
            in_measurement_noise){}

    Eigen::MatrixXd predict(double dt){return ekf.predict(dt).first;}
    Eigen::MatrixXd correct(const Eigen::MatrixXd &z){return ekf.correct(z).first;}
};

void bind_ekf(pybind11::module &m)
{
    py::class_<BindEKF_10_CV_XX>(m, "BindEKF_10_CV_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF_10_CV_XX::predict)
        .def("correct",&BindEKF_10_CV_XX::correct)
        .def("m_predict",&BindEKF_10_CV_XX::getMeasPredict)
        .def("s_predict",&BindEKF_10_CV_XX::getStatePredict);
    py::class_<BindEKF_10_CV_POL>(m, "BindEKF_10_CV_POL")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF_10_CV_POL::predict)
        .def("correct",&BindEKF_10_CV_POL::correct)
        .def("m_predict",&BindEKF_10_CV_POL::getMeasPredict)
        .def("s_predict",&BindEKF_10_CV_POL::getStatePredict);
    py::class_<BindEKF_10_CT_XX>(m, "BindEKF_10_CT_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF_10_CT_XX::predict)
        .def("correct",&BindEKF_10_CT_XX::correct);
    py::class_<BindEKF_10_CT_deg_XX>(m, "BindEKF_10_CT_deg_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF_10_CT_deg_XX::predict)
        .def("correct",&BindEKF_10_CT_deg_XX::correct);
    py::class_<BindEKF_10_CA_XX>(m, "BindEKF_10_CA_XX")
        .def(py::init<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>())
        .def("predict",&BindEKF_10_CA_XX::predict)
        .def("correct",&BindEKF_10_CA_XX::correct);
}
