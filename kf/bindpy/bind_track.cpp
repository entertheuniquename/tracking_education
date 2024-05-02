#include "bind_track.h"

#include<string>
#include<iostream>
#include<variant>
#include<tuple>

namespace py = pybind11;

Measurement<Eigen::MatrixXd> make_meas(const py::tuple& tu)
{
    std::tuple<double,Eigen::MatrixXd> a_tup = py::cast<std::tuple<double,Eigen::MatrixXd>>(tu);
    double t = std::get<0>(a_tup);
    Eigen::MatrixXd m = std::get<1>(a_tup);

    Measurement<Eigen::MatrixXd> meas{t,m};
    return meas;
}

class BindTrackKFE
{
private:
    Track<Eigen::MatrixXd,
          Estimator::KFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
          EstimatorInitKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
          Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>> track;
public:

    BindTrackKFE(const py::tuple& tu):
        track(make_meas(tu)){}

    Eigen::MatrixXd step1(const py::tuple& tu)
    {
        return track.step(make_meas(tu));
    }
    Eigen::MatrixXd step2(double t)
    {
        return track.step(t);
    }
};

class BindTrackEKFE
{
private:
    Track<Eigen::MatrixXd,
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>>,
    EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>>,
    Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>> track;
public:

    BindTrackEKFE(const py::tuple& tu):
        track(make_meas(tu)){}

    Eigen::MatrixXd step1(const py::tuple& tu)
    {
        return track.step(make_meas(tu));
    }
    Eigen::MatrixXd step2(double t)
    {
        return track.step(t);
    }
};

void bind_track(pybind11::module &m)
{
    py::class_<BindTrackKFE>(m, "BindTrackKFE")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackKFE::step1)
        .def("step",&BindTrackKFE::step2);
    py::class_<BindTrackEKFE>(m, "BindTrackEKFE")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKFE::step1)
        .def("step",&BindTrackEKFE::step2);
}
