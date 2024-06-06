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

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
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

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
};

class BindTrackKFE_CT
{
private:
    Track<Eigen::MatrixXd,
          Estimator::KFEx<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
          EstimatorInitKFEx<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
          Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> track;
public:

    BindTrackKFE_CT(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
};

class BindTrackEKFE_xyz_cv
{
private:
    Track<Eigen::MatrixXd,
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
    EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
    Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>> track;
public:

    BindTrackEKFE_xyz_cv(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
};

class BindTrackEKFE_xyz_ct
{
private:
    Track<Eigen::MatrixXd,
    Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    EstimatorInitEKFE_xyz_ct<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> track;
public:

    BindTrackEKFE_xyz_ct(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
    Eigen::MatrixXd GetState(){return track.GetState();}
    Eigen::MatrixXd GetCovariance(){return track.GetCovariance();}
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
    py::class_<BindTrackKFE_CT>(m, "BindTrackKFE_CT")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackKFE_CT::step1)
        .def("step",&BindTrackKFE_CT::step2);
    py::class_<BindTrackEKFE_xyz_cv>(m, "BindTrackEKFE_xyz_cv")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKFE_xyz_cv::step1)
        .def("step",&BindTrackEKFE_xyz_cv::step2);
    py::class_<BindTrackEKFE_xyz_ct>(m, "BindTrackEKFE_xyz_ct")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKFE_xyz_ct::step1)
        .def("step",&BindTrackEKFE_xyz_ct::step2)
        .def("getState",&BindTrackEKFE_xyz_ct::GetState)
        .def("getCov",&BindTrackEKFE_xyz_ct::GetCovariance);
}
