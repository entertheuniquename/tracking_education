#include "bind_tracker_prototype.h"

namespace py = pybind11;

using MatrixType = Eigen::MatrixXd;
using FModelType = Models10::FCV<MatrixType>;
using HModelType = Models10::H<MatrixType>;
using GModelType = Models10::G<MatrixType>;
using MeasurementType = Tracker::Measurement3<MatrixType>;
using EstimatorType = Estimator::KF<MatrixType,FModelType,HModelType,GModelType>;
using EstimatorInitializatorType = Tracker::EstimatorInitializator10<MatrixType,EstimatorType,MeasurementType>;
using TrackType = Tracker::Track<MatrixType,MeasurementType,EstimatorInitializatorType>;

class BindTracker_10_KF_CV_XX
{
private:
    double time;
    double counter;
    Tracker::Tracker_prototype<MatrixType,
                                EstimatorInitializatorType,
                                EstimatorType,
                                TrackType,
                                MeasurementType> tracker;
public:
    BindTracker_10_KF_CV_XX():time(0.)/*,counter(0)*/{}

    MatrixType step(MatrixType zs, double dt)
    {
//        counter++;
        std::vector<MeasurementType> measurements;
        for(int i=0;i<zs.rows();i++)
            measurements.push_back(MeasurementType{time,zs(i,0),zs(i,1),zs(i,2),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.});

        time+=dt;

        MatrixType estimates(zs.rows(),zs.cols());
        estimates.setZero();

        tracker.step(measurements,dt);

        for(int i=0;i<tracker.tracks.size();i++)
        {
            MatrixType state = tracker.tracks.at(i)->getState();
            estimates(i,0) = state(0,0);
            estimates(i,1) = state(3,0);
            estimates(i,2) = state(6,0);
        }

        return estimates;
    }
};

void bind_tracker_prototype(pybind11::module &m)
{
    py::class_<BindTracker_10_KF_CV_XX>(m, "BindTracker_10_KF_CV_XX")
        .def(py::init())
        .def("step",&BindTracker_10_KF_CV_XX::step);
}
