#include "bind_measurement.h"
#include "../source/models.h"

namespace py = pybind11;

class BindMeasurement
{
private:
    Measurement<> meas;
public:
    BindMeasurement(double t, Eigen::MatrixXd p):
        meas({t,p}){}
};

void bind_measurement(pybind11::module &m)
{
    py::class_<BindMeasurement>(m, "BindMeasurement")
        .def(py::init<double,Eigen::MatrixXd>());
}
