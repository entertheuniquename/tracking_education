#include "../source/models.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

void bind_models(py::module &m) {
    m.def("stateModel_CTx", [](Eigen::MatrixXd x,double dt){
        //#CAUSE
        Models::StateModel_CT_Deg<Eigen::MatrixXd> sm;
        //Models::StateModel_CT<Eigen::MatrixXd> sm;
        //~
        return sm(x,dt);});
    m.def("stateModel_CTx_rad", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CT<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("measureModel_XwXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
}
