#include "../source/models.h"

#include <pybind11-global/pybind11/pybind11.h>
#include <pybind11-global/pybind11/numpy.h>
#include <pybind11-global/pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

void bind_models(py::module &m) {
    m.def("stateModel_3A", [](double dt){
        return Models::stateModel_3A<Eigen::MatrixXd>(dt);
    }/*,R"pbdoc(
          stateModel_3A
      )pbdoc",
      py::arg("dt")*/
    );
    m.def("stateModel_3Ax", [](Eigen::MatrixXd x,double dt){
        return Models::stateModel_3Ax<Eigen::MatrixXd>(x,dt);
    }/*,R"pbdoc(
          stateModel_3Ax
      )pbdoc",
      py::arg("dt")*/
    );
    m.def("measureModel_3A", [](){
          return Models::measureModel_3A<Eigen::MatrixXd>();
    }/*,R"pbdoc(
            measureModel3A
        )pbdoc",
        py::arg("x")*/
    );
    m.def("measureModel_3Ax", [](Eigen::MatrixXd x){
          return Models::measureModel_3Ax<Eigen::MatrixXd>(x);
    }/*,R"pbdoc(
            measureModel3A
        )pbdoc",
        py::arg("x")*/
    );
}
