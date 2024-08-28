#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "../source/kf_eigen3.h"
#include "../source/models.h"

void bind_kf_eigen3(pybind11::module &m);
