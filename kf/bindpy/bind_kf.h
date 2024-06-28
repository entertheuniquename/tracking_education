#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "../source/kf.h"
#include "../source/models.h"

void bind_kf(pybind11::module &m);
