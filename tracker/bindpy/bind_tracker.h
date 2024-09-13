#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "../source/kf.h"
#include "../source/ekf.h"
#include "../source/imm.h"
#include "../source/tracker.h"
#include "../source/models.h"

void bind_tracker(pybind11::module &m);