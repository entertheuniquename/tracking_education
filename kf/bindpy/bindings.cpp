#include "bind_kf.h"
#include "bind_ekf.h"
#include "bind_imm_prototype.h"
#include "bind_imm.h"
#include "bind_models.h"

PYBIND11_MODULE(estimator, m) {
    bind_kf(m);
    bind_ekf(m);
    bind_imm_prot(m);
    bind_imm(m);
    bind_models(m);
}
