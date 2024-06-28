#include <gtest/gtest.h>
#include "../source/kf.h"
#include "../source/ekf_qr.h"
#include "../source/ekf.h"
#include "../source/imm.h"
#include "../source/models.h"
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
