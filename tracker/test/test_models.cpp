#include "../source/models.h"
#include "../source/utils.h"
#include <gtest/gtest.h>

TEST (Models10,H_POL_8_squares) {

    using M = Eigen::MatrixXd;

    Models10::H_POL<M> modelHPolar;
    Models10::H_POL_Jacobian<M> modelHPolarJacobian;

    // (1/8)

    M state(10,1);
    state.setZero();

    M zzz(3,1);//#ZAGL
    zzz.setZero();

    //std::cout << "[create]state: " << Utils::transpose(state) << std::endl;

    state(static_cast<int>(Models10::POSITION_X::X)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Y)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Z)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::W)) = 0.;
    //std::cout << "[init-1/8]state: " << Utils::transpose(state) << std::endl;

    M meas = modelHPolar(state,zzz,state);
    //std::cout << "[calculate]meas: " << Utils::transpose(meas) << " (" << meas(1)*(180./M_PI) << ")(" << meas(2)*(180./M_PI) << ")" << std::endl;

    double expectedRange = 173.205808;
    double expectedDegAzimuth = 45.;
    double expectedDegElevation = 35.264;

    ASSERT_NEAR(expectedRange,meas(static_cast<int>(Models10::POSITION_R::R)),0.001);
    ASSERT_NEAR(expectedDegAzimuth,meas(static_cast<int>(Models10::POSITION_R::A))*(180./M_PI),0.001);
    ASSERT_NEAR(expectedDegElevation,meas(static_cast<int>(Models10::POSITION_R::E))*(180./M_PI),0.001);

    // (2/8)

    state(static_cast<int>(Models10::POSITION_X::X)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Y)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Z)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::W)) = 0.;
    //std::cout << "[init-2/8]state: " << Utils::transpose(state) << std::endl;

    meas = modelHPolar(state,zzz,state);
    //std::cout << "[calculate]meas: " << Utils::transpose(meas) << " (" << meas(1)*(180./M_PI) << ")(" << meas(2)*(180./M_PI) << ")" << std::endl;

    expectedRange = 173.205808;
    expectedDegAzimuth = 135.;
    expectedDegElevation = 35.264;

    ASSERT_NEAR(expectedRange,meas(static_cast<int>(Models10::POSITION_R::R)),0.001);
    ASSERT_NEAR(expectedDegAzimuth,meas(static_cast<int>(Models10::POSITION_R::A))*(180./M_PI),0.001);
    ASSERT_NEAR(expectedDegElevation,meas(static_cast<int>(Models10::POSITION_R::E))*(180./M_PI),0.001);

    // (3/8)

    state(static_cast<int>(Models10::POSITION_X::X)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Y)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Z)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::W)) = 0.;
    //std::cout << "[init-3/8]state: " << Utils::transpose(state) << std::endl;

    meas = modelHPolar(state,zzz,state);
    //std::cout << "[calculate]meas: " << Utils::transpose(meas) << " (" << meas(1)*(180./M_PI) << ")(" << meas(2)*(180./M_PI) << ")" << std::endl;

    expectedRange = 173.205808;
    expectedDegAzimuth = 225.;
    expectedDegElevation = 35.264;

    ASSERT_NEAR(expectedRange,meas(static_cast<int>(Models10::POSITION_R::R)),0.001);
    ASSERT_NEAR(expectedDegAzimuth,meas(static_cast<int>(Models10::POSITION_R::A))*(180./M_PI)+360.,0.001);
    ASSERT_NEAR(expectedDegElevation,meas(static_cast<int>(Models10::POSITION_R::E))*(180./M_PI),0.001);

    // (4/8)

    state(static_cast<int>(Models10::POSITION_X::X)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Y)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Z)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::W)) = 0.;
    //std::cout << "[init-4/8]state: " << Utils::transpose(state) << std::endl;

    meas = modelHPolar(state,zzz,state);
    //std::cout << "[calculate]meas: " << Utils::transpose(meas) << " (" << meas(1)*(180./M_PI) << ")(" << meas(2)*(180./M_PI) << ")" << std::endl;

    expectedRange = 173.205808;
    expectedDegAzimuth = 315.;
    expectedDegElevation = 35.264;

    ASSERT_NEAR(expectedRange,meas(static_cast<int>(Models10::POSITION_R::R)),0.001);
    ASSERT_NEAR(expectedDegAzimuth,meas(static_cast<int>(Models10::POSITION_R::A))*(180./M_PI)+360.,0.001);
    ASSERT_NEAR(expectedDegElevation,meas(static_cast<int>(Models10::POSITION_R::E))*(180./M_PI),0.001);

    // (8/8)

    state(static_cast<int>(Models10::POSITION_X::X)) = 100.;
    state(static_cast<int>(Models10::POSITION_X::VX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AX)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Y)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AY)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::Z)) = -100.;
    state(static_cast<int>(Models10::POSITION_X::VZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::AZ)) = 0.;
    state(static_cast<int>(Models10::POSITION_X::W)) = 0.;
    //std::cout << "[init-8/8]state: " << Utils::transpose(state) << std::endl;

    meas = modelHPolar(state,zzz,state);
    //std::cout << "[calculate]meas: " << Utils::transpose(meas) << " (" << meas(1)*(180./M_PI) << ")(" << meas(2)*(180./M_PI) << ")" << std::endl;

    expectedRange = 173.205808;
    expectedDegAzimuth = 315.;
    expectedDegElevation = -35.264;

    ASSERT_NEAR(expectedRange,meas(static_cast<int>(Models10::POSITION_R::R)),0.001);
    ASSERT_NEAR(expectedDegAzimuth,meas(static_cast<int>(Models10::POSITION_R::A))*(180./M_PI)+360.,0.001);
    ASSERT_NEAR(expectedDegElevation,meas(static_cast<int>(Models10::POSITION_R::E))*(180./M_PI),0.001);
}
