#pragma once

#include <QtGlobal>
#include <QTest>
#include <QtCore>
#include <QObject>
#include "../source/kf_eigen3.h"

#include <iostream>
#include <chrono>
#include <random>
#include<eigen3/Eigen/Dense>

#include<unsupported/Eigen/MatrixFunctions>
#include<Eigen/Eigenvalues>

class test_KFE : public QObject
{
    Q_OBJECT
    public:
    struct matrices
    {
        Eigen::MatrixXd A;
        Eigen::MatrixXd H;
        Eigen::MatrixXd Rpos;
        Eigen::MatrixXd Rvel;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd G;
        Eigen::MatrixXd x0;
        Eigen::MatrixXd P0;
        Eigen::MatrixXd B;
        Eigen::MatrixXd u;
    };
    struct xvector
    {
        double x;
        double vx;
        double y;
        double vy;
        double z;
        double vz;
    };
    private slots:
        matrices data(double meas_var,
                      double velo_var,
                      double process_var,
                      Eigen::MatrixXd stateModel,
                      Eigen::MatrixXd measurementModel,
                      Eigen::MatrixXd GModel,
                      Eigen::MatrixXd HposModel,
                      Eigen::MatrixXd HvelModel,
                      xvector x0);
        void estimation();
};

