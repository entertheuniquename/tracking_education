#pragma once

#include <QtGlobal>
#include <QTest>
#include <QtCore>
#include <QObject>

#include <Eigen/Dense>

#include "../source/track.h"

class test_Track : public QObject
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
private slots:
    matrices data(double meas_var,
                  double velo_var,
                  double process_var,
                  Eigen::MatrixXd stateModel,
                  Eigen::MatrixXd measurementModel,
                  Eigen::MatrixXd GModel,
                  Eigen::MatrixXd HposModel,
                  Eigen::MatrixXd HvelModel,
                  Eigen::MatrixXd x0);
    void doit();
};
