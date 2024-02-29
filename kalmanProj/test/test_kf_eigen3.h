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
    private slots:
        void data();
        void estimation();
};

