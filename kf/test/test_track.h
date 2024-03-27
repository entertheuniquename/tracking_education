#pragma once

#include <QtGlobal>
#include <QTest>
#include <QtCore>
#include <QObject>

#include<Eigen/Dense>

#include "../source/track.h"

class test_Track : public QObject
{
Q_OBJECT
public:

private slots:
    void data();
    void doit();
};
