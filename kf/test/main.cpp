#include <QTest>
#include <QObject>


#include "test_kf_eigen3.h"

//class test_kf_eigen3;

int main(int argc, char *argv[])
{
    test_KFE t0;
    QTest::qExec(&t0);
}
