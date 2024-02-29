#include <qt5/QtTest/QTest>
#include <qt5/QtCore/QObject>


#include "test_kf_eigen3.h"

//class test_kf_eigen3;

int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);

    test_KFE t0;
    QTest::qExec(&t0);

    //Test_PUCParser test_pucparser0;
    //QTest::qExec(&test_pucparser0);

    //Test_PUMath test_pumath0;
    //QTest::qExec(&test_pumath0);

    //Test_PUPoint test_pupoint0;
    //QTest::qExec(&test_pupoint0);

    //return a.exec();
}
