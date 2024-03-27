#include <QTest>
#include "test_kf_eigen3.h"
#include "test_track.h"

int main(int argc, char *argv[])
{
    test_Track t0;
    QTest::qExec(&t0);
    test_KFE t1;
    QTest::qExec(&t1);
}
