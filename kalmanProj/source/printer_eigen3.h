#pragma once

#include <iostream>
#include<eigen3/Eigen/Dense>

#include<QApplication>
#include<QChartView>
#include<QLineSeries>
#include<QScatterSeries>

using namespace QtCharts;

enum class TYPE{
    LINE = 0,
    SCATTER=1
};

void print_chart_statistic(Eigen::MatrixXd xx)
{
    //======================================================
    QChart *chartX0 = new QChart();
    chartX0->legend()->hide();

    QLineSeries *seriesX0 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX0->append(i,xx(0,i));
    chartX0->addSeries(seriesX0);

    chartX0->createDefaultAxes();
    chartX0->setTitle("statistic of X errors");

    QChartView *chartViewX0 = new QChartView(chartX0);
    chartViewX0->setRenderHint(QPainter::Antialiasing);
    chartViewX0->setWindowTitle("Simple line chart");
    chartViewX0->resize(800,600);
    chartViewX0->show();
    //======================================================
    QChart *chartX1 = new QChart();
    chartX1->legend()->hide();

    QLineSeries *seriesX1 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX1->append(i,xx(1,i));
    chartX1->addSeries(seriesX1);

    chartX1->createDefaultAxes();
    chartX1->setTitle("statistic of VX errors");

    QChartView *chartViewX1 = new QChartView(chartX1);
    chartViewX1->setRenderHint(QPainter::Antialiasing);
    chartViewX1->setWindowTitle("Simple line chart");
    chartViewX1->resize(800,600);
    chartViewX1->show();
    //======================================================
    QChart *chartX2 = new QChart();
    chartX2->legend()->hide();

    QLineSeries *seriesX2 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX2->append(i,xx(2,i));
    chartX2->addSeries(seriesX2);

    chartX2->createDefaultAxes();
    chartX2->setTitle("statistic of Y errors");

    QChartView *chartViewX2 = new QChartView(chartX2);
    chartViewX2->setRenderHint(QPainter::Antialiasing);
    chartViewX2->setWindowTitle("Simple line chart");
    chartViewX2->resize(800,600);
    chartViewX2->show();
    //======================================================
    QChart *chartX3 = new QChart();
    chartX3->legend()->hide();

    QLineSeries *seriesX3 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX3->append(i,xx(3,i));
    chartX3->addSeries(seriesX3);

    chartX3->createDefaultAxes();
    chartX3->setTitle("statistic of VY errors");

    QChartView *chartViewX3 = new QChartView(chartX3);
    chartViewX3->setRenderHint(QPainter::Antialiasing);
    chartViewX3->setWindowTitle("Simple line chart");
    chartViewX3->resize(800,600);
    chartViewX3->show();
    //======================================================
    QChart *chartX4 = new QChart();
    chartX4->legend()->hide();

    QLineSeries *seriesX4 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX4->append(i,xx(4,i));
    chartX4->addSeries(seriesX4);

    chartX4->createDefaultAxes();
    chartX4->setTitle("statistic of Z errors");

    QChartView *chartViewX4 = new QChartView(chartX4);
    chartViewX4->setRenderHint(QPainter::Antialiasing);
    chartViewX4->setWindowTitle("Simple line chart");
    chartViewX4->resize(800,600);
    chartViewX4->show();
    //======================================================
    QChart *chartX5 = new QChart();
    chartX5->legend()->hide();

    QLineSeries *seriesX5 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX5->append(i,xx(5,i));
    chartX5->addSeries(seriesX5);

    chartX5->createDefaultAxes();
    chartX5->setTitle("statistic of VZ errors");

    QChartView *chartViewX5 = new QChartView(chartX5);
    chartViewX5->setRenderHint(QPainter::Antialiasing);
    chartViewX5->setWindowTitle("Simple line chart");
    chartViewX5->resize(800,600);
    chartViewX5->show();
    //======================================================
}

void print_chart(Eigen::MatrixXd x1,
                 Eigen::MatrixXd x2,
                 Eigen::MatrixXd x3,
                 Eigen::MatrixXd xx)
{
    QChart *chart = new QChart();
    chart->legend()->hide();

    QLineSeries *series1 = new QLineSeries();
    for(int i=0;i<x1.cols();i++)
        series1->append(x1(0,i),x1(2,i));
    chart->addSeries(series1);

    QScatterSeries *series2 = new QScatterSeries();
    for(int i=0;i<x2.cols();i++)
        series2->append(x2(0,i),x2(2,i));
    chart->addSeries(series2);

    QLineSeries *series3 = new QLineSeries();
    for(int i=0;i<x3.cols();i++)
        series3->append(x3(0,i),x3(2,i));
    chart->addSeries(series3);

    chart->createDefaultAxes();
    chart->setTitle("kalman - XY");

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setWindowTitle("Simple line chart");
    chartView->resize(800,600);
    chartView->show();

    QChart *chart1 = new QChart();
    chart1->legend()->hide();

    QLineSeries *series11 = new QLineSeries();
    for(int i=0;i<x1.cols();i++)
        series11->append(x1(0,i),x1(4,i));
    chart1->addSeries(series11);

    QScatterSeries *series12 = new QScatterSeries();
    for(int i=0;i<x2.cols();i++)
        series12->append(x2(0,i),x2(4,i));
    chart1->addSeries(series12);

    QLineSeries *series13 = new QLineSeries();
    for(int i=0;i<x3.cols();i++)
        series13->append(x3(0,i),x3(4,i));
    chart1->addSeries(series13);

    chart1->createDefaultAxes();
    chart1->setTitle("kalman - XZ");

    QChartView *chartView1 = new QChartView(chart1);
    chartView1->setRenderHint(QPainter::Antialiasing);
    chartView1->setWindowTitle("Simple line chart");
    chartView1->resize(800,600);
    chartView1->show();


    QChart *chartX = new QChart();
    chartX->legend()->hide();

    QLineSeries *seriesX1 = new QLineSeries();
    for(int i=0;i<xx.cols();i++)
        seriesX1->append(i,xx(0,i)*xx(0,i));
    chartX->addSeries(seriesX1);

    chartX->createDefaultAxes();
    chartX->setTitle("errors - X step");

    QChartView *chartViewX = new QChartView(chartX);
    chartViewX->setRenderHint(QPainter::Antialiasing);
    chartViewX->setWindowTitle("Simple line chart");
    chartViewX->resize(800,600);
    chartViewX->show();
}

void print_charts_universal(std::vector<Eigen::MatrixXd> vec, TYPE type)
{
    QChart *chart = new QChart();
    chart->legend()->hide();

    switch (type) {
    case TYPE::LINE:
        for(int i=0;i<vec.size();i++)
        {
            QLineSeries *series = new QLineSeries();
            for(int j=0;j<vec[i].cols();j++)
                series->append(vec[i](1,j),vec[i](0,j));
            chart->addSeries(series);
        }
        break;
    case TYPE::SCATTER:
        for(int i=0;i<vec.size();i++)
        {
            QScatterSeries *series = new QScatterSeries();
            for(int j=0;j<vec[i].cols();j++)
                series->append(vec[i](1,j),vec[i](0,j));
            chart->addSeries(series);
        }
        break;
    default:
        for(int i=0;i<vec.size();i++)
        {
            QLineSeries *series = new QLineSeries();
            for(int j=0;j<vec[i].cols();j++)
                series->append(vec[i](1,j),vec[i](0,j));
            chart->addSeries(series);
        }
        break;
    }

    chart->createDefaultAxes();
    chart->setTitle("measurements");

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setWindowTitle("Simple line chart");
    chartView->resize(800,600);
    chartView->show();
}
