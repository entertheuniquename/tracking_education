#pragma once

#include <iostream>
#include <Eigen/Dense>

#include<QApplication>
#include<QChartView>
#include<QLineSeries>
#include<QScatterSeries>
#include<QtCharts>
using namespace QtCharts;

enum class TYPE{LINE = 0,SCATTER=1};
struct MTN2
{
    std::vector<double> v1;
    std::vector<double> v2;
    TYPE t;
    std::string s;
    QColor c;
    MTN2(TYPE t_in,std::string s_in,std::vector<double> v1_in,std::vector<double> v2_in=std::vector<double>(), QColor c_in=Qt::red)
    {v1=v1_in;v2=v2_in;t=t_in;s=s_in;c=c_in;}
};
inline int print_charts_universal3(std::vector<MTN2> vec)
{
    int i=1;
    QApplication a(i,nullptr);

    std::map<std::string,QChart*> charts_map;
    for(int i=0;i<vec.size();i++)
    {
        int sz = vec[i].v1.size();
        if(vec[i].v2.size()<sz && !vec[i].v2.empty())
            sz=vec[i].v2.size();

        QChart *chart;
        if(charts_map.count(vec[i].s))
            chart = charts_map[vec[i].s];
        else
            chart = new QChart();
        chart->legend()->hide();

        switch (vec[i].t) {
        case TYPE::LINE:
        {
            QLineSeries *series = new QLineSeries();
            if(vec[i].v2.empty())
                for(int j=0;j<vec[i].v1.size();j++)
                    series->append(j,vec[i].v1[j]);
            else
                for(int j=0;j<sz;j++)
                    series->append(vec[i].v2[j],vec[i].v1[j]);

            series->setColor(vec[i].c);
            chart->addSeries(series);
            break;
        }
        case TYPE::SCATTER:
        {
            QScatterSeries *series = new QScatterSeries();
            series->setMarkerSize(10.);
            series->setMarkerShape(QScatterSeries::MarkerShape::MarkerShapeCircle);
            if(vec[i].v2.empty())
                for(int j=0;j<vec[i].v1.size();j++)
                    series->append(j,vec[i].v1[j]);
            else
                for(int j=0;j<sz;j++)
                    series->append(vec[i].v2[j],vec[i].v1[j]);

            series->setColor(vec[i].c);
            chart->addSeries(series);
            break;
        }
        default:
        {
            QLineSeries *series = new QLineSeries();
            if(vec[i].v2.empty())
                for(int j=0;j<vec[i].v1.size();j++)
                    series->append(j,vec[i].v1[j]);
            else
                for(int j=0;j<sz;j++)
                    series->append(vec[i].v2[j],vec[i].v1[j]);

            series->setColor(vec[i].c);
            chart->addSeries(series);
            break;
        }
        };
        charts_map[vec[i].s] = chart;
    }


    for(auto chart : charts_map)
    {
        chart.second->createDefaultAxes();
        chart.second->setTitle(chart.first.data());

        QChartView *chartView = new QChartView(chart.second);
        chartView->setRubberBand(QChartView::RubberBand::RectangleRubberBand/*HorizontalRubberBand*/);
        chartView->setRenderHint(QPainter::Antialiasing);
        chartView->setWindowTitle("Simple line chart");
        chartView->resize(600,300);
        chartView->show();
    }

    return a.exec();
}
