/*
 * Adjust Curves
 *
 * Author: JoStudio
 */

#ifndef OPENCV2_PS_CURVES_HPP_
#define OPENCV2_PS_CURVES_HPP_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;

namespace cv {

/**
 * Class of Curve for one channel
 */
class Curve {
protected:
	Scalar color;
	Scalar back_color;
	int tolerance; //��갴�»��ƶ�ʱ���������ߵ����Χ
	bool is_mouse_down;
	vector<Point> points;  //control points ���ߵ����п��Ƶ�
	vector<Point>::iterator current;  //pointer to current point ��ǰ���Ƶ��ָ��

	vector<Point>::iterator  find(int x);
	vector<Point>::iterator  find(int x, int y);
	vector<Point>::iterator  add(int x, int y);

public:
	Curve();
	virtual ~Curve();

	int calcCurve(double *z); //���ڲ����õķ�������������

	void draw(Mat &mat);  //�����߻���mat��
	void mouseDown(int x, int y); //����갴�£������mouseDown()����
	bool mouseMove(int x, int y); //������ƶ��������mouseMove()����
	void mouseUp(int x, int y); //�����̧�������mouseUp()����

	//���·������ڣ��ñ�̷�ʽ��������
	void clearPoints(); //������������еĵ�
	int  addPoint(const Point &p); //����һ����
	int  deletePoint(const Point &p); //ɾ��һ����
	int  movePoint(const Point &p, int x, int y); //�ƶ�һ����
};

/**
 * Class of Curves for all channels
 */
class Curves {
protected:
	void createColorTables(uchar colorTables[][256]);
public:
	Curves();
	virtual ~Curves();

	Curve RGBChannel;   //RGB��ͨ��
	Curve RedChannel;   //Redͨ��
	Curve GreenChannel; //Greenͨ��
	Curve BlueChannel;  //Blueͨ��

	Curve * CurrentChannel; //��ǰͨ����ָ��

	void draw(Mat &mat);  //�����߻���mat��
	void mouseDown(int x, int y); //����갴�£������mouseDown()����
	bool mouseMove(int x, int y); //������ƶ��������mouseMove()����
	void mouseUp(int x, int y); //�����̧�������mouseUp()����

	//ʵʩ���ߵ���
	int adjust(InputArray src, OutputArray dst, InputArray mask = noArray());

};

//��һ������
void dot_line(Mat &mat, Point &p1, Point &p2, Scalar &color, int step = 8);

} /* namespace cv */

#endif /* OPENCV2_PS_CURVES_HPP_ */
