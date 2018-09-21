/*
 * Adjust Levels
 *
 * Author: JoStudio
 */
#ifndef OPENCV2_PS_LEVELS_HPP_
#define OPENCV2_PS_LEVELS_HPP_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

namespace cv {

/**
 * Class of Level for one channel
 */
class Level {
public:
	int   Shadow;  //����ɫ�׺ڵ�ֵ
	float Midtones; //����ɫ�׻ҵ�ֵ��ע���Ǹ�������
	int   Highlight; //����ɫ�װ׵�ֵ

	int   OutputShadow; //���ɫ�׺ڵ�ֵ
	int   OutputHighlight; //���ɫ�װ׵�ֵ

	Level();
	virtual ~Level();

	bool createColorTable(uchar * colorTable);
	void clear();
};

/**
 * Class of Levels for all channels
 */
class Levels {
protected:
	bool createColorTables(uchar colorTables[][256]);

public:
	Level RGBChannel;  //RGB�������
	Level RedChannel;  //��ɫͨ��
	Level GreenChannel; //��ɫͨ��
	Level BlueChannel; //��ɫͨ��

	Levels();
	virtual ~Levels();

	int adjust(InputArray src, OutputArray dst); //ʵʩɫ�׵���
};


} /* namespace cv */

#endif /* OPENCV2_PS_LEVELS_HPP_ */
