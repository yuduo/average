//#include <stdio.h>
//#include <stdlib.h>
//
//#include <opencv2/opencv.hpp>
//#include <opencv/highgui.h>
//
//using namespace cv;
//using namespace std;
//
///*
//
//  This program calculates the average of several images with the same size.
//
// */
//
//void print_errmsg(const char* argv)
//{
//  cerr << "Use: " << argv << "<image1> <image2> [more images ...]" << endl
//       << "This program takes in several images and outputs a single image that is the average of the inputs. Using images of different sizes causes undefined behaviour. It outputs the name of the created file, which is the name of the first file + \"_avg\"\n";
//}
//
//int main(int argc, char** argv )
//{
//  if ( argc < 3 )
//    {
//      print_errmsg(argv[0]);
//      return -1; 
//   }
//
//  Mat acc, avg;
//  for (int i = 1; i < argc; i++)
//    {
//      Mat image = imread(argv[i], 1);
//      if ( !image.data )
//	{
//	  cerr << "No image data \n";
//	  cerr << "Admitted formats reminder: *.bmp, *.dib, *.jpeg, *.jpg, *.jpe, *.jp2, *.png, *.pbm, *.pgm, *.ppm, *.sr, *.ras, *.tiff, *.tif \n";
//	  return -1;
//	}
//      if (acc.empty())
//	acc = Mat(image.size(), CV_32FC3, Scalar(0, 0, 0));
//      accumulate(image, acc);
//    }
//
//  acc.convertTo(avg, CV_32SC3, 1.0/(argc - 1));
//
//
//  std::string writename = argv[1];
//  writename.erase(writename.find_last_of('.'));
//  std::ostringstream extension;
//
//  extension << "_avg.png";
//  writename += extension.str();
//  imwrite(writename, avg);
//  printf("%s\n", writename.c_str());
//
//  return 0;
//}
//#include <stdio.h>
//#include <stdlib.h>
//
//#include <opencv2/opencv.hpp>
//#include <opencv/highgui.h>
//
//using namespace cv;
//using namespace std;
//
///*
//
//This program performs a gaussian blur on an image with the given radius.
//
//The image is converted to grayscale.
//
//*/
//
//int main(int argc, char** argv)
//{
//	if (argc < 2 || argc > 4 || argc == 3)
//	{
//		cerr << "usage: " << argv[0] << " (gauss radius) [-f <Image_path>]\n";
//		cerr << "This function performs a gaussian blur operation on the provided image (after grayscale) and saves the result as a png. It outputs the name of the created file. If the -f argument isn't provided, it takes a filename from standard input.\n";
//		return -1;
//	}
//
//	char* check = 0;
//	int r1 = strtol(argv[1], &check, 10);
//	if (check == argv[1])
//	{
//		cerr << "usage: " << argv[0] << " (gauss radius) [-f <Image_path>]\n";
//		cerr << "This function performs a gaussian blur operation on the provided image (after grayscale) and saves the result as a png. It outputs the name of the created file. If the -f argument isn't provided, it takes a filename from standard input.\n";
//		return -1;
//	}
//
//
//	string filename;
//	if (argc == 4)
//		if (!strcmp(argv[2], "-f"))
//			filename = argv[3];
//		else
//		{
//			cerr << "No filename provided/n";
//			return -1;
//		}
//	else
//		cin >> filename;
//
//	Mat image = imread(filename, 1);
//	if (!image.data)
//	{
//		cerr << "No image data \n";
//		cerr << "Admitted formats reminder: *.bmp, *.dib, *.jpeg, *.jpg, *.jpe, *.jp2, *.png, *.pbm, *.pgm, *.ppm, *.sr, *.ras, *.tiff, *.tif \n";
//		return -1;
//	}
//
//	//Convert to grayscale
//	cvtColor(image, image, CV_BGR2GRAY);
//	if (r1 % 2 == 0) r1++;
//
//	//Gauss blurs
//	Mat g1;
//	GaussianBlur(image, g1, Size(r1, r1), 0);
//
//	//Change filename extension
//	std::string writename = filename;
//	writename.erase(writename.find_last_of('.'));
//	std::ostringstream extension;
//	//add operation tag
//	extension << "_" << r1 << "g" << ".png";
//	writename += extension.str();
//	imwrite(writename, g1);
//	cout << writename << endl;
//
//	return 0;
//}
# if 0
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;

int main(int argc, char** argv)
{
	namedWindow("Before", CV_WINDOW_AUTOSIZE);

	// Load the source image
	Mat src = imread("e:/LEFTIMAGE_.tif", 1);

	// Create a destination Mat object
	Mat dst;

	// display the source image
	imshow("Before", src);
#if 1
	int i = 25;
	//for (int i = 1; i<51; i = i + 2)
	{
		// smooth the image in the "src" and save it to "dst"
		 blur(src, dst, Size(i,i));
		 imshow("blur filter", dst);

		// Gaussian smoothing
		 GaussianBlur( src, dst, Size( i, i ), 0, 0 );
		 imshow("Gaussian filter", dst);

		// Median smoothing
		medianBlur(src, dst, i);

		// show the blurred image with the text
		imshow("Median filter", dst);

		// Bilateral smoothing
		bilateralFilter(src, dst, i, i * 2, i * 2);

		//show the blurred image with the text
		imshow("Bilateral filter", dst);

		// wait for 5 seconds
		waitKey(0);
	}
#endif
#if 0
	// converting to gray scale
	cvtColor(src, src, COLOR_BGR2GRAY);

	Laplacian(src, dst, CV_64F);
	//show the blurred image with the text
	//imshow("Laplacian filter", dst);

	Sobel(src, dst, CV_64F, 1, 0);
	//imshow("Sobel filter", dst);

	int kernel_size = 10;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	filter2D(src, dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

	imshow("filter2D filter", dst);
	// wait for 5 seconds
	waitKey(0);
#endif
}
#endif
#if 0
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

/**
* @function main
*/
int main(int, char** argv)
{
	/// Load source image
	src = imread("f:/zebra.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cerr << "No image supplied ..." << endl;
		return -1;
	}

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// Create Window
	const char* source_window = "Source";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, src);

	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return(0);
}

/**
* @function thresh_callback
*/
void thresh_callback(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}
#endif
#if 0
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;

double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */

int main(int argc, char** argv)
{
	/// Read image given by user
	//Mat image = imread("e:/LEFTIMAGE_.tif", 1);
	////

	//Mat new_image = Mat::zeros(image.size(), image.type());
	//Mat dst;
	///// Initialize values
	//std::cout << " Basic Linear Transforms " << std::endl;
	//std::cout << "-------------------------" << std::endl;
	////std::cout << "* Enter the alpha value [1.0-3.0]: "; std::cin >> alpha;
	////std::cout << "* Enter the beta value [0-100]: "; std::cin >> beta;
	//int alpha = 3.0;
	//int beta = 100;
	///// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	//for (int y = 0; y < image.rows; y++)
	//{
	//	for (int x = 0; x < image.cols; x++)
	//	{
	//		for (int c = 0; c < 1; c++)
	//		{
	//			new_image.at<Vec3b>(y, x)[c] =
	//				saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
	//		}
	//	}
	//}

	///// Create Windows
	//namedWindow("Original Image", 1);
	//namedWindow("New Image", 1);

	///// Show stuff
	//imshow("Original Image", image);
	//imshow("New Image", new_image);

	// Gaussian smoothing
	/*GaussianBlur(new_image, dst, Size(25, 25), 0, 0);
	addWeighted(new_image, 1.5, dst, -0.5, 0, dst);
	imshow("Gaussian filter", dst);*/

	/*int kernel_size = 10;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	filter2D(new_image, dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("filter2D", dst);*/


	Mat img, imgLaplacian, imgResult;

	//------------------------------------------------------------------------------------------- test, first of all
	// now do it by hand
	//img = (Mat_<uchar>(4, 4) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15);

	//// first, the good result
	//Laplacian(img, imgLaplacian, CV_8UC1);
	//std::cout << "let opencv do it" << std::endl;
	//std::cout << imgLaplacian << std::endl;

	//Mat kernel = (Mat_<float>(3, 3) <<
	//	0, 1, 0,
	//	1, -4, 1,
	//	0, 1, 0);
	//int window_size = 3;

	//// now, reaaallly by hand
	//// note that, for avoiding padding, the result image will be smaller than the original one.
	//Mat frame, frame32;
	//Rect roi;
	//imgLaplacian = Mat::zeros(img.size(), CV_32F);
	//for (int y = 0; y<img.rows - window_size / 2 - 1; y++) {
	//	for (int x = 0; x<img.cols - window_size / 2 - 1; x++) {
	//		roi = Rect(x, y, window_size, window_size);
	//		frame = img(roi);
	//		frame.convertTo(frame, CV_32F);
	//		frame = frame.mul(kernel);
	//		float v = sum(frame)[0];
	//		imgLaplacian.at<float>(y, x) = v;
	//	}
	//}
	//imgLaplacian.convertTo(imgLaplacian, CV_8U);
	//std::cout << "dudee" << imgLaplacian << std::endl;

	//// a little bit less "by hand"..
	//// using cv::filter2D
	//filter2D(img, imgLaplacian, -1, kernel);
	//std::cout << imgLaplacian << std::endl;


	//------------------------------------------------------------------------------------------- real stuffs now
#if 1
	img =  imread("e:/LEFTIMAGE_.tif", 0); // load grayscale image
	
								 // ok, now try different kernel
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); 
	/*Mat kernel = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0);*/
		// another approximation of second derivate, more stronger

				  // do the laplacian filtering as it is
				  // well, we need to convert everything in something more deeper then CV_8U
				  // because the kernel has some negative values, 
				  // and we can expect in general to have a Laplacian image with negative values
				  // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
				  // so the possible negative number will be truncated
	filter2D(img, imgLaplacian, CV_32F, kernel);
	img.convertTo(img, CV_32F);
	imgResult = img - imgLaplacian;
	imgResult = imgResult + 180;

	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8U);
	imgLaplacian.convertTo(imgLaplacian, CV_8U);

	namedWindow("laplacian", CV_WINDOW_AUTOSIZE);
	imshow("laplacian", imgLaplacian);

	namedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("result", imgResult);

	/*GaussianBlur(imgResult, imgResult, cv::Size(0, 0), 3);
	addWeighted(imgResult, 1.5, imgResult, -0.5, 0, imgResult);
	imshow("result2", imgResult);*/
	
#endif
	/// Wait until user press some key
	waitKey();
	return 0;
}

#endif

#if 0
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>
using namespace cv;
using namespace std;

void histEqualize(Mat &src, Mat &dst); //Histogram equalization
void drawHist(const Mat &src, Mat &dst); // Draw Histogram

int main()
{
	Mat image, hist1, hist2;
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };


	image = imread("e:/LEFTIMAGE_.tif", CV_LOAD_IMAGE_GRAYSCALE);

	if (!image.data) {// Check for invalid input
		cout << "Could not open or find the image" << endl;
		system("pause");
		return -1;
	}
	Mat out = Mat::zeros(image.size(), image.type());

	histEqualize(image, out); //eq

	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", image);
	waitKey(0);

	calcHist(&image, 1, 0, Mat(), hist1, 1, &histSize, &histRange);
	Mat showHist1(256, 256, CV_8UC1, Scalar(255));
	drawHist(hist1, showHist1);

	namedWindow("Histogram(Original)", CV_WINDOW_AUTOSIZE);
	imshow("Histogram(Original)", showHist1);
	waitKey(0);

	namedWindow("Histogram Equalization", CV_WINDOW_AUTOSIZE);
	imshow("Histogram Equalization", out);
	waitKey(0);


	calcHist(&out, 1, 0, Mat(), hist2, 1, &histSize, &histRange);
	Mat showHist2(256, 256, CV_8UC1, Scalar(255));
	drawHist(hist2, showHist2);

	namedWindow("Histogram(Equalized)", CV_WINDOW_AUTOSIZE);
	imshow("Histogram(Equalized)", showHist2);
	waitKey(0);

	//imwrite("Histogram_equalized.jpg", out);

	return 0;
}

void histEqualize(Mat &src, Mat &dst)
{
	int gray[256]{ 0 }; //Ex: gray[0] = gray value 0 appaer times, gray[1] = gray value 1 appaer times, ...
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) { //count each gray value appaer times
			gray[(int)src.at<uchar>(y, x)]++;
		}
	}

	int mincdf; //Minimum of cdf

	for (int i = 0; i < 255; i++) { //calculate cdf (Cumulative distribution function)
		gray[i + 1] += gray[i];
	}

	for (int i = 0; i < 255; i++) { //find minimum of cdf
		if (gray[i] != 0) {
			mincdf = gray[i];
			break;
		}
	}

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			// h(v) = round(((cdf(v) - mincdf) / (M * N) - mincdf) * (L - 1)) ; L = 2^8
			dst.at<uchar>(y, x) = (uchar)round((((double)gray[(int)src.at<uchar>(y, x)] - mincdf) / (double)(src.rows * src.cols - mincdf)) * (double)255);
		}
	}
}

void drawHist(const Mat &src, Mat &dst)
{
	int histSize = 256;
	float histMax = 0;
	for (int i = 0; i < histSize; i++) {
		float temp = src.at<float>(i);
		if (histMax < temp) {
			histMax = temp;
		}
	}

	float scale = (0.9 * 256) / histMax;
	for (int i = 0; i < histSize; i++) {
		int intensity = static_cast<int>(src.at<float>(i)*scale);
		line(dst, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
	}
}
#endif

#if 0
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Global Variables
cv::Mat hue;
int minHue = 0;
int maxHue = 0;

// Do all the thresholding
void threshold(int, void*)
{
	using namespace cv;

	Mat threshLow;
	Mat threshHigh;
	threshold(hue, threshLow, minHue, 255, THRESH_BINARY);
	threshold(hue, threshHigh, maxHue, 255, THRESH_BINARY_INV);
	Mat threshed = threshLow & threshHigh;

	imshow("threshLow", threshLow);
	imshow("threshHigh", threshHigh);
	imshow("Thresholded", threshed);
}

int main()
{
	

	cv::Mat img = cv::imread("e:/LEFTIMAGE_.tif", 1);

	// Blur the image to smooth it out (especially with JPG's)
	//cv::GaussianBlur(img, img, cv::Size(3, 3), 1, 1);

	//cv::imshow("Full", img);

	// Convert to HSV
	cv::Mat cvted;
	cv::cvtColor(img, cvted, CV_BGR2HSV);

	// Isolate the Hue Channel, and store in global variable
	std::vector<cv::Mat> separated(3);
	cv::split(cvted, separated);
	hue = separated.at(0).clone();

	cv::namedWindow("Thresholded", cv::WINDOW_NORMAL);

	cv::createTrackbar("hueMin", "Thresholded", &minHue, 255, threshold);
	cv::createTrackbar("hueMax", "Thresholded", &maxHue, 255, threshold);

	// Do the image processing once initially (parameters have no significance)
	threshold(1, NULL);

	cv::waitKey(0);

	return 0;
}
#endif

#if 0
#include<opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include<vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	// 读取RBG图片，转成Lab模式
	Mat bgr_image = imread("e:/LEFTIMAGE_.tif",1);


	Mat image = bgr_image;
	//cvtColor(imgResult, img, cv::COLOR_GRAY2RGB, 3);
	Mat new_image = Mat::zeros(image.size(), image.type());
	int alpha = 3.0;
	int beta = 100;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				new_image.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}

	/// Create Windows
	namedWindow("Original Image", 1);
	namedWindow("New Image", 1);

	/// Show stuff
	imshow("Original Image", image);
	imshow("New Image", new_image);
	waitKey();

	bgr_image = new_image;

	if (!bgr_image.rows) {
		cout << "imread failed!" << endl;
		return 0;
	}
	if (bgr_image.channels() == 1) {
		cv::Mat tmp;
		cv::cvtColor(bgr_image, tmp, cv::COLOR_GRAY2BGR, 3);
		bgr_image = tmp;
	}
	Mat lab_image;
	cvtColor(bgr_image, lab_image, CV_BGR2Lab);

	// 提取L通道
	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);

	// CLAHE 算法
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);
	Mat dst;
	clahe->apply(lab_planes[0], dst);
	dst.copyTo(lab_planes[0]);
	merge(lab_planes, lab_image);

	// 恢复RGB图像
	Mat image_clahe;
	cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	// 打印结果
	imshow("原始图片", bgr_image);
	imshow("CLAHE处理", image_clahe);
	waitKey();

#if 1
	Mat img = image_clahe;
	cvtColor(image_clahe, img, cv::COLOR_RGB2GRAY,1);
	Mat imgLaplacian; Mat imgResult;
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1);
	
	// another approximation of second derivate, more stronger

	// do the laplacian filtering as it is
	// well, we need to convert everything in something more deeper then CV_8U
	// because the kernel has some negative values, 
	// and we can expect in general to have a Laplacian image with negative values
	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
	// so the possible negative number will be truncated
	filter2D(img, imgLaplacian, CV_32F, kernel);
	img.convertTo(img, CV_32F);
	imgResult = img - imgLaplacian;
	//imgResult = imgResult + 180;

	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8U);
	imgLaplacian.convertTo(imgLaplacian, CV_8U);

	namedWindow("laplacian", CV_WINDOW_AUTOSIZE);
	imshow("laplacian", imgLaplacian);

	namedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("result", imgResult);

	waitKey();
#else
	short int pe=3;	// primary enhancementFactor
	short int se=-3; // primary enhancementFactor
	Mat inputImage= image_clahe, outputImage, kern;
	//assigning mask to kern variable
	kern = (Mat_<char>(3, 3) <<
		0, se, 0,
		se, pe, se,
		0, se, 0);

	//applying mask to input image
	filter2D(inputImage, outputImage, inputImage.depth(), kern);



	//defying windows and siplaying images
	namedWindow("Input image", CV_WINDOW_AUTOSIZE);
	namedWindow("Output image", CV_WINDOW_AUTOSIZE);
	imshow("Input image", inputImage);
	imshow("Output image", outputImage);
	Mat imgResult = outputImage;
	waitKey();
#endif

	int i = 25;
	Mat src = imgResult; //Mat dst;
	// smooth the image in the "src" and save it to "dst"
	blur(src, dst, Size(i, i));
	imshow("blur filter", dst);

	// Gaussian smoothing
	GaussianBlur(src, dst, Size(i, i), 0, 0);
	imshow("Gaussian filter", dst);

	// Median smoothing
	medianBlur(src, dst, i);

	// show the blurred image with the text
	imshow("Median filter", dst);

	// Bilateral smoothing
	bilateralFilter(src, dst, i, i * 2, i * 2);

	//show the blurred image with the text
	imshow("Bilateral filter", dst);

	// wait for 5 seconds
	waitKey(0);
	return 0;
}
#endif

#if 0
#include <opencv2\opencv.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "suace.h"
using namespace cv;
using namespace std;

int a = 21;
int b = 36;
int intensityMax = 255;

int main()
{
	namedWindow("SUACE", 1);
	createTrackbar("distance", "SUACE", &a, intensityMax);
	createTrackbar("sigma", "SUACE", &b, intensityMax);
	char filename[100];
	Mat suaceResult;
	Mat frame;

	// 读取RBG图片，转成Lab模式
	Mat bgr_image = imread("e:/LEFTIMAGE_.tif", 1);


	Mat image = bgr_image;
	


	cvtColor(bgr_image, frame, cv::COLOR_RGB2GRAY, 1);

	while (true)
	{
		
		//frame = imread("e:/LEFTIMAGE_.tif", CV_LOAD_IMAGE_GRAYSCALE);
		performSUACE(frame, suaceResult, a, (b + 1) / 8.0); //perform SUACE with the parameters
		imshow("SUACE", suaceResult);
		imshow("Original", frame);
		int response = waitKey(0);//press key to update
		if (response == 32) //exit when spacebar key pressed;
			break;
	}

	cvtColor(suaceResult, image, cv::COLOR_GRAY2RGB, 3);
	Mat new_image = image;// Mat::zeros(image.size(), image.type());
	int alpha = 1.0;
	int beta = 50;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				new_image.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
	/// Create Windows
	//namedWindow("Original Image", 1);
	namedWindow("New Image", 1);

	/// Show stuff
	//imshow("Original Image", new_image);
	imshow("New Image", new_image);

	waitKey();

	int i = 5;
	Mat src = new_image; Mat dst;
	cvtColor(new_image, src, cv::COLOR_RGB2GRAY, 1);
	src = src + 30;
						 // smooth the image in the "src" and save it to "dst"
	blur(src, dst, Size(i, i));
	imshow("blur filter", dst);

	// Gaussian smoothing
	GaussianBlur(src, dst, Size(i, i), 0, 0);
	imshow("Gaussian filter", dst);

	// Median smoothing
	medianBlur(src, dst, i);

	// show the blurred image with the text
	imshow("Median filter", dst);

	// Bilateral smoothing
	bilateralFilter(src, dst, i, i * 2, i * 2);

	//show the blurred image with the text
	imshow("Bilateral filter", dst);

	// wait for 5 seconds
	waitKey(0);
	return 0;
}
#endif
#if 0
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo(int, void*);

/**
 * @function main
 */
int main(int argc, char** argv)
{
	/// Load an image
	src = imread("e:/LEFTIMAGE_.tif", 1);

	/// Convert the image to Gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window to display results
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold
	createTrackbar(trackbar_type,
		window_name, &threshold_type,
		max_type, Threshold_Demo);

	createTrackbar(trackbar_value,
		window_name, &threshold_value,
		max_value, Threshold_Demo);

	/// Call the function to initialize
	Threshold_Demo(0, 0);

	/// Wait until user finishes program
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

}


/**
 * @function Threshold_Demo
 */
void Threshold_Demo(int, void*)
{
	/* 0: Binary
	   1: Binary Inverted
	   2: Threshold Truncated
	   3: Threshold to Zero
	   4: Threshold to Zero Inverted
	 */

	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}
#endif
#if 0
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;


void thresholdIntegral(cv::Mat &inputMat, cv::Mat &outputMat)
{
	// accept only char type matrices
	CV_Assert(!inputMat.empty());
	CV_Assert(inputMat.depth() == CV_8U);
	CV_Assert(inputMat.channels() == 1);
	CV_Assert(!outputMat.empty());
	CV_Assert(outputMat.depth() == CV_8U);
	CV_Assert(outputMat.channels() == 1);

	// rows -> height -> y
	int nRows = inputMat.rows;
	// cols -> width -> x
	int nCols = inputMat.cols;

	// create the integral image
	cv::Mat sumMat;
	cv::integral(inputMat, sumMat);

	CV_Assert(sumMat.depth() == CV_32S);
	CV_Assert(sizeof(int) == 4);

	int S = MAX(nRows, nCols) / 8;
	double T = 0.15;

	// perform thresholding
	int s2 = S / 2;
	int x1, y1, x2, y2, count, sum;

	// CV_Assert(sizeof(int) == 4);
	int *p_y1, *p_y2;
	uchar *p_inputMat, *p_outputMat;

	for (int i = 0; i < nRows; ++i)
	{
		y1 = i - s2;
		y2 = i + s2;

		if (y1 < 0) {
			y1 = 0;
		}
		if (y2 >= nRows) {
			y2 = nRows - 1;
		}

		p_y1 = sumMat.ptr<int>(y1);
		p_y2 = sumMat.ptr<int>(y2);
		p_inputMat = inputMat.ptr<uchar>(i);
		p_outputMat = outputMat.ptr<uchar>(i);

		for (int j = 0; j < nCols; ++j)
		{
			// set the SxS region
			x1 = j - s2;
			x2 = j + s2;

			if (x1 < 0) {
				x1 = 0;
			}
			if (x2 >= nCols) {
				x2 = nCols - 1;
			}

			count = (x2 - x1)*(y2 - y1);

			// I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
			sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

			if ((int)(p_inputMat[j] * count) < (int)(sum*(1.0 - T)))
				p_outputMat[j] = 255;
			else
				p_outputMat[j] = 0;
		}
	}
}


int main(int argc, char *argv[])
{
	//! [load_image]
		// Load the image
	cv::Mat src = cv::imread("e:/LEFTIMAGE_.tif",0);

	// Check if image is loaded fine
	if (src.empty()) {
		cerr << "Problem loading image!!!" << endl;
		return -1;
	}

	// Show source image
	cv::imshow("src", src);
	//! [load_image]

	//! [gray]
		// Transform source image to gray if it is not
	cv::Mat gray;

	if (src.channels() == 3)
	{
		cv::cvtColor(src, gray, CV_BGR2GRAY);

		// Show gray image
		cv::imshow("gray", gray);
	}
	else
	{
		gray = src;
	}

	cout << "TEST" << endl;

	//! [gray] 

	//! [bin_1]
	cv::Mat bw1;
	cv::adaptiveThreshold(gray, bw1, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

	// Show binary image
	cv::imshow("binary opencv", bw1);
	//! [bin_1] 


	//! [bin_2]
	cv::Mat bw2 = cv::Mat::zeros(gray.size(), CV_8UC1);
	thresholdIntegral(gray, bw2);

	// Show binary image
	cv::imshow("binary integral", bw2);
	//! [bin_2] 

	cv::waitKey(0);
	return 0;
}
#endif

#if 0
#include <cstdio>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Curves.hpp"

using namespace std;
using namespace cv;

static string window_name = "Photo";
static Mat src;
static Mat dst;
static string curves_window = "Adjust Curves";
static Mat curves_mat;
static int channel = 0;
Curves  curves;

static void invalidate()
{
	curves.draw(curves_mat);
	imshow(curves_window, curves_mat);

	
	curves.adjust(src, dst);
	imshow(window_name, dst);

	int y, x;
	uchar *p;

	y = 150; x = 50;
	p = dst.ptr<uchar>(y) + x * 3;
	cout << "(" << int(p[2]) << ", " << int(p[1]) << ", " << int(p[0]) << ")  ";

	y = 150; x = 220;
	p = dst.ptr<uchar>(y) + x * 3;
	cout << "(" << int(p[2]) << ", " << int(p[1]) << ", " << int(p[0]) << ")  ";

	y = 150; x = 400;
	p = dst.ptr<uchar>(y) + x * 3;
	cout << "(" << int(p[2]) << ", " << int(p[1]) << ", " << int(p[0]) << ")  " << endl;
}

static void callbackAdjustChannel(int, void *)
{
	switch (channel) {
	case 3:
		curves.CurrentChannel = &curves.BlueChannel;
		break;
	case 2:
		curves.CurrentChannel = &curves.GreenChannel;
		break;
	case 1:
		curves.CurrentChannel = &curves.RedChannel;
		break;
	default:
		curves.CurrentChannel = &curves.RGBChannel;
		break;
	}


	invalidate();
}

static void callbackMouseEvent(int mouseEvent, int x, int y, int flags, void* param)
{
	switch (mouseEvent) {
	case CV_EVENT_LBUTTONDOWN:
		curves.mouseDown(x, y);
		invalidate();
		break;
	case CV_EVENT_MOUSEMOVE:
		if (curves.mouseMove(x, y))
			invalidate();
		break;
	case CV_EVENT_LBUTTONUP:
		curves.mouseUp(x, y);
		invalidate();
		break;
	}
	return;
}


int main()
{
	//read image file
	src = imread("e:/LEFTIMAGE_.tif",0);
	if (!src.data) {
		cout << "error read image" << endl;
		return -1;
	}
	cvtColor(src, src, cv::COLOR_GRAY2RGB, 3);
	//create window
	namedWindow(window_name);
	imshow(window_name, src);

	//create Mat for curves
	curves_mat = Mat::ones(256, 256, CV_8UC3);

	//create window for curves
	namedWindow(curves_window);
	setMouseCallback(curves_window, callbackMouseEvent, NULL);
	createTrackbar("Channel", curves_window, &channel, 3, callbackAdjustChannel);


	// 范例：用程序代码在Red通道中定义一条曲线
	//	curves.RedChannel.clearPoints();
	//	curves.RedChannel.addPoint( Point(10,  10) );
	//	curves.RedChannel.addPoint( Point(240, 240) );
	//	curves.RedChannel.addPoint( Point(127, 127) );

	invalidate();


	waitKey();
	cvtColor(src, src, cv::COLOR_RGB2GRAY, 1);
	imwrite("Histogram_equalized.tif", dst);
	return 0;

}
#endif

#if 0
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void imhist(Mat image, int histogram[])
{


	for (int i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}


	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
			histogram[(int)image.at<uchar>(y, x)]++;

}

void cumhist(int histogram[], int cumhistogram[])
{
	cumhistogram[0] = histogram[0];

	for (int i = 1; i < 256; i++)
	{
		cumhistogram[i] = histogram[i] + cumhistogram[i - 1];
	}
}


void cumgoshist(float histogram[], float cumhistogram[])
{
	cumhistogram[0] = histogram[0];

	for (int i = 1; i < 256; i++)
	{
		cumhistogram[i] = histogram[i] + cumhistogram[i - 1];
	}
}

void histDisplay(int histogram[], const char* name)
{
	int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));


	int max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}

	for (int i = 0; i < 256; i++) {
		hist[i] = ((double)hist[i] / max)*histImage.rows;
	}



	for (int i = 0; i < 256; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h),
			Point(bin_w*(i), hist_h - hist[i]),
			Scalar(0, 0, 0), 1, 8, 0);
	}

	namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, histImage);
}

void histDis(float histogram[], const char* name)
{
	float hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));


	float max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}



	for (int i = 0; i < 256; i++) {
		hist[i] = ((double)hist[i] / max)*histImage.rows;
	}



	for (int i = 0; i < 256; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h),
			Point(bin_w*(i), hist_h - hist[i]),
			Scalar(0, 0, 0), 1, 8, 0);
	}


	namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, histImage);
}

int main()
{

	Mat image = imread("e:/LEFTIMAGE_.tif", 0);


	int histogram[256];
	imhist(image, histogram);


	int size = image.rows * image.cols;
	float alpha = 255.0 / size;


	float PrRk[256];
	for (int i = 0; i < 256; i++)
	{
		PrRk[i] = (double)histogram[i] / size;
	}


	int cumhistogram[256];
	float cumgos[256];
	cumhist(histogram, cumhistogram);


	int Sk[256];
	for (int i = 0; i < 256; i++)
	{
		Sk[i] = cvRound((double)cumhistogram[i] * alpha);
	}


	float gos[256];
	float sigma;
	int median;
	cout << "Enter value of sigma : " << endl;
	cin >> sigma;
	cout << "Enter value of median : " << endl;
	cin >> median;


	for (int i = -median; i < 255 - median; i++)
	{
		float value = (1 / sqrt(2 * 3.1416)*sigma)*exp(-(pow(i, 2) / (2 * pow(sigma, 2))));
		gos[i + median] = value;
	}

	histDis(gos, "Gaussian Histogram");
	cumgoshist(gos, cumgos);
	float Gz[256];
	for (int i = 0; i < 256; i++)
	{
		Gz[i] = cvRound((double)cumgos[i] * alpha);
	}

	Mat new_image = image.clone();

	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			if (Sk[y] == Gz[x] || abs(Sk[y] - Gz[x]) == 1)
			{
				Sk[y] = x;
				break;
			}
		}
	}

	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
			new_image.at<uchar>(y, x) = saturate_cast<uchar>(Sk[image.at<uchar>(y, x)]);


	float PsSk[256];
	for (int i = 0; i < 256; i++)
	{
		PsSk[i] = 0;
	}

	for (int i = 0; i < 256; i++)
	{
		PsSk[Sk[i]] += PrRk[i];
	}

	int final[256];
	for (int i = 0; i < 256; i++)
		final[i] = cvRound(PsSk[i] * 255);


	//namedWindow("Original Image");
	//imshow("Original Image", image);

	//histDisplay(histogram, "Original Histogram");



	namedWindow("Equilized Image");
	imshow("Equilized Image", new_image);

	//histDisplay(final, "Equilized Histogram");

	waitKey();
	return 0;
}
#endif



# if 0
#include <cstdio>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "Levels.hpp"

using namespace std;
using namespace cv;

static string window_name = "Photo";
static Mat src;

static Mat levels_mat;
static string levels_window = "Adjust Levels";
static int channel = 0;
Levels  levels;

int   Shadow;
int   Midtones = 100;
int   Highlight;
int   OutputShadow;
int   OutputHighlight;
Mat dst;
static void invalidate()
{
	
	levels.adjust(src, dst);
	imshow(window_name, dst);

	imshow(levels_window, levels_mat);
}

static void channelRead(int which_channel)
{
	channel = which_channel;
	Level * CurrentChannel = NULL;
	switch (channel) {
	case 0: CurrentChannel = &levels.RGBChannel; break;
	case 1: CurrentChannel = &levels.RedChannel; break;
	case 2: CurrentChannel = &levels.GreenChannel; break;
	case 3: CurrentChannel = &levels.BlueChannel; break;
	}
	if (CurrentChannel == NULL) return;

	Shadow = CurrentChannel->Shadow;
	Midtones = int(CurrentChannel->Midtones * 100);
	Highlight = CurrentChannel->Highlight;
	OutputShadow = CurrentChannel->OutputShadow;
	OutputHighlight = CurrentChannel->OutputHighlight;

}

static void channelWrite()
{
	Level * CurrentChannel = NULL;
	switch (channel) {
	case 0: CurrentChannel = &levels.RGBChannel; break;
	case 1: CurrentChannel = &levels.RedChannel; break;
	case 2: CurrentChannel = &levels.GreenChannel; break;
	case 3: CurrentChannel = &levels.BlueChannel; break;
	}

	if (CurrentChannel == NULL)
		return;

	CurrentChannel->Shadow = Shadow;
	CurrentChannel->Midtones = Midtones / 100.0;
	CurrentChannel->Highlight = Highlight;
	CurrentChannel->OutputShadow = OutputShadow;
	CurrentChannel->OutputHighlight = OutputHighlight;

	invalidate();
}


static void callbackAdjust(int, void *)
{
	channelWrite();
	invalidate();
}


static void callbackAdjustChannel(int, void *)
{
	channelRead(channel);
	setTrackbarPos("Shadow", levels_window, Shadow);
	setTrackbarPos("Midtones", levels_window, Midtones);
	setTrackbarPos("Highlight", levels_window, Highlight);
	setTrackbarPos("OutShadow", levels_window, OutputShadow);
	setTrackbarPos("OutHighlight", levels_window, OutputHighlight);
	invalidate();
}


int main()
{
	//read image file
	src = imread("e:/LEFTIMAGE_.tif", CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
	src.convertTo(src, CV_8U, 0.00390625);
	cvtColor(src, src, CV_GRAY2RGB);
	if (!src.data) {
		cout << "error read image" << endl;
		return -1;
	}

	//create window
	namedWindow(window_name);
	imshow(window_name, src);


	//create window for levels
	namedWindow(levels_window);
	levels_mat = Mat::ones(100, 400, CV_8UC3);
	levels_mat.setTo(Scalar(255, 255, 255));
	imshow(levels_window, levels_mat);

	channelRead(0);
	createTrackbar("Channel", levels_window, &channel, 3, callbackAdjustChannel);
	createTrackbar("Shadow", levels_window, &Shadow, 255, callbackAdjust);
	createTrackbar("Midtones", levels_window, &Midtones, 200, callbackAdjust);
	createTrackbar("Highlight", levels_window, &Highlight, 255, callbackAdjust);
	createTrackbar("OutShadow", levels_window, &OutputShadow, 255, callbackAdjust);
	createTrackbar("OutHighlight", levels_window, &OutputHighlight, 255, callbackAdjust);

	waitKey();
	imwrite("des.tif", dst);
	return 0;

}
#endif

#if 0

#include <corecrt_math_defines.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


#define SWAP(a, b, t)  do { t = a; a = b; b = t; } while(0)
#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )
#define COLOR_RANGE(value)  CLIP_RANGE(value, 0, 255)

/**
 * Adjust Brightness and Contrast
 *
 * @param src [in] InputArray
 * @param dst [out] OutputArray
 * @param brightness [in] integer, value range [-255, 255]
 * @param contrast [in] integer, value range [-255, 255]
 *
 * @return 0 if success, else return error code
 */
int adjustBrightnessContrast(InputArray src, OutputArray dst, int brightness, int contrast)
{
	Mat input = src.getMat();
	if (input.empty()) {
		return -1;
	}

	dst.create(src.size(), src.type());
	Mat output = dst.getMat();

	brightness = CLIP_RANGE(brightness, -255, 255);
	contrast = CLIP_RANGE(contrast, -255, 255);

	/**
	Algorithm of Brightness Contrast transformation
	The formula is:
		y = [x - 127.5 * (1 - B)] * k + 127.5 * (1 + B);

		x is the input pixel value
		y is the output pixel value
		B is brightness, value range is [-1,1]
		k is used to adjust contrast
			k = tan( (45 + 44 * c) / 180 * PI );
			c is contrast, value range is [-1,1]
	*/

	double B = brightness / 255.;
	double c = contrast / 255.;
	double k = tan((45 + 44 * c) / 180 * M_PI);

	Mat lookupTable(1, 256, CV_16U);
	uchar *p = lookupTable.data;
	for (int i = 0; i < 256; i++)
		p[i] = COLOR_RANGE((i - 127.5 * (1 - B)) * k + 127.5 * (1 + B));

	LUT(input, lookupTable, output);

	return 0;
}


//=====主程序开始====

static string window_name = "photo";
static Mat src;
static int brightness = 255;
static int contrast = 255;

static void callbackAdjust(int, void *)
{
	Mat dst;
	adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);
	imshow(window_name, dst);
}


int main()
{
	src = imread("e:/LEFTIMAGE_.tif", CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);

	if (!src.data) {
		cout << "error read image" << endl;
		return -1;
	}

	namedWindow(window_name);
	createTrackbar("brightness", window_name, &brightness, 2 * brightness, callbackAdjust);
	createTrackbar("contrast", window_name, &contrast, 2 * contrast, callbackAdjust);
	callbackAdjust(0, 0);

	waitKey();

	return 0;

}
#endif
#if 0

Mat& ScanImageAndReduceC_16UC1(Mat& I, const unsigned short* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	unsigned short* p = (unsigned short*)I.data;
	for (unsigned int i = 0; i < nCols*nRows; ++i)
		*p++ = table[*p];

	return I;
}

int main()
{
	Size Img_Size(320, 240);
	Mat Img_Source_16 = imread("e:/LEFTIMAGE_.tif", CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
	Mat Img_Destination_16;

	unsigned short LookupTable[4096];
	for (int i = 0; i < 4096; i++)
	{
		LookupTable[i] = 4096 - i;
	}

	
	imshow("Img_Source", Img_Source_16);

	
	Img_Destination_16 = ScanImageAndReduceC_16UC1(Img_Source_16.clone(), LookupTable);

	imshow("Img_Destination", Img_Destination_16);
	waitKey();
	return 0;
}

#endif

#if 0
#include "opencv2\opencv.hpp"
#include <iostream>    
#include <stdio.h> 
using namespace std;
using namespace cv;

Mat filter2d(Mat picture, int select) {

	//均值滤波器
	if (select == 1) {
		int div;
		cout << "输入均值滤波器的维数:";
		cin >> div;
		int **averageFilter = new int*[div];//均值滤波器
		for (int i = 0; i < div; i++) {
			averageFilter[i] = new int[div];
		}
		//将均值滤波器每个值初始化为1
		for (int i = 0; i < div; i++) {
			for (int j = 0; j < div; j++) {
				averageFilter[i][j] = 1;
			}
		}
		int row = picture.rows;//原图片行数
		int col = picture.cols;//原图片列数
		int **picPixel = new int*[row];//原图片像素矩阵
		for (int m = 0; m < row; m++) {
			picPixel[m] = new int[col];
		}
		//获取原图片每一点像素值
		for (int n = 0; n < row; n++) {
			for (int z = 0; z < col; z++) {
				picPixel[n][z] = int(picture.at<Vec3b>(n, z)[0]);
			}
		}
		int extendRow = row + div * 2 - 2;//补0后的矩阵行数
		int extendCol = col + div * 2 - 2;//补0后的矩阵列数
		int **extendMatrix = new int*[extendRow];//补0后的矩阵,且初始化为0
		for (int i = 0; i < extendRow; i++) {
			extendMatrix[i] = new int[extendCol];
		}
		//将原图片矩阵放入补0后的矩阵
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				extendMatrix[i + div - 1][j + div - 1] = picPixel[i][j];
			}
		}
		//补边界,否则会出现边缘过亮
		for (int i = 0; i < extendRow; i++) {
			for (int j = 0; j < extendCol; j++) {
				if ((i < div - 1 && j <= col + div - 1) || (i <= row + div - 1 && j < div - 1)) {
					extendMatrix[i][j] = picPixel[0][0];
				}
				else if ((i > row + div - 1 && j <= col + div - 1) || j > col + div - 1) {
					extendMatrix[i][j] = picPixel[row - 1][col - 1];
				}
				continue;
			}
		}
		int a = (div - 1) / 2;
		int b = (div - 1) / 2;
		Mat newPicture = Mat(row, col, CV_8UC3);//目标图片
		//均值滤波
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int grey = 0;
				for (int k = -a; k <= a; k++) {
					for (int l = -b; l <= b; l++) {
						grey += extendMatrix[i + div - 1 + k][j + div - 1 + l] * averageFilter[k + a][l + b];//相关操作
					}
				}
				grey = grey / (div * div);
				if (i == 0) {
					cout << grey << endl;
				}
				newPicture.at<Vec3b>(i, j) = Vec3b(grey, grey, grey);
			}
		}
		return newPicture;
	}

	//拉普拉斯滤波器
	if (select == 2) {
		int filter[3][3] = { 0,1,0,
							 1,-4,1,
							 0,1,0 };
		int row = picture.rows;//原图片行数
		int col = picture.cols;//原图片列数
		int **picPixel = new int*[row];//原图片像素矩阵
		for (int m = 0; m < row; m++) {
			picPixel[m] = new int[col];
		}
		//获取原图片每一点像素值
		for (int n = 0; n < row; n++) {
			for (int z = 0; z < col; z++) {
				picPixel[n][z] = int(picture.at<Vec3b>(n, z)[0]);
			}
		}
		int extendrow = row + 3 * 2 - 2;//补0后的矩阵行数
		int extendcol = col + 3 * 2 - 2;//补0后的矩阵列数
		int **extendMatrix = new int*[extendrow];//补0后的矩阵,且初始化为0
		for (int i = 0; i < extendrow; i++) {
			extendMatrix[i] = new int[extendcol];
		}
		//将原图片矩阵放入补0后的矩阵
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				extendMatrix[i + 3 - 1][j + 3 - 1] = picPixel[i][j];
			}
		}
		//补边界,否则会出现边缘过亮
		for (int i = 0; i < extendrow; i++) {
			for (int j = 0; j < extendcol; j++) {
				if ((i < 3 - 1 && j <= col + 3 - 1) || (i <= row + 3 - 1 && j < 3 - 1)) {
					extendMatrix[i][j] = picPixel[0][0];
				}
				else if ((i > row + 3 - 1 && j <= col + 3 - 1) || j > col + 3 - 1) {
					extendMatrix[i][j] = picPixel[row - 1][col - 1];
				}
				continue;
			}
		}
		int a = (3 - 1) / 2;
		int b = (3 - 1) / 2;
		Mat newPicture = Mat(row, col, CV_8UC3);//目标图片
		//拉普拉斯滤波
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int grey = 0;
				for (int k = -a; k <= a; k++) {
					for (int l = -b; l <= b; l++) {
						grey += extendMatrix[i + 3 - 1 + k][j + 3 - 1 + l] * filter[k + a][l + b];
					}
				}
				grey = grey + picPixel[i][j];
				newPicture.at<Vec3b>(i, j) = Vec3b(grey, grey, grey);
			}
		}
		return newPicture;
	}

	//高提升滤波
	if (select == 3) {
		int div = 3;
		int **averageFilter = new int*[div];//均值滤波器
		for (int i = 0; i < div; i++) {
			averageFilter[i] = new int[div];
		}
		//将均值滤波器每个值初始化为1
		for (int i = 0; i < div; i++) {
			for (int j = 0; j < div; j++) {
				averageFilter[i][j] = 1;
			}
		}
		int row = picture.rows;//原图片行数
		int col = picture.cols;//原图片列数
		int **picPixel = new int*[row];//原图片像素矩阵
		for (int m = 0; m < row; m++) {
			picPixel[m] = new int[col];
		}
		//获取原图片每一点像素值
		for (int n = 0; n < row; n++) {
			for (int z = 0; z < col; z++) {
				picPixel[n][z] = int(picture.at<Vec3b>(n, z)[0]);
			}
		}
		int extendRow = row + div * 2 - 2;//补0后的矩阵行数
		int extendCol = col + div * 2 - 2;//补0后的矩阵列数
		int **extendMatrix = new int*[extendRow];//补0后的矩阵,且初始化为0
		for (int i = 0; i < extendRow; i++) {
			extendMatrix[i] = new int[extendCol];
		}
		//将原图片矩阵放入补0后的矩阵
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				extendMatrix[i + div - 1][j + div - 1] = picPixel[i][j];
			}
		}
		//补边界,否则会出现边缘过亮
		for (int i = 0; i < extendRow; i++) {
			for (int j = 0; j < extendCol; j++) {
				if ((i < div - 1 && j <= col + div - 1) || (i <= row + div - 1 && j < div - 1)) {
					extendMatrix[i][j] = picPixel[0][0];
				}
				else if ((i > row + div - 1 && j <= col + div - 1) || j > col + div - 1) {
					extendMatrix[i][j] = picPixel[row - 1][col - 1];
				}
				continue;
			}
		}
		int a = (div - 1) / 2;
		int b = (div - 1) / 2;
		Mat newPicture = Mat(row, col, CV_8UC3);//目标图片
		//高提升滤波
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int grey = 0;
				for (int k = -a; k <= a; k++) {
					for (int l = -b; l <= b; l++) {
						grey += extendMatrix[i + div - 1 + k][j + div - 1 + l] * averageFilter[k + a][l + b];
					}
				}
				grey = grey / (div * div);//平滑后的图
				grey = picPixel[i][j] - grey;//得到细节gmask(x,y)
				grey = picPixel[i][j] + 2 * grey;//细节增强g(x,y)，k取2
				//防止越界
				if (grey > 255) {
					grey = 255;
				}
				if (grey < 0) {
					grey = 0;
				}
				newPicture.at<Vec3b>(i, j) = Vec3b(grey, grey, grey);
			}
		}
		return newPicture;
	}
}
int main() {
	Mat picture = imread("e:/LEFTIMAGE_.tif");
	cout << "Please enter 1---均值滤波器   2---拉普拉斯滤波器   3---高提升滤波：";
	int select;
	cin >> select;
	Mat newPicture = filter2d(picture, select);

	imshow("newPicture", newPicture);
	waitKey(0);
}
#endif
#if 0
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath> 
#include <corecrt_math_defines.h>
// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cout;
using std::endl;
using std::vector;
using namespace cv;

namespace
{
	/** Global Variables */
	int alpha = 100;
	int beta = 100;
	int gamma_cor = 100;
	Mat img_original, img_corrected, img_gamma_corrected;

	void basicLinearTransform(const Mat &img, const double alpha_, const int beta_)
	{
		Mat new_image = Mat::zeros(img.size(), img.type());
		/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					new_image.at<Vec3b>(y, x)[c] =
						saturate_cast<uchar>(alpha*(img.at<Vec3b>(y, x)[c]) + beta);
				}
			}
		}
		imshow("Brightness and contrast adjustments", new_image);
	}
	Mat& ScanImageAndReduceC_16UC1(Mat& I, const unsigned short* const table)
	{
		// accept only char type matrices
		CV_Assert(I.depth() != sizeof(uchar));

		int channels = I.channels();

		int nRows = I.rows;
		int nCols = I.cols * channels;

		if (I.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}

		int i, j;
		unsigned short* p = (unsigned short*)I.data;
		for (unsigned int i = 0; i < nCols*nRows; ++i)
			*p++ = table[*p];

		return I;
	}
	void histEqualize(Mat &src, Mat &dst)
	{
		int gray[256]{ 0 }; //Ex: gray[0] = gray value 0 appaer times, gray[1] = gray value 1 appaer times, ...
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) { //count each gray value appaer times
				gray[(int)src.at<uchar>(y, x)]++;
			}
		}

		int mincdf; //Minimum of cdf

		for (int i = 0; i < 255; i++) { //calculate cdf (Cumulative distribution function)
			gray[i + 1] += gray[i];
		}

		for (int i = 0; i < 255; i++) { //find minimum of cdf
			if (gray[i] != 0) {
				mincdf = gray[i];
				break;
			}
		}

		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				// h(v) = round(((cdf(v) - mincdf) / (M * N) - mincdf) * (L - 1)) ; L = 2^8
				dst.at<uchar>(y, x) = (uchar)round((((double)gray[(int)src.at<uchar>(y, x)] - mincdf) / (double)(src.rows * src.cols - mincdf)) * (double)255);
			}
		}
	}
	void BrightnessAndContrastAuto(const Mat &src, Mat &dst, float clipHistPercent)
	{
		CV_Assert(clipHistPercent >= 0);
		CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

		int histSize = 256;
		float alpha, beta;
		double minGray = 0, maxGray = 0;

		//to calculate grayscale histogram
		Mat gray;
		if (src.type() == CV_8UC1) gray = src;
		else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
		else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
		if (clipHistPercent == 0)
		{
			// keep full available range
			minMaxLoc(gray, &minGray, &maxGray);
		}
		else
		{
			Mat hist; //the grayscale histogram

			float range[] = { 0, 256 };
			const float* histRange = { range };
			bool uniform = true;
			bool accumulate = false;
			calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

			// calculate cumulative distribution from the histogram
			vector<float> accumulator(histSize);
			accumulator[0] = hist.at<float>(0);
			for (int i = 1; i < histSize; i++)
			{
				accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
			}

			// locate points that cuts at required value
			float max = accumulator.back();
			clipHistPercent *= (max / 100.0); //make percent as absolute
			clipHistPercent /= 2.0; // left and right wings
									// locate left cut
			minGray = 0;
			while (accumulator[minGray] < clipHistPercent)
				minGray++;

			// locate right cut
			maxGray = histSize - 1;
			while (accumulator[maxGray] >= (max - clipHistPercent))
				maxGray--;
		}

		// current range
		float inputRange = maxGray - minGray;

		alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
		beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

											 // Apply brightness and contrast normalization
											 // convertTo operates with saurate_cast
		src.convertTo(dst, -1, alpha, beta);

		// restore alpha channel from source 
		if (dst.type() == CV_8UC4)
		{
			int from_to[] = { 3, 3 };
			mixChannels(&src, 4, &dst, 1, from_to, 1);
		}
		return;
	}
#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )
#define COLOR_RANGE(value)  CLIP_RANGE(value, 0, 255)

	/**
	 * Adjust Brightness and Contrast
	 *
	 * @param src [in] InputArray
	 * @param dst [out] OutputArray
	 * @param brightness [in] integer, value range [-255, 255]
	 * @param contrast [in] integer, value range [-255, 255]
	 *
	 * @return 0 if success, else return error code
	 */
	int adjustBrightnessContrast(InputArray src, OutputArray dst, int brightness, int contrast)
	{
		Mat input = src.getMat();
		if (input.empty()) {
			return -1;
		}

		dst.create(src.size(), src.type());
		Mat output = dst.getMat();

		brightness = CLIP_RANGE(brightness, -255, 255);
		contrast = CLIP_RANGE(contrast, -255, 255);

		/**
		Algorithm of Brightness Contrast transformation
		The formula is:
			y = [x - 127.5 * (1 - B)] * k + 127.5 * (1 + B);

			x is the input pixel value
			y is the output pixel value
			B is brightness, value range is [-1,1]
			k is used to adjust contrast
				k = tan( (45 + 44 * c) / 180 * PI );
				c is contrast, value range is [-1,1]
		*/

		double B = brightness / 255.;
		double c = contrast / 255.;
		double k = tan((45 + 44 * c) / 180 * M_PI);

		Mat lookupTable(1, 256, CV_8U);
		uchar *p = lookupTable.data;
		for (int i = 0; i < 256; i++)
			p[i] = COLOR_RANGE((i - 127.5 * (1 - B)) * k + 127.5 * (1 + B));

		LUT(input, lookupTable, output);

		return 0;
	}
	void gammaCorrection(const Mat &img, const double gamma_)
	{
		CV_Assert(gamma_ >= 0);
		Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		for (int i = 0; i < 256; ++i)
			p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

		Mat res = img.clone();
		LUT(img, lookUpTable, res);
		
		
		imshow("Gamma correction", res);
	}

	void on_linear_transform_alpha_trackbar(int, void *)
	{
		double alpha_value = alpha / 100.0;
		int beta_value = beta - 100;
		basicLinearTransform(img_original, alpha_value, beta_value);
	}

	void on_linear_transform_beta_trackbar(int, void *)
	{
		double alpha_value = alpha / 100.0;
		int beta_value = beta - 100;
		basicLinearTransform(img_original, alpha_value, beta_value);
	}

	void on_gamma_correction_trackbar(int, void *)
	{
		double gamma_value = gamma_cor / 100.0;
		gammaCorrection(img_original, gamma_value);
	}
}
void RawToMat(const char filename[], cv::Mat& dst, int width = 1024, int height = 1024)
{
	size_t nsize = width * height;
	unsigned short *data = new unsigned short[nsize];
	if (data == NULL)
	{
		std::cout << "data space malloc failed" << std::endl;
	}
	FILE *file;
	//	fopen_s(&file, fileName, "rb+");
	file = fopen(filename, "rb+");
	fread(data, sizeof(unsigned short), nsize, file);
	fclose(file);
	cv::Mat temp(height, width, CV_16UC1, data);//单通道的Mat raw数据
	cv::Mat mtep[3];
	temp.copyTo(mtep[0]);
	temp.copyTo(mtep[1]);
	temp.copyTo(mtep[2]);

	cv::Mat mergeM(height, width, CV_16UC3);
	cv::merge(mtep, 3, mergeM);
	dst = mergeM;
	//mergeM.convertTo(dst, CV_32FC3);
	delete[] data;
	return;
}
int main(int argc, char** argv)
{
	RawToMat("f:\\wwww_123－222.raw", img_original, 2560, 2304);
	//img_original = imread("f:/220px-Unsharp_mask_principle.svg.png");
	//normalize(img_original, img_original, 1, 65535, NORM_MINMAX);
	img_original.convertTo(img_original, CV_8U, 1.0 / 256.0);
	if (img_original.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}

	//img_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());
	//img_gamma_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());

	//hconcat(img_original, img_original, img_corrected);
	//hconcat(img_original, img_original, img_gamma_corrected);

	namedWindow("Brightness and contrast adjustments");
	namedWindow("Gamma correction");

	createTrackbar("Alpha gain (contrast)", "Brightness and contrast adjustments", &alpha, 3, on_linear_transform_alpha_trackbar);
	createTrackbar("Beta bias (brightness)", "Brightness and contrast adjustments", &beta, 100, on_linear_transform_beta_trackbar);
	createTrackbar("Gamma correction", "Gamma correction", &gamma_cor, 200, on_gamma_correction_trackbar);

	on_linear_transform_alpha_trackbar(0, 0);
	on_gamma_correction_trackbar(0, 0);

	waitKey();

	imwrite("linear_transform_correction.png", img_corrected);
	imwrite("gamma_correction.png", img_gamma_corrected);

	return 0;
}
#endif
#if 0
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables
Mat src, dst;

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;

char* window_name = "Morphology Transformations Demo";

/** Function Headers */
void Morphology_Operations(int, void*);

/** @function main */
int main(int argc, char** argv)
{
	/// Load an image
	src = imread("e:/ADE FILTER IMAGE_.tif");

	if (!src.data)
	{
		return -1;
	}

	/// Create window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to select Morphology operation
	createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations);

	/// Create Trackbar to select kernel type
	createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
		&morph_elem, max_elem,
		Morphology_Operations);

	/// Create Trackbar to choose kernel size
	createTrackbar("Kernel size:\n 2n +1", window_name,
		&morph_size, max_kernel_size,
		Morphology_Operations);

	/// Default start
	Morphology_Operations(0, 0);

	waitKey(0);
	return 0;
}

/**
 * @function Morphology_Operations
 */
void Morphology_Operations(int, void*)
{
	// Since MORPH_X : 2,3,4,5 and 6
	int operation = morph_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(src, dst, operation, element);
	imshow(window_name, dst);
}
#endif
#if 0
#include<iostream>
#include "opencv2\opencv.hpp"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmdata/dcrledrg.h"
#include "dcmtk/dcmimage/diregist.h"

using namespace std;
using namespace cv;

#pragma comment(lib, "Netapi32.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "Iphlpapi.lib")
int Img_bitCount;

DicomImage* LoadDcmDataSet(std::string file_path)
{

	DcmFileFormat fileformat;
	OFCondition oc = fileformat.loadFile(file_path.c_str());                    //读取Dicom图像    
	if (!oc.good())     //判断Dicom文件是否读取成功    
	{
		std::cout << "file Load error" << std::endl;
		return nullptr;
	}
	DcmDataset *dataset = fileformat.getDataset();                              //得到Dicom的数据集    
	E_TransferSyntax xfer = dataset->getOriginalXfer();                          //得到传输语法    

	OFString patientname;
	dataset->findAndGetOFString(DCM_PatientName, patientname);                   //获取病人姓名    

	unsigned short bit_count(0);
	dataset->findAndGetUint16(DCM_BitsStored, bit_count);                        //获取像素的位数 bit    

	OFString isRGB;
	dataset->findAndGetOFString(DCM_PhotometricInterpretation, isRGB);           //DCM图片的图像模式    

	unsigned short img_bits(0);
	dataset->findAndGetUint16(DCM_SamplesPerPixel, img_bits);                    //单个像素占用多少byte    
	Img_bitCount = (int)img_bits;

	OFString framecount;
	dataset->findAndGetOFString(DCM_NumberOfFrames, framecount);             //DCM图片的帧数    


	//DicomImage* img_xfer = new DicomImage(xfer, 0, 0, 1);                     //由传输语法得到图像的帧    

	unsigned short m_width;                                                     //获取图像的窗宽高    
	unsigned short m_height;
	dataset->findAndGetUint16(DCM_Rows, m_height);
	dataset->findAndGetUint16(DCM_Columns, m_width);

	/////////////////////////////////////////////////////////////////////////    
	const char* transferSyntax = NULL;
	fileformat.getMetaInfo()->findAndGetString(DCM_TransferSyntaxUID, transferSyntax);       //获得传输语法字符串    
	string losslessTransUID = "1.2.840.10008.1.2.4.70";
	string lossTransUID = "1.2.840.10008.1.2.4.51";
	string losslessP14 = "1.2.840.10008.1.2.4.57";
	string lossyP1 = "1.2.840.10008.1.2.4.50";
	string lossyRLE = "1.2.840.10008.1.2.5";
	if (transferSyntax == losslessTransUID || transferSyntax == lossTransUID ||
		transferSyntax == losslessP14 || transferSyntax == lossyP1)
	{
		DJDecoderRegistration::registerCodecs();
		dataset->chooseRepresentation(EXS_LittleEndianExplicit, NULL);                       //对压缩的图像像素进行解压    
		DJDecoderRegistration::cleanup();
	}
	else if (transferSyntax == lossyRLE)
	{
		DcmRLEDecoderRegistration::registerCodecs();
		dataset->chooseRepresentation(EXS_LittleEndianExplicit, NULL);
		DcmRLEDecoderRegistration::cleanup();
	}
	else
	{
		dataset->chooseRepresentation(xfer, NULL);
	}

	DicomImage* m_dcmImage = new DicomImage((DcmObject*)dataset, xfer); //利用dataset生成DicomImage，需要上面的解压方法；    

	return m_dcmImage;
}

std::vector<cv::Mat> GetImageFromDcmDataSet(DicomImage* m_dcmImage)
{
	std::vector<cv::Mat> output_img;              //输出图像向量  
	int framecount(m_dcmImage->getFrameCount()); //获取这个文件包含的图像的帧数  
	for (int k = 0; k < framecount; k++)
	{
		unsigned char *pixelData = (unsigned char*)(m_dcmImage->getOutputData(8, k, 0)); //获得8位的图像数据指针    
		if (pixelData != NULL)
		{
			int m_height = m_dcmImage->getHeight();
			int m_width = m_dcmImage->getWidth();
			cout << "高度：" << m_height << "，长度" << m_width << endl;
			if (3 == Img_bitCount)
			{
				cv::Mat dst2(m_height, m_width, CV_8UC3, cv::Scalar::all(0));
				for (int i = 0; i < m_height; i++)
				{
					for (int j = 0; j < m_width; j++)
					{
						dst2.at<cv::Vec3b>(i, j)[0] = *(pixelData + i * m_width * 3 + j * 3 + 2);   //B channel    
						dst2.at<cv::Vec3b>(i, j)[1] = *(pixelData + i * m_width * 3 + j * 3 + 1);   //G channel    
						dst2.at<cv::Vec3b>(i, j)[2] = *(pixelData + i * m_width * 3 + j * 3);       //R channel    
					}
				}
				output_img.push_back(dst2);
			}
			else if (1 == Img_bitCount)
			{
				cv::Mat dst2(m_height, m_width, CV_8UC1, cv::Scalar::all(0));
				uchar* data = nullptr;
				for (int i = 0; i < m_height; i++)
				{
					data = dst2.ptr<uchar>(i);
					for (int j = 0; j < m_width; j++)
					{
						data[j] = *(pixelData + i * m_width + j);
					}
				}
				output_img.push_back(dst2);
			}

			/*cv::imshow("image", dst2);
			cv::waitKey(0);*/
		}
	}

	return output_img;
}

int main()
{


	string filepath = "f:\\dcm.dcm";
	DicomImage *m_dcmImage;
	m_dcmImage = LoadDcmDataSet(filepath);

	vector<cv::Mat> images;
	images = GetImageFromDcmDataSet(m_dcmImage);
	cout << images.size() << endl;
	imshow("image", images[0]);
	cv::waitKey(0);
	
	return 0;
}

#endif
#if 0
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
void RawToMat(const char filename[], cv::Mat& dst, int width = 1024, int height = 1024)
{
	size_t nsize = width * height;
	unsigned short *data = new unsigned short[nsize];
	if (data == NULL)
	{
		std::cout << "data space malloc failed" << std::endl;
	}
	FILE *file;
	//	fopen_s(&file, fileName, "rb+");
	file = fopen(filename, "rb+");
	fread(data, sizeof(unsigned short), nsize, file);
	fclose(file);
	cv::Mat temp(height, width, CV_16UC1, data);//单通道的Mat raw数据
	cv::Mat mtep[3];
	temp.copyTo(mtep[0]);
	temp.copyTo(mtep[1]);
	temp.copyTo(mtep[2]);

	cv::Mat mergeM(height, width, CV_16UC3);
	cv::merge(mtep, 3, mergeM);
	dst = mergeM;
	//mergeM.convertTo(dst, CV_32FC3);
	delete[] data;
	return;
}
double SpaceFactor(int x1, int y1, int x2, int y2, double sigmaD) {
	double absX = pow(abs(x1 - x2), 2);
	double absY = pow(abs(y1 - y2), 2);

	return exp(-(absX + absY) / (2 * pow(sigmaD, 2)));
}

double ColorFactor(int x, int y, double sigmaR) {
	double distance = abs(x - y) / sigmaR;
	return exp(-0.5 * pow(distance, 2));
}

cv::Mat fastBilateralFilter(cv::Mat inputImg, int filterSize, double sigmaD, double sigmaR) {
	int len; //must be odd number
	cv::Mat gray; // must be 1-channel image
	cv::Mat LabImage; // if channels == 3

	if (filterSize % 2 != 1 || filterSize <= 0) {
		std::cerr << "Filter Size must be a positive odd number!" << std::endl;
		return inputImg;
	}
	len = filterSize / 2;

	if (inputImg.channels() >= 3) {
		cv::cvtColor(inputImg, LabImage, cv::COLOR_BGR2Lab);
		gray = cv::Mat::zeros(LabImage.size(), CV_8UC1);
		for (int i = 0; i < LabImage.rows; i++) {
			for (int j = 0; j < LabImage.cols; j++) {
				gray.ptr<uchar>(i)[j] = LabImage.ptr<uchar>(i, j)[0];
			}
		}
	}
	else if (inputImg.channels() == 1) {
		inputImg.copyTo(gray);
	}
	else {
		std::cerr << "the count of input image's channel can not be 2!" << std::endl;
		return inputImg;
	}

	cv::Mat resultGrayImg = cv::Mat::zeros(gray.size(), CV_8UC1);
	for (int i = 0; i < gray.rows; i++) {
		for (int j = 0; j < gray.cols; j++) {
			double k = 0;
			double f = 0;
			double sum = 0;
			for (int r = i - len; r <= i + len; r++) {
				if (r < 0 || r >= gray.rows)
					continue;
				f = f + gray.ptr<uchar>(r)[j] * SpaceFactor(i, j, r, j, sigmaD) * ColorFactor(gray.ptr<uchar>(i)[j], gray.ptr<uchar>(r)[j], sigmaD);
				k += SpaceFactor(i, j, r, j, sigmaD) * ColorFactor(gray.ptr<uchar>(i)[j], gray.ptr<uchar>(r)[j], sigmaD);
			}
			sum = f / k;
			f = k = 0.0;
			for (int c = j - len; c <= j + len; c++) {
				if (c < 0 || c >= gray.cols)
					continue;
				f = f + gray.ptr<uchar>(i)[c] * SpaceFactor(i, j, i, c, sigmaD) * ColorFactor(gray.ptr<uchar>(i)[j], gray.ptr<uchar>(i)[c], sigmaD);
				k += SpaceFactor(i, j, i, c, sigmaD) * ColorFactor(gray.ptr<uchar>(i)[j], gray.ptr<uchar>(i)[c], sigmaD);
			}
			int value = (sum + f / k) / 2;
			if (value < 0) value = 0;
			else if (value > 255) value = 255;

			resultGrayImg.ptr<uchar>(i)[j] = (uchar)value;
		}
	}

	cv::Mat resultImg;
	if (inputImg.channels() >= 3) {
		for (int i = 0; i < LabImage.rows; i++) {
			for (int j = 0; j < LabImage.cols; j++) {
				LabImage.ptr<uchar>(i, j)[0] = resultGrayImg.ptr<uchar>(i)[j];
			}
		}
		cv::cvtColor(LabImage, resultImg, cv::COLOR_Lab2BGR);
	}
	else {
		resultGrayImg.copyTo(resultImg);
	}

	return resultImg;
}
int main()
{
	cv::Mat images;
	cv::Mat dst;
	int i = 1;
	RawToMat("I:\\33_03\\image_00.raw",images,1536,1920);
	
	try {
		dst=fastBilateralFilter(images, 10, 3,20);
	}
	catch (cv::Exception e){

	}
	double alpha = 400;
	double beta = 100;
	double alpha_value = alpha / 100.0;
	int beta_value = beta - 100;


	dst.convertTo(dst, -1, alpha_value, beta_value);

	imshow("image", dst);
	resizeWindow("image", 600, 600);
	cv::waitKey(0);

	return 0;
}
#endif
#if 1
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int computeOutput(int, int, int, int, int);
void RawToMat(const char filename[], cv::Mat& dst, int width = 1024, int height = 1024)
{
	size_t nsize = width * height;
	unsigned short *data = new unsigned short[nsize];
	if (data == NULL)
	{
		std::cout << "data space malloc failed" << std::endl;
	}
	FILE *file;
	//	fopen_s(&file, fileName, "rb+");
	file = fopen(filename, "rb+");
	fread(data, sizeof(unsigned short), nsize, file);
	fclose(file);
	cv::Mat temp(height, width, CV_16UC1, data);//单通道的Mat raw数据
	cv::Mat mtep[3];
	temp.copyTo(mtep[0]);
	temp.copyTo(mtep[1]);
	temp.copyTo(mtep[2]);

	cv::Mat mergeM(height, width, CV_16UC3);
	cv::merge(mtep, 3, mergeM);
	dst = mergeM;
	//mergeM.convertTo(dst, CV_32FC3);
	delete[] data;
	return;
}
double SpaceFactor(int x1, int y1, int x2, int y2, double sigmaD) {
	double absX = pow(abs(x1 - x2), 2);
	double absY = pow(abs(y1 - y2), 2);

	return exp(-(absX + absY) / (2 * pow(sigmaD, 2)));
}

double ColorFactor(int x, int y, double sigmaR) {
	double distance = abs(x - y) / sigmaR;
	return exp(-0.5 * pow(distance, 2));
}
int main()
{

	cv::Mat image;
	cv::Mat dst;
	int i = 1;
	RawToMat("f:\\image_00.raw", image, 1536, 1920);
	

	namedWindow("New Image", 1);
	imshow("New Image", image);
	resizeWindow("New Image", 600, 600);
	waitKey();

	return 0;
}

int computeOutput(int x, int r1, int s1, int r2, int s2)
{


	float result;
	if (0 <= x && x <= r1) {
		result = s1 / r1 * x;
	}
	else if (r1 < x && x <= r2) {
		result = ((s2 - s1) / (r2 - r1)) * (x - r1) + s1;
	}
	else if (r2 < x && x <= 255) {
		result = ((255 - s2) / (255 - r2)) * (x - r2) + s2;
	}
	return (int)result;
}
#endif


#if 0
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int ac, char *av[])
{
	int width = 1536; int height = 1920;
	size_t nsize = width * height;
	unsigned short *data = new unsigned short[nsize];
	unsigned short *tdata = new unsigned short[nsize];
	//memset(tdata, 0, sizeof(unsigned short)*nsize);
	if (data == NULL)
	{
		std::cout << "data space malloc failed" << std::endl;
	}
	FILE *file;
	
	for (int i = 0; i < 8; i++) {
		string filename = "";
		sprintf((char*)filename.c_str(), "f:\\image_0%d.raw", i);
		file = fopen(filename.c_str(), "rb+");
		fread(data, sizeof(unsigned short), nsize, file);
		fclose(file);
		
		for (int j = 0; j < nsize; j++) {
			
			tdata[j] = tdata[j]+data[j];
		}
		
		//data = new unsigned short[nsize];
	}
	for (int j = 0; j < nsize; j++) {
		
		tdata[j] = tdata[j]/8;
	}
	/*file = fopen("merge.raw", "wb");
	fwrite(tdata, sizeof(unsigned short), nsize, file);
	fclose(file);*/
	//cv::Mat test(height, width, CV_16UC1, tdata);//单通道的Mat raw数据
	//Mat imageBGR = test.clone();
	//cvtColor(test, imageBGR, CV_GRAY2BGR555);
	
	cv::Mat temp(height, width, CV_16UC1, tdata);//单通道的Mat raw数据
	cv::Mat mtep[3];
	temp.copyTo(mtep[0]);
	temp.copyTo(mtep[1]);
	temp.copyTo(mtep[2]);

	cv::Mat mergeM(height, width, CV_16UC3);
	cv::merge(mtep, 3, mergeM);
	
	namedWindow("New Image", 1);
	imshow("New Image", mergeM);
	resizeWindow("New Image", 600, 600);
	waitKey();
	return 0;
}
#endif
