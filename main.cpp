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
	cv::GaussianBlur(img, img, cv::Size(3, 3), 1, 1);

	cv::imshow("Full", img);

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

#if 1
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


	namedWindow("Original Image");
	imshow("Original Image", image);

	histDisplay(histogram, "Original Histogram");



	namedWindow("Equilized Image");
	imshow("Equilized Image", new_image);

	histDisplay(final, "Equilized Histogram");

	waitKey();
	return 0;
}
#endif