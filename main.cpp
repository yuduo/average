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
		1, 1, 1); // another approximation of second derivate, more stronger

				  // do the laplacian filtering as it is
				  // well, we need to convert everything in something more deeper then CV_8U
				  // because the kernel has some negative values, 
				  // and we can expect in general to have a Laplacian image with negative values
				  // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
				  // so the possible negative number will be truncated
	filter2D(img, imgLaplacian, CV_32F, kernel);
	img.convertTo(img, CV_32F);
	imgResult = img - imgLaplacian;
	imgResult = imgResult + 80;

	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8U);
	imgLaplacian.convertTo(imgLaplacian, CV_8U);

	namedWindow("laplacian", CV_WINDOW_AUTOSIZE);
	imshow("laplacian", imgLaplacian);

	namedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("result", imgResult);

	GaussianBlur(imgResult, imgResult, cv::Size(0, 0), 3);
	addWeighted(imgResult, 1.5, imgResult, -0.5, 0, imgResult);
	imshow("result2", imgResult);
	
#endif
	/// Wait until user press some key
	waitKey();
	return 0;
}