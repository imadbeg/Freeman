#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<conio.h>


using namespace cv;
using namespace std;


void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade, double scale);
string cascadeName, nestedCascadeName;

int main(int argc, const char** argv)
{

    int x1, y1;
    int x2, y2;
   // Mat video_stream;//Declaring a matrix hold frames from video stream//
   // VideoCapture real_time(0);//capturing video from default webcam//


	Mat img1 = imread("C:/Users/minim/Desktop/test/test/test/imad1.jpg");
	if (img1.empty())
	{
		cout << "Cannot load image!" << endl;
		return -1;
	}
   // namedWindow("Face Detection");//Declaring an window to show the result//
    string trained_classifier_location = "C:/Users/minim/Downloads/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml";//Defining the location our XML Trained Classifier in a string//
    CascadeClassifier faceDetector;//Declaring an object named 'face detector' of CascadeClassifier class//
    faceDetector.load(trained_classifier_location);//loading the XML trained classifier in the object//
    vector<Rect>faces;//Declaring a rectangular vector named faces//




    for (size_t i = 0; i < 50; i++)
    {

        faceDetector.detectMultiScale(img1, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));//Detecting the faces in 'image_with_humanfaces' matrix//
        //real_time.read(video_stream);// reading frames from camera and loading them in 'video_stream' Matrix//
        for (int i = 0; i < faces.size(); i++) { //for locating the face
            Mat faceROI = img1(faces[i]);//Storing face in the matrix//
            int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
            x1 = x;
            int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
            y1 = y;
            int h = y + faces[i].height;//Calculating the height of the rectangle//
            int w = x + faces[i].width;//Calculating the width of the rectangle//
            //rectangle(video_stream, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the faces//
        }
       // imwrite("screen.png", video_stream);
       // imshow("Face Detection screan ", video_stream);
        //Showing the detected face//
        
    }

	
	Mat img2 = imread("C:/Users/minim/Desktop/test/test/test/imad2.jpg");
	if (img2.empty())
	{
		cout << "Cannot load image!" << endl;
		return -1;
	}



	for (size_t i = 0; i < 50; i++)
	{

		faceDetector.detectMultiScale(img2, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));//Detecting the faces in 'image_with_humanfaces' matrix//
		//real_time.read(video_stream);// reading frames from camera and loading them in 'video_stream' Matrix//
		for (int i = 0; i < faces.size(); i++) { //for locating the face
			Mat faceROI = img2(faces[i]);//Storing face in the matrix//
			int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
			x2 = x;
			int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
			y2 = y;
			int h = y + faces[i].height;//Calculating the height of the rectangle//
			int w = x + faces[i].width;//Calculating the width of the rectangle//
			//rectangle(video_stream, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the faces//
		}
		// imwrite("screen.png", video_stream);
		// imshow("Face Detection screan ", video_stream);
		 //Showing the detected face//

	}











	cvtColor(img1, img1, COLOR_BGR2GRAY);
	imshow("FirstImage", img1);  // Original image1

	cvtColor(img2, img2, COLOR_BGR2GRAY);
	imshow("SecondImage", img2);  // Original image2

	Mat elp1(img1.rows, img1.cols, CV_8UC1, Scalar(0, 0, 0));
	//ellipse(elp1, Point(100, 300), Size(400, 400), 0, 0, 100, Scalar(255, 255, 255), -1, 8);
	  rectangle(elp1, Point(100, 300), Point(400, 400), Scalar(2552, 255, 255), -1, 8);
	//imshow("ellipse1", elp1);  

	Mat elp2(img2.rows, img2.cols, CV_8UC1, Scalar(0, 0, 0));
	//ellipse(elp2, Point(100, 300), Size(400, 400), 0, 0, 100, Scalar(255, 255, 255), -1, 8);
	rectangle(elp1, Point(100, 300), Point(400, 400), Scalar(2552, 255, 255), -1, 8);
	//imshow("ellipse2", elp2); 

	Mat face1;
	bitwise_and(img1, elp1, face1);

	//imshow("Facce1", face1);  //face image
	cv::Mat Nfc1;
	cv::subtract(img1, face1, Nfc1);
	//imshow("NoFace1", Nfc1); //Noface image


	Mat face2;
	bitwise_and(img2, elp2, face2);
	imshow("Face2", face2);  // face image
	cv::Mat Nfc2;
	cv::subtract(img2, face2, Nfc2);
	//imshow("NoFace2", Nfc2); //Noface image




	Size size(450, 800);
	Mat out1, out2; //output images
	Mat resize1, resize3;
	Mat resize2, resize4;

	cv::resize(Nfc2, resize1, size); //resize image
	cv::resize(Nfc1, resize3, size); //resize image
	cv::resize(face1, resize2, size); //resize image
	cv::resize(face2, resize4, size); //resize image

	cv::add(resize1, resize2, out1);
	cv::add(resize3, resize4, out2);

	imshow("output", out1);
	//imshow("reverse", out2);

	waitKey(0);
	return 0;
}




