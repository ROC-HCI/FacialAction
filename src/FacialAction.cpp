#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <FaceTracker/Tracker.h>
#include <opencv2\opencv.hpp>
#include <Image.h>
#include <Shore.h>
#include <CreateFaceEngine.h>
#include <ContentToText.h>

Shore::Engine* engine;
// This will be the result;
Shore::Content const* content = 0;

/* 
Face Tracker by Md. Iftekhar Tanveer (go2chayan@gmail.com)
This is built upon Jason M. Saragih's facetracking API which
can be found here: http://web.mac.com/jsaragih/FaceTracker/FaceTracker.html
This original tracker has been modified by Md. Iftekhar Tanveer
for better stability and a specific need.

Stability is provided by applying some engineering tweaking to the basic
facetracker. Then, 10 Facial features are extracted and written in a CSV
file.
*/





#define atd at<double>

cv::Scalar colorSet[12] = {CV_RGB(255,0,0),CV_RGB(0,255,0),CV_RGB(0,0,255),
	CV_RGB(255,250,0),CV_RGB(0,255,250),CV_RGB(250,0,255),
	CV_RGB(255,50,0),CV_RGB(0,255,50),CV_RGB(50,0,255),
	CV_RGB(255,50,155),CV_RGB(155,255,50),CV_RGB(50,155,255),};

std::string FeatureSet[12] = {"Pitch","Yaw  ","Roll ",
	"inBrL","otBrL","inBrR",
	"otBrR","EyeOL","EyeOR",
	"oLipH","iLipH","LipCDt"};

void Writeblank(std::ofstream &ofstm){
	for (int i=0;i<10;i++){

		ofstm<<",";
	}
	ofstm<<std::endl;
}

/* Find the distance of (x1,y1,z1) to a line formed by the
points (x2,y2,z2) and (x3,y3,z3)*/
double DistFromLine(double x1, double y1, double z1,
	double x2, double y2, double z2,
	double x3, double y3, double z3){

		cv::Point3d X0(x1,y1,z1);
		cv::Point3d X1(x2,y2,z2);
		cv::Point3d X2(x3,y3,z3);

		cv::Point3d num = (X0-X1).cross(X0-X2);
		cv::Point3d den = (X2-X1);

		return cv::norm(num)/cv::norm(den);
}

/* Static Function for getting 10 facial features
% Inner Eye brow (Left)
% Outer Eye Brow (Left)
% Inner Eye Brow (Right)
% Outer Eye Brow (Right)
% Eye Opening (Left)
% Eye Opening (Right)
% Height of outer lip boundary
% Height of inner lip boundary
% Mouth corner angle
% Distance between lip corners */
void FeatureList (cv::Mat &TrackedShape, cv::Mat &RefShape, double * FeatureLst){
	// Annotation metadata can be found here
	// https://lh4.googleusercontent.com/-H5d99m0kZmY/Tq3oWDN-WDI/AAAAAAAADGg/nb9__qoesnU/s512/Face-Annotations.png
	// Remember: The point number here = point number in picture - 1


	double temp1=0.0,temp2=0.0; cv::Point2d tempPoint1, tempPoint2;
	// Feature 0 = Inner Eye brow (Left) = Distance of point 22
	// from the line formed by point 42 and 45
	// = abs(y22 + x22*(y45 - y42)/(x42-x45) + x42*(y42-y45)/(x42-x45) - y42)/
	// sqrt(1 + ((y45-y42)/(x42-x45))^2)
	int x = 0,y=66,z=132;
	temp1 = DistFromLine(TrackedShape.atd(x+22),
		TrackedShape.atd(y+22),
		TrackedShape.atd(z+22),
		TrackedShape.atd(x+42),
		TrackedShape.atd(y+42),
		TrackedShape.atd(z+42),
		TrackedShape.atd(x+45),
		TrackedShape.atd(y+45),
		TrackedShape.atd(z+45));
	temp2 = DistFromLine(RefShape.atd(x+22),
		RefShape.atd(y+22),
		RefShape.atd(z+22),
		RefShape.atd(x+42),
		RefShape.atd(y+42),
		RefShape.atd(z+42),
		RefShape.atd(x+45),
		RefShape.atd(y+45),
		RefShape.atd(z+45));
	FeatureLst[0] = temp1/temp2;

	// Feature 1 = Outer Eye brow (Left) = Distance of point 26
	// from the line formed by point 42 and 45
	// = abs(y26 + x26*(y45 - y42)/(x42-x45) + x42*(y42-y45)/(x42-x45) - y42)/
	// sqrt(1 + ((y45-y42)/(x42-x45))^2)
	temp1 = DistFromLine(TrackedShape.atd(x+26),
		TrackedShape.atd(y+26),
		TrackedShape.atd(z+26),
		TrackedShape.atd(x+42),
		TrackedShape.atd(y+42),
		TrackedShape.atd(z+42),
		TrackedShape.atd(x+45),
		TrackedShape.atd(y+45),
		TrackedShape.atd(z+45));
	temp2 = DistFromLine(RefShape.atd(x+26),
		RefShape.atd(y+26),
		RefShape.atd(z+26),
		RefShape.atd(x+42),
		RefShape.atd(y+42),
		RefShape.atd(z+42),
		RefShape.atd(x+45),
		RefShape.atd(y+45),
		RefShape.atd(z+45));
	FeatureLst[1] = temp1/temp2;

	// Feature 2 = Inner Eye brow (Right) = Distance of point 21
	// from the line formed by point 36 and 39
	// = abs(y21 + x21*(y36 - y39)/(x39-x36) + x39*(y43-y36)/(x39-x36) - y39)/
	// sqrt(1 + ((y36-y39)/(x39-x36))^2)
	temp1 = DistFromLine(TrackedShape.atd(x+21),
		TrackedShape.atd(y+21),
		TrackedShape.atd(z+21),
		TrackedShape.atd(x+36),
		TrackedShape.atd(y+36),
		TrackedShape.atd(z+36),
		TrackedShape.atd(x+39),
		TrackedShape.atd(y+39),
		TrackedShape.atd(z+39));
	temp2 = DistFromLine(RefShape.atd(x+21),
		RefShape.atd(y+21),
		RefShape.atd(z+21),
		RefShape.atd(x+36),
		RefShape.atd(y+36),
		RefShape.atd(z+36),
		RefShape.atd(x+39),
		RefShape.atd(y+39),
		RefShape.atd(z+39));
	FeatureLst[2] = temp1/temp2;

	// Feature 3 = Outer Eye brow (Right) = Distance of point 17
	// from the line formed by point 36 and 39
	// = abs(y17 + x17*(y36 - y39)/(x39-x36) + x39*(y43-y36)/(x39-x36) - y39)/
	// sqrt(1 + ((y36-y39)/(x39-x36))^2)
	temp1 = DistFromLine(TrackedShape.atd(x+17),TrackedShape.atd(y+17),TrackedShape.atd(z+17),
		TrackedShape.atd(x+36),TrackedShape.atd(y+36),TrackedShape.atd(z+36),
		TrackedShape.atd(x+39),TrackedShape.atd(y+39),TrackedShape.atd(z+39));
	temp2 = DistFromLine(RefShape.atd(x+17),RefShape.atd(y+17),RefShape.atd(z+17),
		RefShape.atd(x+36),RefShape.atd(y+36),RefShape.atd(z+36),
		RefShape.atd(x+39),RefShape.atd(y+39),RefShape.atd(z+39));
	FeatureLst[3] = temp1/temp2;

	// Feature 4 = Eye Opening (Left) = avg(Distance of point 43 to 47
	// and point 44 to point 46)
	temp1 = 0.5*(cv::norm(cv::Point3d(TrackedShape.atd(x+43),TrackedShape.atd(y+43),TrackedShape.atd(z+43))-
		cv::Point3d(TrackedShape.atd(x+47),TrackedShape.atd(y+47),TrackedShape.atd(z+47)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+44),TrackedShape.atd(y+44),TrackedShape.atd(z+44))-
		cv::Point3d(TrackedShape.atd(x+46),TrackedShape.atd(y+46),TrackedShape.atd(z+46))));

	temp2 = 0.5*(cv::norm(cv::Point3d(RefShape.atd(x+43),RefShape.atd(y+43),RefShape.atd(z+43))-
		cv::Point3d(RefShape.atd(x+47),RefShape.atd(y+47),RefShape.atd(z+47)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+44),RefShape.atd(y+44),RefShape.atd(z+44))-
		cv::Point3d(RefShape.atd(x+46),RefShape.atd(y+46),RefShape.atd(z+46))));
	FeatureLst[4] = temp1/temp2;

	// Feature 5 = Eye Opening (Right) = avg(Distance of point 37 to 41
	// and point 38 to point 40)
	temp1 = 0.5*(cv::norm(cv::Point3d(TrackedShape.atd(x+37),TrackedShape.atd(y+37),TrackedShape.atd(z+37))-
		cv::Point3d(TrackedShape.atd(x+41),TrackedShape.atd(y+41),TrackedShape.atd(z+41)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+38),TrackedShape.atd(y+38),TrackedShape.atd(z+38))-
		cv::Point3d(TrackedShape.atd(x+40),TrackedShape.atd(y+40),TrackedShape.atd(z+40))));

	temp2 = 0.5*(cv::norm(cv::Point3d(RefShape.atd(x+37),RefShape.atd(y+37),RefShape.atd(z+37))-
		cv::Point3d(RefShape.atd(x+41),RefShape.atd(y+41),RefShape.atd(z+41)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+38),RefShape.atd(y+38),RefShape.atd(z+38))-
		cv::Point3d(RefShape.atd(x+40),RefShape.atd(y+40),RefShape.atd(z+40))));
	FeatureLst[5] = temp1/temp2;

	// Feature 6 = Height of outer lip boundary = avg(dist(50,58)
	//dist(51,57),dist(52,56))
	temp1 = 0.3333333*(cv::norm(cv::Point3d(TrackedShape.atd(x+50),TrackedShape.atd(y+50),TrackedShape.atd(z+50))-
		cv::Point3d(TrackedShape.atd(x+58),TrackedShape.atd(y+58),TrackedShape.atd(z+58)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+51),TrackedShape.atd(y+51),TrackedShape.atd(z+51))-
		cv::Point3d(TrackedShape.atd(x+57),TrackedShape.atd(y+57),TrackedShape.atd(z+57)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+52),TrackedShape.atd(y+52),TrackedShape.atd(z+52))-
		cv::Point3d(TrackedShape.atd(x+56),TrackedShape.atd(y+56),TrackedShape.atd(z+56))));
	temp2 = 0.3333333*(cv::norm(cv::Point3d(RefShape.atd(x+50),RefShape.atd(y+50),RefShape.atd(z+50))-
		cv::Point3d(RefShape.atd(x+58),RefShape.atd(y+58),RefShape.atd(z+58)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+51),RefShape.atd(y+51),RefShape.atd(z+51))-
		cv::Point3d(RefShape.atd(x+57),RefShape.atd(y+57),RefShape.atd(z+57)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+52),RefShape.atd(y+52),RefShape.atd(z+52))-
		cv::Point3d(RefShape.atd(x+56),RefShape.atd(y+56),RefShape.atd(z+56))));
	FeatureLst[6] = temp1/temp2;

	// Feature 7 = Height of inner lip boundary = avg(dist(60,65)
	//dist(61,64),dist(62,63))
	temp1 = 0.3333333*(cv::norm(cv::Point3d(TrackedShape.atd(x+60),TrackedShape.atd(y+60),TrackedShape.atd(z+60))-
		cv::Point3d(TrackedShape.atd(x+65),TrackedShape.atd(y+65),TrackedShape.atd(z+65)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+61),TrackedShape.atd(y+61),TrackedShape.atd(z+61))-
		cv::Point3d(TrackedShape.atd(x+64),TrackedShape.atd(y+64),TrackedShape.atd(z+64)))
		+ cv::norm(cv::Point3d(TrackedShape.atd(x+62),TrackedShape.atd(y+62),TrackedShape.atd(z+62))-
		cv::Point3d(TrackedShape.atd(x+63),TrackedShape.atd(y+63),TrackedShape.atd(z+63))));
	temp2 = 0.3333333*(cv::norm(cv::Point3d(RefShape.atd(x+60),RefShape.atd(y+60),RefShape.atd(z+60))-
		cv::Point3d(RefShape.atd(x+65),RefShape.atd(y+65),RefShape.atd(z+65)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+61),RefShape.atd(y+61),RefShape.atd(z+61))-
		cv::Point3d(RefShape.atd(x+64),RefShape.atd(y+64),RefShape.atd(z+64)))
		+ cv::norm(cv::Point3d(RefShape.atd(x+62),RefShape.atd(y+62),RefShape.atd(z+62))-
		cv::Point3d(RefShape.atd(x+63),RefShape.atd(y+63),RefShape.atd(z+63))));
	FeatureLst[7] = temp1/temp2;

	// Feature 8 = avg angle of the mouth corners

	FeatureLst[8] = 0.0;

	// Feature 9 = Lip Corner Puller =  dist(48,54)
	temp1 = cv::norm(cv::Point3d(TrackedShape.atd(x+48),TrackedShape.atd(y+48),TrackedShape.atd(z+48))-
		cv::Point3d(TrackedShape.atd(x+54),TrackedShape.atd(y+54),TrackedShape.atd(z+54)));
	temp2 = cv::norm(cv::Point3d(RefShape.atd(x+48),RefShape.atd(y+48),RefShape.atd(z+48))-
		cv::Point3d(RefShape.atd(x+54),RefShape.atd(y+54),RefShape.atd(z+54)));
	FeatureLst[9] = (std::pow(temp1,2)/std::pow(temp2,2));

	// Feature 10,11,12,13,14,15 = global parameters
}
//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
	int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

	//draw triangulation
	c = CV_RGB(0,0,0);
	for(i = 0; i < tri.rows; i++){
		if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
			visi.at<int>(tri.at<int>(i,1),0) == 0 ||
			visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
		p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
			shape.at<double>(tri.at<int>(i,0)+n,0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
			shape.at<double>(tri.at<int>(i,1)+n,0));
		cv::line(image,p1,p2,c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
			shape.at<double>(tri.at<int>(i,0)+n,0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
			shape.at<double>(tri.at<int>(i,2)+n,0));
		cv::line(image,p1,p2,c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
			shape.at<double>(tri.at<int>(i,2)+n,0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
			shape.at<double>(tri.at<int>(i,1)+n,0));
		cv::line(image,p1,p2,c);
	}
	//draw connections
	c = CV_RGB(0,0,255);
	for(i = 0; i < con.cols; i++){
		if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
			visi.at<int>(con.at<int>(1,i),0) == 0)continue;
		p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
			shape.at<double>(con.at<int>(0,i)+n,0));
		p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
			shape.at<double>(con.at<int>(1,i)+n,0));
		cv::line(image,p1,p2,c,1);
	}
	//draw points
	for(i = 0; i < n; i++){    
		if(visi.at<int>(i,0) == 0)continue;
		p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
		c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
	}return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv,
	char* ftFile,char* conFile,char* triFile,
	bool &fcheck,double &scale,int &fpd, bool &show, int &jobID, int &camidx)
{
	int i; fcheck = false; scale = 1; fpd = -1;
	for(i = 1; i < argc; i++){
		if((std::strcmp(argv[i],"-?") == 0) ||
			(std::strcmp(argv[i],"--help") == 0)){
				std::cout << "track_face:- Written by Jason Saragih 2010." << std::endl <<
						"Added feature extraction routines by Md. Iftekhar Tanveer (go2chayan@gmail.com) 2014" << std::endl
					<< "Performs automatic face tracking" << std::endl << std::endl
					<< "#" << std::endl 
					<< "# usage: ./face_tracker [options]" << std::endl
					<< "#" << std::endl << std::endl
					<< "Arguments:" << std::endl
					<< "-m <string> -> Tracker model (default: ./model/face2.tracker)"
					<< std::endl
					<< "-c <string> -> Connectivity (default: ./model/face.con)"
					<< std::endl
					<< "-t <string> -> Triangulation (default: ./model/face.tri)"
					<< std::endl
					<< "-s <double> -> Image scaling (default: 1)" << std::endl
					<< "-job <index/int> -> specify the index of the input video file to work with" << std::endl
					<< "-d <int>    -> Frames/detections (default: -1)" << std::endl
					<< "--check     -> Check for failure" << std::endl
					<< "--noshow    -> Hides the video frames and other feedbacks" <<std::endl
					<< "-# <Camera_ID_Number> -> If only one camera, use 0. If want process from file, use -1 (default 0)" << std::endl
					<< "-input <Filename> <Filename> ... <Filename> -> input video files (this must be the last argument)" << std::endl;
				return -1;
		}
	}
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
	}
	if(i >= argc)fcheck = false;
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"--noshow") == 0){show = false; break;}
	}
	if(i >= argc)show = true;
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-s") == 0){
			if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
			break;
		}
	}
	if(i >= argc)scale = 1;
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-d") == 0){
			if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
			break;
		}
	}
	if(i >= argc)fpd = -1;
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-#") == 0){
			if(argc > i+1)camidx = std::atoi(argv[i+1]); else camidx = 0;
			break;
		}
	}
	if(i >= argc)camidx = 0;
	for(i = 1; i < argc; i++){
			if(std::strcmp(argv[i],"-job") == 0){
				if(argc > i+1)jobID = std::atoi(argv[i+1]); else jobID = -1;
				break;
			}
		}
		if(i >= argc)jobID = -1;
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-m") == 0){
			if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
			else strcpy(ftFile,"../model/face2.tracker");
			break;
		}
	}
	if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-c") == 0){
			if(argc > i+1)std::strcpy(conFile,argv[i+1]);
			else strcpy(conFile,"../model/face.con");
			break;
		}
	}
	if(i >= argc)std::strcpy(conFile,"../model/face.con");
	for(i = 1; i < argc; i++){
		if(std::strcmp(argv[i],"-t") == 0){
			if(argc > i+1)std::strcpy(triFile,argv[i+1]);
			else strcpy(triFile,"../model/face.tri");
			break;
		}
	}
	if(i >= argc)std::strcpy(triFile,"../model/face.tri");
	return 0;
}

cv::Mat getFilterKernel(int windowSize,int ArrLen,bool forfilter = true){
	cv::Mat FilterKernel ;
	if(forfilter){
		FilterKernel = cv::Mat::ones(windowSize,ArrLen,CV_64FC1);
	}else{
		cv::vconcat(-1.0*cv::Mat::ones(windowSize/2,ArrLen,CV_64FC1),
			cv::Mat::ones(windowSize/2,ArrLen,CV_64FC1),FilterKernel);
	}
	FilterKernel = FilterKernel/(double)FilterKernel.rows;
	return FilterKernel;
}

void FilterFeatures(double FeatureLst[],cv::Mat &FilteredContent,cv::Mat &Differentiated,
	int ArrLen,int windowSize = 6){

		cv::Mat tempBuff,dottedData;

		static cv::Mat DatBufferforFilter = cv::Mat::ones(windowSize,ArrLen,CV_64FC1);
		static cv::Mat Kernel = getFilterKernel(windowSize,ArrLen);
		static cv::Mat Kernel_Diff = getFilterKernel(windowSize,ArrLen,false);

		// Slides the window vertically for Filtering
		DatBufferforFilter.rowRange(1,windowSize).copyTo(tempBuff);
		tempBuff.copyTo(DatBufferforFilter.rowRange(0,windowSize-1));
		for(int i=0;i<ArrLen;i++)
			DatBufferforFilter.atd(windowSize-1,i)=FeatureLst[i];

		// Apply the Kernel for Filter
		FilteredContent.create(1,ArrLen,CV_64FC1);
		dottedData = DatBufferforFilter.mul(Kernel);
		for (int i=0;i<ArrLen;i++){
			FilteredContent.atd(0,i) = cv::sum(dottedData.col(i))[0];
		}

		// Apply the Kernel for Filter
		Differentiated.create(1,ArrLen,CV_64FC1);
		cv::Mat temp = DatBufferforFilter.mul(Kernel_Diff);
		for (int i=0;i<ArrLen;i++){
			Differentiated.atd(0,i) = cv::sum(temp.col(i))[0];
		}
//		for (int i=0;i<ArrLen;i++){
//			if((cv::sum(dottedData.col(i))[0]>thresHold)&&(SkipEvent[i]==0)){
//				Differentiated.atd(0,i) = 1.0;
//				SkipEvent[i]=windowSize;
//			}else if((cv::sum(dottedData.col(i))[0]<-1.0*thresHold)&&(SkipEvent[i]==0)){
//				Differentiated.atd(0,i) = -1.0;
//				SkipEvent[i]=windowSize;
//			}else
//				Differentiated.atd(0,i) = 0.0;
//		}
}

void plotFeatureswithSlidingWindow(double FeatureLst[],int ArrLen,double eventFeature[],
	int rows = 480,int cols = 640,
	int HorzScale = 4,double vertScale = 75.0){

		int BuffLen = cols/HorzScale;
		static cv::Mat DatBuffer = cv::Mat::ones(ArrLen,BuffLen,CV_64FC1);
		static cv::Mat updownEventBuff = cv::Mat::zeros(ArrLen,BuffLen,CV_64FC1);

		// Preparing the Axis
		cv::Mat PlotArea = cv::Mat::zeros(rows,cols,CV_8UC3);
		cv::line(PlotArea,cv::Point(0,rows/2),cv::Point(cols,rows/2),
			CV_RGB(255,255,255),1); // x axis
		cv::line(PlotArea,cv::Point(cols/2,0),cv::Point(cols/2,rows),
			CV_RGB(255,255,255),1); // y axis

		// Slides the window horizontally for displaying image
		cv::Mat tempBuff;
		DatBuffer.colRange(1,BuffLen).copyTo(tempBuff);
		tempBuff.copyTo(DatBuffer.colRange(0,BuffLen-1));
		for(int i=0;i<ArrLen;i++)
			DatBuffer.atd(i,BuffLen-1)=FeatureLst[i];

		// Slides the window horizontally for Marking Events
		updownEventBuff.colRange(1,BuffLen).copyTo(tempBuff);
		tempBuff.copyTo(updownEventBuff.colRange(0,BuffLen-1));
		for(int i=0;i<ArrLen;i++)
			updownEventBuff.atd(i,BuffLen-1)=eventFeature[i];

		//Draw the plots
		double *tempArray = new double[ArrLen];
		for(int i=0;i<ArrLen;i++)tempArray[i]=0.0;
		for(int i=0;i<(BuffLen-HorzScale);i++){
			for(int j=0;j<3;j++)
			{
				cv::line(PlotArea,cv::Point(HorzScale*i,-1*vertScale
					*tempArray[j]+int(rows*1.5/ArrLen)),
					cv::Point(HorzScale*(i+1),-1*vertScale*
					DatBuffer.atd(j,i)+int(rows*1.5/ArrLen)),
					colorSet[j],1,CV_AA);
				tempArray[j] = DatBuffer.atd(j,i);
			}
			for(int j=3;j<ArrLen;j++)
			{
				// Plot lines
				cv::line(PlotArea,cv::Point(HorzScale*i,-1*0.5*
					vertScale*tempArray[j]+int(rows*float(j+1)/ArrLen)),
					cv::Point(HorzScale*(i+1),-1*0.5*vertScale*
					DatBuffer.atd(j,i)+int(rows*float(j+1)/ArrLen)),
					colorSet[j],1,CV_AA);
				tempArray[j] = DatBuffer.atd(j,i);

				// Write Up/Down Events
				if(updownEventBuff.atd(j,i)==1.0){
					cv::putText(PlotArea,"Up",cv::Point(HorzScale*(i+1),
						-1*0.5*vertScale*DatBuffer.atd(j,i)+int(rows*
						float(j+1)/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.25,
						colorSet[j]);
				}else if(updownEventBuff.atd(j,i)==-1.0){
					cv::putText(PlotArea,"Down",cv::Point(HorzScale*(i+1),
						-1*0.5*vertScale*DatBuffer.atd(j,i)+int(rows*
						float(j+1)/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.25,
						colorSet[j]);
				}
			}
		}
		cv::putText(PlotArea,"Pitch,Yaw,Roll",cv::Point(cols/2,
			int(rows*1.5/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
		for(int j=3;j<ArrLen;j++){
			cv::putText(PlotArea,FeatureSet[j],cv::Point(cols/2,
				int(rows*float(j)/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
		}
		cv::imshow("plot",PlotArea);

}

//=============================================================================
int main(int argc, const char** argv){

		   // Define and initialize the engine parameters for the '-e option'
   float         timeBase          = 0.03;            // Use single image mode
   bool          updateTimeBase    = true;        // Not used in video mode
   unsigned long threadCount       = 2UL;          // Let's take one thread only
   char const*   model             = "Face.Profile"; // Search frontal faces
   float         imageScale        = 1.0f;         // Scale the images
   float         minFaceSize       = 0.0f;         // Find small faces too
   long          minFaceScore      = 9L;           // That's the default value
   float         idMemoryLength    = 0.0f;
   char const*   idMemoryType      = "Spatial";
   bool          trackFaces        = true;
   char const*   phantomTrap       = "Off";
   bool          searchEyes        = true;
   bool          searchNose        = false;
   bool          searchMouth       = false;
   bool          analyzeEyes       = false;
   bool          analyzeMouth      = false;
   bool          analyzeGender     = false;
   bool          analyzeAge        = false;
   bool          analyzeHappy      = false;
   bool          analyzeSad        = false;
   bool          analyzeSurprised  = false;
   bool          analyzeAngry      = false;

   // Now create the engine with the defined parameters.
   engine = Shore::CreateFaceEngine( timeBase,
                                    updateTimeBase,
                                    threadCount,
                                    model,
                                    imageScale,
                                    minFaceSize,
                                    minFaceScore,
                                    idMemoryLength,
                                    idMemoryType,
                                    trackFaces,
                                    phantomTrap,
                                    searchEyes,
                                    searchNose,
                                    searchMouth,
                                    analyzeEyes,
                                    analyzeMouth,
                                    analyzeGender,
                                    analyzeAge,
                                    analyzeHappy,
                                    analyzeSad,
                                    analyzeSurprised,
                                    analyzeAngry );
      // Was the setup successful?
   if ( engine == 0 ){
      std::cerr << "Engine setup failed - exit!\n";
      std::exit(1);
   }

	//parse command line arguments
	char ftFile[256],conFile[256],triFile[256];
	bool fcheck = false; int fpd = -1; bool show = false;
	int jobID = -1; bool inputFiles = false;
	double Size=1.0;int lastResetCount = 0;	std::ofstream ofstm;
	int camidx=0;	//Camera Index (If only one camera available, use 0)
	if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,Size,fpd,show,jobID,camidx)<0)return 0;
	// #### TODO: Include these variables into command line parser
	int rotate = 0; // 0,1,2,3 only

	if (jobID ==-2)
		jobID = std::atoi(std::getenv("SLURM_PROCID"));

	std::vector<std::string> fileList;
	for (int argi = 1;argi<argc;argi++){
		if(!inputFiles){
			if (strcmp(argv[argi],"-input")!=0){
				continue;
			}else{
				inputFiles = true;
				continue;
			}
		}
		fileList.push_back(argv[argi]);
	}
	if(camidx>=0)
		fileList.push_back("");
	for (int idx = 0; idx<fileList.size();idx++){
		std::string filename;
		std::string outFileName;

		if(camidx<0){
			if (jobID<0){
				filename=fileList[idx];
				outFileName = std::string(fileList[idx]).append(".csv");
			}else{
				filename=fileList[jobID];
				outFileName = std::string(fileList[jobID]).append(".csv");
			}


			std::cout<<filename<<std::endl;
			std::cout<<outFileName<<std::endl;
		}

		std::cout<<"startTime = "<<cv::getTickCount()/float(cv::getTickFrequency())<<std::endl;
		// ############## Initialization #####################
			cv::Mat filteredFeature;
			cv::Mat eventFeature;

			// Tracker initiatizing variables
			std::vector<int> wSize1(1); wSize1[0] = 7;std::vector<uchar> encoded;
			std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
			int nIter = 5; double clamp=3,fTol=0.01;

			// Load all the models from file
			FACETRACKER::Tracker model(ftFile);
			cv::Mat trackedShape3d,refShape3d;
			model._clm._pdm._M.copyTo(trackedShape3d);
			trackedShape3d.copyTo(refShape3d);
			double FeatureLst[10]={0.0};

			cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
			cv::Mat con=FACETRACKER::IO::LoadCon(conFile);

			//initialize variables and display window
			cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text;

			// Configuring capture object for extraction of data from video or webcam
			cv::VideoCapture cap;
			if (camidx!=-1)
				cap.open(camidx);
			else
				cap.open(filename);

			if(!cap.isOpened()){
				printf("Video Could Not be loaded");
				return -1;
			}

			int64 t1,t0 = cvGetTickCount(); unsigned long int fnum=0;
			if(show){
				std::cout << "Hot keys: "        << std::endl
					<< "\t ESC or x - skip video"     << std::endl
					<< "\t X - quit"     << std::endl
					<< "\t d   - Redetect" << std::endl;
			}

			//loop until quit (i.e user presses ESC)
			bool failed = true;

			// Write header of the csv
			ofstm.open(outFileName.c_str());
			for (int feat_count = 0; feat_count < 12; feat_count++){
				ofstm<<FeatureSet[feat_count]<<",";
			}
			for (int feat_count = 0; feat_count < 24; feat_count++){
				ofstm<<"dicCoeff_local"<<feat_count<<",";
			}
			ofstm<<"\n";
			// ############## End of Initialization ################
			double FeaturesInArray[14] ={0.0};
			double sumFPS = 0;
			while(1){
					//grab image
					cap>>frame;

					if(frame.data==NULL)break;
					cv::resize(frame,frame,cv::Size(),Size,Size);

					// 0, 90, 180, 270 degrees of rotation
					if(rotate==1){
						frame=frame.t();
						cv::flip(frame,frame,1);
					}else if(rotate==2)
						cv::flip(frame,frame,1);
					else if(rotate==3){
						frame=frame.t();
						cv::flip(frame,frame,-1);
					}

					// Convert to grayscale
					if(frame.channels()!=1)cv::cvtColor(frame,gray,CV_BGR2GRAY);
					else gray = frame;

					// track this image
					std::vector<int> wSize;
					if(failed)wSize = wSize2; else wSize = wSize1;

					// Detect face in the image (using SHORE)
					cv::imencode(".pgm",gray,encoded);
					Image imagePGM(encoded);
					content = engine->Process( imagePGM.LeftTop(),
												 imagePGM.Width(),
												 imagePGM.Height(),
												 1,
												 1,
												 imagePGM.Width(),
												 0,
												 "GRAYSCALE" );
					cv::Rect R1;bool foundFace = false;

					if (content->GetObjectCount() > 0){
						for( unsigned long j = 0; j < content->GetObjectCount(); ++j ){
								Shore::Object const*currentObj = content->GetObject(j);
								if(strcmp(currentObj->GetType(),"Face")==0){
									R1.x = currentObj->GetRegion()->GetLeft();
									R1.y = currentObj->GetRegion()->GetTop();
									R1.width = currentObj->GetRegion()->GetRight() - R1.x;
									R1.height = currentObj->GetRegion()->GetBottom() - R1.y;
									foundFace = true;
									break;
								}
						}
						if(foundFace && model.Track(gray,wSize,R1,fpd,nIter,clamp,fTol,fcheck) == 0){
							int idx = model._clm.GetViewIdx(); failed = false;

							// Extract all the features
							trackedShape3d = model._clm._pdm._M + model._clm._pdm._V*model._clm._plocal;
							FeatureList(trackedShape3d,refShape3d,FeatureLst);
							FeaturesInArray[0] = model._clm._pglobl.at<double>(1); // pitch
							FeaturesInArray[1] = model._clm._pglobl.at<double>(2); // yaw
							FeaturesInArray[2] = model._clm._pglobl.at<double>(3); // roll
							FeaturesInArray[3] = FeatureLst[0];	//inBrL
							FeaturesInArray[4] = FeatureLst[1]; //oBrL
							FeaturesInArray[5] = FeatureLst[2]; //inBrR
							FeaturesInArray[6] = FeatureLst[3]; //oBrR
							FeaturesInArray[7] = FeatureLst[4]; //EyeOL
							FeaturesInArray[8] = FeatureLst[5]; //EyeOR
							FeaturesInArray[9] = FeatureLst[6]; //oLipH
							FeaturesInArray[10] = FeatureLst[7]; //inLipH
							FeaturesInArray[11] = FeatureLst[9]; //LipCDt
							FeaturesInArray[12] = model._clm._pglobl.at<double>(4);
							FeaturesInArray[13] = model._clm._pglobl.at<double>(5);

							// Debug. TODO: Remove the following:
							// Smooth out the features and plot in a separate window
							FilterFeatures(FeaturesInArray,filteredFeature,eventFeature,14);
							if(show)plotFeatureswithSlidingWindow(FeaturesInArray,12,(double*)eventFeature.data);
							double errorTracker = std::sqrt(eventFeature.at<double>(0)*eventFeature.at<double>(0)
									+eventFeature.at<double>(1)*eventFeature.at<double>(1)
									+eventFeature.at<double>(2)*eventFeature.at<double>(2)
									+eventFeature.at<double>(12)*eventFeature.at<double>(12)
									+eventFeature.at<double>(13)*eventFeature.at<double>(13));
							/*if(errorTracker>10 && (fnum - lastResetCount) > 15){
								if(show)std::cout<<"Unlikely movement detected. Resetting Tracker."<<std::endl;
								failed = true;
								model.FrameReset();
								lastResetCount = fnum;
								ofstm<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<
										","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<
										","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<"\n";
							}*/

							// Write the data in file
							for (int feat_count = 0; feat_count < 12; feat_count++){
								ofstm<<filteredFeature.at<double>(0,feat_count)<<",";
							}
							for (int feat_count = 0; feat_count < 24; feat_count++){
								ofstm<<model._clm._plocal.at<double>(feat_count,0)<<",";
							}
							ofstm<<"\n";

							if(show)Draw(frame,model._shape,con,tri,model._clm._visi[idx]);
						}
					
					}else{
						if(show){cv::Mat R(frame,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
							failed = true;
							model.FrameReset();
							ofstm<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<
									","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<
									","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<","<<"\n";
					}
					//count framerate
					if(fnum%10 == 0){
						t1 = cv::getTickCount();
						fps = 10.0/((double(t1-t0)/cv::getTickFrequency()));
						t0 = t1;
					}
					sumFPS = sumFPS + fps;
					fnum += 1;
					//draw framerate on display image
					char frameNo[50] = {" "};
					if(show){
						sprintf(sss,"%d frames/sec",(int)ceil(fps)); text = sss;
						cv::putText(frame,text,cv::Point(10,20),
							CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255),2);
						cv::putText(frame,text,cv::Point(10,20),
							CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(0,0,0),1);
						sprintf(frameNo,"Frame:%d",(int)fnum);
						cv::putText(frame,frameNo,cv::Point(frame.cols - 100,20),
							CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255),3);
						cv::putText(frame,frameNo,cv::Point(frame.cols - 100,20),
							CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(0,0,0),2);
						// Show output picture
						cv::imshow("Face Tracker",frame);
						// Wait for user input
						int c = cvWaitKey(5);
						if(c == 27 || char(c) == 'x')break;
						else if(char(c) == 'd')model.FrameReset();
						else if (char(c) == 'X'){ofstm.close();exit(0);}
					}
			}
			ofstm.close();
			std::cout<<"endTime = "<<cv::getTickCount()/float(cv::getTickFrequency())<<std::endl;
			std::cout<<"FramesProcessed = "<<fnum<<std::endl;
			if (fnum!=0)
				std::cout<<"AvgFPS = "<<sumFPS/fnum<<std::endl;
			else
				std::cout<<"AvgFPS = 0"<<std::endl;
			if (jobID>=0)
				break;
		}
		exit(0);
	}
//=============================================================================