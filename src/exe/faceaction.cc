/* 
 Face Tracker by Md. Iftekhar Tanveer (go2chayan@gmail.com)
 This is built upon Jason M. Saragih's facetracking API which
 can be found here: https://github.com/kylemcdonald/FaceTracker
 This original tracker has been modified by Md. Iftekhar Tanveer
 for better stability and a specific need.

 Stability is provided by applying some engineering tweaking to the basic
 facetracker. Then, 10 Facial features are extracted and written in a CSV
 file.
 */

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <FaceTracker/Tracker.h>

#define atd at<double>

struct FacialExpression{
	bool happy;
	bool surprised;
	bool confused;
	bool eyeBlink;
};

cv::Scalar colorSet[12] = { CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0,
		255), CV_RGB(255, 250, 0), CV_RGB(0, 255, 250), CV_RGB(250, 0, 255),
		CV_RGB(255, 50, 0), CV_RGB(0, 255, 50), CV_RGB(50, 0, 255), CV_RGB(255,
				50, 155), CV_RGB(155, 255, 50), CV_RGB(50, 155, 255), };

std::string FeatureSet[13] = { "Pitch", "Yaw  ", "Roll ",
		"Height of Left Eye Brow", "Left Eye Blink", "Height of Right Eye Brow",
		"Right Eye Blink", "Left Eye Openness (Noisy)",
		"Right Eye Openness (Noisy)", "Smile Intensity", "Mouth Openness", "",
		"" };

/* Find the distance of (x1,y1,z1) to a line formed by the
 points (x2,y2,z2) and (x3,y3,z3)*/
double DistFromLine(double x1, double y1, double z1, double x2, double y2,
		double z2, double x3, double y3, double z3) {

	cv::Point3d X0(x1, y1, z1);
	cv::Point3d X1(x2, y2, z2);
	cv::Point3d X2(x3, y3, z3);

	//cv::Point3d num = (X0-X1).cross(X0-X2);
	//cv::Point3d den = (X2-X1);
	cv::Point3d num = (X2 - X1).cross(X0 - X1);
	cv::Point3d den = (X2 - X1);

	return cv::norm(num) / cv::norm(den);
}

// Static Function for getting 10 facial features
// ----------------------------  Deprecated  --------------------------------
void ExtractFeatures(cv::Mat &TrackedShape, cv::Mat &RefShape,
		double * FeatureLst) {
	// Annotation metadata can be found here
	// https://drive.google.com/file/d/0B9gztQjJ5_xVRzYxMm5yQXpoSkk/view?usp=sharing
	// Remember: The point number here = point number in picture - 1
	double temp1 = 0.0, temp2 = 0.0;
	cv::Point2d tempPoint1, tempPoint2;
	int x = 0, y = 66, z = 132;

	// Feature 0 = Left Eye brow = Distance of point 24
	// from the line formed by point 31 and 35
	temp1 = DistFromLine(TrackedShape.atd(x+24),
	TrackedShape.atd(y+24),
	TrackedShape.atd(z+24),
	TrackedShape.atd(x+31),
	TrackedShape.atd(y+31),
	TrackedShape.atd(z+31),
	TrackedShape.atd(x+35),
	TrackedShape.atd(y+35),
	TrackedShape.atd(z+35));
	temp2 = DistFromLine(RefShape.atd(x+24),
	RefShape.atd(y+24),
	RefShape.atd(z+24),
	RefShape.atd(x+31),
	RefShape.atd(y+31),
	RefShape.atd(z+31),
	RefShape.atd(x+35),
	RefShape.atd(y+35),
	RefShape.atd(z+35));
	FeatureLst[0] = 5. * (temp1 - temp2) / temp2;

	// Feature 1 = Left Eye Blink = Distance of point 33
	// from the line formed by point 42 and 45
	temp1 = DistFromLine(TrackedShape.atd(x+33),
	TrackedShape.atd(y+33),
	TrackedShape.atd(z+33),
	TrackedShape.atd(x+42),
	TrackedShape.atd(y+42),
	TrackedShape.atd(z+42),
	TrackedShape.atd(x+45),
	TrackedShape.atd(y+45),
	TrackedShape.atd(z+45));
	temp2 = DistFromLine(RefShape.atd(x+33),
	RefShape.atd(y+33),
	RefShape.atd(z+33),
	RefShape.atd(x+42),
	RefShape.atd(y+42),
	RefShape.atd(z+42),
	RefShape.atd(x+45),
	RefShape.atd(y+45),
	RefShape.atd(z+45));
	FeatureLst[1] = 7. * std::abs(temp1 - temp2) / temp2;

	// Feature 2 = Right Eye Brow = Distance of point 19
	// from the line formed by point 31 and 35
	temp1 = DistFromLine(TrackedShape.atd(x+19),
	TrackedShape.atd(y+19),
	TrackedShape.atd(z+19),
	TrackedShape.atd(x+31),
	TrackedShape.atd(y+31),
	TrackedShape.atd(z+31),
	TrackedShape.atd(x+35),
	TrackedShape.atd(y+35),
	TrackedShape.atd(z+35));
	temp2 = DistFromLine(RefShape.atd(x+19),
	RefShape.atd(y+19),
	RefShape.atd(z+19),
	RefShape.atd(x+31),
	RefShape.atd(y+31),
	RefShape.atd(z+31),
	RefShape.atd(x+35),
	RefShape.atd(y+35),
	RefShape.atd(z+35));
	FeatureLst[2] = 5. * (temp1 - temp2) / temp2;

	// Feature 3 = Right Eye Blink = Distance of point 33
	// from the line formed by point 36 and 39
	temp1 = DistFromLine(TrackedShape.atd(x+33),
	TrackedShape.atd(y+33),
	TrackedShape.atd(z+33),
	TrackedShape.atd(x+36),
	TrackedShape.atd(y+36),
	TrackedShape.atd(z+36),
	TrackedShape.atd(x+39),
	TrackedShape.atd(y+39),
	TrackedShape.atd(z+39));
	temp2 = DistFromLine(RefShape.atd(x+33),
	RefShape.atd(y+33),
	RefShape.atd(z+33),
	RefShape.atd(x+36),
	RefShape.atd(y+36),
	RefShape.atd(z+36),
	RefShape.atd(x+39),
	RefShape.atd(y+39),
	RefShape.atd(z+39));
	FeatureLst[3] = 7. * std::abs(temp1 - temp2) / temp2;

	// Feature 4 = Eye Opening (Left) = avg(Distance of point 43 to 47
	// and point 44 to point 46)
	temp1 = 0.5 * (cv::norm(cv::Point3d(TrackedShape.atd(x+43),TrackedShape.atd(y+43),TrackedShape.atd(z+43))-
	cv::Point3d(TrackedShape.atd(x+47),TrackedShape.atd(y+47),TrackedShape.atd(z+47)))
	+ cv::norm(cv::Point3d(TrackedShape.atd(x+44),TrackedShape.atd(y+44),TrackedShape.atd(z+44))-
	cv::Point3d(TrackedShape.atd(x+46),TrackedShape.atd(y+46),TrackedShape.atd(z+46))));

	temp2 = 0.5 * (cv::norm(cv::Point3d(RefShape.atd(x+43),RefShape.atd(y+43),RefShape.atd(z+43))-
	cv::Point3d(RefShape.atd(x+47),RefShape.atd(y+47),RefShape.atd(z+47)))
	+ cv::norm(cv::Point3d(RefShape.atd(x+44),RefShape.atd(y+44),RefShape.atd(z+44))-
	cv::Point3d(RefShape.atd(x+46),RefShape.atd(y+46),RefShape.atd(z+46))));
	FeatureLst[4] = temp1 / temp2 - 1.;

	// Feature 5 = Eye Opening (Right) = avg(Distance of point 37 to 41
	// and point 38 to point 40)
	temp1 = 0.5 * (cv::norm(cv::Point3d(TrackedShape.atd(x+37),TrackedShape.atd(y+37),TrackedShape.atd(z+37))-
	cv::Point3d(TrackedShape.atd(x+41),TrackedShape.atd(y+41),TrackedShape.atd(z+41)))
	+ cv::norm(cv::Point3d(TrackedShape.atd(x+38),TrackedShape.atd(y+38),TrackedShape.atd(z+38))-
	cv::Point3d(TrackedShape.atd(x+40),TrackedShape.atd(y+40),TrackedShape.atd(z+40))));

	temp2 = 0.5 * (cv::norm(cv::Point3d(RefShape.atd(x+37),RefShape.atd(y+37),RefShape.atd(z+37))-
	cv::Point3d(RefShape.atd(x+41),RefShape.atd(y+41),RefShape.atd(z+41)))
	+ cv::norm(cv::Point3d(RefShape.atd(x+38),RefShape.atd(y+38),RefShape.atd(z+38))-
	cv::Point3d(RefShape.atd(x+40),RefShape.atd(y+40),RefShape.atd(z+40))));
	FeatureLst[5] = temp1 / temp2 - 1.;

	// Feature 6 = Smile =  dist(48,54)
	temp1 = cv::norm(cv::Point3d(TrackedShape.atd(x+48),TrackedShape.atd(y+48),TrackedShape.atd(z+48))-
	cv::Point3d(TrackedShape.atd(x+54),TrackedShape.atd(y+54),TrackedShape.atd(z+54)));
	temp2 = cv::norm(cv::Point3d(RefShape.atd(x+48),RefShape.atd(y+48),RefShape.atd(z+48))-
	cv::Point3d(RefShape.atd(x+54),RefShape.atd(y+54),RefShape.atd(z+54)));
	FeatureLst[6] = 5. * (temp1 - temp2) / temp2;

	// Feature 7 = Mouth Open = avg(dist(50,58)
	//dist(51,57),dist(52,56))
	temp1 = 0.3333333 * (cv::norm(cv::Point3d(TrackedShape.atd(x+50),TrackedShape.atd(y+50),TrackedShape.atd(z+50))-
	cv::Point3d(TrackedShape.atd(x+58),TrackedShape.atd(y+58),TrackedShape.atd(z+58)))
	+ cv::norm(cv::Point3d(TrackedShape.atd(x+51),TrackedShape.atd(y+51),TrackedShape.atd(z+51))-
	cv::Point3d(TrackedShape.atd(x+57),TrackedShape.atd(y+57),TrackedShape.atd(z+57)))
	+ cv::norm(cv::Point3d(TrackedShape.atd(x+52),TrackedShape.atd(y+52),TrackedShape.atd(z+52))-
	cv::Point3d(TrackedShape.atd(x+56),TrackedShape.atd(y+56),TrackedShape.atd(z+56))));
	temp2 = 0.3333333 * (cv::norm(cv::Point3d(RefShape.atd(x+50),RefShape.atd(y+50),RefShape.atd(z+50))-
	cv::Point3d(RefShape.atd(x+58),RefShape.atd(y+58),RefShape.atd(z+58)))
	+ cv::norm(cv::Point3d(RefShape.atd(x+51),RefShape.atd(y+51),RefShape.atd(z+51))-
	cv::Point3d(RefShape.atd(x+57),RefShape.atd(y+57),RefShape.atd(z+57)))
	+ cv::norm(cv::Point3d(RefShape.atd(x+52),RefShape.atd(y+52),RefShape.atd(z+52))-
	cv::Point3d(RefShape.atd(x+56),RefShape.atd(y+56),RefShape.atd(z+56))));
	FeatureLst[7] = 2.*(temp1 - temp2) / temp2;

	// Deprecated
	// Feature 8 = Mouth Open (inner) = avg(dist(60,65)
	//dist(61,64),dist(62,63))
	/*temp1 = 0.3333333*(cv::norm(cv::Point3d(TrackedShape.atd(x+60),TrackedShape.atd(y+60),TrackedShape.atd(z+60))-
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
	 cv::Point3d(RefShape.atd(x+63),RefShape.atd(y+63),RefShape.atd(z+63))));*/
	FeatureLst[8] = 0.0;

	// Deprecated
	// Feature 9 = avg angle of the mouth corners
	FeatureLst[9] = 0.0;
}

// A heuristic method for defining expressions from the features
FacialExpression extractExpression(const cv::Mat &filteredFeatures,
		const cv::Mat &differentialExpression){
	FacialExpression extractedExpression;

	// When the eyeblink feature is high enough, it is an eyeblink
	extractedExpression.eyeBlink = (filteredFeatures.atd(0,4)>0.25);

	// Smile intensity is higher than 0.8 means happy
	extractedExpression.happy = (filteredFeatures.atd(0,9)>0.95);

	// Eye brow lowered without eyeblink
	extractedExpression.confused = filteredFeatures.atd(0,3)<-0.1;

	// Negative Smile intensity or increased eyebrow
	// while no eyeblink or open mouth = surprise
	extractedExpression.surprised =
			(filteredFeatures.atd(0,3)>0.25 && !extractedExpression.eyeBlink)||
			(filteredFeatures.atd(0,10)>0.1)||
			(filteredFeatures.atd(0,9)<-0.75);
	return extractedExpression;
}

//=============================================================================
void Draw(cv::Mat &image, cv::Mat &shape, cv::Mat &con, cv::Mat &tri,
		cv::Mat &visi) {
	int i, n = shape.rows / 2;
	cv::Point p1, p2;
	cv::Scalar c;

	//draw triangulation
	c = CV_RGB(0, 0, 0);
	for (i = 0; i < tri.rows; i++) {
		if (visi.at<int>(tri.at<int>(i, 0), 0) == 0
				|| visi.at<int>(tri.at<int>(i, 1), 0) == 0
				|| visi.at<int>(tri.at<int>(i, 2), 0) == 0)
			continue;
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0),
				shape.at<double>(tri.at<int>(i, 0) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0),
				shape.at<double>(tri.at<int>(i, 1) + n, 0));
		cv::line(image, p1, p2, c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0),
				shape.at<double>(tri.at<int>(i, 0) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0),
				shape.at<double>(tri.at<int>(i, 2) + n, 0));
		cv::line(image, p1, p2, c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0),
				shape.at<double>(tri.at<int>(i, 2) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0),
				shape.at<double>(tri.at<int>(i, 1) + n, 0));
		cv::line(image, p1, p2, c);
	}
	//draw connections
	c = CV_RGB(0, 0, 255);
	for (i = 0; i < con.cols; i++) {
		if (visi.at<int>(con.at<int>(0, i), 0) == 0
				|| visi.at<int>(con.at<int>(1, i), 0) == 0)
			continue;
		p1 = cv::Point(shape.at<double>(con.at<int>(0, i), 0),
				shape.at<double>(con.at<int>(0, i) + n, 0));
		p2 = cv::Point(shape.at<double>(con.at<int>(1, i), 0),
				shape.at<double>(con.at<int>(1, i) + n, 0));
		cv::line(image, p1, p2, c, 1);
	}
	//draw points
	for (i = 0; i < n; i++) {
		if (visi.at<int>(i, 0) == 0)
			continue;
		p1 = cv::Point(shape.at<double>(i, 0), shape.at<double>(i + n, 0));
		c = CV_RGB(255, 0, 0);
		cv::circle(image, p1, 2, c);
	}
	return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv, char* ftFile, char* conFile,
		char* triFile, bool &fcheck, double &scale, int &fpd, bool &show,
		int &jobID, int &camidx) {
	int i;
	fcheck = false;
	scale = 1;
	fpd = -1;
	for (i = 1; i < argc; i++) {
		if ((std::strcmp(argv[i], "-?") == 0)
				|| (std::strcmp(argv[i], "--help") == 0)) {
			std::cout << "track_face:- Written by Jason Saragih 2010."
					<< std::endl
					<< "Added feature extraction routines by Md. Iftekhar Tanveer (go2chayan@gmail.com) 2014"
					<< std::endl << "Performs automatic face tracking"
					<< std::endl << std::endl << "#" << std::endl
					<< "# usage: ./face_tracker [options]" << std::endl << "#"
					<< std::endl << std::endl << "Arguments:" << std::endl
					<< "-m <string> -> Tracker model (default: ./model/face2.tracker)"
					<< std::endl
					<< "-c <string> -> Connectivity (default: ./model/face.con)"
					<< std::endl
					<< "-t <string> -> Triangulation (default: ./model/face.tri)"
					<< std::endl << "-s <double> -> Image scaling (default: 1)"
					<< std::endl
					<< "-job <index/int> -> specify the index of the input video file to work with"
					<< std::endl
					<< "-d <int>    -> Frames/detections (default: -1)"
					<< std::endl << "--check     -> Check for failure"
					<< std::endl
					<< "--noshow    -> Hides the video frames and other feedbacks"
					<< std::endl
					<< "-# <Camera_ID_Number> -> If only one camera, use 0. If want process from file, use -1 (default 0)"
					<< std::endl
					<< "-input <Filename> <Filename> ... <Filename> -> input video files (this must be the last argument)"
					<< std::endl;
			return -1;
		}
	}
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--check") == 0) {
			fcheck = true;
			break;
		}
	}
	if (i >= argc)
		fcheck = false;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--noshow") == 0) {
			show = false;
			break;
		}
	}
	if (i >= argc)
		show = true;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-s") == 0) {
			if (argc > i + 1)
				scale = std::atof(argv[i + 1]);
			else
				scale = 1;
			break;
		}
	}
	if (i >= argc)
		scale = 1;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-d") == 0) {
			if (argc > i + 1)
				fpd = std::atoi(argv[i + 1]);
			else
				fpd = -1;
			break;
		}
	}
	if (i >= argc)
		fpd = -1;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-#") == 0) {
			if (argc > i + 1)
				camidx = std::atoi(argv[i + 1]);
			else
				camidx = 0;
			break;
		}
	}
	if (i >= argc)
		camidx = 0;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-job") == 0) {
			if (argc > i + 1)
				jobID = std::atoi(argv[i + 1]);
			else
				jobID = -1;
			break;
		}
	}
	if (i >= argc)
		jobID = -1;
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-m") == 0) {
			if (argc > i + 1)
				std::strcpy(ftFile, argv[i + 1]);
			else
				strcpy(ftFile, "../model/face2.tracker");
			break;
		}
	}
	if (i >= argc)
		std::strcpy(ftFile, "../model/face2.tracker");
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-c") == 0) {
			if (argc > i + 1)
				std::strcpy(conFile, argv[i + 1]);
			else
				strcpy(conFile, "../model/face.con");
			break;
		}
	}
	if (i >= argc)
		std::strcpy(conFile, "../model/face.con");
	for (i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "-t") == 0) {
			if (argc > i + 1)
				std::strcpy(triFile, argv[i + 1]);
			else
				strcpy(triFile, "../model/face.tri");
			break;
		}
	}
	if (i >= argc)
		std::strcpy(triFile, "../model/face.tri");
	return 0;
}

cv::Mat getFilterKernel(int windowSize, int ArrLen, bool forfilter = true) {
	cv::Mat FilterKernel;
	if (forfilter) {
		FilterKernel = cv::Mat::ones(windowSize, ArrLen, CV_64FC1);
	} else {
		cv::vconcat(-1.0 * cv::Mat::ones(windowSize / 2, ArrLen, CV_64FC1),
				cv::Mat::ones(windowSize / 2, ArrLen, CV_64FC1), FilterKernel);
	}
	FilterKernel = FilterKernel / (double) FilterKernel.rows;
	return FilterKernel;
}

void FilterFeatures(double FeatureLst[], cv::Mat &FilteredContent,
		cv::Mat &Differentiated, int ArrLen, int windowSize = 2) {

	cv::Mat tempBuff, dottedData;
	int *SkipEvent = new int[ArrLen];
	std::memset(SkipEvent, 0, ArrLen);
	// Threshold zero means the signal doesn't need to differentiate
	// Each differential feature is thresholded for a different value
	//double thresHold[13] = {0,0,0,0.8,0.05,0.45,0.05,0.05,0.2,0.01,0.30,0,0};
	//double eyeblinkThreshold = 0.2;

	// Static data structures
	// To-Do: Move these to global space or in class space
	static cv::Mat DatBufferforFilter = cv::Mat::ones(windowSize, ArrLen,
	CV_64FC1);
	static cv::Mat Kernel = getFilterKernel(windowSize, ArrLen);
	static cv::Mat Kernel_Diff = getFilterKernel(windowSize, ArrLen, false);

	// Slides the window vertically for Filtering
	DatBufferforFilter.rowRange(1, windowSize).copyTo(tempBuff);
	tempBuff.copyTo(DatBufferforFilter.rowRange(0, windowSize - 1));
	for (int i = 0; i < ArrLen; i++)
		DatBufferforFilter.atd(windowSize-1,i)=FeatureLst[i];

		// Apply the Kernel for Smoothing Filter
	FilteredContent.create(1, ArrLen, CV_64FC1);
	dottedData = DatBufferforFilter.mul(Kernel);
	for (int i = 0; i < ArrLen; i++) {
		FilteredContent.atd(0,i) = cv::sum(dottedData.col(i))[0];
	}

	// Apply the Kernel for Differential Filter
	Differentiated.create(1, ArrLen, CV_64FC1);
	cv::Mat temp = DatBufferforFilter.mul(Kernel_Diff);
	for (int i = 0; i < ArrLen; i++) {
		// Scaler multiplier is just scaling the differential results
		Differentiated.atd(0,i) = 5*cv::sum(temp.col(i))[0];
	}
}

void plotFeatures(const double FeatureLst[], int ArrLen,
		const double eventFeature[], std::string plotName, int rows = 480,
		int cols = 640, int HorzScale = 4, double vertScale = 75.0) {
	int BuffLen = cols / HorzScale;
	static cv::Mat DatBuffer = cv::Mat::ones(ArrLen, BuffLen, CV_64FC1);
	static cv::Mat updownEventBuff = cv::Mat::zeros(ArrLen, BuffLen, CV_64FC1);

	// Preparing the Axis
	cv::Mat PlotArea = cv::Mat::zeros(rows, cols, CV_8UC3);
	cv::line(PlotArea, cv::Point(0, rows / 2), cv::Point(cols, rows / 2),
			CV_RGB(255, 255, 255), 1); // x axis
	cv::line(PlotArea, cv::Point(cols / 2, 0), cv::Point(cols / 2, rows),
			CV_RGB(255, 255, 255), 1); // y axis

	// Slides the window horizontally for displaying image
	cv::Mat tempBuff;
	DatBuffer.colRange(1, BuffLen).copyTo(tempBuff);
	tempBuff.copyTo(DatBuffer.colRange(0, BuffLen - 1));
	for (int i = 0; i < ArrLen; i++)
		DatBuffer.atd(i,BuffLen-1)=FeatureLst[i];

		// Slides the window horizontally for Marking Events
	updownEventBuff.colRange(1, BuffLen).copyTo(tempBuff);
	tempBuff.copyTo(updownEventBuff.colRange(0, BuffLen - 1));
	for (int i = 0; i < ArrLen; i++)
		updownEventBuff.atd(i,BuffLen-1)=eventFeature[i];

		//Draw the plots
	double *tempArray = new double[ArrLen];
	for (int i = 0; i < ArrLen; i++)
		tempArray[i] = 0.0;
	// Loop for horizontal movement (sliding window)
	for (int i = 0; i < (BuffLen - HorzScale); i++) {
		// The first three plots (Pitch, Yaw and Roll)
		for (int j = 0; j < 3; j++) {
			cv::line(PlotArea,
					cv::Point(HorzScale * i,
							-1 * vertScale * tempArray[j]
									+ int(rows * 1.5 / ArrLen)),
					cv::Point(HorzScale * (i + 1),
							-1 * vertScale * DatBuffer.atd(j,i)+int(rows*1.5/ArrLen)),
			colorSet[j],1,CV_AA);
			tempArray[j] = DatBuffer.atd(j,i);
		}
		for(int j=3;j<ArrLen;j++) {
			// Plot lines
			cv::line(PlotArea,cv::Point(HorzScale*i,-1*0.5*
			vertScale*tempArray[j]+int(rows*float(j)/ArrLen)),
			cv::Point(HorzScale*(i+1),-1*0.5*vertScale*
			DatBuffer.atd(j,i)+int(rows*float(j)/ArrLen)),
			colorSet[j],1,CV_AA);
			tempArray[j] = DatBuffer.atd(j,i);

			// Write Up/Down Events
			if(updownEventBuff.atd(j,i)==1.0) {
				cv::putText(PlotArea,"Up",cv::Point(HorzScale*(i+1),
				-1*0.5*vertScale*DatBuffer.atd(j,i)+int(rows*
				float(j)/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.25,
				colorSet[j]);
			} else if(updownEventBuff.atd(j,i)==-1.0) {
				cv::putText(PlotArea,"Down",cv::Point(HorzScale*(i+1),
				-1*0.5*vertScale*DatBuffer.atd(j,i)+int(rows*
				float(j)/ArrLen)),CV_FONT_HERSHEY_SIMPLEX,0.25,
				colorSet[j]);
			}
		}
	}

	// Plot labels
	// Put the plot labels outside the i-loop cause we don't want them to shift with the data
	cv::putText(PlotArea, "Pitch,Yaw,Roll",
			cv::Point(cols / 2, int(rows * 1.5 / ArrLen)),
			CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));

	for (int j = 3; j < ArrLen; j++) {
		std::stringstream plotLabel;
		plotLabel << FeatureSet[j] << ": " << std::setprecision(2)
				<< tempArray[j];
		cv::putText(PlotArea, plotLabel.str(),
				cv::Point(cols / 2, int(rows * float(j) / ArrLen)),
				CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));
	}

	cv::imshow(plotName, PlotArea);
}

//=============================================================================
int main(int argc, const char** argv) {
	//parse command line arguments
	char ftFile[256], conFile[256], triFile[256];
	bool fcheck = false;
	int fpd = -1;
	bool show = false;
	int jobID = -1;
	double Size = 1.0;
	int lastResetCount = 0;
	int camidx = 0;	//Camera Index (If only one camera available, use 0)
	if (parse_cmd(argc, argv, ftFile, conFile, triFile, fcheck, Size, fpd, show,
			jobID, camidx) < 0)
		return 0;
	// #### TODO: Include these variables into command line parser
	int rotate = 0; // 0,1,2,3 only

	std::cout << "startTime = "
			<< cv::getTickCount() / float(cv::getTickFrequency()) << std::endl;
	// ############## Initialization #####################
	cv::Mat filteredFeature;
	cv::Mat eventFeature;

	// Tracker initiatizing variables
	std::vector<int> wSize1(1);
	wSize1[0] = 7;
	std::vector<int> wSize2(3);
	wSize2[0] = 11;
	wSize2[1] = 9;
	wSize2[2] = 7;
	int nIter = 5;
	double clamp = 3, fTol = 0.01;

	// Load all the models from file
	FACETRACKER::Tracker model(ftFile);
	cv::Mat trackedShape3d, refShape3d;
	model._clm._pdm._M.copyTo(trackedShape3d);
	trackedShape3d.copyTo(refShape3d);
	double FeatureLst[10] = { 0.0 };

	cv::Mat tri = FACETRACKER::IO::LoadTri(triFile);
	cv::Mat con = FACETRACKER::IO::LoadCon(conFile);

	//initialize variables and display window
	cv::Mat frame, gray, im;
	double fps = 0;
	char sss[256];
	std::string text;

	// Configuring capture object for extraction of data from video or webcam
	cv::VideoCapture cap;
	if (camidx != -1)
		cap.open(camidx);
	else {
		std::cout << "File input is not allowed in this version" << std::endl;
		return -1;
	}

	if (!cap.isOpened()) {
		printf("Video Could Not be loaded");
		return -1;
	}

	int64 t1, t0 = cvGetTickCount();
	unsigned long int fnum = 0;
	if (show) {
		std::cout << "Hot keys: " << std::endl << "\t ESC or x - skip video"
				<< std::endl << "\t X - quit" << std::endl
				<< "\t d   - Redetect" << std::endl;
	}

	//loop until quit (i.e user presses ESC)
	bool failed = true;
	// ############## End of Initialization ################
	double FeaturesInArray[14] = { 0.0 };
	double sumFPS = 0;
	while (1) {
		//grab image
		cap >> frame;

		if (frame.data == NULL)
			break;
		cv::resize(frame, frame, cv::Size(), Size, Size);

		// 0, 90, 180, 270 degrees of rotation
		if (rotate == 1) {
			frame = frame.t();
			cv::flip(frame, frame, 1);
		} else if (rotate == 2)
			cv::flip(frame, frame, 1);
		else if (rotate == 3) {
			frame = frame.t();
			cv::flip(frame, frame, -1);
		}

		// Convert to grayscale
		if (frame.channels() != 1)
			cv::cvtColor(frame, gray, CV_BGR2GRAY);
		else
			gray = frame;

		//track this image
		std::vector<int> wSize;
		if (failed)
			wSize = wSize2;
		else
			wSize = wSize1;
		if (model.Track(gray, wSize, fpd, nIter, clamp, fTol, fcheck) == 0) {
			int idx = model._clm.GetViewIdx();
			failed = false;

			// Extract all the features
			trackedShape3d = model._clm._pdm._M
					+ model._clm._pdm._V * model._clm._plocal;
			ExtractFeatures(trackedShape3d, refShape3d, FeatureLst);
			//Remap in a new format
			FeaturesInArray[0] = model._clm._pglobl.at<double>(1); // pitch
			FeaturesInArray[1] = model._clm._pglobl.at<double>(2); // yaw
			FeaturesInArray[2] = model._clm._pglobl.at<double>(3); // roll
			for(int indx = 0; indx < 9; indx++)
				FeaturesInArray[indx+3] = FeatureLst[indx];
			FeaturesInArray[12] = model._clm._pglobl.at<double>(4); // X poisition of face in picture
			FeaturesInArray[13] = model._clm._pglobl.at<double>(5); // Y poisition of face in picture
			/*"Height of Left Eye Brow","Left Eye Blink","Height of Right Eye Brow",
			 "Right Eye Blink","Left Eye Openness (Noisy)","Right Eye Openness (Noisy)",
			 "Smile Intensity","Mouth Openness"*/

			// Smooth out the features and plot in a separate window
			FilterFeatures(FeaturesInArray, filteredFeature, eventFeature, 14);
			//if (show)plotFeatures(FeaturesInArray, 12, (double*) eventFeature.data,"Features_Raw");
			if(show)plotFeatures((double*)filteredFeature.data,12,(double*)eventFeature.data,"Features_Filtered");
			//if(show)plotFeatures((double*)eventFeature.data,12,FeaturesInArray,"diffFeatures");

			//Extract Expressions
			FacialExpression exp = extractExpression(filteredFeature,eventFeature);
			std::cout<<"Happy: "<<exp.happy<<"  Surprised: "<<exp.surprised<<"  Confused: "<<exp.confused;
			std::cout<<"  Blink: "<<exp.eyeBlink<<std::endl;

			// Catch tracking failure and reset
			double errorTracker = std::sqrt(
					eventFeature.at<double>(0) * eventFeature.at<double>(0)
							+ eventFeature.at<double>(1)
									* eventFeature.at<double>(1)
							+ eventFeature.at<double>(2)
									* eventFeature.at<double>(2)
							+ eventFeature.at<double>(12)
									* eventFeature.at<double>(12)
							+ eventFeature.at<double>(13)
									* eventFeature.at<double>(13));
			if (errorTracker > 10 && (fnum - lastResetCount) > 15) {
				if (show)
					std::cout
							<< "Unlikely movement detected. Resetting Tracker."
							<< std::endl;
				failed = true;
				model.FrameReset();
				lastResetCount = fnum;
			}

			if (show)
				Draw(frame, model._shape, con, tri, model._clm._visi[idx]);
		} else {
			if (show) {
				cv::Mat R(frame, cvRect(0, 0, 150, 50));
				R = cv::Scalar(0, 0, 255);
			}
			failed = true;
			model.FrameReset();
		}

		//count framerate
		if (fnum % 10 == 0) {
			t1 = cv::getTickCount();
			fps = 10.0 / ((double(t1 - t0) / cv::getTickFrequency()));
			t0 = t1;
		}
		sumFPS = sumFPS + fps;
		fnum += 1;
		//draw framerate on display image
		char frameNo[50] = { " " };
		if (show) {
			sprintf(sss, "%d frames/sec", (int) ceil(fps));
			text = sss;
			cv::putText(frame, text, cv::Point(10, 20),
			CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 2);
			cv::putText(frame, text, cv::Point(10, 20),
			CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
			sprintf(frameNo, "Frame:%d", (int) fnum);
			cv::putText(frame, frameNo, cv::Point(frame.cols - 100, 20),
			CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 3);
			cv::putText(frame, frameNo, cv::Point(frame.cols - 100, 20),
			CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 2);
			// Show output picture
			cv::imshow("Face Tracker", frame);
			// Wait for user input
			int c = cvWaitKey(5);
			if (c == 27 || char(c) == 'x')
				break;
			else if (char(c) == 'd')
				model.FrameReset();
			else if (char(c) == 'X') {
				exit(0);
			}
		}
	}
	// Write execution summary
	std::cout << "endTime = "
			<< cv::getTickCount() / float(cv::getTickFrequency()) << std::endl;
	std::cout << "FramesProcessed = " << fnum << std::endl;
	if (fnum != 0)
		std::cout << "AvgFPS = " << sumFPS / fnum << std::endl;
	else
		std::cout << "AvgFPS = 0" << std::endl;
}
//=============================================================================
