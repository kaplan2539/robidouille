/*

 Copyright (c) by Emil Valkov,
 All rights reserved.

 License: http://www.opensource.org/licenses/bsd-license.php

*/

#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <unistd.h>
#include "RaspiCamCV.h"

//// OpenCV port of 'LAPM' algorithm (Nayar89)
//double modifiedLaplacian(const cv::Mat& src)
//{
//    cv::Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
//    cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);
//
//    cv::Mat Lx;
//    cv::sepFilter2D(src, Lx, CV_64F, M, G);
//
//    cv::Mat Ly;
//    cv::sepFilter2D(src, Ly, CV_64F, G, M);
//
//    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);
//
//    double focusMeasure = cv::mean(FM).val[0];
//    return focusMeasure;
//}
//
//// OpenCV port of 'LAPV' algorithm (Pech2000)
//double varianceOfLaplacian(const cv::Mat& src)
//{
//    cv::Mat lap;
//    cv::Laplacian(src, lap, CV_64F);
//
//    cv::Scalar mu, sigma;
//    cv::meanStdDev(lap, mu, sigma);
//
//    double focusMeasure = sigma.val[0]*sigma.val[0];
//    return focusMeasure;
//}
//
//// OpenCV port of 'TENG' algorithm (Krotkov86)
//double tenengrad(const cv::Mat& src, int ksize)
//{
//    cv::Mat Gx, Gy;
//    cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
//    cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);
//
//    cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);
//
//    double focusMeasure = cv::mean(FM).val[0];
//    return focusMeasure;
//}
//
//// OpenCV port of 'GLVN' algorithm (Santos97)
//double normalizedGraylevelVariance(const cv::Mat& src)
//{
//    cv::Scalar mu, sigma;
//    cv::meanStdDev(src, mu, sigma);
//
//    double focusMeasure = (sigma.val[0]*sigma.val[0]) / mu.val[0];
//    return focusMeasure;
//}

short GetSharpness(char* data, unsigned int width, unsigned int height)
{
    // assumes that your image is already in planner yuv or 8 bit greyscale
    IplImage* in = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
    IplImage* out = cvCreateImage(cvSize(width,height),IPL_DEPTH_16S,1);
    memcpy(in->imageData,data,width*height);

    // aperture size of 1 corresponds to the correct matrix
    cvLaplace(in, out, 1);

    short maxLap = -32767;
    short* imgData = (short*)out->imageData;
    int i=0;
    for(i=0;i<(out->imageSize/2);i++)
    {
        if(imgData[i] > maxLap) maxLap = imgData[i];
    }

    cvReleaseImage(&in);
    cvReleaseImage(&out);
    return maxLap;
}

int main(int argc, char *argv[ ]){

	RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
	
	config->width=640;
	config->height=480;
	config->bitrate=0;	// zero: leave as default
	config->framerate=0;
	config->monochrome=1;

	int opt;

	while ((opt = getopt(argc, argv, "lxm")) != -1)
	{
		switch (opt)
		{
			case 'l':					// large
				config->width = 640;
				config->height = 480;
				break;
			case 'x':	   				// extra large
				config->width = 960;
				config->height = 720;
				break;
			case 'm':					// monochrome
				config->monochrome = 1;
				break;
			default:
				fprintf(stderr, "Usage: %s [-x] [-l] [-m] \n", argv[0], opt);
				fprintf(stderr, "-l: Large mode\n");
				fprintf(stderr, "-x: Extra large mode\n");
				fprintf(stderr, "-l: Monochrome mode\n");
				exit(EXIT_FAILURE);
		}
	}

	/*
	Could also use hard coded defaults method: raspiCamCvCreateCameraCapture(0)
	*/
    RaspiCamCvCapture * capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
	free(config);
	
	CvFont font;
	double hScale=0.4;
	double vScale=0.4;
	int    lineWidth=1;

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, 0, lineWidth, 8);

	IplImage* edges =0;
	IplImage* both =0;

	cvNamedWindow("RaspiCamTest", 1);
	int exit =0;
	do {
		IplImage* image = raspiCamCvQueryFrame(capture);
		short sharpness = GetSharpness(image->imageData,image->width, image->height);

        if(!edges) {	
			edges = cvCreateImage(cvSize(image->width,image->height),image->depth,image->nChannels);
		}
        if(!both) {	
			both = cvCreateImage(cvSize(image->width*2,image->height),image->depth,image->nChannels);
		}

		cvCanny(image,edges,1.0,1.0,3);

		char text[200];
		sprintf(
			text
			, "w=%.0f h=%.0f fps=%.0f bitrate=%.0f monochrome=%.0f"
			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FRAME_WIDTH)
			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FRAME_HEIGHT)
			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FPS)
			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_BITRATE)
			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_MONOCHROME)
		);
		cvPutText (image, text, cvPoint(05, 40), &font, cvScalar(255, 255, 0, 0));
	

		sprintf(text, "Sharpness: %d -- Press ESC to exit", sharpness);
		cvPutText (image, text, cvPoint(05, 80), &font, cvScalar(255, 255, 0, 0));

        cvSetImageROI(both, cvRect(0,0,image->width,image->height));
		cvCopy(image,both,0);			
        cvSetImageROI(both, cvRect(image->width,0,image->width,image->height));
		cvCopy(edges,both,0);			
        cvSetImageROI(both, cvRect(0,0,image->width*2,image->height));
	
		cvShowImage("RaspiCamTest", both);
		
		char key = cvWaitKey(10);
		
		switch(key)	
		{
			case 27:		// Esc to exit
				exit = 1;
				break;
			case 60:		// < (less than)
				raspiCamCvSetCaptureProperty(capture, RPI_CAP_PROP_FPS, 25);	// Currently NOOP
				break;
			case 62:		// > (greater than)
				raspiCamCvSetCaptureProperty(capture, RPI_CAP_PROP_FPS, 30);	// Currently NOOP
				break;
		}
		
	} while (!exit);

	cvDestroyWindow("RaspiCamTest");
	raspiCamCvReleaseCapture(&capture);
	return 0;
}
