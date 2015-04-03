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
#include <zbar.h>

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

//short GetSharpness(char* data, unsigned int width, unsigned int height)
CvScalar GetSharpness(IplImage* in)
{
	const  short history_size           = 5;
    static short history_index          = 0;
    static short history[5];//  = {0,0,0,0,0};

    // assumes that your image is already in planner yuv or 8 bit greyscale
//    IplImage* in = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
    static IplImage* out = 0;

	if( ! out ) {
        out=cvCreateImage(cvSize(in->width,in->height),IPL_DEPTH_16S,1);
    }
//    memcpy(in->imageData,data,width*height);

    // aperture size of 1 corresponds to the correct matrix
    cvLaplace(in, out, 1);

    short maxLap = -32767;
    short* imgData = (short*)out->imageData;
    int i=0;
	double avg=0.0;
    for(i=0;i<(out->imageSize/2);i++)
    {
        if(abs(imgData[i]) > maxLap) maxLap = abs(imgData[i]);
//		if(abs(imgData[i]) > 240) in->imageData[i]=255;
//		if(abs(imgData[i]) < 50) in->imageData[i]=0;
		avg += abs(imgData[i]);
    }
	avg /= in->imageSize;

	history[history_index++] = maxLap;
    history_index = (history_index + 1) % history_size;
    float mean = 0.0;
    for(i=0;i<history_size;i++) {
		mean+=history[i];
	}
    mean /= history_size;

//    cvReleaseImage(&in);
//    cvReleaseImage(&out);

	CvScalar r;
	r.val[0] = mean;
    r.val[1] = avg;
	return r;
}

int main(int argc, char *argv[ ]){

	RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
	
//	config->width=640;
//	config->height=480;
	//config->width=1920;
	//config->height=1080;
	config->width=2592;
	config->height=1944;
	config->bitrate=0;	// zero: leave as default
	config->framerate=0;
	config->monochrome=1;

	/*
	Could also use hard coded defaults method: raspiCamCvCreateCameraCapture(0)
	*/
    RaspiCamCvCapture * capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
	free(config);
	
	CvFont font;
	double hScale=0.8;
	double vScale=0.8;
	int    lineWidth=2;

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, 0, lineWidth, 8);

	IplImage* edges =0;
	IplImage* both =0;
    IplImage* image =0;

	cvNamedWindow("RaspiCamTest", 1);
	int exit =0;
	do {
		IplImage* big_img = raspiCamCvQueryFrame(capture);
		//short sharpness = GetSharpness(image->imageData,image->width, image->height);
        if(!image) {	
			image = cvCreateImage(cvSize(640,480),big_img->depth,big_img->nChannels);
		}

		CvRect cropRect=cvRect((big_img->width-640)/2,(big_img->height-480)/2-400,640,480);
		cvSetImageROI(big_img,cropRect);
		cvCopy(big_img,image,NULL);

		CvScalar sharpness = GetSharpness(image);
		double threshold=250.0;


        if(!edges) {	
			edges = cvCreateImage(cvSize(image->width,image->height),image->depth,image->nChannels);
		}
//        if(!both) {	
//			both = cvCreateImage(cvSize(image->width*2,image->height),image->depth,image->nChannels);
//		}
  
		//cvSmooth( image, edges, CV_BLUR, 5,5,0,0 );
		//cvCanny(edges,edges,1.0,1.0,3);

		char text[200];
//		sprintf(
//			text
//			, "w=%.0f h=%.0f fps=%.0f bitrate=%.0f monochrome=%.0f"
//			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FRAME_WIDTH)
//			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FRAME_HEIGHT)
//			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_FPS)
//			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_BITRATE)
//			, raspiCamCvGetCaptureProperty(capture, RPI_CAP_PROP_MONOCHROME)
//		);
//		cvPutText (image, text, cvPoint(05, 40), &font, cvScalar(255, 255, 0, 0));

		sprintf(text , (sharpness.val[0]>threshold ? "** OK **" : "!! keep going !!" ) );	
		cvPutText (image, text, cvPoint(05, 40), &font, cvScalar(255, 255, 0, 0));

		sprintf(text, "Sharpness: %f (%f) -- Press ESC to exit", sharpness.val[0], sharpness.val[1]);
		cvPutText (image, text, cvPoint(05, 80), &font, cvScalar(255, 255, 0, 0));

//        cvSetImageROI(both, cvRect(0,0,image->width,image->height));
//		cvCopy(image,both,0);			
//        cvSetImageROI(both, cvRect(image->width,0,image->width,image->height));
//		cvCopy(edges,both,0);			
//        cvSetImageROI(both, cvRect(0,0,image->width*2,image->height));
	
		cvShowImage("RaspiCamTest", image);
		
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
