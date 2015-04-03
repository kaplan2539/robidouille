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

CvScalar GetSharpness(IplImage* in, IplImage* drawHist=0)
{
	const  short history_size           = 5;
    static short history_index          = 0;
    static short history[5];

    static IplImage* data = 0;
    static IplImage* out = 0;
    static IplImage* out_8bit = 0;

	if( ! out ) {
        out=cvCreateImage(cvSize(in->roi->width,in->roi->height),IPL_DEPTH_16S,1);
		out_8bit=cvCreateImage(cvSize(in->roi->width,in->roi->height),IPL_DEPTH_8U,1);
    }

    // aperture size of 1 corresponds to the correct matrix
    cvLaplace(in, out, 1);

    short maxLap = -32767;
    short* imgData = (short*)out->imageData;
    int i=0;
	double avg=0.0;
    for(i=0;i<(out->imageSize/2);i++)
    {
        if(abs(imgData[i]) > maxLap) maxLap = abs(imgData[i]);
		avg += abs(imgData[i]);
    }
	avg /= out->imageSize;

	history[history_index++] = maxLap;
    history_index = (history_index + 1) % history_size;
    float mean = 0.0;
    for(i=0;i<history_size;i++) {
		mean+=history[i];
	}
    mean /= history_size;

	if(drawHist) {
        cvConvertScale(out,out_8bit);

	    CvHistogram* hist;
	    int hist_size[] = { 256 };
	    float ranges_1[] = { 0, 256 };
	    float* ranges[] = { ranges_1 };
	    hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY, ranges, 1 );

	    cvCalcHist( &out_8bit, hist, 0, 0 ); // Compute histogram
	    cvNormalizeHist( hist, 20*255 ); // Normalize it

        // populate the visualization
	    float max_value = 0;
	    cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );

	    for( int s = 0; s < 256; s++ ){
	    	float bin_val = cvQueryHistValue_1D( hist, s );
			if(bin_val>0.0) {
				cvRectangle( drawHist, cvPoint( s, 100 ),
						cvPoint( s, 100- (bin_val/max_value*100)),
						CV_RGB( 0, 255, 0 ),
						CV_FILLED );
			}
	    }

	    cvReleaseHist(&hist);
    }

	CvScalar r;
	r.val[0] = mean;
    r.val[1] = avg;
	return r;
}

int main(int argc, char *argv[ ])
{
	char SETTINGS_FILE[]="settings.txt";

	RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
	
	config->width=2592;
	config->height=1944;
	config->bitrate=0;	// zero: leave as default
	config->framerate=15;
	config->monochrome=1;

    RaspiCamCvCapture * capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
	free(config);
	
	CvFont font;
	double hScale=0.8;
	double vScale=0.8;
	int    lineWidth=2;

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, 0, lineWidth, 8);

	IplImage* both  = 0;
    IplImage* image = 0;

	cvNamedWindow("RaspiCamTest", 1);
	cvMoveWindow("RaspiCamTest", 100,100);

	int viewport_x = 0;
	int viewport_y = -400;
	int viewport_width = 640;
	int viewport_height = 480;

	CvScalar sharpness= {100, 5.0, 1.0, .10};

	int flip=0;

	FILE * f = fopen(SETTINGS_FILE,"r");
	if(f) {
		fscanf(f,"%d %d %d\n",&viewport_x,&viewport_y,&flip);
		fclose(f);
	}

	int exit =0;
	do {
		IplImage* big_img = raspiCamCvQueryFrame(capture);
        if(!image) {	
			image = cvCreateImage( cvSize(viewport_width, viewport_height), big_img->depth,3 );
		}

		CvRect cropRect = cvRect( (big_img->width-viewport_width)/2+viewport_x
                                 ,(big_img->height-viewport_height)/2+viewport_y
                                 ,viewport_width ,viewport_height );
		cvSetImageROI(big_img,cropRect);
		CvRect destRect=cvRect(0,0,viewport_width,viewport_height);
		cvSetImageROI(image,destRect);
		cvCvtColor(big_img,image,CV_GRAY2BGR);

		destRect=cvRect(0,0,viewport_width,viewport_height);
		cvSetImageROI(image,destRect);

		sharpness = GetSharpness(big_img,image);
		double threshold=5.0;

		char text[200];
		sprintf(text , (sharpness.val[1]>threshold ? "** OK **" : "!! keep going !!" ) );	
		cvPutText (image, text, cvPoint(05, 400), &font, cvScalar(255, 255, 0, 0));
		sprintf(text, "Sharpness: %f (%f) x=%d y=%d", sharpness.val[0], sharpness.val[1], viewport_x, viewport_y);
		cvPutText (image, text, cvPoint(05, 440), &font, cvScalar(255, 255, 0, 0));

		cvLine( image, cvPoint(0,240), cvPoint(639,240), CV_RGB( 255, 0, 0 ));
		cvLine( image, cvPoint(320,0), cvPoint(320,479),   CV_RGB( 255, 0, 0 ));
		cvCircle( image, cvPoint(320,240), 100,   CV_RGB( 255, 0, 0 ));

		if(flip) {
			cvFlip(image);
		}


		cvShowImage("RaspiCamTest", image);

		
		char key = cvWaitKey(10);
		
		switch(key)	
		{
			case 81:        //left
				viewport_x -= 10;
				break;
			case 82:        //up
				viewport_y -= 10;
				break;
			case 83:        //right
				viewport_x += 10;
				break;
			case 84:        //down
				viewport_y += 10;
				break;


			case 's':       //save current position
			{
				FILE *f = fopen(SETTINGS_FILE,"w");
				if(f) {
					fprintf(f,"%d %d %d\n",viewport_x,viewport_y,flip);
					fclose(f);
				}
				break;
			}


			case 'r':       //read position
            {
				FILE *f = fopen(SETTINGS_FILE,"r");
				if(f) {
					fscanf(f,"%d %d %d\n",&viewport_x,&viewport_y,&flip);
					fclose(f);
				}

				break;
			}

			case 'c':
				viewport_x=viewport_y=0;
				break;

			case 'f':
				flip = !flip;
				break;

			case 27:		// Esc to exit
				exit = 1;
				break;
		}
		
	} while (!exit);

	cvDestroyWindow("RaspiCamTest");
	raspiCamCvReleaseCapture(&capture);
	return 0;
}
