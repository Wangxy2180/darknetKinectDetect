#include "KinectPart.h"
#include "darknet.h"
//#include "image.h"

#include "kinect.h"

#include "opencv2/opencv.hpp"
/*此cpp文件说明
 *为尽可能地减少耦合度，将一些函数复制过来并加上_mogai后缀
 *所有需要在外部调用的函数，均以darknetKinect开头
 *目前共有*个
 *（1）darknetKinectOpenC（）
 *（2）darknetKinectOpenReaderC（）
 *（3）darknetKinect_load_imageC（）
 *（4）
 */
//#ifdef _cplueplus

extern "C" {
//#endif // _cplueplus

    const float intrinsicFx = 1051.9;
    const float intrinsicFy = 1050.6;
    const float intrinsicU0 = 965.833;
    const float intrinsicV0 = 531.5299;


CSYS3DInColor globalPixelCoor = { 0 };

struct KinectSensorC :IKinectSensor { int a[0]; };
struct KinectColorFrameReaderC : IColorFrameReader { int a[0]; };
struct KinectDepthFrameReaderC : IDepthFrameReader { int a[0]; };
struct KinectCoordinateMapperC : ICoordinateMapper { int a[0]; };

/******************************************重写***************************************************/
image mat_to_image_mogai(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k*w*h + y * w + x] = data[y*step + x * c + k] / 255.0f;
            }
        }
    }
    return im;
}

cv::Mat image_to_mat_mogai(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y * img.w + x];
                mat.data[y*step + x * img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}

void draw_box_mogai(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1 * a.w + 0 * a.w*a.h] = r;
        a.data[i + y2 * a.w + 0 * a.w*a.h] = r;

        a.data[i + y1 * a.w + 1 * a.w*a.h] = g;
        a.data[i + y2 * a.w + 1 * a.w*a.h] = g;

        a.data[i + y1 * a.w + 2 * a.w*a.h] = b;
        a.data[i + y2 * a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i * a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i * a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i * a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i * a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i * a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i * a.w + 2 * a.w*a.h] = b;
    }
}

/******************************************重写over************************************************/
/*留白*/
/*留白*/
/******************************************只在内部调用****************************************************************/
bool getKinectColorImageC(KinectColorFrameReaderC** pColorFrameReaderC, int mColorImgPoint, cv::Mat* mColorImage)
{
    IColorFrameReader** pColorFrameReader = (IColorFrameReader**)pColorFrameReaderC;
    IColorFrame* pColorFrame = nullptr;
    if ((*pColorFrameReader)->AcquireLatestFrame(&pColorFrame) == S_OK)
    {
        if (pColorFrame->CopyConvertedFrameDataToArray(mColorImgPoint * 4 * sizeof(BYTE), (*mColorImage).data, ColorImageFormat_Bgra) == S_OK)
        {
            pColorFrame->Release();
            pColorFrame = nullptr;
            return true;
        }
        else
        {
            std::cout << "彩色转换出现错误" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "彩色图读取最新一帧失败" << std::endl;
        return false;
    }
    pColorFrame = nullptr;
    return true;
}

bool getKinectDepthImageC(KinectDepthFrameReaderC** pDepthFrameReaderC, int mDepthImgPoint, cv::Mat* mDepthImage)
{
    IDepthFrameReader** pDepthFrameReader = (IDepthFrameReader**)pDepthFrameReaderC;
    IDepthFrame* pDepthFrame = nullptr;
    if ((*pDepthFrameReader)->AcquireLatestFrame(&pDepthFrame) == S_OK)
    {
        if (pDepthFrame->CopyFrameDataToArray(mDepthImgPoint, reinterpret_cast<UINT16*>((*mDepthImage).data)) == S_OK)
        {

            pDepthFrame->Release();
            pDepthFrame = nullptr;
            return true;
        }
        else
        {
            std::cout << "彩色转换出现错误" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "彩色图读取最新一帧失败" << std::endl;
        return false;
    }
    pDepthFrame = nullptr;
    return true;
}

void putTextInLBPoint(std::string displayedText, cv::Mat* innerImage, cv::Point innerTextLeftBottomPoint)
{
    std::cout << " displayedText" << displayedText << std::endl;
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 1;
    int thickness = 2;
    int baseline;
    cv::Size text_size = cv::getTextSize(displayedText, font_face, font_scale, thickness, &baseline);
    putText((*innerImage), displayedText, innerTextLeftBottomPoint, font_face, font_scale, cv::Scalar(0, 0, 0), thickness, 8, 0);
}

bool CSYSPixelToCameraInColor(CSYS3DInColor* CameraCSYSPoint, CSYS3DInColor pixelCSYSPoint3D)
{//CSYS Coordinate System
    int Xc = 0;
    int Yc = 0;
    int Zc = 0;
    Zc = pixelCSYSPoint3D.z / 1;
    //Zc = tempZc / (255.0/4500);
    Xc = (Zc * (pixelCSYSPoint3D.x - intrinsicU0)) / intrinsicFx;
    Yc = (Zc * (pixelCSYSPoint3D.y - intrinsicV0)) / intrinsicFy;
   (*CameraCSYSPoint).x = Xc;
   (*CameraCSYSPoint).y = Yc;
   (*CameraCSYSPoint).z = Zc;
   std::cout << "cam obj cen  (" << (*CameraCSYSPoint).x << "," << (*CameraCSYSPoint).y << "," << (*CameraCSYSPoint).z << ")" << std::endl;
   return true;
}

bool CSYSCameraToWorldInColor(CSYS3DInColor* innerWorldCoordinate, CSYS3DInColor cameraCSYSPoint)
{
    (*innerWorldCoordinate).x = cameraCSYSPoint.x*(-1) + cameraCSYSPoint.y*(-1) + cameraCSYSPoint.z*(-1) + (-999);
    (*innerWorldCoordinate).y = cameraCSYSPoint.x*(-1) + cameraCSYSPoint.y*(-1) + cameraCSYSPoint.z*(-1) + (-999);
    (*innerWorldCoordinate).z = cameraCSYSPoint.x*(-1) + cameraCSYSPoint.y*(-1) + cameraCSYSPoint.z*(-1) + (-999);
    return true;
}

/*****************************************只在内部调用over*************************************************************/
/*留白*/
/*留白*/
/***********************************外部调用************************************************/
KinectColorFrameReaderC* darknetKinectOpenColorReaderC(KinectSensorC** pSensorC)
{
	IKinectSensor** pSensor = (IKinectSensor**)pSensorC;
	Sleep(3000);//我太迷了，非得打开了才行吗

	IColorFrameReader* pColorFrameReader = nullptr;//放在外边，一会获取最新帧还需要
    std::cout << "开始获取彩色数据获取" << std::endl;
	{
		//获取彩色数据Source
		IColorFrameSource* pColorFrameSource = nullptr;
        std::cout << "darknetSensor " << &(*pSensor) << std::endl;
		(*pSensor)->get_ColorFrameSource(&pColorFrameSource);

		//打开彩色阅读器并释放Source
		if (pColorFrameSource->OpenReader(&pColorFrameReader) == S_OK)
		{
            std::cout << "打开reader成功" << std::endl;
            std::cout << "readerin" << &pColorFrameReader << std::endl;
		}
		else std::cout << "打开reader失败" << std::endl;
		pColorFrameSource->Release();
		pColorFrameSource = nullptr;
	}
	KinectColorFrameReaderC* readerC = (KinectColorFrameReaderC*)pColorFrameReader;
	return readerC;
}

KinectDepthFrameReaderC* darknetKinectOpenDepthReaderC(KinectSensorC** pSensorC)
{
	IKinectSensor** pSensor = (IKinectSensor**)pSensorC;
	Sleep(3000);//我太迷了，非得打开了才行吗

	IDepthFrameReader* pDepthFrameReader = nullptr;//放在外边，一会获取最新帧还需要
    std::cout << "开始获取深度数据获取" << std::endl;
	{
		//获取彩色数据Source
		IDepthFrameSource* pDepthFrameSource = nullptr;
        std::cout << "darknetSensor " << &(*pSensor) << std::endl;
		(*pSensor)->get_DepthFrameSource(&pDepthFrameSource);

		//打开彩色阅读器并释放Source
		if (pDepthFrameSource->OpenReader(&pDepthFrameReader) == S_OK)
		{
            std::cout << "打开深度reader成功" << std::endl;
            std::cout << "readerin" << &pDepthFrameReader << std::endl;
		}
		else std::cout << "打开深度reader失败" << std::endl;
		pDepthFrameSource->Release();
		pDepthFrameSource = nullptr;
	}
	KinectDepthFrameReaderC* readerC = (KinectDepthFrameReaderC*)pDepthFrameReader;
	return readerC;
}

void darknetKinectSensorOpenC(KinectSensorC** pSensorC)
{
    IKinectSensor** pSensorInner = (IKinectSensor**)pSensorC;
    std::cout << "Sensor " << &(*pSensorInner) << std::endl;
    std::cout << "SensorC " << &(*pSensorC) << std::endl;
    if (GetDefaultKinectSensor(&(*pSensorInner)) == S_OK)
    {
        std::cout << "Get Sensor OK" << std::endl;
    }
    else
    {
        std::cerr << "Get Sensor failed" << std::endl;
    }
    (*pSensorInner)->Open();
    std::cout << "pSensor addr is " << (*pSensorInner) << std::endl;
    pSensorC = (KinectSensorC**)pSensorInner;
    std::cout << "2Sensor " << &(*pSensorInner) << std::endl;
    std::cout << "2SensorC " << &(*pSensorC) << std::endl;
    std::cout << "**********************pSensor Open**********************" << std::endl;

}

image darknetKinect_load_imageC(KinectColorFrameReaderC** pColorFrameReaderC)
{
    cv::Mat mColorImage = cv::Mat(1080, 1920, CV_8UC4);
    int mColorPoint = 1920 * 1080;

    if (getKinectColorImageC(&(*pColorFrameReaderC), mColorPoint, &mColorImage) == true)
    {
        cv::Mat CIResize = cv::Mat::zeros(360, 640, CV_8UC3);
        //colorImage resize
        resize(mColorImage, CIResize, CIResize.size());
        imshow("resize", CIResize);
        //cv::waitKey(33);
    }

    cv::Mat BGRColorImage = cv::Mat(1080, 1920, CV_8UC3);
    cv::cvtColor(mColorImage, BGRColorImage, cv::COLOR_BGRA2RGB);//这句似乎是多余的
    ////{//只为了输出BGR图片
    ////    cv::Mat outImage = cv::Mat::zeros(540, 960, CV_8UC3);
    ////    cv::resize(BGRColorImage, outImage, outImage.size());
    ////    cv::imshow("RGB", outImage);
    ////}
    //这个函数应该返回im类型
    cv::Mat BGRflipColorImage = cv::Mat(1080, 1920, CV_8UC3);
    //反转彩色图像，然后识别，深度图像坐标反转在读取缓冲区做的
    cv::flip(BGRColorImage, BGRflipColorImage, 1);
    return mat_to_image_mogai(BGRflipColorImage);
    //return mat_to_image_mogai(BGRColorImage);
}

KinectCoordinateMapperC* darknetKinectOpenMapperC(KinectSensorC** pSensorC)
{
    IKinectSensor** pSensor = (IKinectSensor**)pSensorC;
    ICoordinateMapper* pCoordinateMapper = nullptr;
    if ((*pSensor)->get_CoordinateMapper(&pCoordinateMapper) == S_OK)
    {
        std::cout << "Mapper打开成功" << std::endl;
        KinectCoordinateMapperC* mapperC = (KinectCoordinateMapperC*)pCoordinateMapper;
        return mapperC;
    }
    else
    {
        std::cerr << "get_CoordinateMapper failed" << std::endl;
        return 0;
    }
}

void darknetKinectGetDepthC(KinectDepthFrameReaderC* pDepthFrameReaderC,KinectCoordinateMapperC* pMapperC,CSYS3DInColor* cen)
{
	cv::Mat mDepthImage = cv::Mat(424, 512, CV_16UC1);
	//cv::Mat mDepthImage ;
	const int mDepthPoint = 512 * 424;
	int mColorPoint = 1920 * 1080;

	ICoordinateMapper* pMapper = (ICoordinateMapper*)pMapperC;

	IDepthFrameReader* pDepthFrameReader = (IDepthFrameReader*)pDepthFrameReaderC;

	IDepthFrame* pDepthFrame = nullptr;

	UINT    uBufferSize = 0;
	UINT16*    tempDepthBuffer = nullptr;

	int depthBuffer[mDepthPoint] = {0};


	if ((pDepthFrameReaderC)->AcquireLatestFrame(&pDepthFrame) == S_OK)
	{
        //std::cout << "获取深度最新一帧成功" << std::endl;
		//存入缓冲区并输出中心点深度
		if (pDepthFrame->AccessUnderlyingBuffer(&uBufferSize, &tempDepthBuffer) == S_OK)
		{
			for (int i = 0; i < mDepthPoint; i++)
			{	
				depthBuffer[i] = tempDepthBuffer[i];
			}	
		}
		else
		{
			std::cout << "深度图存入缓冲失败" << std::endl;
		}

		if (pDepthFrame->CopyFrameDataToArray((UINT)mDepthPoint, reinterpret_cast<UINT16*>(mDepthImage.data)) == S_OK)
		{
			pDepthFrame->Release();
			pDepthFrame = nullptr;
		}
		else
		{
			std::cerr << "深度数据存入矩阵失败" << std::endl;
		}
	}
	else
	{
		std::cerr << "深度图最新一帧获取失败" << std::endl;
	}
	pDepthFrame = nullptr;

	DepthSpacePoint* colorSiteInDepth = new DepthSpacePoint[mColorPoint];
	if ((pMapperC)->MapColorFrameToDepthSpace(mDepthPoint, reinterpret_cast<UINT16*>(mDepthImage.data), mColorPoint, colorSiteInDepth) == S_OK)
    {
		DepthSpacePoint objCenPointColor2Depth = { 0 };
        //std::cout << "cen is" << (*cen).x << " " << (*cen).y << " " << (*cen).z << std::endl;
		objCenPointColor2Depth = colorSiteInDepth[(*cen).y * 1920 +1920- (*cen).x];
        //x坐标翻转
        std::cout << "depth cen is ( " << objCenPointColor2Depth.X << " , " << objCenPointColor2Depth.Y << " )" << std::endl;


		if ((isfinite(objCenPointColor2Depth.X)==0)|| (isfinite(objCenPointColor2Depth.X) == 0))
		{
            std::cout << "无穷小" << std::endl;
			(*cen).z = -999;
		}
		else if (objCenPointColor2Depth.X >= 0 && objCenPointColor2Depth.X < mDepthImage.cols && objCenPointColor2Depth.Y >= 0 && objCenPointColor2Depth.Y < mDepthImage.rows)
		{
			(*cen).z = depthBuffer[(int)(objCenPointColor2Depth.Y)*mDepthImage.cols + (int)(objCenPointColor2Depth.X)];
		}
	}
	else std::cout << "坐标变换失败" << std::endl;

	delete colorSiteInDepth;

	{
		cv::Mat img8Bit = cv::Mat(424, 512, CV_8UC1);
		mDepthImage.convertTo(img8Bit, CV_8U);
		cv::imshow("de", img8Bit);
		//cv::waitKey(33);
        //统一最后去wait
	}
}

/*
 *这个函数似乎应当承担起读深度图的责任
 */
void darknetKinectDrawCenterLabel(image im, int left, int top, int bot, int right)
{
    int x_center_wxy = (left + right) / 2;
    int y_center_wxy = (top + bot) / 2;
    ////printf("center(x,y) is (%d,%d)\n", x_center_wxy, y_center_wxy);
    draw_box_mogai(im, x_center_wxy, y_center_wxy, x_center_wxy + 1, y_center_wxy + 1, 0, 0, 0);
    //这里为啥要draw_box_mogai(）呢，没道理，来到了我的地盘，是不是，明明随便画一下就好了
    //////char center_buff[50] = { 0 };
    //////sprintf(center_buff, "pixel ( %d , %d )", x_center_wxy, y_center_wxy);
    ////std::string center2DPixel = "pixel( " + std::to_string(x_center_wxy) + "," + std::to_string(y_center_wxy) + ")";

    ////cv::Mat drawIt = image_to_mat_mogai(im);
    ////cv::Point textLeftBottomPoint;
    ////textLeftBottomPoint.x = x_center_wxy;
    ////textLeftBottomPoint.y = y_center_wxy;
    ////putTextInLBPoint(center2DPixel, &drawIt, textLeftBottomPoint);
    ////im = mat_to_image_mogai(drawIt);
    //////show_image(im, "qwe");
    ////cv::imshow("123", drawIt);
}

void darknetCSYSPixelToWorld(CSYS3DInColor PixelObjCenInner, CSYS3DInColor* WorldObjCen)
{
    CSYS3DInColor cameraCSYSPoint = { -1 };
    CSYSPixelToCameraInColor(&cameraCSYSPoint, PixelObjCenInner);
    CSYSCameraToWorldInColor(WorldObjCen, cameraCSYSPoint);
}

/************************************外部调用over********************************************/






bool KinectClose(IKinectSensor** pSensor)
{
    std::cout << "******************pSensor close******************" << std::endl;
    (*pSensor)->Close();
    std::cout << "pSensor close" << (*pSensor) << std::endl;
    (*pSensor)->Release();
    (*pSensor) = nullptr;
    std::cout << "pSensor close nullptr" << (*pSensor) << std::endl;
    return true;
}
































































//
//image darknetKinectGetColorImg(IColorFrameReader** pColorFrameReader)
//{
//    image im;
//    cv::Mat mColorImage;
//    mColorImage = cv::Mat(1080, 1920, CV_8UC4);
//    int mColorPoint = 1920 * 1080;
//
//    getKinectColorImage(&(*pColorFrameReader), mColorPoint, &mColorImage);
//
//    cv::Mat outImage;
//    outImage = cv::Mat(1080, 1920, CV_8UC4);
//    cv::cvtColor(mColorImage, outImage, cv::COLOR_BGRA2BGR);
//    {//这里只为了看图片是否正确
//        cv::Mat outImageResize=cv::Mat::zeros(540,960,CV_8UC3);
//        cv::resize(outImage, outImageResize, outImageResize.size());
//        cv::imshow("BGR", outImageResize);
//    }
//    im = mat_to_image_mogai(mColorImage);
//    //这里本应该调用mat_to_image_cv()更好一点，但是我不太懂这个转换关系 鬼知道下边那么写会不会出问题
//    //还把源码注释掉的头文件声明给放出来了
//
//    //mat_cv* mat = (mat_cv*)outImage;
//    //im = mat_to_image_cv(mat);
//
//    return im;
//}


//#ifdef _cplueplus


}//extern "C"
//#endif // _cplueplus


