#pragma once
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct {
        int x;
        int y;
        int z;
    }CSYS3DInColor;
    typedef struct KinectColorFrameReaderC KinectColorFrameReaderC;
    typedef struct KinectDepthFrameReaderC KinectDepthFrameReaderC;
    typedef struct KinectCoordinateMapperC KinectCoordinateMapperC;
    typedef struct KinectSensorC KinectSensorC;

    image darknetKinect_load_imageC(KinectColorFrameReaderC** pColorFrameReaderC);
    void darknetKinectSensorOpenC(KinectSensorC** pSensor);
    KinectColorFrameReaderC* darknetKinectOpenColorReaderC(KinectSensorC** pSensorC);
    KinectDepthFrameReaderC* darknetKinectOpenDepthReaderC(KinectSensorC** pSensorC);
    KinectCoordinateMapperC* darknetKinectOpenMapperC(KinectSensorC** pSensorC);
    void darknetKinectDrawCenterLabel(image im, int left, int top, int bot, int right);
    void darknetKinectGetDepthC(KinectDepthFrameReaderC* pDepthFrameReaderC, KinectCoordinateMapperC* pMapperC, CSYS3DInColor* cen);


#ifdef __cplusplus
}
#endif