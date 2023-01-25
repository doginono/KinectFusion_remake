#pragma once

#include "Volume.h"
#include "PointCloud.h"
#include "Ray.h"

class RayCaster {
private:
    Volume& vol;
    PointCloud frame;

public:

    RayCaster();
    RayCaster(Volume& vol);
    RayCaster(Volume& vol, PointCloud& frame);

    void changeFrame(PointCloud& frame);
    void changeVolume(Volume& vol);
    PointCloud& rayCast();
};