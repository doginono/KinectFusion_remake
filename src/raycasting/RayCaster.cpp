#include "RayCaster.h"

RayCaster::RayCaster(Volume& vol_) : vol(vol_) {}

RayCaster::RayCaster(Volume& vol_, PointCloud& frame_) : vol(vol_), frame(frame_) {}

void RayCaster::changeFrame(PointCloud& frame_) {
    frame = frame_;
}

void RayCaster::changeVolume(Volume& vol_) {
    vol = vol_;
}

// TODO