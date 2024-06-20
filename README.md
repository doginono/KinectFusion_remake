# KinectFusion_remake
Course Project from 3D Scanning and Motion Capture at TUM

# Project Structure
<pre>
└── ProjectKinectFusion\ 
   ├── KinectFusionRemake\ The git cloned folder 
   │   ├── main.cpp\ 
   │   └─── CMakeLists.txt  
   ├── Libs 
   │   ├── Ceres
   │   ├── Eigen
   │   ├── FreeImage
   │   ├── Glog
   │   └── Flann
   ├── Data
   └── build
</pre>

You also need to have CUDA installed. The main entry point is the main.cpp file which calls the necessary methods. There is only one option now, it being the reconstruct room function. Then in this function the pose estimation and TSDF updating function will be called for every frame. Raycasting part is not finished that it is commented out.

If you do not want or think we are not going to need any file feel free to change CMakelist.txt as you wish.
