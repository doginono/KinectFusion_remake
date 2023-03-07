# KinectFusion_remake
Course Project from 3D Scanning and Motion Capture at TUM
# Latex report
https://sharelatex.tum.de/project/639468018f773b1b855819e5

# Project Structure
<pre>
└── ProjectKinectFusion\ <br />
   ├── KinectFusionRemake\ The git cloned folder <br />
   │   ├── main.cpp\ <br />
   │   └─── CMakeLists.txt  <br />
   ├── Libs <br />
   │   ├── Ceres<br />
   │   ├── Eigen<br />
   │   ├── FreeImage<br />
   │   ├── Glog<br />
   │   └── Flann  <br />
   ├── Data<br />
   └── build<br />
   <br />
</pre>

You also need to have CUDA installed. The main entry point is the main.cpp file which calls the necessary methods. There is only one option now, it being the reconstruct room function. Then in this function the pose estimation and TSDF updating function will be called for every frame. Raycasting part is not finished that it is commented out.

If you do not want or think we are not going to need any file feel free to change CMakelist.txt as you wish.
