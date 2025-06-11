# DemBones-python
It's a python module of [DemBones](https://github.com/electronicarts/dem-bones) with little modify. It can now process pointcloud data which did not use the skinning weights smoothing regularization in paper [Robust and Accurate Skeletal Rigging from Mesh Sequences](http://binh.graphics/papers/2014s-ske/) .


## Requirements 
1. OpenExr 
2. IMath 
3. Eigen 
4. pybind11
5. FBXSDX 

## Compiling
Tested platforms:
- g++ 9.3.0 on Ubuntu Linux 20.04
- g++ 11.4.0 on Ubuntu Linux 22.04

Compiling steps:
1. Install [cmake](https://cmake.org/)
2. Copy the following libraries to their respective folders in `ExtLibs` so that [cmake](https://cmake.org/) can find these paths:
    - [Eigen 3.3.9](https://eigen.tuxfamily.org/) with path `ExtLibs/Eigen/Eigen/Dense`,
    - [Alembic (from Maya 2020 Update 4 DevKit)](https://www.autodesk.com/developer-network/platform-technologies/maya) with path `ExtLibs/Alembic/include/Alembic/Abc/All.h`,
    - [FBXSDK 2020.0.1](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0) with path `ExtLibs/FBXSDK/include/fbxsdk.h`,
    - [tclap 1.2.4](http://tclap.sourceforge.net/) with path `ExtLibs/tclap/include/tclap/CmdLine.h`,
3. Run cmake:
```
mkdir build
cd build
cmake ..
```
4. Build: 
```
cmake --build . --config Release --target install
```

> **Notes for Linux** 
>   - You may need to install some libriries: [libxml2-dev](http://xmlsoft.org/) (run `$ sudo apt-get install libxml2-dev`) and [zlib-dev](https://zlib.net/) (run `$ sudo apt-get install zlib1g-dev`).