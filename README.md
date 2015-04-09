FacialAction
============
Extracts major facial features from videos. Works for only one face per frame. Optimized for OpenMP. 

For Linux
==========
To compile, go in the main folder and use:
make

Needs OpenCV. Check out the instructions in the following page. Make sure you use correct versions for each package.
http://www.ozbotz.org/opencv-installation/

Specify the full video filename(s) with path in the argument.

FOR WINDOWS
===========
To install opencv in windows follow the official opencv documentation.

To successfully compile and run this program with Visual Studio 10, make sure the following for both projects:

1. Right click on a project name > Properties > Configuration Properties > VC++ Directories. Then from right pane, make sure Include Directories and Library Directories are pointing to correct path to opencv include and library directories. If you specify the corresponding folders in OPENCV_INCLUDE and OPENCV_LIB environment variables, then it should automatically work.
2. Right click on a project name > Properties > Configuration Properties > Linker > Input. Check the *.lib files are representing the correct opencv version. For example opencv_core249d.lib represents OpenCV 2.4.9. The "d" after the version number represents configuration (d means debug, absence of d means release). Make sure that matches with your configuration.
3. Make sure you compile the project "Tracker" before compiling "FacialAction"

Note on bluehive-workable branch
================================
Code in this branch is mainly suitable for command line operation for feature extraction. The terminal commands are tested for accurate working. The output csv file contains more information. A few typically useful commands are as follows. However, please refer to the help (-? or --help) for more info.

FacialAction --noshow -esen 18 -crop 800 0 1000 600 -input video1.mp4 video2.mp4 video3.mp4
FacialAction --noshow -esen 18 -job 0 -crop 800 0 1000 600 -input video1.mp4 video2.mp4 video3.mp4