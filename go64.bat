wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
copy c:\opencv\build\x64\vc15\bin\opencv_world454.dll .
copy c:\opencv\build\x64\vc15\bin\opencv_videoio_msmf454_64.dll .
@set oldpath=%Path%
@set oldinclude=%INCLUDE%
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64
c:\harbour\bin\win\msvc64\hbmk2 opencv.hbp -comp=msvc64
@set Path=%oldpath%
@set INCLUDE=%oldinclude%
