@echo off

set BUILD_DIR=vs2019-x64

if not exist %BUILD_DIR% md %BUILD_DIR%

cd %BUILD_DIR%

cmake ../.. -G "Visual Studio 16 2019" -A x64 ^
    -DOpenCV_DIR=D:/lib/opencv/4.5.2-pre

cd ..

pause