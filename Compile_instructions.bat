@echo off
IF NOT EXIST build (
echo Creating build folder ...
mkdir .\build
)

IF NOT EXIST lib (
echo Creating lib folder ...
mkdir .\lib
)

SET libname=libpatchextraction.lib
SET dllname=libpatchextraction.dll
SET required_include_paths=
SET required_libs=
SET macros_definitions=/DBUILDING_PATCHEXTRACTION_DLL
SET version=release
SET install=false
SET installation_path=

if "%1" == "python" (
    SET required_include_paths=%required_include_paths% /ID:\Apps\Anaconda3\include /ID:\Apps\Anaconda3\Lib\site-packages\numpy\core\include
    SET required_libs=%required_libs% D:\Apps\Anaconda3\libs\python36.lib
    SET macros_definitions=%macros_definitions% /DBUILDING_PYTHON_MODULE
    SET dllname=patchextraction.pyd
    SET installation_path=%PYTHONPATH%
    if "%2" == "debug" SET version="debug"
    if "%3" == "install" SET install=true
) ELSE (
    if "%1" == "debug" SET version="debug"
    if "%2" == "install" (
        SET install=true
        SET installation_path=%3
    )
)
if "%version%" == "release" SET macros_definitions=%macros_definitions% /DNDEBUG

cl /LD %macros_definitions% %required_include_paths% include\random_numbers_generator.c include\patchextraction.c %required_libs% /link /DLL /IMPLIB:.\lib\%libname% /out:.\lib\%dllname%
move patchextraction.obj .\build
move random_numbers_generator.obj .\build

if "%install%" == "true" (
    echo Copy module to: %installation_path%\
    copy .\lib\%dllname% %installation_path%\
)
