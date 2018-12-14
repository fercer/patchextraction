@echo off
IF NOT EXIST build (
echo Creating build folder ...
mkdir .\build
)

IF NOT EXIST bin (
echo Creating bin folder ...
mkdir .\bin
)

SET required_include_paths=/I..\include
SET required_libs=..\lib\libpatchextraction.lib
SET macros_definitions=/DBUILDING_PATCHEXTRACTION_DLL
SET version=release

if "%version%" == "release" SET macros_definitions=%macros_definitions% /DNDEBUG

cl %macros_definitions% %required_include_paths% /Fo:build\test_patchextraction.o src\test_patchextraction.c %required_libs% /link /out:.\bin\test_patchextraction.exe


