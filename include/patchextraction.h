/***********************************************************************************************************************************
CENTRO DE INVESTIGACION EN MATEMATICAS
DOCTORADO EN CIENCIAS DE LA COMPUTACION
FERNANDO CERVANTES SANCHEZ

FILE NAME : patchextraction.h

PURPOSE : Declares the functions required to extract patches using c++ instead of python. 
The pyinterface are mediators between procedures using python objects and the c implementations.

FILE REFERENCES :
Name        I / O        Description
None----       ----------

ABNORMAL TERMINATION CONDITIONS, ERROR AND WARNING MESSAGES :
None
************************************************************************************************************************************/
#ifndef PATCHEXTRACTION_DLL_H_INCLUDED
#define PATCHEXTRACTION_DLL_H_INCLUDED

#ifdef BUILDING_PYTHON_MODULE
    #include <Python.h>
    #include <numpy/ndarraytypes.h>
    #include <numpy/ufuncobject.h>
    #include <numpy/npy_3kcompat.h>
    #define PATCHEXTRACTION_DLL_PUBLIC
    #define PATCHEXTRACTION_DLL_LOCAL 
#else
    #if defined(_WIN32) || defined(_WIN64)
        #ifdef BUILDING_PATCHEXTRACTION_DLL
            #ifdef __GNUC__
                #define PATCHEXTRACTION_DLL_PUBLIC __attribute__ ((dllexport))
            #else
                #define PATCHEXTRACTION_DLL_PUBLIC __declspec(dllexport)
            #endif
        #else
            #ifdef __GNUC__
                #define PATCHEXTRACTION_DLL_PUBLIC __attribute__ ((dllimport))
            #else
                #define PATCHEXTRACTION_DLL_PUBLIC __declspec(dllimport)
            #endif
        #endif
        #define PATCHEXTRACTION_DLL_LOCAL
    #else
        #if __GNUC__ >= 4
            #define PATCHEXTRACTION_DLL_PUBLIC __attribute__ ((visibility ("default")))
            #define PATCHEXTRACTION_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
        #else
            #define PATCHEXTRACTION_DLL_PUBLIC
            #define PATCHEXTRACTION_DLL_LOCAL
        #endif
    #endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "random_numbers_generator.h"

#ifndef NDEBUG
#define DEBMSG(MESSAGE) printf(MESSAGE)
#define DEBNUMMSG(MESSAGE, NUM) printf(MESSAGE, NUM);
#else
#define DEBMSG(MESSAGE) 
#define DEBNUMMSG(MESSAGE, NUM) 
#endif

double * PATCHEXTRACTION_DLL_LOCAL defineClass_Full(double * source, const unsigned int height, const unsigned int width, const unsigned int patch_size, unsigned int patch_stride, const double threshold_count);

double * PATCHEXTRACTION_DLL_LOCAL defineClass_Center(double * source, const unsigned int height, const unsigned int width, const unsigned int patch_size, unsigned int patch_stride);

double * PATCHEXTRACTION_DLL_LOCAL extractPatches_impl(double * source, unsigned int * samples_count, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, unsigned int patch_stride);

double * PATCHEXTRACTION_DLL_LOCAL extractSampledPatches_impl(double * source, unsigned int * sample_list, const unsigned int sample_size, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size);

void PATCHEXTRACTION_DLL_LOCAL markValidPatches_Full(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size, const double threshold_count);
void PATCHEXTRACTION_DLL_LOCAL fillValidPatches_Full(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size, const double threshold_count);

void PATCHEXTRACTION_DLL_LOCAL markValidPatches_Center(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size);
void PATCHEXTRACTION_DLL_LOCAL fillValidPatches_Center(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size);

double * PATCHEXTRACTION_DLL_LOCAL defineBackground(double * class_labels, unsigned int * background_count, const unsigned int height, const unsigned int width, const unsigned int patch_size);

double * PATCHEXTRACTION_DLL_LOCAL defineForeground(double * class_labels, const unsigned char patch_extraction_mode, unsigned int * foreground_count, const unsigned int height, const unsigned int width, const unsigned int patch_size);

unsigned int PATCHEXTRACTION_DLL_LOCAL balanceSamples(unsigned int * foreground_count, unsigned int * background_count);

unsigned int * PATCHEXTRACTION_DLL_LOCAL samplePatches(double * class_labels, unsigned int labels_count, unsigned int sample_size, const unsigned int height, const unsigned int width);

#ifdef BUILDING_PYTHON_MODULE
static PyObject* computeClasses(PyObject *self, PyObject *args);
static PyObject* extractPatches(PyObject *self, PyObject *args);
static PyObject* computeSampledClasses(PyObject *self, PyObject *args);
static PyObject* extractSampledPatches(PyObject *self, PyObject *args);
static PyObject* defineClasses(PyObject *self, PyObject *args);
#endif


#endif //PATCHEXTRACTION_DLL_H_INCLUDED
