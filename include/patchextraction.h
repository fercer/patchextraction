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

void PATCHEXTRACTION_DLL_LOCAL extractPatchClass_same(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractPatchClass_center(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractPatchClass_mean(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractPatchClass_max(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractSinglePatchAndClass_same(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractSinglePatchAndClass_center(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractSinglePatchAndClass_mean(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractSinglePatchAndClass_max(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);
void PATCHEXTRACTION_DLL_LOCAL extractSinglePatch(double * input, double * output, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);

void PATCHEXTRACTION_DLL_PUBLIC computeClasses_impl(double * input, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode);
unsigned int PATCHEXTRACTION_DLL_PUBLIC generatePatchesSample_impl(unsigned int ** sample_indices, double * class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const double sample_percentage);
void PATCHEXTRACTION_DLL_PUBLIC samplePatches_impl(double * input, double ** output, double * class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage, unsigned int ** sample_indices_ptr);
void PATCHEXTRACTION_DLL_PUBLIC extractSampledPatchesAndClasses_impl(double * input, double ** output, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, unsigned int * sample_indices, const unsigned int sample_size);
void PATCHEXTRACTION_DLL_PUBLIC extractSampledPatches_impl(double * input, double ** output, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, unsigned int * sample_indices, const unsigned int sample_size);
void PATCHEXTRACTION_DLL_PUBLIC extractAllPatchesAndClasses_impl(double * input, double ** output, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode);
void PATCHEXTRACTION_DLL_PUBLIC extractAllPatches_impl(double * input, double ** output, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride);

#ifdef BUILDING_PYTHON_MODULE
static PyObject* computeClasses(PyObject *self, PyObject *args);

static PyObject* generatePatchesSample(PyObject *self, PyObject *args, PyObject *kw);

static PyObject* extractSampledPatchesAndClasses_pyinterface(PyArrayObject *input, PyArrayObject *labels, PyArrayObject *patches_sample, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage);
static PyObject* extractSampledPatchesAndClasses(PyObject *self, PyObject *args, PyObject *kw);

static PyObject* extractSampledPatches_pyinterface(PyArrayObject *input, PyArrayObject *labels, PyArrayObject *patches_sample, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage);
static PyObject* extractSampledPatches(PyObject *self, PyObject *args, PyObject *kw);

static PyObject* extractAllPatchesAndClasses_pyinterface(PyArrayObject *input, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode);
static PyObject* extractAllPatchesAndClasses(PyObject *self, PyObject *args, PyObject *kw);

static PyObject* extractAllPatches_pyinterface(PyArrayObject *input, const unsigned int patch_size, const unsigned int patch_stride);
static PyObject* extractAllPatches(PyObject *self, PyObject *args);

static PyObject* samplePatches(PyObject *self, PyObject *args, PyObject *kw);
#endif

#endif //PATCHEXTRACTION_DLL_H_INCLUDED
