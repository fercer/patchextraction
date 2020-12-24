/***********************************************************************************************************************************
CENTRO DE INVESTIGACION EN MATEMATICAS
DOCTORADO EN CIENCIAS DE LA COMPUTACION
FERNANDO CERVANTES SANCHEZ

FILE NAME : patchextraction.h

PURPOSE : Declares the functions required to extract patches using c/c++ api of python.
************************************************************************************************************************************/

#ifndef PATCHEXTRACTION_DLL_H_INCLUDED
#define PATCHEXTRACTION_DLL_H_INCLUDED

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#define PATCHEXTRACTION_DLL_PUBLIC
#define PATCHEXTRACTION_DLL_LOCAL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "random_numbers_generator.h"

inline void PATCHEXTRACTION_DLL_LOCAL extract_patch(char* input, char* output, const int patch_id, const int x, const int y, const int channels, const int patch_ini, const int patch_end, npy_intp* strides_src, npy_intp* strides_dst);

inline unsigned int PATCHEXTRACTION_DLL_LOCAL mark_patch(char* input, const int x, const int y, const int patch_ini, const int patch_end, const char * background_label, const size_t background_label_size, npy_intp* strides_src);

void PATCHEXTRACTION_DLL_LOCAL mark_patches(char* input, char* output, const int height, const int width, const int patch_size, const int patch_stride, const char* background_label, const size_t background_label_size, const unsigned int threshold_count, npy_intp* strides_src);

void PATCHEXTRACTION_DLL_LOCAL mark_patches_center(char* input, char* output, const int height, const int width, const int patch_size, const int patch_stride, const char* background_label, const size_t background_label_size, npy_intp* strides_src);

char* PATCHEXTRACTION_DLL_LOCAL define_foreground(char* input, unsigned int* foreground_count, const int height, const int width, const int patch_size, const int patch_stride, const unsigned int threshold_count, const char* background_label, const size_t background_label_size, npy_intp* strides_src, const int extract_full_patch);

void PATCHEXTRACTION_DLL_LOCAL sample_patches(char* input, unsigned int* sample_indices, unsigned int labels_count, unsigned int sample_size, const int marked_height, const int marked_width);

unsigned int* PATCHEXTRACTION_DLL_PUBLIC compute_patches(char* input, const int height, const int width, const int patch_size, const int patch_stride, const unsigned int threshold_count, const char* background_label, const size_t background_label_size, const double sample_percentage, unsigned int* sample_size, npy_intp* strides_src, const int extract_full_patch);

void PATCHEXTRACTION_DLL_PUBLIC extract_sampled_patches(char* input, char* output, unsigned int* sampled_indices, unsigned int sample_size, const int channels, const int height, const int width, const int patch_size, const int patch_stride, int extract_full_patch, npy_intp* strides_src, npy_intp* strides_dst);

void PATCHEXTRACTION_DLL_PUBLIC merge_patches(char* input, char* output, unsigned int n_patches, const int channels, const int height, const int width, const int patch_size, npy_intp* strides_src, npy_intp* strides_dst);


static PyObject* compute_classes_api(PyObject* self, PyObject* args, PyObject* kw);
static PyObject* extract_patches_api(PyObject* self, PyObject* args, PyObject* kw);
static PyObject* merge_patches_api(PyObject* self, PyObject* args);

#endif //PATCHEXTRACTION_DLL_H_INCLUDED
