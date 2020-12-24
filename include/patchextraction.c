/***********************************************************************************************************************************
CENTRO DE INVESTIGACION EN MATEMATICAS
DOCTORADO EN CIENCIAS DE LA COMPUTACION
FERNANDO CERVANTES SANCHEZ

FILE NAME : patchextraction.c

PURPOSE : Defines the functions required to extract patches using c/c++ api of python.
************************************************************************************************************************************/
#include "patchextraction.h"


inline void extract_patch(char* input, char* output, const int patch_id, const int x, const int y, const int channels, const int patch_ini, const int patch_end, npy_intp* strides_src, npy_intp* strides_dst)
{
    int i, j, pi, pj, pc;
    for (i = patch_ini, pi = 0; i < patch_end; i++, pi++)
    {
        for (j = patch_ini, pj = 0; j < patch_end; j++, pj++)
        {
            for (pc = 0; pc < channels; pc++)
            {
                // The current patch contains at least one positive pixel
                memcpy(output + patch_id * strides_dst[0] + pc * strides_dst[1] + pi * strides_dst[2] + pj * strides_dst[3],
                    input + pc * strides_src[0] + (y + i) * strides_src[1] + (x + j) * strides_src[2], (size_t)strides_src[2]);
            }
        }
    }
}


inline unsigned int mark_patch(char* input, const int x, const int y, const int patch_ini, const int patch_end, const char* background_label, const size_t background_label_size, npy_intp* strides_src)
{
    unsigned int patch_sum = 0;

    if (background_label)
    {
        for (int i = patch_ini; i < patch_end; i++)
        {
            for (int j = patch_ini; j < patch_end; j++)
            {
                // The current patch contains at least one positive pixel
                patch_sum += memcmp(input + (y + i) * strides_src[1] + (x + j) * strides_src[2], background_label, background_label_size) != 0;
            }
        }
    }
    else
    {
        patch_sum = (unsigned int)((patch_end - patch_ini) * (patch_end - patch_ini));
    }

    return patch_sum;
}


void mark_patches(char* input, char* output, const int height, const int width, const int patch_size, const int patch_stride, const char* background_label, const size_t background_label_size, const unsigned int threshold_count, npy_intp* strides_src)
{
    const int offset_y = patch_size / 2;
    const int offset_x = patch_size / 2;

    const int patch_ini = -patch_size / 2;
    const int patch_end = patch_size / 2 + patch_size % 2;

    const int marked_width = (width - patch_size + patch_stride) / patch_stride;
    const int marked_height = (height - patch_size + patch_stride) / patch_stride;

    memset(output, -1, marked_height * marked_width * sizeof(char));
    unsigned int patch_sum;
    for (int y = offset_y, py = 0; y < height - offset_y + (patch_size + 1) % 2; y += patch_stride, py++)
    {
        for (int x = offset_x, px = 0; x < width - offset_x + (patch_size + 1) % 2; x += patch_stride, px++)
        {
            patch_sum = mark_patch(input, x, y, patch_ini, patch_end, background_label, background_label_size, strides_src);
            // If the current patch contains at least 'threshold_count' pixels, the current patch is considered for extraction
            if (patch_sum > threshold_count)
            {
                *(output + py * marked_width + px) = 1;
            }
        }
    }
}


void mark_patches_center(char* input, char* output, const int height, const int width, const int patch_size, const int patch_stride, const char* background_label, const size_t background_label_size, npy_intp* strides_src)
{
    const int offset_y = patch_size / 2;
    const int offset_x = patch_size / 2;

    const int marked_width = (width - patch_size + patch_stride) / patch_stride;
    const int marked_height = (height - patch_size + patch_stride) / patch_stride;

    if (background_label)
    {
        memset(output, -1, marked_height * marked_width * sizeof(char));
        for (int y = offset_y, py = 0; y < height - offset_y + (patch_size + 1) % 2; y += patch_stride, py++)
        {
            for (int x = offset_x, px = 0; x < width - offset_x + (patch_size + 1) % 2; x += patch_stride, px++)
            {
                *(output + py * marked_width + px) = memcmp(input + y * strides_src[1] + x * strides_src[2], background_label, background_label_size) != 0;
            }
        }
    }
    else
    {
        memset(output, 1, marked_height * marked_width * sizeof(char));
    }
}


char* define_foreground(char* input, unsigned int* foreground_count, const int height, const int width, const int patch_size, const int patch_stride, const unsigned int threshold_count, const char* background_label, const size_t background_label_size, npy_intp* strides_src, const int extract_full_patch)
{
    const int marked_width = (width - patch_size + patch_stride) / patch_stride;
    const int marked_height = (height - patch_size + patch_stride) / patch_stride;

    // Define foreground as all patches containing any labeled pixel
    char* output = (char*)malloc(marked_height * marked_width * sizeof(char));

    if (extract_full_patch)
    {
        mark_patches(input, output, height, width, patch_size, patch_stride, background_label, background_label_size, threshold_count, strides_src);
    }
    else
    {
        mark_patches_center(input, output, height, width, patch_size, patch_stride, background_label, background_label_size, strides_src);
    }

    if (foreground_count) {
        *foreground_count = 0;
        for (int xy = 0; xy < marked_height * marked_width; xy++)
        {
            *foreground_count = *foreground_count + (*(output + xy) > 0);
        }
    }

    return output;
}


void sample_patches(char* input, unsigned int* sample_indices, unsigned int labels_count, unsigned int sample_size, const int marked_height, const int marked_width)
{
    // The first pass indexes the positions of each class pixel
    unsigned int indexed_count = 0;
    for (int xy = 0; (xy < marked_height * marked_width) && (indexed_count < labels_count); xy++)
    {
        if (*(input + xy) > 0)
        {
            *(sample_indices + indexed_count++) = xy;
        }
    }

    if (sample_size == labels_count)
    {
        return;
    }

    STAUS* rng_seed = initSeed(0);
    unsigned int j, swap_idx;
    // Perform a random permutation of the indices, and keep only 'sample_size' indices
    for (unsigned int i = 0; i < sample_size; i++)
    {
        j = (int)floor(HybTaus((double)i, (double)labels_count - 1e-1, rng_seed));
        swap_idx = *(sample_indices + i);
        *(sample_indices + i) = *(sample_indices + j);
        *(sample_indices + j) = swap_idx;
    }
    free(rng_seed);
}


unsigned int* compute_patches(char* input, const int height, const int width, const int patch_size, const int patch_stride, const unsigned int threshold_count, const char* background_label, const size_t background_label_size, const double sample_percentage, unsigned int* sample_size, npy_intp* strides_src, const int extract_full_patch)
{
    const int marked_width = (width - patch_size + patch_stride) / patch_stride;
    const int marked_height = (height - patch_size + patch_stride) / patch_stride;

    // Define the foreground and background:
    unsigned int foreground_count;
    char* foreground = define_foreground(input, &foreground_count, height, width, patch_size, patch_stride, threshold_count, background_label, background_label_size, strides_src, extract_full_patch);

    unsigned int* sample_indices = (unsigned int*)malloc(foreground_count * sizeof(unsigned int));

    const unsigned int foreground_sample_size = (unsigned int)ceil(foreground_count * sample_percentage);
    sample_patches(foreground, sample_indices, foreground_count, foreground_sample_size, marked_height, marked_width);
    free(foreground);

    *sample_size = foreground_sample_size;
    return sample_indices;
}


void extract_sampled_patches(char* input, char* output, unsigned int* sampled_indices, unsigned int sample_size, const int channels, const int height, const int width, const int patch_size, const int patch_stride, int extract_full_patch, npy_intp* strides_src, npy_intp* strides_dst)
{
    const int offset_y = patch_size / 2;
    const int offset_x = patch_size / 2;

    const int patch_ini = -patch_size / 2;
    const int patch_end = patch_size / 2 + patch_size % 2;

    const int marked_width = (width - patch_size + patch_stride) / patch_stride;
    const int marked_height = (height - patch_size + patch_stride) / patch_stride;

    //#pragma omp parallel num_threads(4) default(none) firstprivate(sample_size, marked_width, offset_y, offset_x, patch_ini, patch_end, patch_stride, channels) shared(input, output, sampled_indices, strides_src, strides_dst)
    unsigned int x, y;
    if (extract_full_patch)
    {
        for (unsigned int s = 0; s < sample_size; s++)
        {
            y = (*(sampled_indices + s) / marked_width) * patch_stride + offset_y;
            x = (*(sampled_indices + s) % marked_width) * patch_stride + offset_x;
            extract_patch(input, output, s, x, y, channels, patch_ini, patch_end, strides_src, strides_dst);
        }
    }
    else
    {
        for (unsigned int s = 0; s < sample_size; s++)
        {
            y = (*(sampled_indices + s) / marked_width) * patch_stride + offset_y;
            x = (*(sampled_indices + s) % marked_width) * patch_stride + offset_x;

            // The current patch contains at least one positive pixel
            memcpy(output + s * strides_dst[0], input + y * strides_src[1] + x * strides_src[2], (size_t)strides_src[2]);
        }
    }
}


void merge_patches(char* input, char* output, unsigned int n_patches, const int channels, const int height, const int width, const int patch_size, npy_intp* strides_src, npy_intp* strides_dst)
{
    const unsigned int n_patches_per_width = width / patch_size;

    unsigned int left_offset = 0, top_offset = 0;
    for (unsigned int n = 0; n < n_patches; n++)
    {
        top_offset = (n / n_patches_per_width) * patch_size;
        left_offset = (n % n_patches_per_width) * patch_size;

        for (int py = 0; py < patch_size; py++)
        {
            for (int px = 0; px < patch_size; px++)
            {
                for (int pc = 0; pc < channels; pc++)
                {
                    memcpy(output + pc * strides_dst[0] + (top_offset + py) * strides_dst[1] + (left_offset + px) * strides_dst[2],
                        input + n * strides_src[0] + pc * strides_src[1] + py * strides_src[2] + px * strides_src[3], strides_src[3]);
                }
            }
        }
    }
}


static PyObject* compute_classes_api(PyObject* self, PyObject* args, PyObject* kw)
{
    PyArrayObject* source;
    int channels, height, width;
    int patch_size, patch_stride;

    int threshold_count = -1;
    double sample_percentage = 1.0;

    PyArrayObject* background_label = NULL;
    int extract_full_patch = 1;

    static char* keywords[] = { "","","","background_label","sample_percentage", "threshold_count", "extract_patch", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!ii|O!dip", keywords, &PyArray_Type, &source, &patch_size, &patch_stride, &PyArray_Type, &background_label, &sample_percentage, &threshold_count, &extract_full_patch))
    {
        return NULL;
    }

    if (threshold_count < 0)
    {
        threshold_count = (int)floor((double)(patch_size * patch_size) / 9.0);
    }

    npy_intp strides_src[3];
    if (source->nd > 2)
    {
        channels = (unsigned int)source->dimensions[0];
        height = (unsigned int)source->dimensions[1];
        width = (unsigned int)source->dimensions[2];

        strides_src[0] = source->strides[0];
        strides_src[1] = source->strides[1];
        strides_src[2] = source->strides[2];
    }
    else
    {
        channels = 1;
        height = (unsigned int)source->dimensions[0];
        width = (unsigned int)source->dimensions[1];

        strides_src[0] = 0;
        strides_src[1] = source->strides[0];
        strides_src[2] = source->strides[1];
    }

    const unsigned int n_samples_per_width = (width - patch_size + patch_stride) / patch_stride;
    const unsigned int n_samples_per_height = (height - patch_size + patch_stride) / patch_stride;

    unsigned int sample_size;
    size_t background_label_size = 0;

    char* background_label_data = NULL;
    if (background_label)
    {
        background_label_data = background_label->data;
        background_label_size = (size_t)(int)background_label->strides[0];
    }

    unsigned int* sample_indices = compute_patches(source->data, height, width, patch_size, patch_stride, (unsigned int)threshold_count, background_label_data, background_label_size, sample_percentage, &sample_size, strides_src, extract_full_patch);

    npy_intp samples_shape[] = { sample_size };
    PyArrayObject* sample_indices_array = (PyArrayObject*)PyArray_SimpleNew(1, &samples_shape[0], NPY_UINT32);

    for (unsigned int i = 0; i < sample_size; i++)
    {
        *(unsigned int*)(sample_indices_array->data + i * sample_indices_array->strides[0]) = *(sample_indices + i);
    }

    free(sample_indices);

    return (PyObject*)sample_indices_array;
}


static PyObject* extract_patches_api(PyObject* self, PyObject* args, PyObject* kw)
{
    PyArrayObject* source;
    PyArrayObject* sampled_indices = NULL;
    unsigned int height, width, channels;
    unsigned int patch_size, patch_stride;
    int extract_full_patch = 1;

    static char* keywords[] = { "","","", "sample_indices", "extract_patch", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!ii|O!p", keywords, &PyArray_Type, &source, &patch_size, &patch_stride, &PyArray_Type, &sampled_indices, &extract_full_patch))
    {
        return NULL;
    }

    npy_intp strides_src[3];
    if (source->nd > 2)
    {
        channels = (unsigned int)source->dimensions[0];
        height = (unsigned int)source->dimensions[1];
        width = (unsigned int)source->dimensions[2];

        strides_src[0] = source->strides[0];
        strides_src[1] = source->strides[1];
        strides_src[2] = source->strides[2];
    }
    else
    {
        channels = 1;
        height = (unsigned int)source->dimensions[0];
        width = (unsigned int)source->dimensions[1];

        strides_src[0] = 0;
        strides_src[1] = (unsigned int)source->strides[0];
        strides_src[2] = (unsigned int)source->strides[1];
    }

    unsigned int n_patches = 0;
    unsigned int* sampled_indices_data = NULL;
    if (sampled_indices)
    {
        n_patches = (unsigned int)sampled_indices->dimensions[0];
        sampled_indices_data = (unsigned int*)(sampled_indices->data);
    }
    else
    {
        const int marked_width = (width - patch_size + patch_stride) / patch_stride;
        const int marked_height = (height - patch_size + patch_stride) / patch_stride;
        n_patches = marked_width * marked_height;
        sampled_indices_data = (unsigned int*)malloc(marked_height * marked_width * sizeof(unsigned int));
        for (unsigned int i = 0; i < marked_height * marked_width; i++)
        {
            *(sampled_indices_data + i) = i;
        }
    }

    npy_intp samples_shape[] = { n_patches, channels, extract_full_patch ? patch_size : 1, extract_full_patch ? patch_size : 1 };
    PyArrayObject* patch_samples = (PyArrayObject*)PyArray_SimpleNew(4, &samples_shape[0], source->descr->type);

    extract_sampled_patches(source->data, patch_samples->data, sampled_indices_data, n_patches, channels, height, width, patch_size, patch_stride, extract_full_patch, strides_src, patch_samples->strides);
    if (!sampled_indices)
    {
        free(sampled_indices_data);
    }
    return (PyObject*)patch_samples;
}


static PyObject* merge_patches_api(PyObject* self, PyObject* args)
{
    PyArrayObject* source;
    unsigned int height, width, channels, n_patches;
    unsigned int patch_size;

    if (!PyArg_ParseTuple(args, "O!II", &PyArray_Type, &source, &height, &width))
    {
        return NULL;
    }

    npy_intp strides_src[4];
    if (source->nd > 3)
    {
        n_patches = (unsigned int)source->dimensions[0];
        channels = (unsigned int)source->dimensions[1];
        patch_size = (unsigned int)source->dimensions[2];

        strides_src[0] = source->strides[0];
        strides_src[1] = source->strides[1];
        strides_src[2] = source->strides[2];
        strides_src[3] = source->strides[3];
    }
    else
    {
        n_patches = 1;
        channels = (unsigned int)source->dimensions[0];
        patch_size = (unsigned int)source->dimensions[1];

        strides_src[0] = 0;
        strides_src[1] = (unsigned int)source->strides[0];
        strides_src[2] = (unsigned int)source->strides[1];
        strides_src[3] = (unsigned int)source->strides[2];
    }

    npy_intp samples_shape[] = { channels, height, width };
    PyArrayObject* merged_patches = (PyArrayObject*)PyArray_SimpleNew(3, &samples_shape[0], source->descr->type);
    memset(merged_patches->data, 0, channels * height * width * merged_patches->strides[2] * sizeof(char));

    merge_patches(source->data, merged_patches->data, n_patches, channels, height, width, patch_size, strides_src, merged_patches->strides);

    return (PyObject*)merged_patches;
}


static PyMethodDef patchextraction_methods[] = {
    /* Casting to (PyCFunction)(void(*)(void)) is necessary due to compute_classes_api and extract_patches_api receive three arguments instead of two */
    { "computePatches", (PyCFunction)(void(*)(void)) compute_classes_api, METH_VARARGS | METH_KEYWORDS, "Compute and sample the labeled patches of given size and patch_stride." },
    { "extractPatches", (PyCFunction)(void(*)(void)) extract_patches_api, METH_VARARGS | METH_KEYWORDS, "Extract indexed patches." },
    { "mergePatches", merge_patches_api, METH_VARARGS, "Merge patches." },
    { NULL, NULL, 0, NULL }
};


static struct PyModuleDef patchextraction_moduledef = {
    PyModuleDef_HEAD_INIT,
    "patchextraction",
    NULL,
    -1,
    patchextraction_methods
};


PyMODINIT_FUNC PyInit_patchextraction(void)
{
    PyObject* m;
    m = PyModule_Create(&patchextraction_moduledef);
    if (!m) {
        return NULL;
    }
    import_array();

    return m;
}
