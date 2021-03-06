/***********************************************************************************************************************************
CENTRO DE INVESTIGACION EN MATEMATICAS
DOCTORADO EN CIENCIAS DE LA COMPUTACION
FERNANDO CERVANTES SANCHEZ

FILE NAME : patchextraction.c

PURPOSE : Defines the functions required to extract patches using c++ instead of python.

FILE REFERENCES :
Name        I / O        Description
None----       ----------

ABNORMAL TERMINATION CONDITIONS, ERROR AND WARNING MESSAGES :
None
************************************************************************************************************************************/
#include "patchextraction.h"

double * defineClass_Full(double * source, const unsigned int height, const unsigned int width, const unsigned int patch_size, unsigned int patch_stride, const double threshold_count)
{
    const unsigned int n_samples_per_width = (width - patch_size + patch_stride) / patch_stride;
    const unsigned int n_samples_per_height = (height - patch_size + patch_stride) / patch_stride;
    
    const int offset = patch_size/2;
    double * patches_classes = (double*) malloc(n_samples_per_width*n_samples_per_height*sizeof(double));
	memset(patches_classes, 0, n_samples_per_width*n_samples_per_height * sizeof(double));

    unsigned int x, y;
    for (unsigned int ys = 0; ys < n_samples_per_height; ys++)
    {
        y = ys*patch_stride + offset;
        for (unsigned int xs = 0; xs < n_samples_per_width; xs++)
        {
            x = xs*patch_stride + offset;
                        
			double patch_sum = 0.0;

            for (int i = -offset; (i < offset) && (patch_sum < threshold_count); i++)
            {
                for (int j = -offset; (j < offset) && (patch_sum < threshold_count); j++)
                {
                    patch_sum += *(source + (y+i)*width + x+j);
                }
            }

			if (patch_sum < threshold_count)
			{
				continue;
			}

			*(patches_classes + ys*n_samples_per_width + xs) = 1.0;
        }
    }
    
    return patches_classes;
}


double * defineClass_Center(double * source, const unsigned int height, const unsigned int width, const unsigned int patch_size, unsigned int patch_stride)
{
	const unsigned int n_samples_per_width = (width - patch_size + patch_stride) / patch_stride;
	const unsigned int n_samples_per_height = (height - patch_size + patch_stride) / patch_stride;

    printf("Extracting samples %i/height, %i/width\n", n_samples_per_height, n_samples_per_width);

	const int offset = patch_size / 2;
	double * patches_classes = (double*)malloc(n_samples_per_width*n_samples_per_height * sizeof(double));

	for (unsigned int ys = 0; ys < n_samples_per_height; ys++)
	{
		for (unsigned int xs = 0; xs < n_samples_per_width; xs++)
		{

			*(patches_classes + ys*n_samples_per_width + xs) = *(source + (ys*patch_stride + offset)*width + xs*patch_stride + offset);
		}
	}

	return patches_classes;
}


double * extractPatches_impl(double * source, unsigned int * samples_count, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, unsigned int patch_stride)
{
	const unsigned int n_samples_per_width = (width - patch_size + patch_stride) / patch_stride;
	const unsigned int n_samples_per_height = (height - patch_size + patch_stride) / patch_stride;
	*samples_count = n_samples_per_width * n_samples_per_height;

	const int offset = patch_size / 2;
	double * sampled_patches = (double*)malloc(n_samples_per_width*n_samples_per_height*patch_size*patch_size*n_channels * sizeof(double));
	for (unsigned int ys = 0; ys < n_samples_per_height; ys++)
	{
		for (unsigned int xs = 0; xs < n_samples_per_width; xs++)
		{
			for (unsigned int z = 0; z < n_channels; z++)
			{
				for (int i = -offset; i < offset; i++)
				{
					for (int j = -offset; j < offset; j++)
					{
						*(sampled_patches + (ys*n_samples_per_width + xs)*n_channels*patch_size*patch_size + z*patch_size*patch_size + (i + offset)*patch_size + j+offset) =
							*(source + z*height*width + (ys*patch_stride + offset + i)*width + xs*patch_stride + offset + j);
                    }
				}
			}
		}
	}

	return sampled_patches;
}


double * extractSampledPatches_impl(double * source, unsigned int * sample_list, const unsigned int sample_size, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size)
{
    const int offset = patch_size/2;
    double * sampled_patches = (double*) malloc(sample_size*patch_size*patch_size*n_channels*sizeof(double));
    double * sampled_patches_ptr;
    unsigned int xy, x, y;
    for (unsigned int s = 0; s < sample_size; s++)
    {
        xy = *(sample_list + s);
        x = xy % width;
        y = xy / width;

        sampled_patches_ptr = sampled_patches + s*n_channels*patch_size*patch_size;
        
        for (unsigned int z = 0; z < n_channels; z++)
        {
            for (int i = -offset; i < offset; i++)
            {
                for (int j = -offset; j < offset; j++)
                {
                    *(sampled_patches_ptr + z*patch_size*patch_size + (i+offset)*patch_size + j+offset) =
                    *(source + z*width*height + (y + i)*width + x + j);
                }
            }
                
        }
    }
    
    return sampled_patches;
}



void markValidPatches_Full(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size, const double threshold_count)
{
    memset((void*)output, 0, height*width*sizeof(double));
    const int offset = patch_size/2;
    int i, j;
    
    for (unsigned int y = 0; y < height; y++)
    {
        if (((int)y < offset) || ((int)(height - y) < offset))
        {
            continue;
        }
        
        for (unsigned int x = 0; x < width; x++)
        {
            
            if (((int)x < offset) || ((int)(width - x) < offset))
            {
                continue;
            }
            
            double patch_sum = 0.0;
            
            for (i = -offset; (i < offset) && (patch_sum < threshold_count); i++)
            {
                for (j = -offset; (j < offset) && (patch_sum < threshold_count); j++)
                {
                    // If the current patch contains at least one positive pixel, fill the complete output corresponding patch.
                    patch_sum += *(class_labels + (y + i)*width + x + j);
                }
            }
            
            if (patch_sum < threshold_count)
            {
                continue;
            }
            
            // Fill the output center if the threhsold is hit.
            *(output + y*width + x) = 1.0;
        }
    }
}


void fillValidPatches_Full(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size, const double threshold_count)
{
    memset(output, 0, height*width*sizeof(double));

    const int offset = patch_size/2;
    int i, j;
    
    for (int y = offset; y < ((int)height-offset); y++)
    {
        for (int x = offset; x < ((int)width-offset); x++)
        {
            double patch_sum = 0.0;
            
            for (i = -offset; (i < offset) && (patch_sum < threshold_count); i++)
            {
                for (j = -offset; (j < offset) && (patch_sum < threshold_count); j++)
                {
                    // If the current patch contains at least one positive pixel, fill the complete output corresponding patch.
                    patch_sum += *(class_labels + (y + i)*width + x + j);
                }
            }
            
            if (patch_sum < threshold_count)
            {
                continue;
            }
            
            // Fill the output center if the threhsold is hit.
            for (i = -offset; i < offset; i++)
            {
                for (j = -offset; j < offset; j++)
                {
                    *(output + (y+i)*width + x+j) = 1.0;
                }
            }
        }
    }
}


void markValidPatches_Center(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size)
{
    memset((void*)output, 0, height*width*sizeof(double));
    const int offset = patch_size/2;
    
    for (unsigned int y = 0; y < height; y++)
    {
        if (((int)y < offset) || ((int)(height - y) < offset))
        {
            continue;
        }
        
        for (unsigned int x = 0; x < width; x++)
        {
            if (((int)x < offset) || ((int)(width - x) < offset))
            {
                continue;
            }
            
            *(output + y*width + x) = *(class_labels + y*width + x);
        }
    }
}


void fillValidPatches_Center(double * class_labels, double * output, const unsigned int height, const unsigned int width, const unsigned int patch_size)
{
    memset((void*)output, 0, height*width*sizeof(double));
    const int offset = patch_size/2;
    
    for (unsigned int y = 0; y < height; y++)
    {
        if (((int)y < offset) || ((int)(height - y) < offset))
        {
            continue;
        }
        
        for (unsigned int x = 0; x < width; x++)
        {
            if (((int)x < offset) || ((int)(width - x) < offset))
            {
                continue;
            }
            
            for (int i = -offset; i<offset; i++)
            {
                for (int j=-offset; j<offset; j++)
                {                    
                    *(output + (y+i)*width + x+j) = 1.0;
                }
            }
        }
    }
}



double * defineBackground(double * class_labels, unsigned int * background_count, const unsigned int height, const unsigned int width, const unsigned int patch_size)
{
    // Define background as all patches that dows not contain any foreground pixel
    double * output_1 = (double*)malloc(height*width*sizeof(double));
    double * output_2 = (double*)malloc(height*width*sizeof(double));

    fillValidPatches_Full(class_labels, output_1, height, width, patch_size, 1.0);
    fillValidPatches_Full(output_1, output_2, height, width, 3, 1.0);
    
    memset((void*)output_1, 0, height*width*sizeof(double));
    for (unsigned int xy = 0; xy < height*width; xy++)
    {
        *(output_1 + xy) = 1.0 - *(output_2 + xy);
    }  
    markValidPatches_Full(output_1, output_2, height, width, patch_size, (double)(patch_size*patch_size));
    free(output_1);

    if (background_count){
        for (unsigned int xy = 0; xy < height*width; xy++)
        {
            *background_count = *background_count + (unsigned int)*(output_2 + xy);
        }
    }

    return output_2;
}



double * defineForeground(double * class_labels, const unsigned char patch_extraction_mode, unsigned int * foreground_count, const unsigned int height, const unsigned int width, const unsigned int patch_size)
{
    // Define background as all patches that dows not contain any foreground pixel
    double * output = (double*)malloc(height*width*sizeof(double));
    printf("Define foreground for patch extraction mode: %i\n", (int)patch_extraction_mode);
    switch ((int)patch_extraction_mode)
    {
        case 0:
            markValidPatches_Full(class_labels, output, height, width, patch_size, 1.0);
            break;
        
        case 1:
            markValidPatches_Center(class_labels, output, height, width, patch_size);
            break;
    }
    
    if (foreground_count){
        for (unsigned int xy = 0; xy < height*width; xy++)
        {
            *foreground_count = *foreground_count + (unsigned int)*(output + xy);
        }
    }
    
    return output;
}



unsigned int balanceSamples(unsigned int * foreground_count, unsigned int * background_count)
{
    unsigned int max_size;
    if (*foreground_count > *background_count)
    {
        *foreground_count = *background_count;
        max_size = *background_count;
    }
    else
    {
        *background_count = *foreground_count;
        max_size = *foreground_count;
    }
    
    return max_size;
}



unsigned int * samplePatches(double * class_labels, unsigned int labels_count, unsigned int sample_size, const unsigned int height, const unsigned int width)
{
    unsigned int * sample_list = (unsigned int *)malloc(labels_count * sizeof(unsigned int));
    
    printf("Sampling %i from %i patches.\n", sample_size, labels_count);
    unsigned int min_index = height*width + 1;
    // The first pass indexes the positions of each class pixel
    unsigned int indexed_count = 0;
    unsigned int xy;
    for (xy = 0; (xy < height*width) && (indexed_count < labels_count); xy++)
    {
        if (*(class_labels + xy) > 0.5)
        {
            *(sample_list + indexed_count++) = xy;
            if (min_index > xy)
            {
                min_index = xy;
            }
        }
    }
    
    printf("Min index before: %i [%i]\n", min_index, xy);

    STAUS *rng_seed = initSeed(0);
    unsigned int j, swap_idx;
    min_index = height*width + 1;
    // Perform a random permutation of the indices, and keep only 'sample_size' indices
    for (unsigned int i = 0; i < sample_size; i++)
    {
        j = (int)floor(HybTaus((double)i, (double)indexed_count-1e-3, rng_seed));
        swap_idx = *(sample_list + i);
        *(sample_list + i) = *(sample_list + j);
        *(sample_list + j) = swap_idx;

        if (min_index > *(sample_list + i))
        {
            printf("Swapping point: %i/%i\n", j, indexed_count);
            min_index = *(sample_list + i);
        }
    }
    printf("Min index after: %i\n", min_index);
    
    free(rng_seed);
    return sample_list;
}


#ifdef BUILDING_PYTHON_MODULE
static PyObject* computeClasses(PyObject *self, PyObject *args)
{
	PyArrayObject *source;
	unsigned int height, width, n_channels;

	unsigned int patch_size, patch_stride;
	unsigned char patch_extraction_mode;

	if (!PyArg_ParseTuple(args, "O!IIb", &PyArray_Type, &source, &patch_size, &patch_stride, &patch_extraction_mode))
	{
		printf("Incomplete arguments\n");
		return NULL;
	}

	if (source->nd > 2)
	{
		n_channels = (unsigned int)source->dimensions[0];
		height = (unsigned int)source->dimensions[1];
		width = (unsigned int)source->dimensions[2];
	}
	else
	{
		n_channels = 1;
		height = (unsigned int)source->dimensions[0];
		width = (unsigned int)source->dimensions[1];
	}

	const unsigned int n_samples_per_width = (width - patch_size + patch_stride) / patch_stride;
	const unsigned int n_samples_per_height = (height - patch_size + patch_stride) / patch_stride;

	double * patches_classes = NULL;
	switch ((int)patch_extraction_mode)
	{
	case 0:
		patches_classes = defineClass_Full((double*)(source->data), height, width, patch_size, patch_stride, 1.0);
		break;

	case 1:
		patches_classes = defineClass_Center((double*)(source->data), height, width, patch_size, patch_stride);
		break;
	}
	npy_intp samples_shape[] = { n_samples_per_height, n_samples_per_width };

	PyArrayObject* patch_samples = (PyArrayObject*)PyArray_SimpleNew(2, &samples_shape[0], NPY_DOUBLE);

	memcpy((double*)(patch_samples->data), patches_classes, n_samples_per_height*n_samples_per_width*sizeof(double));

	free(patches_classes);

	return (PyObject *)patch_samples;
}



static PyObject* extractPatches(PyObject *self, PyObject *args)
{
	PyArrayObject *source;
	unsigned int height, width, n_channels;
	unsigned int patch_size, patch_stride;

	if (!PyArg_ParseTuple(args, "O!II", &PyArray_Type, &source, &patch_size, &patch_stride))
	{
		printf("Incomplete arguments\n");
		return NULL;
	}

	if (source->nd > 2)
	{
		n_channels = (unsigned int)source->dimensions[0];
		height = (unsigned int)source->dimensions[1];
		width = (unsigned int)source->dimensions[2];
	}
	else
	{
		n_channels = 1;
		height = (unsigned int)source->dimensions[0];
		width = (unsigned int)source->dimensions[1];
	}

	unsigned int samples_count = 0;
	double * samples = extractPatches_impl((double*)source->data, &samples_count, height, width, n_channels, patch_size, patch_stride);
	npy_intp samples_shape[] = { samples_count, n_channels, patch_size, patch_size };

	PyArrayObject* patch_samples = (PyArrayObject*)PyArray_SimpleNew(4, &samples_shape[0], NPY_DOUBLE);

	memcpy((double*)(patch_samples->data), samples, samples_count*n_channels*patch_size*patch_size * sizeof(double));

	free(samples);

	return (PyObject *)patch_samples;
}


static PyObject* computeSampledClasses(PyObject *self, PyObject *args)
{
	PyArrayObject *class_labels;
	unsigned int height, width, n_channels;

	unsigned int patch_size;
	unsigned char patch_extraction_mode;

	if (!PyArg_ParseTuple(args, "O!Ib", &PyArray_Type, &class_labels, &patch_size, &patch_extraction_mode))
	{
		printf("Incomplete arguments\n");
		return NULL;
	}

	if (class_labels->nd > 2)
	{
		n_channels = (unsigned int)class_labels->dimensions[0];
		height = (unsigned int)class_labels->dimensions[1];
		width = (unsigned int)class_labels->dimensions[2];
	}
	else
	{
		n_channels = 1;
		height = (unsigned int)class_labels->dimensions[0];
		width = (unsigned int)class_labels->dimensions[1];
	}

	unsigned int background_count = 0;
	double * background_mask = defineBackground((double*)(class_labels->data), &background_count, height, width, patch_size);

	unsigned int foreground_count = 0;
	double * foreground_mask = defineForeground((double*)(class_labels->data), patch_extraction_mode, &foreground_count, height, width, patch_size);

    printf("Foreground samples: %i, background samples: %i\n", foreground_count, background_count);

	unsigned int sample_size = balanceSamples(&foreground_count, &background_count);
    printf("Balanced sample size: %i\n", sample_size);
	unsigned int * foreground_samples = samplePatches(foreground_mask, foreground_count, sample_size, height, width);
	unsigned int * background_samples = samplePatches(background_mask, background_count, sample_size, height, width);

	free(background_mask);
	free(foreground_mask);

	npy_intp samples_shape[] = { sample_size };
	unsigned int samples_dimension = 1;

	PyArrayObject* foreground_sample_indices = (PyArrayObject*)PyArray_SimpleNew(samples_dimension, &samples_shape[0], NPY_UINT);
	PyArrayObject* background_sample_indices = (PyArrayObject*)PyArray_SimpleNew(samples_dimension, &samples_shape[0], NPY_UINT);

	memcpy((unsigned int*)(foreground_sample_indices->data), foreground_samples, sample_size * sizeof(unsigned int));
	memcpy((unsigned int*)(background_sample_indices->data), background_samples, sample_size * sizeof(unsigned int));

	free(foreground_samples);
	free(background_samples);

	PyObject *classes_tuple = PyTuple_New(2);
	PyTuple_SetItem(classes_tuple, 0, (PyObject*)foreground_sample_indices);
	PyTuple_SetItem(classes_tuple, 1, (PyObject*)background_sample_indices);

    printf("Samples computed successfully...\n");

	return classes_tuple;
}


static PyObject* extractSampledPatches(PyObject *self, PyObject *args)
{
    PyArrayObject *source;
    PyArrayObject *sampled_indices;
    unsigned int height, width, n_channels;
    unsigned int patch_size;
    
    if (!PyArg_ParseTuple(args, "O!O!I", &PyArray_Type, &source, &PyArray_Type, &sampled_indices, &patch_size))
    {
        printf("Incomplete arguments\n");
        return NULL;
    }
    
    if (source->nd > 2)
    {
        n_channels = (unsigned int)source->dimensions[0];
        height = (unsigned int)source->dimensions[1];
        width = (unsigned int)source->dimensions[2];
    }
    else
    {
        n_channels = 1;
        height = (unsigned int)source->dimensions[0];
        width = (unsigned int)source->dimensions[1];
    }
    
    unsigned int sample_size = sampled_indices->dimensions[0];
    printf("Extracting %i samples [%i,%i,%i]...\n", sample_size, n_channels, height, width);
    double * samples = extractSampledPatches_impl((double*)source->data, (unsigned int *)sampled_indices->data, sample_size, height, width, n_channels, patch_size);
    npy_intp samples_shape[] = { sample_size, n_channels, patch_size, patch_size };
    
    PyArrayObject* patch_samples = (PyArrayObject*)PyArray_SimpleNew(4, &samples_shape[0], NPY_DOUBLE);
    
    memcpy((double*)(patch_samples->data), samples, sample_size*n_channels*patch_size*patch_size*sizeof(double));
    
    free(samples);
    
    printf("Sampled patches extracted successfully...\n");

    return (PyObject *)patch_samples;
}


static PyObject* defineClasses(PyObject *self, PyObject *args)
{
    PyArrayObject *source;
	unsigned char patch_extraction_mode;
    unsigned int height, width, n_channels;
    unsigned int patch_size;
    
    if (!PyArg_ParseTuple(args, "O!Ib", &PyArray_Type, &source, &patch_size, &patch_extraction_mode))
    {
        printf("Incomplete arguments\n");
        return NULL;
    }
    
    if (source->nd > 2)
    {
        n_channels = (unsigned int)source->dimensions[0];
        height = (unsigned int)source->dimensions[1];
        width = (unsigned int)source->dimensions[2];
    }
    else
    {
        n_channels = 1;
        height = (unsigned int)source->dimensions[0];
        width = (unsigned int)source->dimensions[1];
    }

	double * background_mask = defineBackground((double*)(source->data), NULL, height, width, patch_size);
	double * foreground_mask = defineForeground((double*)(source->data), patch_extraction_mode, NULL, height, width, patch_size);
    
    npy_intp samples_shape[] = { height, width };
	PyArrayObject* background_mask_arr = (PyArrayObject*)PyArray_SimpleNew(2, &samples_shape[0], NPY_DOUBLE);
	PyArrayObject* foreground_mask_arr = (PyArrayObject*)PyArray_SimpleNew(2, &samples_shape[0], NPY_DOUBLE);
	
    memcpy((double*)(background_mask_arr->data), background_mask, height*width*sizeof(double));
	memcpy((double*)(foreground_mask_arr->data), foreground_mask, height*width*sizeof(double));
	
    free(background_mask);
	free(foreground_mask);

	PyObject *classes_tuple = PyTuple_New(2);
	PyTuple_SetItem(classes_tuple, 0, (PyObject*)background_mask_arr);
	PyTuple_SetItem(classes_tuple, 1, (PyObject*)foreground_mask_arr);

	return classes_tuple;
}


static PyMethodDef patchextraction_methods[] = {
	{ "computeClasses", computeClasses, METH_VARARGS, "Compute classes from all possible patches." },
	{ "computeSampledClasses", computeSampledClasses, METH_VARARGS, "Compute classes from a balanced sample of all possible positions." },
    { "extractSampledPatches", extractSampledPatches, METH_VARARGS, "Extract sampled positions." },
    { "defineClasses", defineClasses, METH_VARARGS, "Define classes areas." },
    { "extractPatches", extractPatches, METH_VARARGS, "Extract all positions." },
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
	PyObject *m;
	m = PyModule_Create(&patchextraction_moduledef);
	if (!m) {
		return NULL;
	}
	import_array();

	return m;
}
#endif
