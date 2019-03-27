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


void extractPatchClass_same(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int y = 0; y < patch_size; y++)
	{
		for (unsigned int x = 0; x < patch_size; x++)
		{
			for (unsigned int z = 0; z < n_channels; z++)
			{
				*(class_labels + y*patch_size*n_channels + x*n_channels + z) = 
					*(input + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels + z);
			}
		}
	}
}


void extractPatchClass_center(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int z = 0; z < n_channels; z++)
	{
		*(class_labels + z) = *(input + (patch_size/2 + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + patch_size/2)*n_channels);
	}
}


void extractPatchClass_mean(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{		
	for (unsigned int z = 0; z < n_channels; z++)
	{
		double mean_input = 0.0;
	
		for (unsigned int y = 0; y < patch_size; y++)
		{
			for (unsigned int x = 0; x < patch_size; x++)
			{
				mean_input += *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
			}
		}
		
		 *(class_labels + z) = mean_input / (double)(patch_size*patch_size);
	}
}


void extractPatchClass_max(double * input, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int z = 0; z < n_channels; z++)
	{
		double *max_class = input + z*height*width + patch_y*patch_stride*width + patch_x*patch_stride;
		
		for (unsigned int y = 0; y < patch_size; y++)
		{
			for (unsigned int x = 0; x < patch_size; x++)
			{
				if (*max_class <  *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels))
				{
					max_class = input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels;
				}
			}
		}
		
		*(class_labels + z) = *max_class;
	}
}


void extractSinglePatchAndClass_same(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int y = 0; y < patch_size; y++)
	{
		for (unsigned int x = 0; x < patch_size; x++)
		{
			for (unsigned int z = 0; z < n_channels; z++)
			{
				*(output + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
				*(class_labels + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
			}
		}
	}
}


void extractSinglePatchAndClass_center(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int z = 0; z < n_channels; z++)
	{
		for (unsigned int y = 0; y < patch_size; y++)
		{
			for (unsigned int x = 0; x < patch_size; x++)
			{
				*(output + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
			}
		}
		
		*(class_labels + z) = *(input + z + (patch_size/2 + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + patch_size/2)*n_channels);
	}
}


void extractSinglePatchAndClass_mean(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int z = 0; z < n_channels; z++)
	{
		double mean_class = 0.0;
		for (unsigned int y = 0; y < patch_size; y++)
		{
			for (unsigned int x = 0; x < patch_size; x++)
			{
				*(output + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
				mean_class += *(input + z + (y + patch_y*patch_stride)*width*n_channels + patch_x*patch_stride + x)*n_channels);
			}
		}
		
		*(class_labels + z) = mean_class / (double)(patch_size*patch_size);
	}
}


void extractSinglePatchAndClass_max(double * input, double * output, double * class_labels, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	
	for (unsigned int z = 0; z < n_channels; z++)
	{
		double *max_class = input + z + patch_y*patch_stride*width*n_channels + patch_x*patch_stride*n_channels;
		
		for (unsigned int y = 0; y < patch_size; y++)
		{
			for (unsigned int x = 0; x < patch_size; x++)
			{
				*(output + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
				if (*max_class <  *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels))
				{
					max_class = input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels;
				}
			}
		}
		
		*(class_labels + z) = *max_class;
	}
}


void extractSinglePatch(double * input, double * output, const unsigned int patch_x, const unsigned int patch_y, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	for (unsigned int y = 0; y < patch_size; y++)
	{
		for (unsigned int x = 0; x < patch_size; x++)
		{
			for (unsigned int z = 0; z < n_channels; z++)
			{
				*(output + z + y*patch_size*n_channels + x*n_channels) = *(input + z + (y + patch_y*patch_stride)*width*n_channels + (patch_x*patch_stride + x)*n_channels);
			}
		}
	}
}


void computeClasses_impl(double * input, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode)
{
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	void (*extractPatchClass_function)(double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
	
	unsigned int class_labels_offset = n_channels;
	
	switch (patch_extraction_mode)
	{
		// The class corresponds to the same patch
		case 0:
			extractPatchClass_function = extractPatchClass_same;
			class_labels_offset = patch_size*patch_size*n_channels;
			break;
			
		// The class corresponds to the center value
		case 1:
			extractPatchClass_function = extractPatchClass_center;
			break;
			
		// The class corresponds to the mean value
		case 2:
			extractPatchClass_function = extractPatchClass_mean;
			break;
			
		// The class corresponds to the maxima value
		case 3:
			extractPatchClass_function = extractPatchClass_max;
			break;
			
		default:
			fprintf(stderr, "<<Error: The class extraction mode %i is unknown>>\n", (int)patch_extraction_mode);
			return;
	}
	
	if (*class_labels == NULL)
	{
		*class_labels = (double*)malloc(n_patches_in_height*n_patches_in_width*class_labels_offset*sizeof(double));
	}
	
	for (unsigned int patch_y = 0; patch_y < n_patches_in_height; patch_y++)
	{
		for (unsigned int patch_x = 0; patch_x < n_patches_in_width; patch_x++)
		{
			extractPatchClass_function(input, *class_labels + (patch_y*n_patches_in_width + patch_x)*class_labels_offset, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
		}
	}
}


unsigned int generatePatchesSample_impl(unsigned int ** sample_indices, double * class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const double sample_percentage)
{	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
		
	unsigned int positive_patches_count = 0;
	unsigned int negative_patches_count = 0;
	
	unsigned int patch_xy;
	
	unsigned int sample_size;
	// By now, the class is only taken according to the first channel:
	for (patch_xy = 0; patch_xy < n_patches_in_height*n_patches_in_width; patch_xy++)
	{
		if (*(class_labels + patch_xy*n_channels) > 0.0)
		{
			positive_patches_count++;
		}
		else
		{
			negative_patches_count++;
		}
	}
	DEBNUMMSG("Positive patches count: %i\n", positive_patches_count);
	DEBNUMMSG("Negative patches count: %i\n", negative_patches_count);
	
	// Balance the probability of take a patch of each class and insert it into the sample:
	double negative_patch_sampling_percentage = ((negative_patches_count <= positive_patches_count) ? 1.0 : (double)positive_patches_count/(double)negative_patches_count)*sample_percentage;
	double positive_patch_sampling_percentage = ((positive_patches_count <= negative_patches_count) ? 1.0 : (double)negative_patches_count/(double)positive_patches_count)*sample_percentage;
	
	if (sample_percentage > 1.0)
	{
		sample_size = (unsigned int)floor(sample_percentage);
		negative_patch_sampling_percentage/=(double)n_patches_in_height*n_patches_in_width;
		positive_patch_sampling_percentage/=(double)n_patches_in_height*n_patches_in_width;
	}
	else
	{
		sample_size = 2*(unsigned int)floor(sample_percentage*((positive_patches_count < negative_patches_count) ? (double)positive_patches_count : (double)negative_patches_count));
	}
	
	DEBNUMMSG("Sample size: %i\n", sample_size);
	*sample_indices = (unsigned int*)malloc(sample_size*sizeof(unsigned int));
	
	STAUS * rng_seed = initSeed(0);
	
	int remaining_sample_indices = sample_size;
	
	
	DEBNUMMSG("Negative patch sampling percentage: %f\n", negative_patch_sampling_percentage);
	DEBNUMMSG("Positive patch sampling percentage: %f\n", positive_patch_sampling_percentage);
	
	double current_patch_class_sampling_percentage;
	unsigned int * sample_indices_ptr = *sample_indices;
	patch_xy = 0;
	while ((remaining_sample_indices >= 1) && ((remaining_sample_indices < positive_patches_count) || (remaining_sample_indices < negative_patches_count)))
	{
		if (*(class_labels + patch_xy*n_channels) > 0.0)
		{
			current_patch_class_sampling_percentage = positive_patch_sampling_percentage;
			positive_patches_count--;
		}
		else
		{
			current_patch_class_sampling_percentage = negative_patch_sampling_percentage;
			negative_patches_count--;
		}
		
		if (HybTaus(0.0, 1.0, rng_seed) <= current_patch_class_sampling_percentage)
		{
			remaining_sample_indices--;
			*(sample_indices_ptr++) = patch_xy;
		}
		
		patch_xy++;
	}
	DEBNUMMSG("Remaining samples to fill: %i ", remaining_sample_indices );
	DEBNUMMSG("%i\n", patch_xy );
	
	// Assign the remaining patches to complete the sample
	while (remaining_sample_indices-- >= 1)
	{
		*(sample_indices_ptr++) = patch_xy;
		patch_xy++;
	}
	
	DEBNUMMSG("Remaining samples to fill: %i ", remaining_sample_indices );
	DEBNUMMSG("%i\n", patch_xy );
	free(rng_seed);
	
	return sample_size;
}


void samplePatches_impl(double * input, double ** output, double * class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage, unsigned int ** sample_indices_ptr)
{
	computeClasses_impl(input, &class_labels, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);
	unsigned int * sample_indices = NULL;
	const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, class_labels, height, width, n_channels, patch_size, patch_stride, sample_percentage);
		
	if (*output == NULL)
	{
		*output = (double*)malloc(sample_size*patch_size*patch_size*n_channels*sizeof(double));
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
		
	unsigned int patch_x, patch_y;
	for (unsigned int i = 0; i < sample_size; i++)
	{
		patch_x = *(sample_indices+i) % n_patches_in_width;
		patch_y = *(sample_indices+i) / n_patches_in_width;
		extractSinglePatch(input, *output + i*n_channels*patch_size*patch_size, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
	}
	
	if (sample_indices_ptr)
	{
		*sample_indices_ptr = sample_indices;
	}
	else
	{
		free(sample_indices);
	}
}


void extractSampledPatchesAndClasses_impl(double * input, double ** output, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, unsigned int * sample_indices, const unsigned int sample_size)
{
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
		
	void (*extractSinglePatchAndClass_function)(double*, double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
	
	unsigned int class_labels_offset = n_channels;
	switch (patch_extraction_mode)
	{
		// The class corresponds to the same patch
		case 0:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_same;
			class_labels_offset = patch_size*patch_size*n_channels;
			break;
			
		// The class corresponds to the center value
		case 1:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_center;
			break;
			
		// The class corresponds to the mean value
		case 2:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_mean;
			break;
			
		// The class corresponds to the maxima value
		case 3:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_max;
			break;
			
		default:
			fprintf(stderr, "<<Error: The class extraction mode %i is unknown>>\n", (int)patch_extraction_mode);
			return;
	}

	if (*output == NULL)
	{
		*output = (double*)malloc(sample_size*patch_size*patch_size*n_channels*sizeof(double));
	}
	
	if (*class_labels == NULL)
	{
		*class_labels = (double*)malloc(sample_size*class_labels_offset*sizeof(double));
	}
	
	unsigned int patch_x, patch_y;
	for (unsigned int i = 0; i < sample_size; i++)
	{
		patch_x = *(sample_indices+i) % n_patches_in_width;
		patch_y = *(sample_indices+i) / n_patches_in_width;
		
		extractSinglePatchAndClass_function(input, *output + i*patch_size*patch_size*n_channels, *class_labels + i*class_labels_offset, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
	}
}


void extractAllPatchesAndClasses_impl(double * input, double ** output, double ** class_labels, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode)
{
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	void (*extractSinglePatchAndClass_function)(double*, double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
	
	unsigned int class_labels_offset = n_channels;
	switch (patch_extraction_mode)
	{
		// The class corresponds to the same patch
		case 0:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_same;
			class_labels_offset = patch_size*patch_size*n_channels;
			break;
			
		// The class corresponds to the center value
		case 1:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_center;
			break;
			
		// The class corresponds to the mean value
		case 2:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_mean;
			break;
			
		// The class corresponds to the maxima value
		case 3:
			extractSinglePatchAndClass_function = extractSinglePatchAndClass_max;
			break;
			
		default:
			fprintf(stderr, "<<Error: The class extraction mode %i is unknown>>\n", (int)patch_extraction_mode);
			return;
	}

	if (*output == NULL)
	{
		*output = (double*)malloc(n_patches_in_height*n_patches_in_width*patch_size*patch_size*n_channels*sizeof(double));
	}
	
	if (*class_labels == NULL)
	{
		*class_labels = (double*)malloc(n_patches_in_height*n_patches_in_width*class_labels_offset*sizeof(double));
	}
	
	for (unsigned int patch_y = 0; patch_y < n_patches_in_height; patch_y++)
	{
		for (unsigned int patch_x = 0; patch_x < n_patches_in_width; patch_x++)
		{
			extractSinglePatchAndClass_function(input, *output + (patch_y*n_patches_in_width + patch_x)*patch_size*patch_size*n_channels, *class_labels + (patch_y*n_patches_in_width + patch_x)*class_labels_offset, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
		}
	}
}


void extractAllPatches_impl(double * input, double ** output, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride)
{
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	if (*output == NULL)
	{
		*output = (double*)malloc(n_patches_in_height*n_patches_in_width*patch_size*patch_size*n_channels*sizeof(double));
	}

	for (unsigned int patch_y = 0; patch_y < n_patches_in_height; patch_y++)
	{
		for (unsigned int patch_x = 0; patch_x < n_patches_in_width; patch_x++)
		{
			extractSinglePatch(input, *output + (patch_y*n_patches_in_width + patch_x)*patch_size*patch_size*n_channels, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
		}
	}
}


void extractSampledPatches_impl(double * input, double ** output, const unsigned int height, const unsigned int width, const unsigned int n_channels, const unsigned int patch_size, const unsigned int patch_stride, unsigned int * sample_indices, const unsigned int sample_size)
{
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	if (*output == NULL)
	{
		*output = (double*)malloc(sample_size*patch_size*patch_size*n_channels*sizeof(double));
	}
	
	unsigned int patch_x, patch_y;
	for (unsigned int i = 0; i < sample_size; i++)
	{
		patch_x = *(sample_indices+i) % n_patches_in_width;
		patch_y = *(sample_indices+i) / n_patches_in_width;
		
		extractSinglePatch(input, *output + i*patch_size*patch_size*n_channels, patch_x, patch_y, height, width, n_channels, patch_size, patch_stride);
	}
}


#ifdef BUILDING_PYTHON_MODULE
static PyObject* computeClasses(PyObject *self, PyObject *args)
{	
	PyArrayObject *input;
    unsigned int height, width, n_channels;
    
    unsigned int patch_size, patch_stride;
    unsigned char patch_extraction_mode;
    
    if (!PyArg_ParseTuple(args, "O!IIb", &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode))
    {
		return NULL;
	}

	if (input->nd > 2)
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = (unsigned int)input->dimensions[2];
	}
	else
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = 1;
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
			
	const unsigned int class_labels_height = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_width = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_offset = (patch_extraction_mode == 0) ? class_labels_height*class_labels_width*n_channels : n_channels;
		
	double * class_labels = (double*)malloc(n_patches_in_height*n_patches_in_width*class_labels_offset*sizeof(double));
	computeClasses_impl((double*)input->data, &class_labels, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);
	
	npy_intp labels_shape[] = { n_patches_in_height*n_patches_in_width, class_labels_height, class_labels_width, n_channels };
	PyArrayObject* labels = (PyArrayObject*)PyArray_SimpleNew(4, &labels_shape[0], NPY_DOUBLE);
	
	memcpy((double*)labels->data, class_labels, n_patches_in_height*n_patches_in_width*class_labels_offset*sizeof(double));
	
	free(class_labels);
	return (PyObject*)labels;
}


static PyObject* generatePatchesSample(PyObject *self, PyObject *args, PyObject *kw)
{	
	PyArrayObject *input;
    unsigned int height, width, n_channels;            
    unsigned int patch_size, patch_stride;
    
    PyArrayObject * labels = NULL;
    unsigned char patch_extraction_mode = 1;
    double sample_percentage = 1.0;
    
    static char *keywords[] = {"input", "patch_size", "patch_stride", "patch_extraction_mode", "sample_percentage", "labels", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!II|$bdO!", keywords, &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode, &sample_percentage, &PyArray_Type, &labels))
    {
		return NULL;
	}

			
	if (input->nd > 2)
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = (unsigned int)input->dimensions[2];
	}
	else
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = 1;
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	const unsigned int class_labels_offset = (patch_extraction_mode == 0) ? patch_size*patch_size*n_channels : n_channels;
	
	double * class_labels = NULL;
	if (labels)
	{
		class_labels = (double*)labels->data;
	}
	else
	{
		class_labels = (double*)malloc(n_patches_in_height*n_patches_in_width*class_labels_offset*sizeof(double));
		computeClasses_impl((double*)(input->data), &class_labels, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);
	}
	
	unsigned int * sample_indices = NULL;
	const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, class_labels, height, width, n_channels, patch_size, patch_stride, sample_percentage);
	
	npy_intp patches_sample_shape[] = { sample_size };
	PyArrayObject * patches_sample = (PyArrayObject*)PyArray_SimpleNew(1, &patches_sample_shape[0], NPY_UINT32);	
	memcpy((unsigned int*)patches_sample->data, sample_indices, sample_size*sizeof(unsigned int));
	
	free(sample_indices);
	
	if (!labels)
	{
		free(class_labels);
	}
	
	return (PyObject*)patches_sample;
}


static PyObject* extractSampledPatchesAndClasses_pyinterface(PyArrayObject *input, PyArrayObject *labels, PyArrayObject *patches_sample, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage)
{	
	unsigned int height, width, n_channels;	
	if (input->nd > 2)
	{
		n_channels = (unsigned int)input->dimensions[0];
		height = (unsigned int)input->dimensions[1];
		width = (unsigned int)input->dimensions[2];
	}
	else
	{
		n_channels = 1;
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
		
	const unsigned int class_labels_height = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_width = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_offset = (patch_extraction_mode == 0) ? class_labels_height*class_labels_width*n_channels : n_channels;

	PyArrayObject *patches = NULL;
    PyArrayObject *patches_classes = NULL;
    PyObject *sample_patches_tuple = NULL;
    
    if (!patches_sample)
    {
		if (!labels)
		{
			// Compute all the patches classes:
			double * labels_data = (double*)malloc(n_patches_in_width*n_patches_in_height*n_channels*sizeof(double));
			computeClasses_impl((double*)input->data, &labels_data, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);
			
			unsigned int * sample_indices = NULL;
			const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, labels_data, height, width, n_channels, patch_size, patch_stride, sample_percentage);
			
			npy_intp patches_sample_shape[] = {sample_size};
			patches_sample = (PyArrayObject*)PyArray_SimpleNew(1, &patches_sample_shape[0], NPY_UINT32);
			memcpy((unsigned int*)patches_sample->data, sample_indices, sample_size*sizeof(unsigned int));
			
			npy_intp patches_shape[] = { sample_size, patch_size, patch_size, n_channels };
			patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
						
			npy_intp patches_classes_shape[] = { sample_size, class_labels_height, class_labels_width, n_channels };
			patches_classes = (PyArrayObject*)PyArray_SimpleNew(4, &patches_classes_shape[0], NPY_DOUBLE);
			
			extractSampledPatchesAndClasses_impl((double*)(input->data), (double**)&(patches->data), (double**)&patches_classes->data, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode, sample_indices + 1, *sample_indices);
			
			free(sample_indices);
			free(labels_data);
			
			sample_patches_tuple = PyTuple_New(3);
			PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
			PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_classes);
			PyTuple_SetItem(sample_patches_tuple, 2, (PyObject*)patches_sample);
		}
		else
		{
			// Compute all the patches classes:
			unsigned int * sample_indices = NULL;
			const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, (double*)labels->data, height, width, n_channels, patch_size, patch_stride, sample_percentage);
			
			npy_intp patches_sample_shape[] = {sample_size};
			patches_sample = (PyArrayObject*)PyArray_SimpleNew(1, &patches_sample_shape[0], NPY_UINT32);
			memcpy((unsigned int*)patches_sample->data, sample_indices, sample_size*sizeof(unsigned int));
			
			npy_intp patches_shape[] = { sample_size, patch_size, patch_size, n_channels };
			patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
			
			npy_intp patches_classes_shape[] = { sample_size, class_labels_height, class_labels_width, n_channels };
			patches_classes = (PyArrayObject*)PyArray_SimpleNew(4, &patches_classes_shape[0], NPY_DOUBLE);
			
			extractSampledPatchesAndClasses_impl((double*)input->data, (double**)&patches->data, (double**)&patches_classes->data, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode, sample_indices + 1, *sample_indices);
			
			free(sample_indices);
			
			sample_patches_tuple = PyTuple_New(3);
			PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
			PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_classes);
			PyTuple_SetItem(sample_patches_tuple, 2, (PyObject*)patches_sample);
		}
	}
	else
	{
		npy_intp patches_shape[] = { patches_sample->dimensions[0], patch_size, patch_size, n_channels };
		patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
		
		npy_intp patches_classes_shape[] = { patches_sample->dimensions[0], class_labels_height, class_labels_width, n_channels };
		patches_classes = (PyArrayObject*)PyArray_SimpleNew(4, &patches_classes_shape[0], NPY_DOUBLE);
		
		extractSampledPatchesAndClasses_impl((double*)input->data, (double**)&patches->data, (double**)&patches_classes->data, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode, (unsigned int *)patches_sample->data, (unsigned int)patches_sample->dimensions[0]);
				
		sample_patches_tuple = PyTuple_New(2);
		PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
		PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_classes);
	}

	return sample_patches_tuple;
}


static PyObject* extractSampledPatchesAndClasses(PyObject *self, PyObject *args, PyObject *kw)
{
	PyArrayObject *input;
    unsigned int height, width, n_channels;            
    unsigned int patch_size, patch_stride;
    
    PyArrayObject * labels = NULL;
    PyArrayObject * patches_sample = NULL;
    unsigned char patch_extraction_mode = 1;
    double sample_percentage = 1.0;
    
    static char *keywords[] = {"input", "patch_size", "patch_stride", "patch_extraction_mode", "sample_percentage", "patches_sample", "labels", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!II|$bdO!O!", keywords, &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode, &sample_percentage, &PyArray_Type, &patches_sample, &PyArray_Type, &labels))
    {
		return NULL;
	}

	return extractSampledPatchesAndClasses_pyinterface(input, labels, patches_sample, patch_size, patch_stride, patch_extraction_mode, sample_percentage);
}


static PyObject* extractSampledPatches_pyinterface(PyArrayObject *input, PyArrayObject *labels, PyArrayObject *patches_sample, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode, const double sample_percentage)
{	
	unsigned int height, width, n_channels;
	if (input->nd > 2)
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = (unsigned int)input->dimensions[2];
	}
	else
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = 1;
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	PyArrayObject *patches = NULL;
    PyObject *sample_patches_tuple = NULL;
    
    if (!patches_sample)
    {
		if (!labels)
		{
			// Compute all the patches classes:
			double * labels_data = (double*)malloc(n_patches_in_width*n_patches_in_height*n_channels*sizeof(double));
			computeClasses_impl((double*)input->data, &labels_data, height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);
			
			unsigned int * sample_indices = NULL;
			const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, labels_data, height, width, n_channels, patch_size, patch_stride, sample_percentage);
			
			npy_intp patches_sample_shape[] = {sample_size};
			patches_sample = (PyArrayObject*)PyArray_SimpleNew(1, &patches_sample_shape[0], NPY_UINT32);
			memcpy((unsigned int*)patches_sample->data, sample_indices, sample_size*sizeof(unsigned int));
			
			npy_intp patches_shape[] = { sample_size, patch_size, patch_size, n_channels };
			patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
			
			extractSampledPatches_impl((double*)input->data, (double**)&(patches->data), height, width, n_channels, patch_size, patch_stride, sample_indices, sample_size);
			
			free(sample_indices);
			free(labels_data);
			
			sample_patches_tuple = PyTuple_New(2);
			PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
			PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_sample);
		}
		else
		{
			// Compute all the patches classes:
			unsigned int * sample_indices = NULL;
			const unsigned int sample_size = generatePatchesSample_impl(&sample_indices, (double*)labels->data, height, width, n_channels, patch_size, patch_stride, sample_percentage);
			
			npy_intp patches_sample_shape[] = {sample_size};
			patches_sample = (PyArrayObject*)PyArray_SimpleNew(1, &patches_sample_shape[0], NPY_UINT32);
			memcpy((unsigned int*)patches_sample->data, sample_indices, sample_size*sizeof(unsigned int));
			
			npy_intp patches_shape[] = { sample_size, patch_size, patch_size, n_channels };
			patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
						
			extractSampledPatches_impl((double*)input->data, (double**)&(patches->data), height, width, n_channels, patch_size, patch_stride, sample_indices, sample_size);
			
			free(sample_indices);
			
			sample_patches_tuple = PyTuple_New(2);
			PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
			PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_sample);
		}
	}
	else
	{
		npy_intp patches_shape[] = { patches_sample->dimensions[0], n_channels, patch_size, patch_size };
		patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
		
		DEBNUMMSG("Extracting %i samples, ", (int)patches_sample->dimensions[0]);
		DEBNUMMSG("of size: %ix", patch_size);
		DEBNUMMSG("%i and", patch_size);
		DEBNUMMSG(" %i channels\n", n_channels);
		extractSampledPatches_impl((double*)(input->data), (double**)&(patches->data), height, width, n_channels, patch_size, patch_stride, (unsigned int *)(patches_sample->data), (unsigned int)(patches_sample->dimensions[0]));

		sample_patches_tuple = (PyObject*)patches;
	}

	return sample_patches_tuple;
}


static PyObject* extractSampledPatches(PyObject *self, PyObject *args, PyObject *kw)
{
	PyArrayObject *input;
    unsigned int height, width, n_channels;            
    unsigned int patch_size, patch_stride;
    
    PyArrayObject * labels = NULL;
    PyArrayObject * patches_sample = NULL;
    unsigned char patch_extraction_mode = 1;
    double sample_percentage = 1.0;
    
    static char *keywords[] = {"input", "patch_size", "patch_stride", "patch_extraction_mode", "sample_percentage", "patches_sample", "labels", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!II|$bdO!O!", keywords, &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode, &sample_percentage, &PyArray_Type, &patches_sample, &PyArray_Type, &labels))
    {
		return NULL;
	}

	return extractSampledPatches_pyinterface(input, labels, patches_sample, patch_size, patch_stride, patch_extraction_mode, sample_percentage);
}


static PyObject* extractAllPatchesAndClasses_pyinterface(PyArrayObject *input, const unsigned int patch_size, const unsigned int patch_stride, const unsigned char patch_extraction_mode)
{	
	unsigned int height, width, n_channels;
	if (input->nd > 2)
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = (unsigned int)input->dimensions[2];
	}
	else
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = 1;
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	const unsigned int class_labels_height = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_width = (patch_extraction_mode == 0) ? patch_size : 1;
	const unsigned int class_labels_offset = (patch_extraction_mode == 0) ? class_labels_height*class_labels_width*n_channels : n_channels;

	PyArrayObject *patches = NULL;
    PyArrayObject *patches_classes = NULL;
    
	npy_intp patches_shape[] = { n_patches_in_height*n_patches_in_width, patch_size, patch_size, n_channels };
	patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
	
	npy_intp patches_classes_shape[] = { n_patches_in_height*n_patches_in_width, class_labels_height, class_labels_width, n_channels };
	patches_classes = (PyArrayObject*)PyArray_SimpleNew(4, &patches_classes_shape[0], NPY_DOUBLE);
	
	extractAllPatchesAndClasses_impl((double*)(input->data), (double**)&(patches->data), (double**)&(patches_classes->data), height, width, n_channels, patch_size, patch_stride, patch_extraction_mode);

    PyObject *sample_patches_tuple = sample_patches_tuple = PyTuple_New(2);
	PyTuple_SetItem(sample_patches_tuple, 0, (PyObject*)patches);
	PyTuple_SetItem(sample_patches_tuple, 1, (PyObject*)patches_classes);

	return sample_patches_tuple;
}


static PyObject* extractAllPatchesAndClasses(PyObject *self, PyObject *args, PyObject *kw)
{
	PyArrayObject *input;
    unsigned int height, width, n_channels;            
    unsigned int patch_size, patch_stride;
    
    unsigned char patch_extraction_mode = 1;
    
    static char *keywords[] = {"input", "patch_size", "patch_stride", "patch_extraction_mode", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!II|$b", keywords, &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode))
    {
		return NULL;
	}

	return extractAllPatchesAndClasses_pyinterface(input, patch_size, patch_stride, patch_extraction_mode);
}


static PyObject* extractAllPatches_pyinterface(PyArrayObject *input, const unsigned int patch_size, const unsigned int patch_stride)
{
    unsigned int height, width, n_channels;  
	if (input->nd > 2)
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = (unsigned int)input->dimensions[2];
	}
	else
	{
		height = (unsigned int)input->dimensions[0];
		width = (unsigned int)input->dimensions[1];
		n_channels = 1;
	}
	
	const unsigned int n_patches_in_width = (unsigned int)floor((double)(width - patch_size + patch_stride) / (double)patch_stride);
	const unsigned int n_patches_in_height = (unsigned int)floor((double)(height - patch_size + patch_stride) / (double)patch_stride);
	
	DEBNUMMSG("Extracting %i patches\n", n_patches_in_height*n_patches_in_width);
	npy_intp patches_shape[] = { n_patches_in_height*n_patches_in_width, patch_size, patch_size, n_channels};
	PyArrayObject *patches = (PyArrayObject*)PyArray_SimpleNew(4, &patches_shape[0], NPY_DOUBLE);
	
	extractAllPatches_impl((double*)(input->data), (double**)&(patches->data), height, width, n_channels, patch_size, patch_stride);

	return (PyObject*)patches;
}


static PyObject* extractAllPatches(PyObject *self, PyObject *args)
{
	PyArrayObject *input;
    unsigned int height, width, n_channels;            
    unsigned int patch_size, patch_stride;
        
    if (!PyArg_ParseTuple(args, "O!II", &PyArray_Type, &input, &patch_size, &patch_stride))
    {
		return NULL;
	}

	return extractAllPatches_pyinterface(input, patch_size, patch_stride);
}


static PyObject* samplePatches(PyObject *self, PyObject *args, PyObject *kw)
{	
	PyArrayObject *input;
	
    unsigned int patch_size, patch_stride;
    
    PyArrayObject * labels = NULL;
    PyArrayObject * patches_sample = NULL;
    
    unsigned char patch_extraction_mode = 4;
    double sample_percentage = -1.0;
    
    static char *keywords[] = {"input", "patch_size", "patch_stride", "patch_extraction_mode", "sample_percentage", "patches_sample", "labels", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!II|$bdO!O!", keywords, &PyArray_Type, &input, &patch_size, &patch_stride, &patch_extraction_mode, &sample_percentage, &PyArray_Type, &patches_sample, &PyArray_Type, &labels))
    {
		return NULL;
	}
	
	if (sample_percentage > 0.0)
	{
		if (patch_extraction_mode < 4)
		{
			// Extract the patches and the corresponding class labels
			return extractSampledPatchesAndClasses_pyinterface(input, labels, patches_sample, patch_size, patch_stride, patch_extraction_mode, sample_percentage);
		}
		else
		{
			// Extract only the patches
			return extractSampledPatches_pyinterface(input, labels, patches_sample, patch_size, patch_stride, patch_extraction_mode, sample_percentage);
		}
	}
	else if (patch_extraction_mode < 4)
	{
		return extractAllPatchesAndClasses_pyinterface(input, patch_size, patch_stride, patch_extraction_mode);
	}
	
	return extractAllPatches_pyinterface(input, patch_size, patch_stride);
}


static PyMethodDef patchextraction_methods[] = {
	{ "computeClasses", computeClasses, METH_VARARGS, "Compute the class labels of each possible patch according to the patch extraction mode." },
	{ "generatePatchesSample", (PyCFunction)generatePatchesSample, METH_VARARGS | METH_KEYWORDS, "Generates a sample of patches selected according to the sample percentage." },
	{ "samplePatches", (PyCFunction)samplePatches, METH_VARARGS|METH_KEYWORDS, "Applies the corresponding extraction procedure, according to the sample percentage, and patch extractio mode defined." },
	{ "extractSampledPatchesAndClasses", (PyCFunction)extractSampledPatchesAndClasses, METH_VARARGS|METH_KEYWORDS, "Extracts a sample of patches and their corresponding classes, according to the sample percentage defined." },
	{ "extractSampledPatches", (PyCFunction)extractSampledPatches, METH_VARARGS|METH_KEYWORDS, "Extracts a sample of patches, according to the sample percentage defined." },
	{ "extractAllPatchesAndClasses", (PyCFunction)extractAllPatchesAndClasses, METH_VARARGS|METH_KEYWORDS, "Extracts all the patches and their corresponding classes." },
	{ "extractAllPatches", extractAllPatches, METH_VARARGS, "Extracts all the patches." },
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
