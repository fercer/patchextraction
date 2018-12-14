#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <patchextraction.h>

void savePGM(const char* filename, const int height, const int width, const int n_channels, double * img_src);
double * loadPGM(const char * filename, int* height, int* width, int *n_channels, const double normalize_min, const double normalize_max);

int main(int argc, char * argv[])
{    
    if (argc < 2)
    {
        printf("\nExample: Extracts a sample of patches of size 32x32 from the input image\n");
        printf("First argument is the path to a .PGM (grayscale) image, or a .PPM (color) image\n");
        printf("Second argument is the path to a .PBM (black and white) image\n");
        printf("Third argument is the patch size\n");
        printf("Fourth argument is the patch stride\n");
        printf("Fifth argument is the patch extraction mode\n");
        return -1;
    }

    FILE * fp_img = fopen(argv[1], "r");
    if (!fp_img)
    {
        printf("<<Error: The file \"%s\" could not be loaded>>\n", argv[1]);
        return -1;
    }
    
    int height, width, n_channels;
    double * gt = loadPGM(argv[2], &height, &width, &n_channels, -1.0, 1.0);
    savePGM("loaded_gt", height, width, n_channels, gt);

    double * img = loadPGM(argv[1], &height, &width, &n_channels, 0.0, 1.0);

    if (!img)
    {
        return -1;
    }

    printf("Loaded image dimensions: %ix%ix%i\n", height, width, n_channels);
    
    unsigned int patch_size = 32, patch_stride = 32;
    unsigned char patch_exctraction_mode = 0;
    
    if (argc > 3)
    {
        patch_size = atoi(argv[3]);
    }
    
    if (argc > 4)
    {
        patch_stride = atoi(argv[4]);
    }
    
    if (argc > 5)
    {
        patch_exctraction_mode = (char)atoi(argv[5]);
    }
    
    unsigned int class_offset = n_channels;
    if (patch_exctraction_mode == 0)
    {
        class_offset = n_channels*patch_size*patch_size;
    }
    
    // Test the patch extaction:
    double * class_labels = NULL;
    double * patches = NULL;
    computeClasses_impl(gt, &class_labels, height, width, 1, patch_size, patch_stride, patch_exctraction_mode);
    
    printf("Classes computed successfully\n");
    
    savePGM("label_1", patch_size, patch_size, 1, class_labels);
    savePGM("label_2", patch_size, patch_size, 1, class_labels + n_channels*patch_size*patch_size);
    savePGM("label_3", patch_size, patch_size, 1, class_labels + 2*n_channels*patch_size*patch_size);
    savePGM("label_4", patch_size, patch_size, 1, class_labels + 3*n_channels*patch_size*patch_size);
    
    unsigned int * sample_indices = NULL;
    unsigned int sample_size = generatePatchesSample_impl(&sample_indices, class_labels, height, width, 1, patch_size, patch_stride, 4);
    printf("Extracted %i patches\n", sample_size);
    
    extractSampledPatches_impl(img, &patches, height, width, n_channels, patch_size, patch_stride, sample_indices, sample_size);
    
    savePGM("sample_1", patch_size, patch_size, n_channels, patches);
    savePGM("sample_2", patch_size, patch_size, n_channels, patches + n_channels*patch_size*patch_size);
    savePGM("sample_3", patch_size, patch_size, n_channels, patches + 2*n_channels*patch_size*patch_size);
    savePGM("sample_4", patch_size, patch_size, n_channels, patches + 3*n_channels*patch_size*patch_size);
    //savePGM("img", height, width, n_channels, img);
    
    free(gt);
    free(img);
    free(class_labels);
    free(patches);
    free(sample_indices);
    
    return 0;
}


void savePGM(const char * filename, const int height, const int width, const int n_channels, double * img_src)
{
    double * max_value = img_src;
    double * min_value = img_src;
    
    for (unsigned int xy = 0; xy < height*width; xy++)
    {
        if (*max_value < *(img_src + xy))
        {
            max_value = img_src + xy;
        }
        
        if (*min_value > *(img_src + xy))
        {
            min_value = img_src + xy;
        }
    }
    
    printf("Max value: %f, min value: %f\n", *max_value, *min_value);
    
    char filename_and_extension[128];
    if (n_channels == 1)
    {
        sprintf(filename_and_extension, "%s.pgm", filename);
        FILE * fp = fopen(filename_and_extension, "w");
        fprintf(fp, "P2\n# Created by FerCer\n%i %i\n255\n", height, width);
                
        for (unsigned int x = 0; x < width; x++)
        {
            for (unsigned int y = 0; y < height; y++)
            {
                fprintf(fp, "%i\n", (int)(255.0 * (*(img_src + x + y*width) - *min_value) / (*max_value - *min_value)));
            }
        }
        
        fclose(fp);
    }
    else
    {   
        sprintf(filename_and_extension, "%s.ppm", filename);
        FILE * fp = fopen(filename_and_extension, "w");
        
        fprintf(fp, "P3\n# Created by FerCer\n%i %i\n255\n", height, width);
                
        for (unsigned int x = 0; x < width; x++)
        {
            for (unsigned int y = 0; y < height; y++)
            {
                fprintf(fp, "%i\n", (int)(255.0 * (*(img_src + x + y*width) - *min_value) / (*max_value - *min_value)));
                fprintf(fp, "%i\n", (int)(255.0 * (*(img_src + height*width + x + y*width) - *min_value) / (*max_value - *min_value)));
                fprintf(fp, "%i\n", (int)(255.0 * (*(img_src + 2*height*width + x + y*width) - *min_value) / (*max_value - *min_value)));
            }
        }
        
        fclose(fp);
    }
    
}


double * loadPGM(const char * filename, int* height, int* width, int *n_channels, const double normalize_min, const double normalize_max)
{
    FILE* fp_img = fopen(filename, "r");
    
    // Read the magic number:
    char magic_number[4];
    fgets(magic_number, 3, fp_img);
    fgetc(fp_img);
    
    double * img_data;
    
    unsigned char load_mode = 0;
    if (strcmp(magic_number, "P1") == 0)
    {
        load_mode = 0;
        *n_channels = 1;
    }
    else if (strcmp(magic_number, "P2") == 0)
    {        
        load_mode = 1; // The image is in grayscale and ascii mode
        *n_channels = 1;
    }
    else if (strcmp(magic_number, "P3") == 0)
    {
        load_mode = 2; // The image is in color and ascii mode
        *n_channels = 3;
    }
    else if (strcmp(magic_number, "P5") == 0)
    {
        load_mode = 3; // The image is in grayscale and raw mode
        *n_channels = 1;
    }
    else if (strcmp(magic_number, "P6") == 0)
    {
        load_mode = 4; // The image is in color and raw mode
        *n_channels = 3;
    }
    else
    {    
        fclose(fp_img);
        printf("<<Error: The input file is not of the required format, its format is: \"%s\" instead of \"P2/3/5/6\">>\n", magic_number);
        return NULL;
    }
    
    // Read the commentary if it exists:
    char commentary[512];
    fgets(commentary, 512, fp_img);
    if (commentary[0] == '#')
    {
        fscanf(fp_img, "%i", height);
        fscanf(fp_img, "%i", width);
    }
    else
    {
        *height = atoi(commentary);
        char * width_ptr = strchr(commentary, ' ') + 1;
        *width = atoi(width_ptr);
    }
    
    double max_intensity;
    
    int ascii_intensity;
    
    switch (load_mode)
    {
        case 0:
            img_data = (double*)malloc(*height**width*sizeof(double));
            fgetc(fp_img);
            for (unsigned int x = 0; x < *width; x++)
            {
                for (unsigned int y = 0; y < *height; y++)
                {                    
                    ascii_intensity = fgetc(fp_img) - 48;
                    while (ascii_intensity < 0)
                    {
                        ascii_intensity = fgetc(fp_img) - 48;
                    }
                    *(img_data+x+y**width) = (double)ascii_intensity * (normalize_max - normalize_min) + normalize_min;
                }                
            }

            break;
            
        case 1:
            fscanf(fp_img, "%lf", &max_intensity);
            img_data = (double*)malloc(*height**width*sizeof(double));

            for (unsigned int x = 0; x < *width; x++)
            {
                for (unsigned int y = 0; y < *height; y++)
                {
                    fscanf(fp_img, "%i", &ascii_intensity);
                    *(img_data+x+y**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;
                }                
            }

            break;
            
        case 2:
            fscanf(fp_img, "%lf", &max_intensity);
            img_data = (double*)malloc(3**height**width*sizeof(double));

            for (unsigned int x = 0; x < *width; x++)
            {
                for (unsigned int y = 0; y < *height; y++)
                {
                    fscanf(fp_img, "%i", &ascii_intensity);
                    *(img_data+x+y**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;

                    fscanf(fp_img, "%i", &ascii_intensity);
                    *(img_data+x+y**width+*height**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;

                    fscanf(fp_img, "%i", &ascii_intensity);
                    *(img_data+x+y**width+2**height**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;
                }
            }
            break;
            
        case 3:
            fscanf(fp_img, "%lf", &max_intensity);
            img_data = (double*)malloc(*height**width*sizeof(double));
            for (unsigned int x = 0; x < *width; x++)
            {
                for (unsigned int y = 0; y < *height; y++)
                {
                    ascii_intensity = fgetc(fp_img);
                    *(img_data+x+y**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;
                }                
            }

            break;
            
        case 4:
            fscanf(fp_img, "%lf", &max_intensity);
            img_data = (double*)malloc(3**height**width*sizeof(double));
            
            for (unsigned int x = 0; x < *width; x++)
            {
                for (unsigned int y = 0; y < *height; y++)
                {
                    ascii_intensity = fgetc(fp_img);
                    *(img_data+x+y**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;

                    ascii_intensity = fgetc(fp_img);
                    *(img_data+x+y**width+*height**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;

                    ascii_intensity = fgetc(fp_img);
                    *(img_data+x+y**width+2**height**width) = (double)ascii_intensity/max_intensity * (normalize_max - normalize_min) + normalize_min;
                }
            }
            
            break;
    }
    
    fclose(fp_img);
    
    return img_data;
}
