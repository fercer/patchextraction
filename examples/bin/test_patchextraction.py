import cv2
import numpy as np
import matplotlib.pyplot as plt
from patchextraction import *
import sys

img = cv2.imread('D:/test_data/Angios_134/1.png', 0) / 255.0
gt = cv2.imread('D:/test_data/Angios_134/1_gt.png', 0) / 127.5 - 1.0

patch_size = 128
patch_stride = 1
patch_extraction_mode = 1

labels = computeClasses(gt, patch_size, patch_stride, patch_extraction_mode)
print(labels.shape)

sample_indices = generatePatchesSample(gt, patch_size, patch_stride, patch_extraction_mode = patch_extraction_mode, sample_percentage = 50, labels = labels)
print(sample_indices.shape)

patches_gt, patches_classes = extractSampledPatchesAndClasses(gt, patch_size, patch_stride, patch_extraction_mode = patch_extraction_mode, patches_samples = sample_indices)
patches = extractSampledPatches(img, patch_size, patch_stride, patches_samples = sample_indices)

print(patches.shape)
print(patches_gt.shape)
print(patches_classes.shape)

print(patches_classes)

print(patches_classes[3,0,:,:])
print(patches_classes[7,0,:,:])

plt.subplot(2, 2, 1)
plt.imshow(np.squeeze(patches[3,0,:,:]))
plt.subplot(2, 2, 2)
plt.imshow(np.squeeze(patches_gt[3,0,:,:]))
plt.subplot(2, 2, 3)
plt.imshow(np.squeeze(patches[7,0,:,:]))
plt.subplot(2, 2, 4)
plt.imshow(np.squeeze(patches_gt[7,0,:,:]))

plt.show()
