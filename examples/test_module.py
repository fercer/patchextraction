import numpy as np
import patchextraction as pe
import matplotlib.pyplot as plt
import sys

U, V = np.meshgrid(np.arange(300), np.arange(300))
a = U**2 + V**2
b = np.logical_and(a>20000,a<21000).astype(np.float64)

fg_sample_indices, bg_sample_indices = pe.computeSampledClasses(b, 32, 1)

print(bg_sample_indices.shape, bg_sample_indices.max())
print(fg_sample_indices.shape, fg_sample_indices.max())

patches = pe.extractSampledPatches(a, fg_sample_indices, 32)
gt_patches = pe.extractSampledPatches(b, fg_sample_indices, 32)

print(patches.shape)
print(gt_patches.shape)

plt.subplot(1, 2, 1)
plt.imshow(patches[0,0])
plt.subplot(1, 2, 2)
plt.imshow(gt_patches[0,0])
plt.scatter(16,16)
plt.show()


U, V = np.meshgrid(np.arange(16), np.arange(16))
a = U**2 + V**2
b = np.logical_and(a>110,a<170).astype(np.float64)

img_classes = pe.computeClasses(b, 4, 4, 0)

print(img_classes.shape)
print(img_classes)

plt.subplot(1, 2, 1)
plt.imshow(a)
plt.subplot(1, 2, 2)
plt.imshow(b)
plt.show()


patches = pe.extractPatches(a, 4, 4)
gt_patches = pe.extractPatches(b, 4, 4)

print(patches.shape)
print(gt_patches.shape)

plt.subplot(4, 2, 1)
plt.imshow(patches[0,0])

plt.subplot(4, 2, 2)
plt.imshow(gt_patches[0,0])

plt.subplot(4, 2, 3)
plt.imshow(patches[1,0])

plt.subplot(4, 2, 4)
plt.imshow(gt_patches[1,0])

plt.subplot(4, 2, 5)
plt.imshow(patches[2,0])

plt.subplot(4, 2, 6)
plt.imshow(gt_patches[2,0])

plt.subplot(4, 2, 7)
plt.imshow(patches[3,0])

plt.subplot(4, 2, 8)
plt.imshow(gt_patches[3,0])
plt.show()