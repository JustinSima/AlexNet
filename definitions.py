"""Define constants."""
import os
import numpy as np

ROOT_DIR = os.path.dirname(__file__)

# Constants required for RGB PCA augmentation.
# Run utilities.pca_augmentation.create_pca_terms(**args) to generate terms.
RGB_MEANS = [154.35630689, 150.78023407, 129.73602081]

EIGENVECTORS = np.asarray(
    [[ 0.5409767,   0.82162038, -0.17967793],
    [ 0.48480786, -0.47921353, -0.73165274],
    [ 0.6872449,  -0.30869781,  0.65757137]]
 )
EIGENVALUES = np.asarray([4767.60760252, 1572.61412334, 1004.7856227])
