#
#  %%
import numpy as np
from PIL import Image
import imageio
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                #  SSDMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,RotationTransform2D,
                                   AffineTransform2D)


import scipy.ndimage
#
import matplotlib.pyplot as plt

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# %%
"""
Read image
"""
def read_image(file):
    image = imageio.imread(file)
    return image

"""
Show image
"""
def show_image(image,title=""):
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


"""
Plot two images
"""
def plot_two_images(static, moving, display_transformed = True):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(static)
    plt.title("Reference image (Image_20449.tif)")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(moving)
    if display_transformed:
        plt.title("Transformed image")
    else:
        plt.title("Target image")

    plt.axis("off")


# image1
image1 = read_image("Image_20449.tif")
image2 = read_image("Image_20450.tif")

plot_two_images(image1, image2,False)


#%%
static = np.copy(image1)
moving = np.copy(image2)
## Threshold image
def threshold_image(im):
    im[im<np.mean(im)]=0
    return im

static = threshold_image(static)
moving = threshold_image(moving)

print(f"Images thresholded at the mean")
plot_two_images(static, moving,False)

#%%
# Transform - center of mass of images
c_of_mass = transform_centers_of_mass(static, None, moving, None)
print("Images with center of mass transformed")
transformed = c_of_mass.transform(moving)

moving = np.copy(transformed)
plot_two_images(static, transformed,False)
# %%
from scipy import optimize
import scipy.ndimage as ndimage

"""
Method to compute ssd between two arrays
"""
def ssd(arr1,arr2):
    """ Compute the sum squared difference metric """
    x = min(arr1.shape[0],arr2.shape[0])
    y = min(arr1.shape[1],arr2.shape[1])
    return np.sum((arr1[:x,:y]-arr2[:x,:y])**2)

"""
Method to apply rotation between two images
"""
def apply_rotation(angle, img):
    pivot = scipy.ndimage.center_of_mass(img)
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    pad_width = np.array([padY, padX],dtype='i')
    imgP = np.pad(img, pad_width, 'constant')
    imgR = ndimage.rotate(imgP, angle,order=1)
    return imgR

"""
Method computes the cost for the optimizer
"""
def cost_ssd(param,reference_image, target_image):
    # = static, moving
    transformed = apply_rotation(param,target_image)
    cost =  ssd(reference_image, transformed)
    print("Param",param,"cost",cost)
    return cost

## Initial param
initial = [np.rad2deg(1)]
rotation_transform = RotationTransform2D()


def cost_function_with_dipy_affine_transform(params,static, moving,transform):
    reference_image, target_image = static, moving

    ## compute affine transform for parameters
    current_affine = transform.param_to_matrix(params)
    affine_map = AffineMap (current_affine , reference_image.shape, None,
                                      target_image.shape, None )
    ## Apply affine transform
    transformed = affine_map.transform ( target_image )

    ## compute cost
    cost =  ssd(reference_image, transformed)
    return cost


best_params = optimize.minimize(cost_function_with_dipy_affine_transform, initial, (static, moving,rotation_transform) ,method='L-BFGS-B', tol=1e-6)
## rotate image
print("Best params",best_params)

## compute affine transform for parameters
current_affine = rotation_transform.param_to_matrix(best_params["x"])
affine_map = AffineMap(current_affine, static.shape, None, moving.shape, None)
print(affine_map.affine)
## Apply affine transform
transformed2 = affine_map.transform(moving)

transformed2bis = scipy.ndimage.interpolation.rotate(moving,(best_params["x"]),reshape=False)
plot_two_images(static, transformed2, False)
plot_two_images(static, transformed2bis, False)

# %%
