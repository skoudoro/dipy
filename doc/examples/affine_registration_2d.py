
# coding: utf-8

# In[2]:


import numpy as np
from PIL import Image
import imageio
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,SSDMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,regtransforms,
                                   AffineTransform2D, RotationTransform2D)

import scipy
#
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


plt.style.use("fivethirtyeight")


# In[3]:


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
def plot_two_images(static, moving, text=""):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(static)
    plt.title("Static image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(moving)
    plt.title("Moving image" + " " + text)
    plt.axis("off")
    plt.show()
    
    
# image1
image1 = read_image("Image_20449.tif")
image2 = read_image("Image_20450.tif")

plot_two_images(image1, image2)


# In[4]:
static = np.copy(image1)
moving = np.copy(image2)


#%%
## Threshold image
def threshold_image(im):
    im[im<np.mean(im)]=0
    return im

static = threshold_image(static)
moving = threshold_image(moving)


sampling_prop = None
metric = MutualInformationMetric(32,sampling_prop)

# In[22]:
level_iters = [10000]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

"""
Now we go ahead and instantiate the registration class with the configuration
we just prepared
"""
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=None,
                            factors=None)
initial = [np.rad2deg(1)]
transform = RotationTransform2D()
affine_map, params, fopt =  affreg.optimize(static, moving, transform, initial,
                              None, None,
                              starting_affine=None, ret_metric = True)
print(params)


transformed = affine_map.transform(moving)

# In[19]:
plot_two_images(static,transformed)
plt.show()
