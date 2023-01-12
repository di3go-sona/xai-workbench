#%%
from fastai.vision.all import *

#%% 
IMAGES_PATH = '../data/images/'
fnames = get_image_files(IMAGES_PATH)
dls = ImageDataLoaders.from_path_func(IMAGES_PATH, fnames, lambda x: x.parent.name, item_tfms=Resize(224))
dls.show_batch()
#%% 
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)



# %%
from fastai.callback.captum import *
captum = CaptumInterpretation (learn, cmap_name='custom blue', colors=None, N=256,
                       methods=('original_image', 'heat_map'),
                       signs=('all', 'positive'), outlier_perc=1)

#%%
import numpy as np
# %%
for id in range(5,100):
    fig = plt.figure(frameon=False)
    enc_data,dec_data, attributions = captum.visualize(fnames[id],metric='Occl',baseline_type='gauss')

    attr = attributions[0].transpose(2,0).transpose(0,1).detach().sum(-1)
    image = dec_data[0].transpose(2,0).transpose(0,1).detach()
    
    attr = np.ma.masked_array(attr, attr > 0.2)

    plt.imshow(image, cmap='brg')
    plt.imshow(attr, 
            alpha=.6, 
            cmap='viridis', 
            interpolation='bilinear')
    plt.show()