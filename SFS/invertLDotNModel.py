import numpy as np

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f    
import matplotlib.pyplot as plt

@jit(nopython = True, cache = True)
def invertLDotN(image_tensor_pixel, illum_model_pixel, shadow_tensor_pixel ):
    '''Implementing per pixel l-dot-n shading 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1233898
    eq. 2- 3 and section 4.1 '''
    # n_light_channels = image_tensor_pixel.shape[-1]
    # implement ((WL)^T WL)^{-1}(WL)

    WL = np.diag(shadow_tensor_pixel) @ illum_model_pixel
    WI = np.diag(shadow_tensor_pixel) @ image_tensor_pixel 
    # print(WL.shape)
    # print(WI.shape)
    # print((np.linalg.inv(WL.T@WL)@WL.T@WI).shape)
    # exit()
    return (np.linalg.inv(WL.T@WL)@WL.T@WI)

@jit(nopython = True, cache = True)
def perPixelLinearShading(image_tensor, illum_model, shadow_tensor):
    ''' Apply invertLDotN to each pixel
    Inputs:
    image_tensor: np.ndarray HxWxC
    shadow_tensor: np.ndarray HxWxC
    illum_model = np.ndarray HxWxCx3
    Returns:
    normals: np.ndarray HxWx3  '''

    n_rows, n_cols = image_tensor.shape[0], image_tensor.shape[1]
    normals = np.zeros((n_rows, n_cols, 3))

    # assert image_tensor.data.c_contiguous
    # assert illum_model.data.c_contiguous
    # assert shadow_tensor.data.c_contiguous
    # assert normals.data.c_contiguous

    for row in range(n_rows):
        for col in range(n_cols):
            normals[row, col, :] = invertLDotN(image_tensor[row, col, :],\
                                               illum_model[row, col, :,:],\
                                                 shadow_tensor[row, col,:] )
            
    return normals


def main(plot = True):

    image = np.random.uniform(0, 1, (100,100,4))
    H,W = image.shape[0], image.shape[1]
    illum_model = np.random.uniform(0, 1, (100,100,4,3)) # already per pixel

    # illum_model = [np.random.uniform(0,1,(3,))]*4 # list of 4 vectors
    # if isinstance(illum_model, list):
    #     illum_model_list = []
    #     for model in illum_model_list:
    #         illum_model_this_channel = np.dstack([np.ones_like(H,W) * model[0], \
    #                                               np.ones_like(H,W) * model[1],
    #                                               np.ones_like(H,W) * model[2]])
    #         illum_model_list.append(illum_model_this_channel)
    #         print(len(illum_model_list))
    #     illum_model = np.dstack(illum_model_list)

    image_mean = image.mean(axis =  -1, keepdims = True)
    image_mean_frac = np.divide(image, image_mean)
    shadow_mask = np.clip(image_mean_frac, a_min = 0.25, a_max= 1.0)
    
    # normalize illuminaion model
    illum_model = illum_model/ np.linalg.norm(illum_model, axis=-1, keepdims = True)
    
    normals = perPixelLinearShading(image, illum_model, shadow_mask)

    fig, ax = plt.subplots(3,3)
    ax = ax.flatten()
    for i, thing, title in zip(np.arange(9).tolist(), [*np.dsplit(image, 4), *np.dsplit(shadow_mask, 4), (normals+1)/2.0], \
                                     ['img_1', 'img_2', 'img_3', 'img_4', 
                                      'shdw_1', 'shdw_2', 'shdw_3', 'shdw_4',
                                      'normals']):

        ax[i].imshow(thing.squeeze())
        ax[i].set_title(title)
    plt.show()



    return normals

if __name__ == "__main__":
    main()
    






