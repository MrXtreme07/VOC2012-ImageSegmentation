import numpy as np

VOC_COLORMAP = np.array([
    [0,0,0],
    [128,0,0],
    [0,128,0],
    [128,128,0],
    [0,0,128],
    [128,0,128],
    [0,128,128],
    [128,128,128],
    [64,0,0],
    [192,0,0],
    [64,128,0],
    [192,128,0],
    [64,0,128],
    [192,0,128],
    [64,128,128],
    [192,128,128],
    [0,64,0],
    [128,64,0],
    [0,192,0],
    [128,192,0],
    [0,64,128]
])

def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(len(VOC_COLORMAP)):
        rgb[mask == cls] = VOC_COLORMAP[cls]
    
    return rgb