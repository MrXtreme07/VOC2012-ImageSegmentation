# Image Segmentation using U-Net (VOC2012)

## Overview

This project implements a U-Net based model for semantic image segmentation on the Pascal VOC 2012 dataset. The objective is to assign a class label to each pixel in the image across 21 categories (including background).

---

## Project Structure

```
.
├── data/
│   └── voc_dataset.py
├── models/
│   └── unet.py
├── checkpoints/
├── train.py
├── predict.py
├── outputs/
└── README.md
```

---

## Model Architecture

The model follows the U-Net encoder–decoder design:

**Encoder (Downsampling Path):**

* Repeated blocks of:

  * 3×3 Convolution
  * ReLU activation
  * Max Pooling

**Decoder (Upsampling Path):**

* Transposed Convolution for upsampling
* Skip connections from encoder layers
* Convolution + ReLU

**Output Layer:**

* 1×1 convolution mapping features to 21 classes

---

## Training Details

* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Learning Rate: 1e-4
* Batch Size: 4
* Epochs: 200

---

## Training Loss Curve

<img width="1673" height="873" alt="image" src="https://github.com/user-attachments/assets/7ee89693-3486-41a4-a853-c84d109484a1" />


---

## Segmentation Results

### Example 1

<img width="1724" height="883" alt="image" src="https://github.com/user-attachments/assets/34804d70-92a9-4324-9566-dab3bab9b2dd" />


---

### Example 2

<img width="1654" height="865" alt="image" src="https://github.com/user-attachments/assets/76842e2a-53ea-4e35-9aa2-a11f6942722c" />


---

## Observations

* The training loss decreases steadily with minor oscillations due to mini-batch updates.
* The model performs well on large and well-defined objects.
* Performance degrades for small or complex objects such as humans in cluttered scenes.
* Some prediction noise is observed, likely due to limited model capacity and absence of augmentation.

---

## Learned Features

* Initial layers capture low-level features such as edges and textures.
* Deeper layers encode higher-level semantic information.
* Skip connections help retain spatial resolution and fine details.

---

## Final Kernel Weights

<img width="905" height="913" alt="image" src="https://github.com/user-attachments/assets/750adaf8-2354-4886-b7c3-d74b68677bc5" />


---

## How to Run

### Train the model

```
python train.py
```

### Resume training from checkpoint

```
python train.py --resume checkpoints/model_epoch_X.pth
```

### Run inference

```
python predict.py
```

---

## Future Improvements

* Introduce data augmentation (flipping, cropping, scaling)
* Add Batch Normalization layers
* Increase network depth
* Train on GPU for improved performance
* Explore advanced architectures such as DeepLab or SegNet

---

## Dataset

Pascal VOC 2012:

* 21 semantic classes (20 objects + background)
* Real-world images with complex scenes

---

## Conclusion

The U-Net model successfully learns semantic segmentation and produces meaningful predictions on the VOC dataset. While performance is strong for large objects, improvements are needed for finer structures and complex scenes.
