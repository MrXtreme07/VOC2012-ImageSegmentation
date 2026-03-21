# Semantic Image Segmentation with PASCAL VOC 2012

## Overview

This project implements a deep learning model for **semantic image segmentation** using the PASCAL VOC 2012 dataset. The goal is to classify each pixel in an image into one of the predefined object classes.

## Dataset

The project uses the **PASCAL VOC 2012** dataset, a standard benchmark dataset for computer vision tasks such as object detection and semantic segmentation.

Dataset details:

* ~2913 images with segmentation annotations
* 20 object classes + background
* Pixel-wise ground truth masks

Object classes include:
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tv/monitor.

## Method

A convolutional neural network (CNN) based segmentation model is used to predict pixel-wise class labels. The model learns from labeled segmentation masks provided in the dataset.

## Usage

1. Download the PASCAL VOC 2012 dataset.
2. Place the dataset in the `dataset/` directory.
3. Run the training script:

python train.py

## Goal

The objective of this project is to train a segmentation model capable of accurately identifying and segmenting objects in images from the dataset.
