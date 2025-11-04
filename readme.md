# Isolated Glyph Recognition Pipeline



This project implements a pipeline for isolated glyph recognition. The pipeline is organized into **five main steps**, designed to systematically process and classify individual glyphs with high accuracy.

![Pipeline Diagram](imgs/Pipeline.png)

## Pipeline Overview

The pipeline includes the following stages:

1. **Data Preparation**  
   - Data Collection and organization of glyph images.
   - Each image  is annotated and labeled by a subject matter expert.
   - The isolated glyphs are then split into training and test sets.
   - All training and testing sets can be downloaded [here](http://amadi.univ-lr.fr/ICFHR2018_Contest/index.php/download-1234-all)

2. **Preprocessing**  
   - Grayscale conversion 
   - Squaring, and padding are applied to standardize the input data for the CNN model
   
3. **Data Augmentation**  
   - Techniques such as rotation, scaling, shearing, and shifting to expand training data.
   - Helps improve model generalization.
   
4. **Classification**  
   - Implementation Transfer Learning and Fine tuning hyperparameters.
   - Trained Ensemble Customized CNN models.
  
5. **Evaluation**  
   - Performance measured using metrics like balanced accuracy, confusion matrix, .
   - Error analysis.
  

Our experimental results demonstrate that the proposed model outperforms existing approaches in isolated glyph classification accuracy. This work lays a solid foundation for optimizing deep learning models in resource-constrained scenarios and contributes to advancing the digitization and preservation of historical documents.

![benchmark](imgs/sota1.png)

## Getting Started

To run this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/isolated-glyph-recognition.git
cd isolated-glyph-recognition

# (Optional) Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate  # or venv\

# (Optional) Create and activate a virtual environment via conda
conda create -n myenv python=3.12.3
conda activate myenv

# install libraries
pip install -r requirements.txt
```



## How to run training
```
python ExpModifiedCNNV1.py --dataset ICFHR18_OB --nb_class 133 --model_index 12 --resize_meth bilinear --color_pad gray_white --img_size 75  --path_out expExpModifiedCNNV1OB_gray_white --weight 'imagenet'
```

## How to run inference
```
python InferenceEnsembelModifiedCNNV1ConvMetNonPersent.py --dataset ICFHR18_OS --nb_class 60 --resize_meth bilinear --color_pad gray_white --img_size 75  --path_out InferenceEnsembelModifiedCNNV1ConvMetNonPersent_gray_white --weight imagenet
```

The code in this repository was used for the publication  mentioned below. If you find this code useful, please cite [our paper](https://doi.org/10.1016/j.patcog.2025.112616)

```
to be informed soon
@article{PAULUS2025,
title = {Supervised Learning for Low-Resource Isolated Glyph Recognition in Palm Leaf Manuscripts},
author = {Erick Paulus and Jean-Christophe Burie and Fons J. Verbeek},
journal = {Pattern Recognition},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112616},
}
```


