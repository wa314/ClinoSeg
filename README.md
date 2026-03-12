# ClinoSeg
ClinoSeg is a Python library for clinoform segmentation in seismic.
## Official repository for the paper "Synthetic Seismic Data Generation for Deep Learning Segmentation of Clinoforms: Integrating Process-Based Forward Stratigraphic Modelling and Ray-Based Seismic Modelling"

### Abstract
Clinoform identification plays a major role in sequence stratigraphy and applied geosciences offering key insights into sediment fluxes, basin evolution, and natural resource exploration. However, manual interpretation of clinoform geometries in seismic data is time-consuming and labour-intensive. This study proposes a deep learning framework to automate clinoform detection and segmentation. To address the scarcity of labelled training data, we generated 20,000 synthetic seismic images using process-based forward stratigraphic modelling followed by ray-based seismic modelling, ensuring both geological realism and variability of data. The generated synthetic seismic images were split into training (70%), validation (15%), and testing (15%) sets. The clinoform binary masks were labelled using the top and base layers from the forward stratigraphic models. Then, a YOLOv8m model was trained on this dataset achieving 86% precision, 82% recall, and 83% Dice coefficient on segmenting clinoforms in the synthetic testing dataset. Qualitative validation on real seismic images confirmed accurate bounding box detection and segmentation of clinoform bodies. These results highlight the potential of deep learning to accelerate seismic interpretation, particularly during early scanning phases, where rapid clinoform analysis supports stratigraphic assessment and reservoir characterization. The proposed workflow bridges the synthetic-to-real domain gap in seismic data, offering a scalable approach to train algorithms for advancing subsurface analysis.

### Example image
![Fig](https://github.com/user-attachments/assets/e64826ca-826f-4fd2-8c9b-9db5b4c382c0)

### Licence
This package is released under the MIT licence.

### Data
Data are avaliable at: https://zenodo.org/records/18977525.

### Citation


## Acknowledgement
The work is part of Waleed AlGharbi’s PhD research at Imperial College London.
