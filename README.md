# Sartorius---Cell-Instance-Segmentation
https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation
*  This is a Code Competition.
*  GPU Notebook <= 9 hours run-time.
*  Internet access disabled.
## NCKU "Machine Learning" - (Competition)
### Competition Description
  For this semester's final project, our focus is on the instance segmentation competition for neural cells hosted by the internationally renowned biotechnology and pharmaceutical company, Sartorius, on Kaggle [1]. Neurological disorders remain a leading cause of global mortality and disability. The lack of effective quantitative methods for assessing the therapeutic efficacy of diseases poses challenges in devising treatment plans and developing new drugs. Traditional examination methods involve observing the morphology of neural cells through optical microscopes, but the vast number of neural cells makes precise judgment and quantification by human efforts challenging.
  
  Therefore, AI assistance is crucial for analysis, and the initial step in AI analysis involves segmenting individual neural cells. Accurate segmentation of different neural cells enables quantitative data analysis with the aid of AI. Researchers can measure the impact of therapeutic drugs on neural cells, facilitating the advancement of new drug development.
  
  The primary objective of this competition is to address the diverse appearance changes in the most challenging SH-SY5Y type neural cells.

### Dataset Download
```shell
kaggle competitions download -c sartorius-cell-instance-segmentation
```
![](./readme_img/kaggle_img.png)

In order to prevent the database from being deleted, I first backed it up to my personal cloud space.

* [(Dataset-GoogleDrive Link)](https://drive.google.com/file/d/1n76PHLwMhEj7LdhSUDbYDPOv75D06bkU/view?usp=sharing) & [(Annotation-json_file)](https://drive.google.com/drive/folders/15_k-MsnejPnD18CqypdHxiF3MBW0hu1D?usp=sharing)

![](./readme_img/image_ann.png)
### Dependencies
```shell
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install pycocotools
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install fastcore
pip install ensemble-boxes
pip install nms
```

### code
#### showimg.py
* Check the images and the annotation inside the dataset.
#### train.py
* training the model.
#### inference.py
* Inference the model with the final model .pth file.
#### ensemble_inference.py
* Ensemble multiple models or difference test time augmentation.

### Method
#### **Detailed description of each part is in [PDF File](https://github.com/chia-shein/Sartorius---Cell-Instance-Segmentation/blob/main/sartorius_methods.pdf).**
1. Mask R-CNN
2. Model Ensembling
3. Image Normalizarion
4. Augmentation
5. Test Time Augmentation

### Experiment Results

