# Sartorius---Cell-Instance-Segmentation
https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation
*  This is a Code Competition.
*  GPU Notebook <= 9 hours run-time.
*  Internet access disabled.
## NCKU "Machine Learning" - Competition
### Competition Description
Neurological disorders, including Alzheimer's and brain tumors,
are major global health issues causing death and disability.
Evaluating the effectiveness of treatments for these disorders is challenging.
One common method involves examining neuronal cells through light microscopy,
but segmenting these cells accurately is difficult and time-consuming.
Improved instance segmentation through computer vision could facilitate drug discovery.

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
```

### code
#### showimg.py
* Check the images and the annotation inside the dataset.
#### 
