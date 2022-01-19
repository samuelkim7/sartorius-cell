# sartorius-cell

## About the Kaggle Competition
- Title: Sartorius - Cell Instance Segmentation
- Link: https://www.kaggle.com/c/sartorius-cell-instance-segmentation
- Date: 2021.10.15 - 2021.12.31

## Setup for detectron2
- Create a new conda environment
- Install pytorch and some needed packages
- Install detectron2 (don't do pip install detectron2)
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- If there is a cuda version issue in installation, try pre-built detectron2 for linux at  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

## Train the model with the jupyter notebook
- Use train_mask_rcnn.ipynb
- Revise the paths of dataset and coco-format json files
- Run each cell in the notebook

## Train the model with the python script
- Use train.py
- Revise the paths accordingly
- run the python script
```
python train.py
```
- multi-gpu training
```
python train.py --num-gpus 4
```

## Make pseudo labels using a trained model
- Use pseudo_labelling.py
- Prepare the images for pseudo labeling
- Revise the paths of dataset and model checkpoint accordingly
- run the python script