# 2D Skin layers and Keratinocytes Segmentation
This project descibes the codes for 2D skin layers (air gap, SC, epidermis, dermis) and keratinocytes segmentation model skin images. 

The segementation model is based on the **U-Net** architecture with applying **deep supvision** and our proposed **deep feature sharing** method.

## Dataset Format
The file name of each image and its corresponding layer and cell labeling should be the same.

```
./[your own path]/dataset/dataset
----/image   % Image files [png]
--------/1_20180129094629_cheek.png
--------/1_20180209120922_normal.png
--------/1_20180326_111332_foot.png
----/layer   % skin layers labeling [png]
--------/1_20180129094629_cheek.png
--------/1_20180209120922_normal.png
--------/1_20180326_111332_foot.png
----/cell    % cell nuclei labeling [png]
--------/1_20180129094629_cheek.png
--------/1_20180209120922_normal.png
--------/1_20180326_111332_foot.png
```

## Available Model

* U-Net
* U-Net with Deep Supervision
* U-Net with Deep Feature Sharing
* U-Net with Deep Supervision and Deep Feature Sharing

## Installation

* Clone this repo:
```
git clone https://github.com/tomohiroliu22/skin-segmentation-with-DS-and-DFS
cd skin-segmentation-with-DS-and-DFS
```

## Model Comparison and Evaluation
* Training all types of the models with 5 folds cross-validation by running 
```
bash train.sh
```
* Testing all types of the models with 5 folds cross-validation by running 
```
bash test.sh
```

## Model Training

### Training for U-Net with DS and DFS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DFS_w_DS --phase train --lr 0.001 --step 10 --epoch 25
```

### Training for U-Net with DFS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DFS --phase train --lr 0.001 --step 10 --epoch 25
```

### Training for U-Net with DS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DS --phase train --lr 0.001 --step 10 --epoch 25
```

### Training for U-Net
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model NONE --phase train --lr 0.001 --step 10 --epoch 25
```

## Model Testing

### Testing for U-Net with DS and DFS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DFS_w_DS --phase test --modelpath ./[path to your model]
```

### Testing for U-Net with DFS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DFS --phase test --modelpath ./[path to your model]
```

### Testing for U-Net with DS
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model DDS --phase test --modelpath ./[path to your model]
```

### Testing for U-Net
```
python main.py --dataroot ./[your own path]/datasets --name [your experiment name] --model NONE --phase test --modelpath ./[path to your model]
```

## Model Performance

The mean DICE coefficient comparison for 4 types of models

| Model | U-Net  | U-Net+DS | U-Net+DFS | U-Net+DS+DFS |
| ----- | ------ | ------ | ------ | ------ |
| Cell  | 0.7125 | 0.7138 | 0.7185 | 0.7199 |
| Layer | 0.8883 | 0.8904 | 0.8915 | 0.8928 |


