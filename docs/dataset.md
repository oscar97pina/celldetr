# Dataset

Both training and evaluation requires datasets to be in the COCO format. In the corresponding files [pannuke.py](celldetr/data/pannuke.py), [consep.py](celldetr/data/consep.py) and [monuseg.py](celldetr/data/monuseg.py) we provide the functions to transform the original datasets into COCO format.

Generally, the datasets come with a pre-defined split into distinct folds. Each fold must be processed independently, so that an independent COCO dataset is created for each fold. For example, the pannuke datasets is split into three independent folds. Therefore, the file structure will be:

```bash
pannuke/
|-- fold1/                   
|   |-- images/
|   |-- annotations.json
|-- fold2/                   
|   |-- images/
|   |-- annotations.json
|-- fold3/ 
|   |-- images/
|   |-- annotations.json
```

In the experiment config file, there must be a ```dataset``` entry specifying the name, root directory, number of classes and fold configuration (which includes train, val and test) of the dataset. For example, for the pannuke dataset training with the first fold, validating on the second and testing on the third, the configuration must be:

```yaml
dataset:
    train:
        name: pannuke
        root: /path/to/pannuke_coco/
        num_classes: 5
        fold: fold1
    val:
        name: pannuke
        root: /path/to/pannuke_coco/
        num_classes: 5
        fold: fold2
    test:
        name: pannuke
        root: /path/to/pannuke_coco/
        num_classes: 5
        fold: fold3
```
We provide base configuration files for the dataset configurations in [configs/base/data](../configs/base/data/).

In case you want to use your own datasets, you just need to name the dataset as ```cell``` in the config file, and then specify the fold partitioning as in the example below. The code expects the image folders and annotations in: ```/path/to/your/coco/dataset/foldname/images/```and ```/path/to/your/coco/dataset/foldname/annotations.json```, respectively.

```
dataset:
    train:
        name: cell
        root: /path/to/my/dataset/train/
        num_classes: 5
    val:
        name: pannuke
        root: /path/to/my/dataset/val/
        num_classes: 5
    test:
        name: pannuke
        root: /path/to/my/dataset/test/
        num_classes: 5
```

If you only want to perform cell detection (rather than detection and classification), you can specify it by adding ```detection: True``` to the model. This will wrap your dataset and convert all labels to 1.