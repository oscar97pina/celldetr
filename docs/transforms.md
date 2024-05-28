# Transforms

Transforms and data augmentation are crucial in deep learning. We use torchvision transforms (v2) in this project and enable to specify the transforms and augmentations from the configuration files. The transforms must contain the ```augmentations``` and ```normalize``` keywords, which will include the data augmentations and the channel normalization parameters, respectively. Here's an example:

```yaml
transforms:
  augmentations:
    - name: elastic
      p: 0.2
      alpha: 0.5
      sigma: 0.25
    - name: hflip
      p: 0.5
  normalize:
    mean: [0.5,0.5,0.5]
    std:  [0.5,0.5,0.5]
```
As you can see, ```augmentations```must be a list, whose elements contains a ```name```and ```p```(probability of being applied), as well as any other parameter that should be sent to the corresponding transform. The list of available augmentations can be found in the in the source code: [```AugmentationFactory```](../celldetr/data/transforms.py#AugmentationFactory). Data augmentation transforms will only be applied to the train dataset, and the same normalization mean and standard deviation will be applied in train, val and test sets of that experiment.

You can find some default augmentation configurations in [configs/base/data/augmentations](../configs/base/data/augmentations/).
