# Cell-DETR: Efficient cell detection and classification in WSIs with transformers

## Description

Cell-DETR is a novel approach for efficient cell detection and classification in Whole Slide Images (WSIs) using transformers. Unlike traditional segmentation methods, Cell-DETR focuses on detecting and classifying nuclei, crucial for understanding cell interactions and distributions. With state-of-the-art performance on established benchmarks, our method enables scalable diagnosis pipelines by significantly reducing computational burdens and achieving x3.4 faster inference times on large WSI datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Create a Virtual environment
```bash
python3 -m venv myenv
```

2. Activate the environment
```bash
source myenv/bin/activate
```

3. Install torch>=2.1.1 and torchvision>=0.16.1. We've used cuda 11.8. You can find more information in the [official website](https://pytorch.org/get-started/locally/).
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Install the other requirements
```bash
pip3 install -r requirements.txt
```

5. Build the MultiScaleDeformAttn module (you will need GPU, follow the authors' [instructions](https://github.com/fundamentalvision/Deformable-DETR))
```bash
cd celldetr/models/deformable_detr/ops
./make.sh
``` 

## Usage

### Project structure
```bash
celldetr/
|-- tools/                   # Contains scripts for training, evaluation, and inference
|   |-- train.py             # Training on COCO format dataset
|   |-- eval.py              # Evaluation on COCO format dataset
|   |-- infer.py             # Inference on WSIs
|-- eval/                    # Evaluation module for evaluating model performance (COCO and Cell detection)
|-- util/                    # Utility functions and modules used throughout the project
|-- data/                    # Datasets, transforms and augmentations for cell detection
|-- models/                  # Deep learning models used in CellDetr
|   |-- deformable_detr/     # Implementation of Deformable DETR model
|   |-- window/              # Implementation of window-based DETR model
|   |-- backbone/            # Backbone networks used in the models
|-- engine.py                # Main engine script for coordinating the training and evaluation process

```
### Configuration files
The entire code is based on yaml configuration files. You can find an explanation of how they work in [config.md](docs/config.md).
- In the [configs/base/](configs/base/) folder you can find default configurations for the datasets, loss functions, model architectures, and others.
- In the [configs/experiments/](configs/experiments/) folder you can find the configuration files for the experiments reported in the paper.

If you want to use your own data, models, or others, consider writing the re-usable independent configuration pieces such as datasets into [configs/base/](configs/base/) and then the configuration for your experiments in [configs/experiments/](configs/experiments/).

### Training and testing
For training and testing, you must create a configuration file in which you specify basic configuration of the [experiment](docs/experiment.md), the [dataset](docs/dataset.md), the [model](docs/model.md), the loss and the [loaders](docs/). You can find an example on [configs/experiments/pannuke/swin/deformable_detr_swinL_35lvl_split123.yaml](configs/experiments/pannuke/swin/deformable_detr_swinL_35lvl_split123.yaml). To run the training, we recommend creating your own configuration file, in which you'll have the modify the data directory and model checkpoints, and run:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/train.py \
                        --config-file /path/to/config/file
```

Alternatively, you could also modify the paths from the command line:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/train.py \
                        --config-file /path/to/config/file \
                        --opts dataset.train.root=/path/to/dataset/ \
                               dataset.val.root=/path/to/dataset/ \
                               dataset.test.root=/path/to/dataset/ \
                               model.checkpoint=/path/to/checkpoint \
                               model.backbone.checkpoint=/path/to/backbone/checkpoint
```

Testing is based on the same configuration file that has been used for training, but calling the [tools/eval.py](tools/eval.py) script rather than the training one. Note that for the evaluation, the training must have been run previously and ended successfully. Now, the checkpoints specified to the model and backbone configurations will be ignored, but the output checkpoint will be used when initializing the model.

### Inference on WSIs
Inference on WSIs is very easy! You just need to create a configuration file that extends (with ```__base__```) an existing configuration file used for training your model, and then include the configuration for the WSI, the model window parameters (for window detection) and the ```infer``` loader configuration. See [docs/inference.md](docs/inference.md). You can find an example here for a slide of the camelyon dataset [configs/camelyon/deformable_detr_swinL.yaml](configs/experiments/camelyon/deformable_detr_swinL_4lvl.yaml).

Then, running is as easy as:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/infer.py \
                        --config-file /path/to/config/file
```

## Citation
If you find this work helpful in your research, please consider citing us:
```bash
@inproceedings{
       pina2024celldetr,
       title={Cell-{DETR}: Efficient cell detection and classification in {WSI}s with transformers},
       author={Oscar Pina and Eduard Dorca and Veronica Vilaplana},
       booktitle={Submitted to Medical Imaging with Deep Learning},
       year={2024},
       url={https://openreview.net/forum?id=H4KbJlAHuq},
       note={under review} }
```

## License
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project is licensed under the [MIT License](LICENSE).
