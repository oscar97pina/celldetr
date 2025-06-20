# From Pixels to Cells: Rethinking Cell Detection in Histopathology with Transformers

## Description

Accurate and efficient cell nuclei detection and classification in histopathological Whole Slide Images (WSIs) are essential for enabling large-scale, quantitative pathology workflows. While segmentation-based methods are commonly used for this task, they introduce substantial computational overhead and produce detailed masks that are often unnecessary for clinical interpretation. In this work, we propose a paradigm shift from segmentation to direct detection, introducing CellNuc-DETR, a transformer-based model that localizes and classifies nuclei without relying on segmentation masks or expensive post-processing. By focusing on clinically relevant outputs—nuclear location and type—CellNuc-DETR achieves significant gains in both accuracy and inference speed. To scale to full-slide inference, we develop a novel strategy that partitions feature maps, rather than images, enabling efficient processing of large tiles with improved context aggregation. We evaluate CellNuc-DETR on PanNuke, CoNSeP, and MoNuSeg, demonstrating state-of-the-art performance, strong generalization across tissue and stain variations, and up to 10x faster inference than segmentation-based methods. Our approach bridges the gap between generic detection frameworks and the practical demands of digital pathology, offering a scalable, accurate, and clinically viable solution for cell-level analysis in WSIs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
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
|   |-- finetune.py          # Fine-tuning on COCO dataset, can be split into stages.
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

### Pre-trained models
The pre-trained model we provide for inference is trained on 80% of PanNuke, taking samples from each of the folds. The detection F1-Score corresponds to the remaining 20% of the data.

| Resolution | Backbone | Backbone levels | DETR levels | Layers/Queries | F1-det | config | weights |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 0.25mpp  |   Swin-T  |   3  |  4  | 3/600 |   82.67  |   [config](configs/public/deformable_detr_swinT_35lvl_S_pannuke.yaml) | [weights](https://drive.google.com/file/d/18aO6SwQ6bdDKusTbpl6trbtp5cIWm2DC) |
| 0.50mpp  |   Swin-T  |   4  |  4  | 3/600 |   81.77  |   [config](configs/public/deformable_detr_swinB_4lvl_S_050mpp_pannuke.yaml)| [weights](https://drive.google.com/file/d/1Atnsv6DdrhbjNDE8hW1kdQPkLFUWrA0O) |
| 0.25mpp  |   Swin-L  |   4  |  4  | 6/900 | 83.06  |   [config](configs/public/deformable_detr_swinL_4lvl_pannuke.yaml)  | [weights](https://drive.google.com/file/d/13ud0-KD2f70p7x_c4WdtWXvLR-0YFVaH) |

### Training and testing
For training and testing, you must create a configuration file in which you specify basic configuration of the [experiment](docs/experiment.md), the [dataset](docs/dataset.md), the [model](docs/model.md), the loss and the [loaders](docs/). You can find an example on [this config file](configs/experiments/pannuke/swin/deformable_detr_swinL_35lvl_split123.yaml). To run the training, we recommend creating your own configuration file, in which you'll have the modify the data directory and model checkpoints, and run:

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
Inference on WSIs is very easy! You just need to create a configuration file that extends (with ```__base__```) an existing configuration file provided jointly with the model weights (see Pre-trained models table above), the model window parameters (for window detection) and the ```infer``` loader configuration. See [docs/inference.md](docs/inference.md).

Then, running is as easy as:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/infer.py \
                        --config-file /path/to/config/file \
                        --experiment.output_dir /path/model/weights/folder \
                        --experiment.output_name model_weights_name.pth
```

## Citation
If you find this work helpful in your research, please consider citing us:
```bash
@misc{pina2025cellnucleidetectionclassification,
      title={Cell Nuclei Detection and Classification in Whole Slide Images with Transformers}, 
      author={Oscar Pina and Eduard Dorca and Verónica Vilaplana},
      year={2025},
      eprint={2502.06307},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.06307}, 
}
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
