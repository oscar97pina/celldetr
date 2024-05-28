# Model

The Cell-DETR model consists of a hierarchical backbone and a deformable transformer. This  is the how the model's config looks like:

```yaml
model:
    name: deformable_detr
    num_classes: 5
    num_queries: 900
    postprocess: label_topk
    with_box_refine: True
    two_stage: True
    hidden_dim: 256
    num_feature_levels: 4
    aux_loss: True
    position_embedding: 
    name: sine
    transformer:
    dim_feedforward : 1024
    nheads: 8
    dropout: 0.1
    enc_layers: 6
    dec_layers: 6
    enc_n_points: 4
    dec_n_points: 4
    checkpoint: /path/to/checkpoint.pt
    backbone:
        name: swin_L_384_22k    # name of the backbone
        dilation: False         # whether to use dilation in the last layer
        frozen_stages: -1       # stages that won't be trained
        return_layers:          # stages to output the features for
        - 1
        - 2
        - 3
        - 4
        checkpoint: /path/to/checkpoint.pt # pre-trained weights
    window:                     # pptional for window-DETR
        size: 256               # size of the windows
        stride: 192             # stride between windows
```
Let's understand its components:


## Deforamble transformer
The implementation of the deformable transformer is taken from the original [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR). Only the initialization functions have been modified to use the configuration files. The configuration of the Deformable-DETR with 4 feature levels, two stage and bounding box refine is the following:

```yaml
model:
    name: deformable_detr
    num_classes: 5
    num_queries: 900
    postprocess: label_topk
    with_box_refine: True
    two_stage: True
    hidden_dim: 256
    num_feature_levels: 4
    aux_loss: True
    position_embedding: 
    name: sine
    transformer:
    dim_feedforward : 1024
    nheads: 8
    dropout: 0.1
    enc_layers: 6
    dec_layers: 6
    enc_n_points: 4
    dec_n_points: 4
    checkpoint: /path/to/checkpoint.pt
```
This configuration is also provided in the base configuration files: [configs/base/model/detr/](../configs/base/model/detr/deformable_detr_4lvl.yaml). The newly added parameters are the ```postprocess```and the ```checkpoint```. The former specifies how to perform the postprocessing. Whereas the original source code performs a *top-k* selection for all queries and labels, we firstly select the most probable label for each query and then select the *top-k* queries. By default, *k* is set as num_queries/3, as in the original code. Secondly, the checkpoint refers to the checkpoint to a pre-trained model. _This checkpoint is only used when initializing the model for training, not for inference or evaluation_. In our experiments, we used the model pre-trained on the COCO dataset provided by the authors of Deformable-DETR.

## Backbone
As we have experimented with distinct backbones in our experiments, we have decoupled the backbone from the original model. Concretely, both ```resnet50``` and ```swin-large (swinL)```can be used. For the swin backbone, the configuration file looks like:

```yaml
model:
    backbone:
        name: swin_L_384_22k    # name of the backbone
        dilation: False         # whether to use dilation in the last layer
        frozen_stages: -1       # stages that won't be trained
        return_layers:          # stages to output the features for
        - 1
        - 2
        - 3
        - 4
        checkpoint: /path/to/checkpoint.pt # pre-trained weights
```

See [configs/base/model/backbone/](../configs/base/model/backbone/) for the different backbone configurations. Notably, another checkpoint for the backbone can be included. If a specific checkpoint for the backbone is provided, the backbone state from the checkpoint of the overall model will be overwriten, as well as the neck parameters.

## Window-DETR
To deal with large images, we perform the *window detection* procedure. To do so, you just need to specify the corresponding configuration into the model. For instance, when dealing with images of 1024x1024 px, and models that are trained in images of size 256x256 px (recommended), the run window detection with overlap you can add the following:
```yaml
model:
    window:
        size: 256
        stride: 192
```