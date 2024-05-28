# Experiment

To train and evaluate Cell-DETR, some base configuration must be provided:

```yaml
experiment:
  wandb: True                               # logging to wandb
  project: project_name              # project (wandb)
  name: experiment_name         # name of the experiment (wandb)
  output_dir: /path/to/output/dir # folder where to place the checkpoints
  output_name: model_name.pth # name of the checkpoint
  seed: 42  # random seed
  resume: False # whether to resume training or not
```

Also, the evaluation criteria must be included. This will apply to the evaluation on the validation set during training, as well as to both the validation and testing set during evaluation:

```yaml
evaluation:
  interval: 10  # interval for evaluation
  thresholds:   # confidence thresholds at which consider predictions
    - 0.5
```