# Data loaders

The configurations for the data loaders is simple. You just need to specify the loader configuration for each fold: ```train```, ```val```, ```test``` and optionally ```infer```. For example:

```yaml
loader:
  train:
    batch_size: 2
    num_workers: 8
    shuffle: True
    drop_last: True
  val:
    batch_size: 2
    num_workers: 8
    shuffle: False
    drop_last: False
  test:
    batch_size: 2
    num_workers: 8
    shuffle: False
    drop_last: False
```