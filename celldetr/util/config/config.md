# Config

This folder contains the configuration utilities for the project.

- The ```AttrDict``` replicates a dict that can get and set items as attributes.
- The ```ConfigDict``` extends ```AttrDict``` and can be initialized from yaml configuration files.

There are two main keywords that have different functionalities:


1. Base files

```yaml
__base__:
    - /path/to/another/config/file
```
At the beginning of a config file, the ```__base__``` keyworkd can be included, and it must contain a list of paths to other configs files. When initializing a ```ConfigDict```containing this keyword, the elements in ```__base__``` will be iteratively loaded at the beginning, and then the content will be updated/override with the content of the main file itself. This is similar to class inheritance.

2. Extension files
The extension files are similar to the base files, but while the base files are intended to extend whole configuration files, extension files' purpose is to reuse small configuration pieces accross experiments.

Assume we have a yaml file: ```pannuke.yaml``` file with the following content:
```yaml
    name: pannuke
    root: /path/to/data/
    num_classes: 5
```

Then, from another configuration file, we can retrieve this data.
```yaml
dataset:
    __file__: /path/to/pannuke.yaml
    folds:
        train: 1
        val: 2
        test: 3
```

The resulting config dict will have the following structure:

```python
{
    'dataset': {
        'name': 'pannuke'
        'root': '/path/to/data/'
        'num_classes': 5
        'folds': {
            'train': 1
            'val': 2
            'test': 3
        }
    }
}
```

If you want to override some variables from the original file, just place them below the __file__:
```yaml
dataset:
    __file__: /path/to/pannuke.yaml
    num_classes: 1
    folds:
        train: 1
        val: 2
        test: 3
```

Additionally, the ConfigDict can also be initialized from a string of options (i.e. from the options sent to command line).


