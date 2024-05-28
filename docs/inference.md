For inference on a folder of patches or WSIs the configuration files must be extended. We recommend extending the base configuration used to train the model you're planning to use.

## Folder of patches
For inference on a folder of patches, you only need to include the following configuration:
```bash
inference:
  input_path: /path/to/input/folder
  output_path: /path/to/output/folder
```
Where both ```input_path``` and ```output_path``` must be paths to existing directories.

## Whole Slide Image
Firstly, you must specify the slide config, which includes the path to the file (which must be compatible with OpenSlide), the patch size for inference, the stride between patches, as well as the output directory and name. 

```bash
inference:
  input_path: /path/to/input/file
  output_path: /path/to/output/file
  patch:
    size: 1024
    stride: 968
```
Where ```input_path``` is the path to a OpenSlide compatible file and ```output_path``` must be a JSON file. 

### Pre-processing
You have two options for the post-processing, either submitting the list of top-left coordinates of the patches to be processed (level 0) via ```inference.preprocessing.coords```:

```bash
inference:
  coords: /path/to/coords/json/file
```
In this case, note that ```inference.patch.stride``` will be ignored. Or, alternatively, you can provide the parameters to run our own pre-processing pipeline. We have developed a pipeline that:
1. Downsamples the WSI by a factor of ```inference.preprocessing.downsample```.
2. Converts the thumbnail to the Hematoxylin, Eosin and DAB (HED) space.
3. Computes a tissue binary mask by thresholding the H, E and DAB space with ```inference.preprocessing.h_tresh```, ```inference.preprocessing.e_tresh``` and ```inference.preprocessing.d_tresh```, respectively.
4. Applies morphological operations to remove holes and small objects, with a disk size of ```inference.preprocessing.disk_size```
5. List all patches containing tissue information with a stride of ```inference.patch.stride```.

### Processing
The processing of the slide (or its tiles) is done by the model itself. You just need to make sure that your model config includes the ```model.window.size``` and ```model.window.stride``` parameters. Additionally, you have to include the parameters of the inference loader in the config loader:

```bash
loader:
  infer:
    batch_size: 1
    num_workers: 8
    drop_last: False
```

### Postprocessing
The postprocessing basically consists on combining the predictions of the different tiles. By default, we only keep those detections with a confidence above ```wsi.postprocessing.threshold```.