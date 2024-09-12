For inference on a folder of patches or WSIs the configuration files must be extended. We recommend extending the base configuration used to train the model you're planning to use.

## Whole Slide Image
Firstly, you must specify the slide config, which includes the path to the input file (which must be compatible with OpenSlide), the patch size for inference, the stride between patches, the microns per pixel (mpp) as well as the output directory. 
```bash
inference:
  input_path: /path/to/input/file
  output_dir: /path/to/output/folder
  patch:
    size: 1024
    stride: 960
    mpp: 0.25
```
Where ```input_path``` is the path to a OpenSlide compatible file and ```output_path``` is a directory. The script will save a CSV file with the same name of the slide to ```output_path```.

### Pre-processing
You have two options for the pre-processing, either submitting the list of top-left coordinates of the patches to be processed (at level 0) via ```inference.preprocessing.coords```:

```bash
inference:
  preprocessing:
    coords: /path/to/coords/csv/file
```
The file must contain the top-left coordinates of the patches to be processed at level 0. In this case, note that ```inference.patch.stride``` will be ignored during the processing of the patches, but you need to provide the patch information (size, stride and mpp) for the post-processing. 

Alternatively, you can provide the parameters to run our own pre-processing pipeline. We have developed a pipeline that:
1. Downsamples the WSI by a factor of ```inference.preprocessing.downsample```.
2. Converts the thumbnail to the Hematoxylin, Eosin and DAB (HED) space.
3. Computes a tissue binary mask by thresholding the H, E and DAB space with ```inference.preprocessing.h_tresh```, ```inference.preprocessing.e_tresh``` and ```inference.preprocessing.d_tresh```, respectively.
4. Applies morphological operations to remove holes and small objects, with a disk size of ```inference.preprocessing.disk_size```
5. List all patches containing tissue information with a stride of ```inference.patch.stride```.

If no pre-processing information is provided or some parameters are missing, the default values are as follows:

```bash
inference:
  patch:
    size: 1024
    stride: 960
    mpp: 0.25
  preprocessing:
    downsample: 32
    h_thresh: 1.0
    e_thresh: 0.0
    d_thresh: 0.5
    disk_size: 20
```

### Processing
The processing of the slide (or its tiles) is done by the model itself. You just need to make sure that your model config includes the ```model.window.size``` and ```model.window.stride``` parameters. Additionally, you have to include the parameters of the inference loader in the config loader:

```bash
model:
  window:
    size: 256
    stride: 192
loader:
  infer:
    batch_size: 16
    num_workers: 8
```

### Postprocessing
The postprocessing basically consists on combining the predictions of the different tiles. By default, we only keep those detections with a confidence above ```inference.postprocessing.threshold```. Additionally, if you want the output to be in a qupath format, you can set ```inference.postprocessing.qupath``` to ```True```.

## Folder of patches
For inference on a folder of patches, you only need to include the following configuration:
```bash
inference:
  input_path: /path/to/input/folder
  output_dir: /path/to/output/folder
```
Where both ```input_path``` and ```output_path``` must be paths to existing directories. The files inside ```input_path``` must be image patches to which perform inference to.
