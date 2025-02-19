# Cell classification with single-cell spatial transcriptomics (Xenium) data and HoVer-Net

## Usage

The following data are required for each training sample (sequenced whole slide tissue):

- Xenium data
- High resolution H&E image

Additionally, a single-cell reference atlas is needed to generate the cell type labels.

### 0. Installation
The `environment.yml` contains the required dependencies for the conda environment.

Computing performance metrics relies on some code from https://github.com/meszlili96/PanNukeChallenge. This repository should be cloned into the same folder as this source.


### 1. Annotate Xenium data
Cell types are assigned to the Xenium data based on a single-cell reference atlas (e.g for breast cancer, https://www.nature.com/articles/s41588-021-00911-1). This can be done in Seurat, using label transfer. The annotated cells should be provided as a csv file with `predicted.id` as the cell type column, indexed by the original Xenium cell indices. 

These labels are then processed and stored into an anndata object for the Xenium data. For this, the following script should be run with the paths to the relevant files:

```
python load_annotations.py --xenium XENIUM_DIR --annotated ANNOTATED_PATH --out-dir OUT_DIR

```

Alternatively, the labels can be provided in the form of a geopandas dataframe. See `generate_masks.py` for details.


### 2. Register H&E image
The H&E image must be registered to the Xenium DAPI data, which can be done with SIFT or similar methods. An affine transformation matrix should be saved as a csv file, with the standard 2x3 format. Example:

```
-0.002511365031637, -0.776020225795623, 27762.2568957
0.775820476630406, -0.00237876719423, -1158.81828889
```


### 3. Generate training data
The following command generates the training data (cell masks and H&E tiles) given the Xenium data and H&E image. 

```
python generate_masks.py \
    --xenium XENIUM_DIR \
    --tif-path TIF_PATH \
    --out-dir OUT_DIR \
    --annotated ANNOTATED_PATH \
    --transform TRANSFORM_PATH \
    --num-categories N_CLASSES
```

Here, `ANNOTATED_PATH` should be the path to the `.h5ad` file generated from `load_annotations.py`; `TRANSFORM_PATH` is the path to the affine transformation matrix.

For more custom registration workflows (e.g. non-linear registration), one can provide the registered segmentation masks as a geopandas dataframe, as described above.


### 4. Preprocess training data

Preprocessing and filtering the tiles is done with a separate command:

```
python preprocess_images.py --data TILE_DIR --nuclei-threshold N
```

where `TILE_DIR` is the `OUT_DIR` from `generate_masks.py`.


### 5. Train model
Multiple training samples can be added to the training data by comma separating the paths.

```
python train_hovernet.py \
    --dataset DATASET_PATH_1,DATASET_PATH_2 \
    --out-dir OUT_DIR \
    --n-classes N_CLASSES \
    --split 0.85,0.15,0.00 \
    --batch-size 8 \
```

The trained model weights are saved in `OUT_DIR`.

### 6. Evaluate model
To generate predictions on the test set, run

```
python test_hovernet.py --dataset DATASET_PATH --path MODEL_PATH  --out-dir RESULTS_DIR --n-classes N_CLASSES
```

To generate the performance metrics, run

```
python compute_acc.py --pred_dir RESULTS_DIR/centroids_pred/ --true_dir RESULTS_DIR/centroids_truth/ --mode instance
```

for the instance segmentation metrics, and 

```
python compute_acc.py --pred_dir RESULTS_DIR/centroids_pred/ --true_dir RESULTS_DIR/centroids_truth/ --mode type
```

for the classification metrics.


## Credits
We use the implementation of HoVer-Net by [PathML](https://pathml.readthedocs.io/en/latest/) (GNU GPL 2.0). Some code for computing performance metrics used here is due to https://github.com/meszlili96/PanNukeChallenge.
