#!/usr/bin/python3
# coding: utf-8

import spatialdata_io
import squidpy as sq
import spatialdata as sd
import spatialdata_plot
from spatialdata import SpatialData
import numpy as np
from matplotlib import pyplot as plt

import sys
import argparse
import cv2
import pandas as pd
import math
import time
import os
import warnings
import gzip

from spatialdata.models import Image2DModel
import multiscale_spatial_image as msi
from spatialdata.transformations import (
    Affine,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    get_transformation_between_coordinate_systems,
    set_transformation,
)
from spatialdata import transform
from squidpy.im import ImageContainer
from scanpy import read_h5ad
from PIL import Image
from xenium_utils import load_registered_image, sdata_load_img_mask, sdata_load_img_mask2
from tqdm import tqdm
import geopandas as gpd
import pickle

def main(args):
    # Create output directory
    out_dir = args.out_dir
    img_dir = f"{out_dir}/images"
    mask_dir = f"{out_dir}/masks"
    for dir in [out_dir, img_dir, mask_dir]:
        if not os.path.exists(dir):
            warnings.warn(f'Creating output directory {dir}')
            os.makedirs(dir)

    # Load data
    warnings.warn("Loading data...")
    if "zarr" in args.xenium:
        warnings.warn("Reading zarr for e.g. rep1 is deprecated!")
        sdata = sd.read_zarr(args.xenium)
    else:
        sdata = spatialdata_io.xenium(args.xenium,
                                      n_jobs=8,
                                      cells_as_shapes=True,
                                          )
    tif_path = args.tif_path
    full_image_parsed = load_registered_image(tif_path)
    if args.annotated:
        adata_annotated = read_h5ad(args.annotated)
        # Need reset_index here because sometimes only NaN's are assigned
        assert sdata.table.obs.shape[0] == adata_annotated.obs.shape[0]
        assert args.annotated_key in adata_annotated.obs.columns
        sdata.table.obs[["celltype_major"]] = adata_annotated.obs.reset_index()[
            [args.annotated_key]
        ]
    elif not args.shapes:
        assert "celltype_major" in sdata.table.obs, "`celltype_major` not found in xenium data" 
        
        sdata.table.obs['celltype_major'] = sdata.table.obs['celltype_major'].astype('category')
        assert len(sdata.table.obs['celltype_major'].cat.categories) == args.num_categories
        # Drop NA values
        # FIXME: See comments in load_annotations
        # We should not have any NA values
        # sdata.table.obs = sdata.table.obs.dropna()
        # sdata.table.write(f"{out_dir}/adata_filtered.h5ad")
        print(sdata.table.obs['celltype_major'].head(15))
        assert sdata.table.obs['celltype_major'].isna().sum() == 0

    if args.shapes:
        with open(args.shapes, 'rb') as f:
            shapes = pickle.load(f)
        # sdata.shapes["nucleus_boundaries"] = shapes
    
    merged = SpatialData(
        images={
            "he": full_image_parsed,
        },
        shapes={
            "cell_circles": sdata.shapes["cell_circles"], # Required for bbox queries otherwise adata table disappears
            "cell_boundaries": sdata.shapes["cell_boundaries"],
            "nucleus_boundaries": sdata.shapes["nucleus_boundaries"],
        },
        table=sdata.table,
    )

    # Load transformation matrix
    if args.transform:
        A = pd.read_csv(args.transform,header=None).to_numpy()
        if A.shape[0] == 2:
            A = np.append(A, [[0,0,1]], axis=0)
        affineT = Affine(A,
                          input_axes=("x", "y"),
                          output_axes=("x", "y"))
    else:
        affineT = None
    
    # Generate masks
    height, width = full_image_parsed['scale0']['image'].shape[-2:]

    if args.split:
        sys.exit("This is not working right now")
        # Split up image into quadrants
        coords = [[[0, 0],[25600, 25600]],
                  [[25600, 0],[height, 25600]],
                  [[0, 25600],[25600, width]],
                  [[25600, 25600],[height, width]]]
    else:
        coords = [[[0, 0],[height, width]]]

    for j, (min_coordinate, max_coordinate) in enumerate(coords):
        # NOTE: Memory stills seems to blow up, the following block skips processing already completed splits
        if args.resume and args.split:
            if j < args.resume:
                continue

        if len(coords) > 1:
            merged_sub = merged.query.bounding_box(
                min_coordinate=min_coordinate,
                max_coordinate=max_coordinate,
                axes=["y", "x"],
                target_coordinate_system="global",
            )
        else:
            merged_sub = merged
        
        # This must be run after any bbox query otherwise some categories will be deleted 
        if not args.shapes:
            merged_sub.table.obs["celltype_major"] = sdata.table.obs.reset_index().iloc[merged_sub.table.obs.index]["celltype_major"].values

        warnings.warn(f"Generating masks for split {j}...")
        start = time.time()
        if args.shapes:
            img_mask = sdata_load_img_mask2(merged_sub, shapes, img_key='he', expand_px=args.expand_px)
        else:
            img_mask = sdata_load_img_mask(merged_sub, affineT=affineT, img_key='he', expand_px=args.expand_px)
        np.save(f"{out_dir}/img_mask.npy", img_mask)
        end = time.time()
        warnings.warn(f"Generated masks for split {j}, time elapsed: {end - start}")
        
        start = time.time()
        warnings.warn(f"Tiling split {j}...")
        
        imgc = ImageContainer(img_mask)
        gen = imgc.generate_equal_crops(size=256, as_array='image', squeeze=True)

        for i, tile in enumerate(tqdm(gen)):
            # Quick filter
            mask = np.moveaxis(tile[:,:,3:], 2, 0)
            image = Image.fromarray(tile[:,:,:3].astype(np.uint8))

            if args.no_save_bg and len(np.unique(mask)) < 3: 
                print(f"Skipped tile {j}_{i}")
            else:
                np.save(f"{mask_dir}/{j}_{i}.npy", mask)
                image.save(f"{img_dir}/{j}_{i}.png")
                print(f"Saved tile {j}_{i}")


        end = time.time()
        print(f"Finished tiling split {j}, time elapsed: {end - start}")
        
        del imgc
        del img_mask
        del merged_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xenium", type=str, default=None, help="Path to xenium data directory (can be zarr or raw)"
    )
    parser.add_argument(
        "--annotated", type=str, default=None, help="Path to annotated anndata (if separate)"
    )
    parser.add_argument(
        "--annotated-key", type=str, default='predicted.id', help="Key to read annotations from"
    )
    parser.add_argument(
        "--out-dir", type=str, default='./tile_data/', help="Output directory"
    )
    parser.add_argument(
        "--tif-path", type=str, default=None, help="Path to registered TIF image"
    )
    parser.add_argument(
        "--expand-px", type=int, default=0, help="Number of pixels to expand nucleus mask"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="Split large image to process separately",
    )
    parser.add_argument(
        "--no-save-bg",
        action="store_true",
        default=False,
        help="Do not save background tiles",
    )

    parser.add_argument(
        "--resume", type=int, default=None, help="Index to resume tile generation at for split data"
    )
    parser.add_argument(
        "--transform", type=str, default=None, help="Path to transformation matrix (csv)"
    )
    parser.add_argument(
        "--shapes", type=str, default=None, help="Path to custom shapes"
    )
    parser.add_argument(
        "--num-categories", type=int, default=9, help="Number of categories"
    )    

    args = parser.parse_args()
    print(args)
    main(args)
