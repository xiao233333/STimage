from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import _read_counts
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

# from https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/visium.py#L141
def load_registered_image(tif_path):
    transform_original = Identity()
    full_image = imread(tif_path).squeeze() #.transpose(2, 0, 1)
    # Check
    if full_image.shape[-1] == 3:
        full_image = full_image.transpose(2, 0, 1)
    full_image = DataArray(full_image, dims=("c", "y", "x"), name="TIF image")
    full_image_parsed = Image2DModel.parse(
        # drop alpha channel
        full_image.sel({"c": slice(0, 3)}),
        scale_factors=[2, 2, 2, 2],
        transformations={"global": transform_original},
    )
    return full_image_parsed

import cv2
import pandas as pd
import torch
from torch import Tensor
import math
import matplotlib.pyplot as plt

import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
import multiscale_spatial_image as msi
import spatialdata_plot
from spatialdata.datasets import raccoon
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
from skimage.segmentation import expand_labels
from tqdm import tqdm

def mask_for_polygons(polygons, im_size, vals):
    """Convert a polygon or multipolygon list to
       an image mask ndarray"""
    if not isinstance(vals, (list, tuple, np.ndarray)):
        vals = np.ones_like(polygons)
    img_mask = np.zeros(im_size, np.float64)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) if poly.geom_type == 'Polygon' 
             else int_coords(poly.convex_hull.exterior.coords)
             for poly in polygons]
    interiors = [poly.interiors if poly.geom_type == 'Polygon'
                 else poly.convex_hull.interiors
                 for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in interiors for pi in poly] # interiors should be [] anyway
    for i in range(len(exteriors)):
        cv2.fillPoly(img_mask, [exteriors[i]], vals[i])
    for i in range(len(interiors)):
        cv2.fillPoly(img_mask, [interiors[i]], 0)
    return img_mask

def sdata_load_img_mask2(sdata, shapes, img_key='he',
                  label_key='celltype_major', expand_px=0, return_sep=False):
    # For geopandas df that contains geometry and class columns
    shapes.index = shapes.index.astype(int)
    shapes_df = shapes
    shapes_df['label'] = shapes_df[label_key].cat.codes
    # Following lbl dicts are unused for now
    int2lbl = dict(enumerate(shapes_df[label_key].cat.categories))
    lbl2int = dict(zip(int2lbl.values(), int2lbl.keys()))
    shapes_df_dict = {k: v for k, v in shapes_df.groupby(label_key)}                     
    img = sdata.images[img_key]
    if isinstance(img, msi.multiscale_spatial_image.MultiscaleSpatialImage):
        # Note that this clears any transformation attribute
        img = Image2DModel.parse(img["scale0"].ds.to_array().squeeze(axis=0))
    img = img.values # NOTE: this errors on int16 dtypes
    masks = [mask_for_polygons(v['geometry'].tolist(),
                               img.shape[-2:],
                               vals=(v.index.to_numpy() + 1).tolist()) # Add 1 here in case val is 0 (background)
             # https://stackoverflow.com/questions/60484383/typeerror-scalar-value-for-argument-color-is-not-numeric-when-using-opencv
            for k, v in shapes_df_dict.items()]
    if expand_px:
        masks = [expand_labels(mask, distance=expand_px) for mask in masks]
    masks = np.stack(masks)    
    # Add the background mask
    mask_bg = (np.sum(masks, axis=0) == 0)*1.
    mask = np.concatenate((masks, np.expand_dims(mask_bg, axis=0)))
    img_mask = np.concatenate((img, mask))
    if return_sep:
        return img, mask
    else:
        return img_mask



def sdata_load_img_mask(sdata, affineT=None,
                  img_key='morphology_focus', shape_key='nucleus_boundaries',
                  label_key='celltype_major', expand_px=0, return_sep=False):
    # If using bbox, transform shapes then translate back to origin
    t_shapes = Sequence([get_transformation(sdata.shapes[shape_key]),
                         get_transformation(sdata.images[img_key]).inverse()])
                         
    # set_transformation(sdata.shapes[shape_key], t_shapes)
    # spatialdata 0.1.2 version
    #shapes = transform(sdata.shapes[shape_key], to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)
    # spatialdata 0.0.9 version
    shapes = transform(sdata.shapes[shape_key], t_shapes)
    # Affine transform
    if affineT:
        # spatialdata 0.1.2 version
        #set_transformation(shapes, affineT)
        #shapes = transform(shapes, to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)
        # spatialdata 0.0.9 version
        shapes = transform(shapes, affineT)
    # spatialdata 0.0.9 version
    shapes.index = shapes.index.astype(int)
    # NOTE: Bounding box query resets (removes) categories for the adata table
    # so, can use the original labels instead
    # labels = labels.loc[sdata.table.obs.index][label_key].to_frame()
    # NOTE: this has been taken care of outside this function in generate_masks
    labels = sdata.table.obs[label_key].to_frame()
    labels.index = labels.index.astype(int)
    shapes_df = shapes.merge(labels, how = 'inner', right_index = True, left_index = True)
    shapes_df['label'] = shapes_df[label_key].cat.codes
    # Following lbl dicts are unused for now
    # int2lbl = dict(enumerate(shapes_df[label_key].cat.categories))
    # lbl2int = dict(zip(int2lbl.values(), int2lbl.keys()))
    shapes_df_dict = {k: v for k, v in shapes_df.groupby(label_key)}                     
    img = sdata.images[img_key]
    if isinstance(img, msi.multiscale_spatial_image.MultiscaleSpatialImage):
        # Note that this clears any transformation attribute
        img = Image2DModel.parse(img["scale0"].ds.to_array().squeeze(axis=0))
    img = img.values # NOTE: this errors on int16 dtypes    
    masks = [
        mask_for_polygons(
            v['geometry'].tolist(),
            img.shape[-2:],
            # Add 1 here in case val is 0 (background)
            vals=(v.index.to_numpy() + 1).tolist()
        )
        # https://stackoverflow.com/questions/60484383/typeerror-scalar-value-for-argument-color-is-not-numeric-when-using-opencv
        for k, v in shapes_df_dict.items()
    ]
    if expand_px:
        masks = [expand_labels(mask, distance=expand_px) for mask in masks]
    # Ideally expand_labels should be run on the merged mask, but the loop
    # is too slow.
    ######################################################################
    # # Remove overlaps
    # merged_mask = masks[0].copy()
    # for mask in masks[1:]:
    #     overlapping = (mask != 0) & (merged_mask != 0)  # Find overlapping regions
    #     merged_mask = np.where(overlapping > 0, mask, merged_mask + mask)
    # # Split back to channels
    # value_to_channel = {}

    # for channel_idx in range(len(masks)):
    #     channel_values = np.unique(masks[channel_idx])
    #     for value in channel_values:
    #         value_to_channel[value] = channel_idx
    # if expand_px:
    #     exp_mask = expand_labels(merged_mask, distance=expand_px)
    # else:
    #     # total_mask = np.sum(masks, axis=0)
    #     exp_mask = merged_mask
    # new_mask = np.zeros_like(np.stack(masks))

    # print("Re-assigning channels")
    # for value in tqdm(np.unique(exp_mask)):
    #     channel_idx = value_to_channel[value]
    #     new_mask[channel_idx] += exp_mask * (exp_mask == value)
          
    # masks = new_mask
    ######################################################################
    masks = np.stack(masks)    
    # Add the background mask
    mask_bg = (np.sum(masks, axis=0) == 0)*1.
    mask = np.concatenate((masks, np.expand_dims(mask_bg, axis=0)))
    img_mask = np.concatenate((img, mask))
    if return_sep:
        return img, mask
    else:
        return img_mask

import os
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
from loguru import logger

from pathml.datasets.base_data_module import BaseDataModule
from pathml.datasets.utils import pannuke_multiclass_mask_to_nucleus_mask
from pathml.ml.hovernet import compute_hv_map
from pathml.utils import download_from_url

def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    # assert (
    #     multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6
    # ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # assert (
    #     multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256
    # ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis=0)
    return out
    
def print_mem():
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())
    print(torch.cuda.memory_allocated()/(1024 * 1024 * 1024))
    print(torch.cuda.memory_reserved()/(1024 * 1024 * 1024))

