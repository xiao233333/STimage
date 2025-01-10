#!/usr/bin/python3
# coding: utf-8

import sys
import argparse
import cv2
import os

import squidpy as sq
import pandas as pd
import spatialdata as sd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import scanpy as sc

def impute_labels(adata, labels, method="max"):
    """
    Impute missing labels in a `anndata` object using partial labels.
    
    method: (str or list): The imputation method to use. Supported methods: ["average", "max"].
        - "average": Impute missing labels by taking the average of neighboring cell labels.
        - "max": Impute missing labels by assigning the most common label among neighboring cells.    
    """
    # Join labels
    labels['predicted.id'] = labels['predicted.id'].astype('category')

    assert set(labels.index) & set(adata.obs.index), "Indexes of labels and adata do not match"
    adata.obs = adata.obs.join(labels)
    unlbl_idx = np.where(adata.obs['predicted.id'].isnull())[0]
    
    # Find neighbors
    if not 'spatial_neighbors' in adata.uns:
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True) # delaunay=False may get more neighbors
    
    # Infer labels from neighboring cells, ensuring that neighbors have at least 1 label
    # Have not used k (num neighboring cells) as the delaunay method results in variable neighbors
    if not isinstance(method, list):
        method = [method]
    
    if "average" in method:
        assert 'prediction' in str(labels.columns.tolist()), "adata.obs does not contain prediction scores"
        categories = adata.obs.columns[adata.obs.columns.str.startswith('prediction')].drop('prediction.score.max').tolist()
    
    if "max" in method:
        assert 'predicted.id' in labels.columns.tolist(), "adata.obs does not contain key `predicted.id`"
        if not "average" in method:
            categories = ['predicted.id']

    labels_imputed = labels.copy()
    for cell_idx in tqdm(unlbl_idx):
        cell_id = adata.obs.index[cell_idx]
        _, idx = adata.obsp["spatial_connectivities"][cell_idx, :].nonzero()
        neighbor_scores = adata[idx, :].obs.loc[:, categories].dropna()
        neighbor_labels = adata[idx, :].obs.dropna()['predicted.id']

        # TODO: get levi to rerun without filtering
        # Reason: can not assign a filtered table to sdata object
        # - don't think it can subset the shapes
        # if neighbor_labels.empty:
        #     imputed_label = np.nan  # What to do with this? Maybe filter out sdata.table before generating mask
        # else:
        if "average" in method:
            if neighbor_scores.empty:
                imputed_label = np.full(len(categories), 1/len(categories))
            else:
                imputed_label = neighbor_scores.mean().values
            adata.obs.loc[cell_id, categories] = imputed_label
            labels_imputed.loc[cell_id, categories] = imputed_label
        if "max" in method:
            if neighbor_labels.empty:
                import random
                imputed_label = random.choice(labels['predicted.id'].cat.categories.tolist())
            else:
                imputed_label = neighbor_labels.mode().iloc[0]
            adata.obs.loc[cell_id, 'predicted.id'] = imputed_label
            labels_imputed.loc[cell_id, 'predicted.id'] = imputed_label
    
    # labels_imputed.to_csv("complete_labels.csv", index=False)
    
    return adata, labels_imputed
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xenium", type=str, default=None, help="Path to xenium data directory (doesn't support zarr)"
    )
    parser.add_argument(
        "--annotated", type=str, default=None, help="Path to annotated anndata (if separate)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="annotated_labels/tmp", help="Output directory to save files"
    )
    parser.add_argument(
        "--method", type=str, default="max,average", help="Output directory to save files"
    )   

    args = parser.parse_args()
    print(args)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We read adata instead of sdata, as the labels come with string indexes, not integer which
    # is default for sdata
    adata = sc.read_10x_h5(
    filename=Path(f"{args.xenium}")/"cell_feature_matrix.h5"
)

    df = pd.read_csv(
        Path(f"{args.xenium}")/"cells.csv.gz"
    )
    
    df.set_index(adata.obs_names, inplace=True)
    adata.obs = df.copy()
    
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    labels = pd.read_csv(args.annotated, index_col=0)
    if labels.index.dtype == 'int64':
        labels.index = labels.index.astype(str)
    assert adata.obs.index.dtype == 'O', "Adata should have string index"
    
    method = args.method.split(",")

    print(labels.head())
    print(adata.obs.head())
    adata_new, labels_imputed = impute_labels(adata, labels, method=method)

    adata_new.write(out_dir/"imputed_annotated.h5ad")
    labels_imputed.to_csv(out_dir/"imputed_labels.csv")

    print("Finished imputation")
