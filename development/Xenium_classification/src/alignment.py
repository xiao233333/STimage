#!/usr/bin/python3
# coding: utf-8

import os as os
import argparse as ag
import warnings as w
import skimage.io as ski
import spatialdata_io as sd_io

def main(args):
    out_dir = args.out_dir
    transform_matrix = args.transform
    tif_path = args.tif_path
    for file in [transform_matrix, tif_path]:
        if not os.path.exists(file):
            w.warn(f'File doesn\'t exist: {file}')
            exit(1)
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        w.warn(f"Created output directory: {out_dir}")

    # SpatialData IO to align image https://spatialdata.scverse.org/projects/io/en/latest/generated/spatialdata_io.xenium_aligned_image.html#spatialdata_io.xenium_aligned_image        
    output = sd_io.xenium_aligned_image(tif_path, transform_matrix)
    ski.imsave(f'{out_dir}/output_image.tif', output)


if __name__ == "__main__":
    parser = ag.ArgumentParser()
    parser.add_argument(
        "--transform", type=str, required=True, default=None, help="Path to transformation matrix (csv)"
    )
    parser.add_argument(
        "--tif-path", type=str, required=True, default=None, help="Path of TIF image to register"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, default=None, help="Output directory"
    )

    args = parser.parse_args()
    print(args)
    main(args)
