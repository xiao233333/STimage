#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python register.py <he_image_path> <dapi_image_path>")
    sys.exit(1)
he_image_path = sys.argv[1]
dapi_image_path = sys.argv[2]
he_image = cv2.imread(he_image_path, cv2.IMREAD_GRAYSCALE)
dapi_image = cv2.imread(dapi_image_path, cv2.IMREAD_GRAYSCALE)
if he_image is None:
    print(f"Failed to load H&E image: {he_image_path}")
    sys.exit(1)
if dapi_image is None:
    print(f"Failed to load DAPI image: {dapi_image_path}")
    sys.exit(1)
sift = cv2.SIFT_create()
keypoints_he, descriptors_he = sift.detectAndCompute(he_image, None)
keypoints_dapi, descriptors_dapi = sift.detectAndCompute(dapi_image, None)
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors_he, descriptors_dapi, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
src_pts = np.float32([keypoints_he[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_dapi[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
csv_output = '\n'.join([','.join(map(str, row)) for row in M])
print(csv_output)
