# Pose Utils

Collection of useful tools for cross dataset analysis and plotting for human poses. Supports only 14 and 16 joints for now. This is clearly made to facilitate my own research work. So, updates will be made when I deem it necessary.

Package can be found here: [pip](https://pypi.org/project/poseutils/)

## Constants

Constants can be pulled from the ```poseutils.constants```. ```EDGES_XX, LEFTS_XX, RIGHTS_XX, NAMES_XX, EDGE_NAMES_XXJNTS``` contain all the necessary information for arranging poses to use this package (XX=14 or 16). Contains supporting function ```adjacency_list``` and ```dataset_indices```

***In a perfect world, it would be good to modify this to your liking but when you are working with 5+ dataset, consistency is the last thing you want to waste time on.***

## Metrics

General metrics will be included for calculating errors. For now, only JPE is included.

## Props

Contains methods for calculating some properties of the pose or dataset. Included -

1. Limb length
2. Camera angle calcuation
3. Joint angle conversion
4. Body centered axis

## View

Contains methods for plotting and viewing different stats of the pose. Included - 

1. Pose (2d or 3d, 14 or 16)
2. Axis

## Transform

Transformation or normalization applied on the pose. Included -

1. Torso normalization
2. Skeletal normalization

## Common

Contains supporting methods for any task in general.