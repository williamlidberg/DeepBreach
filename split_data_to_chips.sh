#!/bin/bash 
echo "split training tiles"
python /workspace/code/tools/split_vrt_to_chips.py /workspace/data/concatenated/training /workspace/data/chips/rgb_downslope_256/training/images/img --tile_size=256
python /workspace/code/tools/split_vrt_to_chips.py /workspace/data/masks/training_mask_tiles /workspace/data/chips/rgb_downslope_256/training/labels/img --tile_size=256
echo "Remove unlabaled chips"
python /workspace/code/tools/remove_empty_chips.py /workspace/data/chips/rgb_downslope_256/training/images/img /workspace/data/chips/rgb_downslope_256/training/labels/img

echo "split testing tiles"
python /workspace/code/tools/split_vrt_to_chips.py /workspace/data/concatenated/testing /workspace/data/chips/rgb_downslope_256/testing/images/img --tile_size=256
python /workspace/code/tools/split_vrt_to_chips.py /workspace/data/masks/testing_mask_tiles /workspace/data/chips/rgb_downslope_256/testing/labels/img --tile_size=256
echo "Remove unlabaled chips"
python /workspace/code/tools/remove_empty_chips.py /workspace/data/chips/rgb_downslope_256/testing/images/img /workspace/data/chips/rgb_downslope_256/testing/labels/img

