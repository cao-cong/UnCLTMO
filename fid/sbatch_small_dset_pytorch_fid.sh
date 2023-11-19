#!/bin/bash

path_real="/Users/yaelvinker/PycharmProjects/lab/fid/ldr"
path_fake="/Users/yaelvinker/PycharmProjects/lab/fid/fake_jpg"
number_of_images=1000
batch=20
#sbatch --mem=4000m -c2 --gres=gpu:2 --time=2-0 small_dset_run_fid.sh \
#  $path_real $path_fake $number_of_images $batch
sh small_dset_run_fid.sh \
  $path_real $path_fake $number_of_images $batch