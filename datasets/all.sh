#!/usr/bin/env bash
set -euo pipefail

ask() {
  while true; do
    read -rp "$1" yn
    case $yn in
    [Yy]*)
      echo y
      break
      ;;
    [Nn]*) echo n ;;
    *) echo "Please answer yes or no." ;;
    esac
  done
}

if [[ $(ask "The Human3.6M, MPII, Pascal VOC, MPI-INF-3DHP and INRIA Holidays datasets are each from third parties.
 Have you read and agreed to their respective licenses? [y/n] ") != 'y' ]]; then
  echo "Then no cookies for you! Go read all the licenses!"
  exit 1
fi

# Find out the location of this script and cd into it
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

./get_h36m.sh
./extract_frames_and_boxes_h36m.py

./get_mpii.sh

./get_pascal_voc.sh

./get_mpi_inf_3dhp.sh
./extract_frames_and_masks_3dhp.py

./setup_darknet.sh
darknet/run_yolo.sh --image-root "$DATA_ROOT/3dhp" --out-path "$DATA_ROOT/3dhp/yolov3_person_detections.pkl" --jobs 3 --hflip

./get_inria_holidays.sh
./prepare_images_inria_holidays.py
darknet/run_yolo.sh --image-root "$DATA_ROOT/inria_holidays/jpg_small" --out-path "$DATA_ROOT/inria_holidays.pkl" --jobs 3 --hflip
./find_nonperson_images_inria_holidays.py
