#!/usr/bin/env bash
set -euo pipefail

sudo apt install build-essential --yes wget curl gfortran git ncurses-dev unzip tar

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

conda create --yes --name metro-pose3d python=3.6 Cython matplotlib pillow imageio ffmpeg scikit-image scikit-learn tqdm numba
conda activate metro-pose3d
conda install --yes opencv3 -c menpo
pip install tensorflow-gpu==1.13.1 attrdict jpeg4py transforms3d more_itertools spacepy

git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI
make
python setup.py install
cd ../..
rm -rf cocoapi

wget https://spdf.sci.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_1/linux/cdf37_1-dist-cdf.tar.gz
tar xf cdf37_1-dist-cdf.tar.gz
rm cdf37_1-dist-cdf.tar.gz
cd cdf37_1-dist
make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all

export LD_LIBRARY_PATH=$PWD/src/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Optional:
# wget https://sourceforge.net/projects/libjpeg-turbo/files/2.0.4/libjpeg-turbo-2.0.4.tar.gz