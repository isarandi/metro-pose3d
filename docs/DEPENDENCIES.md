## Dependencies

***Note: All instructions were tested on Ubuntu 18.04.3.***

Anaconda is the easiest way to install the dependencies. If you don't have it installed yet, open a Bash shell and install Miniconda as follows: 

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
```

Create a new environment and install the dependencies:

```bash
conda create --name metro-pose3d python=3.6 Cython matplotlib pillow imageio ffmpeg scikit-image scikit-learn tqdm numba
conda activate metro-pose3d
conda install opencv3 -c menpo
pip install tensorflow-gpu==1.13.1 attrdict jpeg4py transforms3d more_itertools spacepy
```
### COCO tools

Install the [COCO tools](https://github.com/pdollar/coco) (used for managing runlength-encoded masks):

```
git clone https://github.com/cocodataset/cocoapi
pushd coco/PythonAPI
make
python setup.py install
popd
rm -rf cocoapi
```

### CDF
If you also want to train the model, you'll need to install the CDF library because 
Human3.6M supplies the annotations as cdf files. We read them using the [SpacePy](https://spacepy.github.io/) Python library,
which in turn depends on the CDF library.

```bash
wget https://spdf.sci.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_0/linux/cdf37_1-dist-cdf.tar.gz
tar xf cdf37_1-dist-cdf.tar.gz
rm cdf37_1-dist-cdf.tar.gz
cd cdf37_1-dist
make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all
```

If you have sudo rights, simply run `sudo make install`. If you have no `sudo` rights, make sure to add the
 `cdf37_1-dist/src/lib` to the `LD_LIBRARY_PATH` environment variable (add to ~/.bashrc for permanent effect), or use GNU Stow.
 
 ### libjpeg-turbo
 
 Install libjpeg-turbo to make JPEG loading faster. TODO