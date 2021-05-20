# Setup

## Requirements

### Pytorch and Cuda
We recommend pytorch 1.7.1 with cuda 11.0, while older versions of pytorch should work we have not tested and so cannot
confirm this.

#### Cuda
In order to install cuda 11.0 you can use the official cuda toolkit archive: https://developer.nvidia.com/cuda-toolkit-archive
You can find version 11.0 here: https://developer.nvidia.com/cuda-11.0-update1-download-archive , you can use the
deb(network) which for ubuntu 20.04 is just:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-0
```
You can install multiple cuda versions if you need another version for a different project, see:
https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae and
https://stackoverflow.com/questions/41330798/install-multiple-versions-of-cuda-and-cudnn

#### Pytorch
In order to install the cuda 11.0 version of pytorch 1.7.1 we can use the previous versions instructions: https://pytorch.org/get-started/previous-versions/
```
# Removed torchaudio since it's not needed for this project
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

Note that if the cuda version your pytorch installation was compiled with doesn't match your local cuda
version you might have issues building apex, so verify this (can check local cuda version with
`nvcc --version # try /usr/local/cuda/bin/nvcc if not found`)

### Python libraries
Install python requirements:

```
pip3 install -r requirements.txt
```

**NOTE** If your install is taking a long time with a bunch of messages about "pip is looking at multiple versions"
you can try adding `--use-deprecated=legacy-resolver` to the install command to use the old pip behavior, see:
https://stackoverflow.com/questions/65122957/resolving-new-pip-backtracking-runtime-issue .
If you get an error like: `AttributeError: type object 'Callable' has no attribute '_abc_registry'`
then try `pip uninstall typing` , see here: https://stackoverflow.com/questions/55833509/attributeerror-type-object-callable-has-no-attribute-abc-registry

Note that the exact versions of software used were relaxed to avoid unnecessary conflicts, but if you want to see the
exact versions we used to validate results you can check `requirements_exact.txt`.

### maskrcnn-benchmark
The project is built on maskrcnn-benchmark, which you can install following the instructions from the official repo
(see below if using pytorch >=1.5.0):
https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md

**NOTE**: If using pytorch >= 1.5.0 you will need to patch maskrcnn-benchmark in order for it to compile.
After cloning the repo change into the git root folder and run these commands in the terminal:
```
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
# You can then run the regular setup command
python3 setup.py build develop
```
Then you can proceed with the setup.py command as normal.

Afer setting up maskrcnn-benchmark and the python dependencies you should be able to run the project code.