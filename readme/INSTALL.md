# Setup

## Requirements

### Pytorch and Cuda
We recommend pytorch 1.7.1 with cuda 11.0, while older versions of pytorch should work we have not tested and so cannot
confirm this.

### Cuda
#### AWS Deep Learning AMI - Cuda already present
If you are using AWS then the latest AWS deep learning AMI already has cuda versions 10.0 to 11.1 installed, so we can
just set 11.0 as the cuda version to use by setting these variables in the terminal:
```bash
export CUDA_HOME="/usr/local/cuda-11.0"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/efa/lib:$LD_LIBRARY_PATH"
```
You can skip over to Python Environment
#### Cuda Installation
If you are not on the deep learning AMI then in order to install cuda 11.0 you can use the official cuda toolkit
archive: https://developer.nvidia.com/cuda-toolkit-archive . You can find version 11.0 here:
https://developer.nvidia.com/cuda-11.0-update1-download-archive , you can use the deb(network).

For **---- Ubuntu 20.04 ----** this is just:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```
For **---- Ubuntu 18.04 ----**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
```
Then for both run:
```bash
sudo apt-get update
sudo apt-get -y install cuda-11-0
# Try running nvidia-smi afterward, if it returns 'Failed to initialize NVML: Driver/library version mismatch'
# then reboot before proceeding
nvidia-smi
```
For other OSes please check nvidia's page for instructions: https://developer.nvidia.com/cuda-11.0-update1-download-archive

You can install multiple cuda versions if you need another version for a different project, see:
https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae and
https://stackoverflow.com/questions/41330798/install-multiple-versions-of-cuda-and-cudnn

### Python Environment
If you're using the AWS deep learning AMI we recommend using the `base` environment as it avoids some of the pip issues
listed below that can happen with the `pytorch_latest_p37` environment. You can clone the base environment for this:
```bash
conda create --name siammot --clone base
conda activate siammot
```
Otherwise we advise using a virtualenv with something like `python3 -m venv VENV_PATH && source VENV_PATH/bin/activate`

### Pytorch
In order to install the cuda 11.0 version of pytorch 1.7.1 we can use the previous versions instructions: https://pytorch.org/get-started/previous-versions/
```bash
# Removed torchaudio since it's not needed for this project
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

Note that if the cuda version your pytorch installation was compiled with doesn't match your local cuda
version you might have issues building apex, so verify this (can check local cuda version with
`nvcc --version # try /usr/local/cuda/bin/nvcc if not found`)

### Python libraries
Install python requirements:

```bash
pip3 install -r requirements.txt
```

**NOTE** If your install is taking a long time with a bunch of messages about "pip is looking at multiple versions"
you can try adding `--use-deprecated=legacy-resolver` to the install command to use the old pip behavior, see:
https://stackoverflow.com/questions/65122957/resolving-new-pip-backtracking-runtime-issue .
If you get an error like: `AttributeError: type object 'Callable' has no attribute '_abc_registry'`
then try `pip3 uninstall typing` , see here: https://stackoverflow.com/questions/55833509/attributeerror-type-object-callable-has-no-attribute-abc-registry

Note that the exact versions of software used were relaxed to avoid unnecessary conflicts, but if you want to see the
exact versions we used to validate results you can check `requirements_exact.txt`.

### maskrcnn-benchmark
The project is built on maskrcnn-benchmark, which you can install following the instructions from the official repo
(see below if using pytorch >=1.5.0):
https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md

**NOTE**: When building apex on a deep learning AMI remember to set the CUDA_HOME as mentioned above if you get an error
about mismatched pytorch and cuda versions

**NOTE**: If using pytorch >= 1.5.0 you will need to patch maskrcnn-benchmark in order for it to compile.
After cloning the repo change into the git root folder and run these commands in the terminal:
```bash
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
# You can then run the regular setup command
python3 setup.py build develop
```
Then you can proceed with the setup.py command as normal.

Afer setting up maskrcnn-benchmark and the python dependencies you should be able to run the project code.
Make sure though that you can run nvidia-smi before trying to run the project code and reboot if necessary.