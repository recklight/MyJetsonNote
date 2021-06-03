# MyJetsonNote
Jetson Series - TX2, Xavier AGX, Xavier NX
------
------

目錄
------
* [安装](#安装)
    * [NX](#NX)
    * [TX2](#TX2)
* [檢查安裝](#檢查安裝)  
* [修改默認python版本](#修改默認python版本)
* [安裝Tensorflow](#安裝Tensorflow) 
* [安裝多版本Tensorflow](#安裝多版本Tensorflow) 
* [Opencv3](#Opencv3) 
* [PyAudio](#PyAudio) 
* [Sounddevice](#Sounddevice) 
* [librosa0.6.3](#librosa0.6.3)
* [exFAT_driver](#exFAT_driver) 
* [JAVA](#JAVA) 
* [kolourpaint4](#kolourpaint4) 
* [PyQt5](#PyQt5) 
* [PyTorch](#PyTorch) 
* [CloneSDcard](#CloneSDcard) 

安裝
------
### NX
* [NVIDIA官網](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)
* 下載image, 燒至SD 卡

### TX2, AGX
**Host電腦端 (電腦端):** 須先準備一台linux系統的電腦/或是在虛擬機（所需硬碟空間>100GB）上安裝linux系統

1. 在電腦端下載並安裝[SDK Manager](https://developer.nvidia.com/embedded/jetpack-archive)

2. 硬體連接
- 使用盒內usb線連接電腦端與AGX（連接AGX電源燈旁邊的插口）, 將AGX連接上電腦螢幕.
- 進入工程模式, AGX接上電源並保持關機狀態下, 按住中間的按鍵（Force Recovery）不放開, 再按下左邊的電源（Power）不放開, 等待約兩秒後同時放開. 
- 在電腦端上輸入lsusb查看是否連線上AGX（出現Nvidia corp.）

3. 開啟SDK Manager, 登入NVIDIA 帳戶
![1](https://user-images.githubusercontent.com/53622566/82420244-0d868f80-9ab2-11ea-9524-b45e54fe9656.png)

4. 如下圖設定 **Jetson TX2** ,  **JetPack 4.3** , **DeepStream**,  點擊**CONTINUE**
- AGX 建議安裝 Jetpack 4.4
![2](https://user-images.githubusercontent.com/53622566/82420328-2abb5e00-9ab2-11ea-8238-c298972a7197.png)

5. 勾選左下**I accept ...**後點擊 **CONTINE**
![3](https://user-images.githubusercontent.com/53622566/82420414-46266900-9ab2-11ea-9ba9-ded4b2738953.png)

6. 跳出一個視窗要求輸入系統密碼, 輸入後以繼續安裝
![4](https://user-images.githubusercontent.com/53622566/82420458-52122b00-9ab2-11ea-9a25-64874768a69f.png)

7. 進入 **STEP 03**, 等待下載及安裝至出現出視窗如下, 點選**Manual Setup**並按照提示操作TX2進入Recovery(工程模式)後, 點擊**Flash**(將TX2連接上螢幕)
![6](https://user-images.githubusercontent.com/53622566/82420572-7a018e80-9ab2-11ea-9a16-fafd75bd70cd.png)

8. 等待出現視窗如下後, 移至**TX2端**, 可以看到系統登入畫面, 請設定系統名稱, 密碼等, 最後畫面停留至桌面
![7](https://user-images.githubusercontent.com/53622566/82420653-9b627a80-9ab2-11ea-819e-2e414d6ff317.png)

9. 回到**Host電腦端**輸入剛才在TX2的設定的使用者名稱與密碼, 點擊**Install**

10. 安裝結束, 點擊**Finish and exit**, 安裝完成
![8](https://user-images.githubusercontent.com/53622566/82421655-e204a480-9ab3-11ea-9e90-7b8e7b0c4e99.png)


檢查安裝（可省）
------
1. CUDA
在Tegra上nvidia-smi是沒有作用的, 直接使用指令查看CUDA版本
```Bash
sudo find / -name  cuda
```
或
```Bash
## set an environment variable for cuda
export PATH=$PATH:/usr/local/cuda/bin
nvcc -V
```
可以看到已經安裝CUDA 10.0 的版本
TX2, Jetpack 4.3, CUDA 10.0

![nvcc](https://user-images.githubusercontent.com/53622566/82421932-5fc8b000-9ab4-11ea-8b4b-8cd5ca6868b0.png)

NX, Jetpack 4.4, CUDA 10.2

![nvcc 4 4](https://user-images.githubusercontent.com/53622566/84370316-aa000580-ac0a-11ea-9051-002be8246910.png)

2. cuDNN
```Bash
sudo find / -name libcudnn*
```

3. TensorRT
```Bash
sudo find / -name tensorrt
```

4. TX2 開啟攝影機鏡頭
```Bash
nvgstcapture --prev-res=2
```

修改默認python版本
------
使用alternatives管理多版本軟體
```Bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 100
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 150
sudo update-alternatives --config python
```
根據提示選用預設版本, 再次運行python確定設定成功

### Install new versions of software
```Bash
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
```

### Dynamic interface where is showed the status of your NVIDIA Jetson (command:  jtop)
```Bash
sudo pip3 install jetson-stats
```


安裝Tensorflow
------
* [NVIDIA DOC.](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
* Note: As of the 20.02 TensorFlow release, the package name has changed from tensorflow-gpu to tensorflow.

1. Install system packages required by TensorFlow:
```Bash
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U pip testresources setuptools==49.6.0
sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 \
h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
```

2. Install TensorFlow using the pip3 command. 
* 注意安裝的 Jetpack 版本
* Check the [TENSORFLOW VERSION](https://developer.download.nvidia.com/compute/redist/jp/)

**TX2**
* tensorflow-gpu 可以安裝 v1.15.0 或 v2.0.0
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu==1.15.0+nv19.12
# sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu==2.0.0+nv20.1
```

**Xavier NX, AGX**
- JetPack 4.4.1
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'
```
or
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.2+nv20.6
```
- JetPack 4.5.1
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 'tensorflow<2'
```

3. Verifying The Installation
To verify that TensorFlow has been successfully installed,  you'll need to launch a Python prompt and import TensorFlow.
```Bash
python
import tensorflow
```

安裝多版本Tensorflow
------
1. Set up the Virtual Environment
First, install the virtualenv package and create a new Python 3 virtual environment:
```Bash
sudo apt-get install virtualenv
python3 -m virtualenv -p python3 <chosen_venv_name>
```

2. Activate the Virtual Environment
Next, activate the virtual environment:
```Bash
source <chosen_venv_name>/bin/activate
```

3. Install the desired version of TensorFlow and its dependencies:
```Bash
pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==$TF_VERSION+nv$NV_VERSION
```

4. Deactivate the Virtual Environment
```Bash
deactivate
```

5. Run a Specific Version of TensorFlow
After the virtual environment has been set up, simply activate it to have access to the specific version of TensorFlow. Make sure to deactivate the environment after use:
```Bash
source <chosen_venv_name>/bin/activate 
<Run the desired TensorFlow scripts>
deactivate
```

6. Upgrading TensorFlow
Note: Due to the recent package renaming, if the TensorFlow version you currently have installed is older than the 20.02 release, you must uninstall it before upgrading to avoid conflicts. See the section on Uninstalling TensorFlow for more information.
To upgrade to a more recent release of TensorFlow, if one is available, run the install command with the ‘upgrade’ flag:
```Bash
sudo pip3 install --upgrade --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$44 tensorflow
```



Opencv3
------
> [Reference1](https://jkjung-avt.github.io/opencv3-on-tx2/)
> [Reference2](https://www.jetsonhacks.com/2018/11/08/build-opencv-3-4-on-nvidia-jetson-agx-xavier-developer-kit/)

```Bash
sudo apt update
sudo apt dist-upgrade
sudo apt install --only-upgrade g++-5 cpp-5 gcc-5
sudo apt install build-essential make cmake cmake-curses-gui \
g++ libavformat-dev libavutil-dev \
libswscale-dev libv4l-dev libeigen3-dev \
libglew-dev libgtk2.0-dev libdc1394-22-dev libxine2-dev \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libjpeg8-dev libjpeg-turbo8-dev libtiff5-dev \
libavcodec-dev libxvidcore-dev libx264-dev libgtk-3-dev \
libatlas-base-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev qt5-default      
```

#### Install dependencies for python3
```Bash
sudo apt install python3-dev python3-pip python3-tk
sudo pip3 install numpy
sudo pip3 install matplotlib
```

#### Modify matplotlibrc (line #41) as 'backend: TkAgg'
```Bash
sudo vim /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
```

#### Before downloading and building opencv-3.4.0, I’d first do some modifications according to this post, in order to fix OpenGL related compilation problems . More specifically, I’d modify /usr/local/cuda/include/cuda_gl_interop.h and fix the symbolic link of libGL.so.
```Bash
sudo vim /usr/local/cuda/include/cuda_gl_interop.h
```
#### Here’s how the relevant lines (line #62~68) of cuda_gl_interop.h look like after the modification.
```Bash
//#if defined(__arm__) || defined(__aarch64__)
//#ifndef GL_VERSION
//#error Please include the appropriate gl headers before including cuda_gl_interop.h
//#endif
//#else
 #include <GL/gl.h>
//#endif
```

#### After Jetpack 4.2 and we don't have to do
```Bash
# cd /usr/lib/aarch64-linux-gnu/
## sudo ln -sf tegra/libGL.so libGL.so
# sudo ln -sf libGL.so.1.0.0 libGL.so
```

#### Next, download opencv-3.4.0 source code, cmake and compile. Note that opencv_contrib modules (cnn/dnn stuffs) would cause problem on pycaffe, so after some experiments I decided not to include those modules at all.
```Bash
mkdir -p ~/src && cd ~/src
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
unzip opencv-3.4.0.zip
```

#### Build opencv (CUDA_ARCH_BIN="6.2" for TX2, or "5.3" for TX1)
```Bash
cd ~/src/opencv-3.4.0
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_CUDA=ON -D CUDA_ARCH_BIN="5.3,6.2,7.2" -D CUDA_ARCH_PTX="" \
        -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
        -D ENABLE_NEON=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
        -D WITH_QT=ON -D WITH_OPENGL=ON ..

sudo make -j4
sudo make install
```

#### To verify the installation:
```Bash
ls /usr/local/lib/python3.6/dist-packages/cv2.*
/usr/local/lib/python3.6/dist-packages/cv2.cpython-35m-aarch64-linux-gnu.so
    
ls /usr/local/lib/python2.7/dist-packages/cv2.*
/use/local/lib/python2.7/dist-packages/cv2.so

python3 -c 'import cv2; print(cv2.__version__)'
python2 -c 'import cv2; print(cv2.__version__)'
```

##### TX2, Jetpack 4.3
* Before

![cv1](https://user-images.githubusercontent.com/53622566/84003020-f7c30680-a99b-11ea-8ed0-cb3fa51a9037.png)

* After

![cv2](https://user-images.githubusercontent.com/53622566/84003024-f8f43380-a99b-11ea-830e-ff9f7ce664ed.png)


##### Xavier NX, Jetpack 4.4
* Before

![nx1](https://user-images.githubusercontent.com/53622566/84370728-42968580-ac0b-11ea-8f93-e265768e748f.png)

* After

![nx2](https://user-images.githubusercontent.com/53622566/84370906-7e314f80-ac0b-11ea-9a39-c1ded95e81d8.png)



PyAudio
------
##### Method 1
```Bash
sudo apt install python-all-dev portaudio19-dev
sudo pip3 install pyaudio
```

##### Method2
```Bash
# Download pa_stable_v190600_20161030.tgz from http://www.portaudio.com/download.html
# Unzip

sudo apt install libasound-dev
./configure && make
sudo make install
sudo apt install python-pyaudio python3-pyaudio
```

Sounddevice
------
```Bash
sudo apt install libffi-dev
sudo pip install sounddevice
```

librosa0.6.3
------
> [How to install the Librosa library in Jetson Nano or aarch64 module](https://learninone209186366.wordpress.com/2019/07/24/how-to-install-the-librosa-library-in-jetson-nano-or-aarch64-module/)

> [numba/llvmlite](https://github.com/numba/llvmlite)

> [llvmlite documents](https://llvmlite.readthedocs.io/en/latest/)

#### llvmlite Compatibility 
llvmlite versions  |compatible LLVM versions
--------- | --------|
0.34.0 - ...       |10.0.x (9.0.x for  ``aarch64`` only)
0.33.0             |9.0.x
0.29.0 - 0.32.0    |7.0.x, 7.1.x, 8.0.x
0.27.0 - 0.28.0    |7.0.x
0.23.0 - 0.26.0    |6.0.x
0.21.0 - 0.22.0    |5.0.x
0.17.0 - 0.20.0    |4.0.x
0.16.0 - 0.17.0    |3.9.x
0.13.0 - 0.15.0    |3.8.x
0.9.0 - 0.12.1     |3.7.x
0.6.0 - 0.8.0      |3.6.x
0.1.0 - 0.5.1      |3.5.x

#### Upgrade the SETUP tools:
```Bash
sudo pip3 install -U setuptools
sudo pip3 install cython
```

#### Install LLVM & LLVMLITE:
```Bash
wget http://releases.llvm.org/7.0.1/llvm-7.0.1.src.tar.xz
tar -xvf llvm-7.0.1.src.tar.xz
cd llvm-7.0.1.src
mkdir llvm_build_dir && cd llvm_build_dir/
cmake ../ -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64"
sudo make -j4
sudo make install
```
```
cd bin/
echo "export LLVM_CONFIG=\""`pwd`"/llvm-config\"" >> ~/.bashrc
echo "alias llvm='"`pwd`"/llvm-lit'" >> ~/.bashrc
source ~/.bashrc
```
##### Finally
```Bash
sudo pip3 install llvmlite==0.32.1
```
#### Install Numba, Scipy, Joblib, Scikit-learn
```Bash
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo pip3 install numba==0.38.0
sudo pip3 install scipy==1.1.0 joblib==0.12 scikit-learn==0.21.1
```
#### Install Librosa
```Bash
sudo pip3 install librosa==0.6.3
```


exFAT_driver
------
```Bash
sudo add-apt-repository universe
sudo apt install exfat-fuse exfat-utils
```


JAVA
------
##### JRE
```Bash
sudo apt-get install default-jre
```
##### JDK
```Bash
sudo apt-get install default-jdk
```


kolourpaint4
------
```Bash
sudo apt install kolourpaint4
```

PyQt5
------
#### Method 1
```Bash
sudo apt-get install qt5-default
sudo apt install python3-pyqt5
sudo apt install pyqt5-dev-tools qttools5-dev-tools
sudo pip3 install pyqt5
```
* If build wheel error, try
```Bash
# sudo pip3 install sip==5.0.1
```

#### Method 2

##### If you have an old version of SIP
```Bash
sudo apt remove python3-sip
sudo apt autoremove
```
##### Install sip
> [What is SIP?](https://pypi.org/project/SIP/)

> [Download Sip source](https://riverbankcomputing.com/software/sip/download)
```Bash
python3 configure.py
make
sudo make install
```

##### Install pyqt5
> [PyQt Reference Guide](https://www.riverbankcomputing.com/static/Docs/PyQt5/installation.html)

> [Download PyQt5 source](https://riverbankcomputing.com/software/pyqt/download5)
```Bash
python3 configure.py
make
sudo make install
sudo pip3 install PyQt5
```
##### Install  ie multimedia
```Bash
sudo apt install python3-pyqt5.qtmultimedia
```


PyTorch
------
- [pytorch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048)
#### PyTorch 1.6.0
```Bash
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
sudo apt install python3-pip libopenblas-base libopenmpi-dev
sudo pip3 install Cython torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
```
#### Torchvision 0.7.0
```Bash
sudo apt install libjpeg-dev zlib1g-dev
git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision && sudo python setup.py install
```
#### PyTorch 1.7.0
```Bash
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo apt install python3-pip libopenblas-base libopenmpi-dev
sudo pip3 install Cython torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
```
#### Torchvision 0.8.1
```Bash
sudo apt install libjpeg-dev zlib1g-dev
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
cd torchvision && sudo python setup.py install
```
#### Verification
```Bash
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
```
```Bash
import torchvision
print(torchvision.version)
```

CloneSDcard
------
-[dd 指令](https://blog.gtwang.org/linux/dd-command-examples/)
##### find your SD card
```Bash
sudo fdisk -l
```
##### clone SD card
```Bash
sudo dd bs=4M if=/dev/mmcblk0 of=<name>.img status=progress
```

