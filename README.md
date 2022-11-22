# MyJetsonNote

For Jetson Series - TX2, Xavier AGX, Xavier NX

# 目錄

* [安装系統](#安装系統)
* [檢查安裝](#檢查安裝)
* [修改默認python版本](#修改默認python版本)
* [Jetson stats](#Jetson-stats)
* [Tensorflow](#Tensorflow)
* [Opencv 4.5](#opencv4.5)
* [Opencv 3.4 (No opencv_contrib)](#opencv3.4)
* [PyAudio & Sounddevice](#PyAudio&Sounddevice)
* [Librosa 0.6.3](#Librosa0.6.3)
* [PyTorch](#PyTorch)
* [Torchaudio](#Torchaudio)
* [PyQt5](#PyQt5)
* [Others](#Others)
    * [FTP server](#FTPserver)
    * [kolourpaint4](#kolourpaint4)
    * [Firefox](#Firefox)
    * [exFAT driver](#exFATdriver)
    * [Java](#Java)
* [Clone SD card](#CloneSDcard)
* [AGX Fast Backup](#AGXBackup)

# 安装系統 <a name="安装系統"></a>

### NX

* [NVIDIA官網](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)
* [JETPACK SDK](https://developer.nvidia.com/embedded/jetpack)

### TX2, AGX

**Host電腦端 (電腦端):** 準備一台系統為linux的電腦

1. 在電腦端下載並安裝[SDK Manager](https://developer.nvidia.com/embedded/jetpack-archive)

2. 硬體連接

- 使用盒內usb線連接電腦端與AGX（連接AGX電源燈對面的Type-c）, 將AGX連接上電腦螢幕.
- 進入工程模式, AGX接上電源並保持關機狀態下, 按住中間的按鍵（Force Recovery）不放開, 再按下左邊的電源（Power）不放開, 等待約兩秒後同時放開.
- 在電腦端上輸入lsusb查看是否連線上AGX（出現Nvidia corp.）

3. 開啟SDK Manager, 登入NVIDIA 帳戶
   ![1](https://user-images.githubusercontent.com/53622566/82420244-0d868f80-9ab2-11ea-9524-b45e54fe9656.png)

4. 如下圖設定 **Jetson TX2** ,  **JetPack 4.3** , **DeepStream**, 點擊**CONTINUE**

- AGX 建議安裝 Jetpack 4.4, 可不勾選DeepStream
  ![2](https://user-images.githubusercontent.com/53622566/82420328-2abb5e00-9ab2-11ea-8238-c298972a7197.png)

5. 勾選左下**I accept ...**後點擊 **CONTINE**
   ![3](https://user-images.githubusercontent.com/53622566/82420414-46266900-9ab2-11ea-9ba9-ded4b2738953.png)

6. 跳出一個視窗要求輸入系統密碼, 輸入後以繼續安裝
   ![4](https://user-images.githubusercontent.com/53622566/82420458-52122b00-9ab2-11ea-9a25-64874768a69f.png)

7. 進入 **STEP 03**, 等待下載及安裝至出現出視窗如下, 點選**Manual Setup**並按照提示操作TX2進入Recovery(工程模式)後, 點擊**Flash**(將TX2連接上螢幕)
   ![6](https://user-images.githubusercontent.com/53622566/82420572-7a018e80-9ab2-11ea-9a16-fafd75bd70cd.png)

8. 等待出現視窗如下後, 移至**TX2端 (AGX)**, 可以看到系統登入畫面, 請設定系統名稱、密碼等, 最後畫面停留至桌面
   ![7](https://user-images.githubusercontent.com/53622566/82420653-9b627a80-9ab2-11ea-819e-2e414d6ff317.png)

9. 回到**Host電腦端**輸入剛才在TX2的設定的使用者名稱與密碼, 點擊**Install**

10. 安裝結束, 點擊**Finish and exit**, 安裝完成
    ![8](https://user-images.githubusercontent.com/53622566/82421655-e204a480-9ab3-11ea-9e90-7b8e7b0c4e99.png)

# 檢查安裝

1. CUDA

```Bash
sudo find / -name  cuda
```

Or

```Bash
## set an environment variable for cuda
export PATH=$PATH:/usr/local/cuda/bin
nvcc -V
```

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

# 修改默認python版本 <a name="修改默認python版本"></a>

使用alternatives管理多版本軟體 將python版本指定為python2.7

```Bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 100
```

將python版本指定為python3.6

```Bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 150
```

選用python版本

```Bash
sudo update-alternatives --config python
```

# Install new versions of software

```Bash
sudo apt update && sudo apt upgrade
```

# Jetson stats <a name="Jetson-stats"></a>

[jetson-stats](https://github.com/rbonghi/jetson_stats) is a package for monitoring and control
your [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) [Xavier NX, Nano, AGX Xavier, TX1, TX2]
Works with all NVIDIA Jetson ecosystem. Install pip

```Bash
sudo apt install python3-pip
```

Install

```Bash
sudo pip3 install jetson-stats
```

# Tensorflow

* [NVIDIA DOC.](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

1. Install [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson device.
2. Install system packages required by TensorFlow:

```Bash
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev \
zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```

3. Install and upgrade pip3

```Bash
sudo apt-get install python3-pip
```

```Bash
sudo pip3 install -U pip testresources setuptools==49.6.0 
```

4. Install the Python package dependencies

```Bash
sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 \
keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 \
protobuf pybind11 cython pkgconfig
```

```Bash
sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
```

5. Install TensorFlow using the pip3 command.

- 對應 Jetpack 版本 - [TENSORFLOW VERSION](https://developer.download.nvidia.com/compute/redist/jp/)

- [安裝範例1] 在 Jetpack 4.6 安裝最新版 TensorFlow

```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
```

- [安裝範例2] 在 Jetpack 4.6 安裝 TensorFlow 版本小於2

```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'
```

- [安裝範例3] 安裝指定版本的 TensorFlow

```Bash
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow==$TF_VERSION+nv$NV_VERSION```
```

# Opencv 4.5 <a name="opencv4.5"></a>

```Bash
sudo apt install \
python3-dev python3-pip python3-tk \
build-essential make cmake cmake-curses-gui \
g++ libavformat-dev libavutil-dev \
libswscale-dev libv4l-dev libeigen3-dev \
libglew-dev libgtk2.0-dev libdc1394-22-dev libxine2-dev \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libjpeg8-dev libjpeg-turbo8-dev libtiff5-dev \
libavcodec-dev libxvidcore-dev libx264-dev libgtk-3-dev \
libatlas-base-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev qt5-default      
```

```Bash
sudo apt purge libopencv*
```

```Bash
wget https://github.com/opencv/opencv/archive/4.5.0.zip -O opencv-4.5.0.zip \
&& unzip opencv-4.5.0.zip && cd opencv-4.5.0
```

```Bash
wget https://github.com/opencv/opencv_contrib/archive/4.5.0.zip -O opencv_contrib-4.5.0.zip \
&& unzip opencv_contrib-4.5.0.zip
```

```Bash
mkdir build && cd build
```

```Bash
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local/ \
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.0/modules \
-D CUDA_ARCH_BIN='7.2' \
-D WITH_CUDA=1 \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_FFMPEG=ON \
-D WITH_OPENGL=ON \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D OPENCV_GENERATE_PKGCONFIG=1 \
-D BUILD_opencv_python2=1 \
-D BUILD_opencv_python3=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-D WITH_GTK_2_X=ON ..
```

```Bash
sudo make -j$(nproc) && sudo make install
```

# Opencv 3.4 (No opencv_contrib)<a name="opencv3.4"></a>

- 由於AGX系統刷完後空間剩下10GB, 沒有足夠空間安裝opencv額外套件(opencv_contrib)
- 參考 [JK Jung's blog](https://jkjung-avt.github.io/opencv3-on-tx2/)
- 參考 [Jetsonhacks](https://www.jetsonhacks.com/2018/11/08/build-opencv-3-4-on-nvidia-jetson-agx-xavier-developer-kit/)

```Bash
sudo apt purge libopencv*
```

```Bash
sudo apt install --only-upgrade g++-5 cpp-5 gcc-5
```

```Bash
sudo apt install \
python3-dev python3-pip python3-tk \
build-essential make cmake cmake-curses-gui \
g++ libavformat-dev libavutil-dev \
libswscale-dev libv4l-dev libeigen3-dev \
libglew-dev libgtk2.0-dev libdc1394-22-dev libxine2-dev \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libjpeg8-dev libjpeg-turbo8-dev libtiff5-dev \
libavcodec-dev libxvidcore-dev libx264-dev libgtk-3-dev \
libatlas-base-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev qt5-default      
```

```Bash
sudo pip3 install matplotlib==3.3.4  # 注意： 會自動安裝其他matplotlib所需套件, 如果要指定套件版本請自行個別安裝
```

#### Modify matplotlibrc (line #41) as 'backend: TkAgg'

```Bash
sudo vim /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
```

#### Modify /usr/local/cuda/include/cuda_gl_interop.h and fix the symbolic link of libGL.so.

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

#### Next, download opencv-3.4.0 source code, cmake and compile. Note that opencv_contrib modules (cnn/dnn stuffs) would cause problem on pycaffe, so after some experiments I decided not to include those modules at all.

```Bash
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
```

```Bash
unzip opencv-3.4.0.zip && cd opencv-3.4.0
```

```Bash
mkdir build && cd build
```

#### Build opencv (CUDA_ARCH_BIN="6.2" for TX2, or "5.3" for TX1)

```Bash
cmake -D CMAKE_BUILD_TYPE=RELEASE  \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON  \
      -D CUDA_ARCH_BIN="7.2"  \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUBLAS=ON  \
      -D ENABLE_FAST_MATH=ON  \
      -D CUDA_FAST_MATH=ON \
      -D ENABLE_NEON=ON  \
      -D WITH_LIBV4L=ON  \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF  \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_QT=ON  \
      -D WITH_OPENGL=ON ..

```

```Bash
sudo make -j$(nproc) && sudo make install
```

#### To verify the installation:

```Bash
ls /usr/local/lib/python3.6/dist-packages/cv2.*
ls /usr/local/lib/python2.7/dist-packages/cv2.*
python3 -c 'import cv2; print(cv2.__version__)'
python2 -c 'import cv2; print(cv2.__version__)'
```

# PyAudio & Sounddevice <a name="PyAudio&Sounddevice"></a>

```Bash
sudo apt install python-all-dev portaudio19-dev libffi-dev
```

```Bash
sudo pip3 install pyaudio sounddevice

# pip install pipwin
# pipwin install pyaudio
```

# Librosa 0.6.3 <a name="Librosa0.6.3"></a>

- [How to install the Librosa library in Jetson Nano or aarch64 module](https://learninone209186366.wordpress.com/2019/07/24/how-to-install-the-librosa-library-in-jetson-nano-or-aarch64-module/)
- [numba/llvmlite](https://github.com/numba/llvmlite)
- [llvmlite documents](https://llvmlite.readthedocs.io/en/latest/)

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
wget http://releases.llvm.org/7.0.1/llvm-7.0.1.src.tar.xz \
tar -xvf llvm-7.0.1.src.tar.xz \
cd llvm-7.0.1.src && mkdir llvm_build_dir && cd llvm_build_dir/ \
cmake ../ -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64" \
sudo make -j$(nproc) && sudo make install
```

```Bash
cd bin/ \
echo "export LLVM_CONFIG=\""`pwd`"/llvm-config\"" >> ~/.bashrc \
echo "alias llvm='"`pwd`"/llvm-lit'" >> ~/.bashrc \
source ~/.bashrc
```

##### Finally

```Bash
sudo pip3 install llvmlite==0.32.1
```

#### Install Numba, Scipy, Joblib, Scikit-learn

```Bash
sudo apt install libblas-dev liblapack-dev libatlas-base-dev gfortran
```

```Bash
sudo pip3 install numba==0.38.0  # numba==0.48.0
```

```Bash
sudo pip3 install scipy==1.1.0  # scipy==1.4.1
```

```Bash
sudo pip3 install joblib==0.12  # joblib==0.14
```

```Bash
sudo pip3 install scikit-learn==0.21.1
```

#### Install Librosa

```Bash
sudo pip3 install librosa==0.6.3  # librosa==0.7.2
```

* 如果 joblib 有問題 可以嘗試更新 joblib

# PyTorch <a name="PyTorch"></a>

- [pytorch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048)

###

```Bash
sudo apt install python3-pip libopenblas-base libopenmpi-dev
sudo apt install libjpeg-dev zlib1g-dev
sudo pip3 install Cython

```

### PyTorch 1.7.0 + Torchvision 0.8.1

```Bash
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
cd torchvision && sudo python setup.py install
```

### PyTorch 1.8.0 + Torchvision 0.9.0

```Bash
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision && sudo python setup.py install
```

### PyTorch 1.10.0 + Torchvision 0.11.3

```Bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.11.3 https://github.com/pytorch/vision torchvision
cd torchvision && sudo python setup.py install

```

# Torchaudio <a name="Torchaudio"></a>

- [torchaudio](https://github.com/pytorch/audio)

```Bash
git clone https://github.com/pytorch/audio.git torchaudio --branch v0.7.2
```

```Bash
sudo apt install sox libsox-dev libsox-fmt-all
```

```Bash
cd torchaudio && sudo python setup.py install
```

# PyQt5 <a name="PyQt5"></a>

#### Method 1

_If build wheel error, try_ `Method 2`

```Bash
sudo apt install qt5-default python3-pyqt5 pyqt5-dev-tools qttools5-dev-tools
```

```Bash
sudo pip3 install pyqt5==5.15.6 
```

```Bash
# 測試中
# pyqt5-sip==12.9.1
# sudo pip3 install sip==5.0.1
```

#### Method 2

###### _Install SIP_

- [What is SIP?](https://pypi.org/project/SIP/)
- [Download Sip source](https://riverbankcomputing.com/software/sip/download)

_If you have an old version of SIP_

```Bash
sudo apt remove python3-sip
```

```Bash
wget https://www.riverbankcomputing.com/static/Downloads/sip/4.19.25/sip-4.19.25.tar.gz
tar -xvf sip-4.19.25.tar.gz
cd sip-4.19.25
```

```Bash
python3 configure.py
make -j$(nproc) && sudo make install
```

###### _Install PyQt5_

- [PyQt Reference Guide](https://www.riverbankcomputing.com/static/Docs/PyQt5/installation.html)
- [Download PyQt5 source](https://riverbankcomputing.com/software/pyqt/download5)

###### _使用 版本 PyQt5-5.15.4.tar.gz, 下載後並解壓_

```Bash
python3 configure.py
make -j$(nproc) && sudo make install
```

#### Install ie multimedia

```Bash
sudo apt install python3-pyqt5.qtmultimedia
```

# Others

### Matplotlib

```
sudo apt install python3-matplotlib  # 不推薦, 但是如果你真的裝不起來再試試這個
```

### others

```Bash
sudo pip install pyqtgraph==0.11.1
```

```Bash
sudo pip install pyusb click xlsxwriter tqdm imutils qdarkstyle
```

```Bash
sudo pip install pandas==1.1.4 PyYAML==5.3.1 --ignore-installed
```

```Bash
sudo pip install seaborn==0.11.0 # 注意, 會自動安裝相關套件!!
```

```Bash
sudo dpkg-reconfigure dash
```

```Bash
sudo apt install filezilla curl
```

### Anonymous FTP server <a name="FTPserver"></a>

```Bash
sudo apt install vsftpd
# systemctl status vsftpd
# netstat -tupln | grep ftp
```

```Bash
sudo vi /etc/vsftpd.conf
# Allow anonymous FTP? (Disabled by default).
anonymous_enable=YES
# Uncomment this to enable any form of FTP write command.
write_enable=YES

# Uncomment this to allow the anonymous FTP user to upload files. This only
# has an effect if the above global write enable is activated. Also, you will
# obviously need to create a directory writable by the FTP user.
anon_upload_enable=YES
```

```Bash
sudo systemctl restart vsftpd
```

### Firefox <a name="Firefox"></a>

```Bash
sudo apt install firefox
```

### Kolourpaint4 <a name="Kolourpaint4"></a>

```Bash
sudo apt install kolourpaint4
```

### exFAT driver <a name="exFATdriver"></a>

```Bash
sudo add-apt-repository universe && sudo apt install exfat-fuse exfat-utils
```

### Java <a name="Java"></a>

```Bash
sudo apt install default-jre # sudo apt install default-jdk
```

# Clone SD card <a name="CloneSDcard"></a>

- [dd command](https://blog.gtwang.org/linux/dd-command-examples/)

###### find your SD card

```Bash
sudo fdisk -l
```

###### clone SD card to img

```Bash
sudo dd if=/dev/sda conv=sync,noerror bs=1M status=progress | gzip -c > ./backup.img.gz
```

###### clone img to SD card

```Bash
sudo gunzip -c ./backup.img.gz | sudo dd of=/dev/sda bs=1M status=progress
```

# AGX Fast Backup <a name="AGXBackup"></a>

- AGX快速備份與平展, 不須透過 NVIDIA SDK Manager, 平均20分鐘內可安裝完成一台

```Bash
Coming soon
```
