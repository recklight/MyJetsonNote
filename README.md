# MyJetsonNote
Jetson Series - TX2, Xavier AGX, Xavier NX
------
------

目錄
------
* [安装系統](#安装系統)
* [檢查安裝](#檢查安裝)  
* [修改默認python版本](#修改默認python版本)
* [Jetson stats](#Jetson\tstats)
* [Tensorflow](#Tensorflow) 
* [Opencv 4.5](#Opencv\t4.5) 
* [PyAudio & Sounddevice](#PyAudio\t&\tSounddevice) 
* [Librosa 0.6.3](#Librosa\t0.6.3)
* [PyTorch](#PyTorch)
* [PyQt5](#PyQt5)
* [Others](#Others) 


安装系統
------
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

4. 如下圖設定 **Jetson TX2** ,  **JetPack 4.3** , **DeepStream**,  點擊**CONTINUE**
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


檢查安裝
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
將python版本指定為python2.7
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


Install new versions of software
------
```Bash
sudo apt update && sudo apt upgrade
```

Jetson stats
------
[jetson-stats](https://github.com/rbonghi/jetson_stats) is a package for monitoring and control your [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) [Xavier NX, Nano, AGX Xavier, TX1, TX2] Works with all NVIDIA Jetson ecosystem.
Install pip 
```Bash
sudo apt install python3-pip
```
Install 
```Bash
sudo pip3 install jetson-stats
```


Tensorflow
------
* [NVIDIA DOC.](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
1. Install [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson device.
2. Install system packages required by TensorFlow:
```Bash
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
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
sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
```
```Bash
sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
```

5. Install TensorFlow using the pip3 command. 
- 注意安裝的 Jetpack 版本
> [v44 TENSORFLOW VERSION](https://developer.download.nvidia.com/compute/redist/jp/v44/tensorflow/)

> [v45 TENSORFLOW VERSION](https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/)

> [v46 TENSORFLOW VERSION](https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/)

- This command will install the latest version of TensorFlow compatible with JetPack 4.6.
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
```

- Note: TensorFlow version 2 was recently released and is not fully backward compatible with TensorFlow 1.x. If you would prefer to use a TensorFlow 1.x package, it can be installed by specifying the TensorFlow version to be less than 2, as in the following command
```Bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'
```

- If you want to install the latest version of TensorFlow supported by a particular version of JetPack, issue the following command:
```Bash
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow
```


Opencv 4.5
------
```Bash
sudo apt install python3-dev python3-pip python3-tk \
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
cmake -D CMAKE_BUILD_TYPE=RELEASE \
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
sudo make -j4 && sudo make install
```


PyAudio & Sounddevice
------
```Bash
sudo apt install python-all-dev portaudio19-dev libffi-dev
```
```Bash
sudo pip3 install pyaudio sounddevice
```

Librosa 0.6.3
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
cd llvm-7.0.1.src && mkdir llvm_build_dir && cd llvm_build_dir/
cmake ../ -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64"
sudo make -j4 && sudo make install
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
sudo apt install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo pip3 install numba==0.38.0 scipy==1.1.0 joblib==0.12 scikit-learn==0.21.1
```
#### Install Librosa
```Bash
sudo pip3 install librosa==0.6.3
```


PyTorch
------
- [pytorch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048)
#### PyTorch 1.6.0 + Torchvision 0.7.0
```Bash
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
sudo apt install python3-pip libopenblas-base libopenmpi-dev
sudo pip3 install Cython torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl

sudo apt install libjpeg-dev zlib1g-dev
git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision && sudo python setup.py install
```

#### PyTorch 1.7.0 + Torchvision 0.8.1
```Bash
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo apt install python3-pip libopenblas-base libopenmpi-dev
sudo pip3 install Cython torch-1.7.0-cp36-cp36m-linux_aarch64.whl

sudo apt install libjpeg-dev zlib1g-dev
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
cd torchvision && sudo python setup.py install
```


PyQt5
------
#### Method 1
```Bash
sudo apt install qt5-default python3-pyqt5 pyqt5-dev-tools qttools5-dev-tools
sudo pip3 install pyqt5
```
- If build wheel error, try
```Bash
# sudo pip3 install sip==5.0.1
```

#### Method 2

- If you have an old version of SIP
```Bash
sudo apt remove python3-sip
sudo apt autoremove
```
- Install sip
> [What is SIP?](https://pypi.org/project/SIP/)

> [Download Sip source](https://riverbankcomputing.com/software/sip/download)
```Bash
python3 configure.py
make
sudo make install
```

- Install pyqt5
> [PyQt Reference Guide](https://www.riverbankcomputing.com/static/Docs/PyQt5/installation.html)

> [Download PyQt5 source](https://riverbankcomputing.com/software/pyqt/download5)
```Bash
python3 configure.py
make
sudo make install
```
- Install  ie multimedia
```Bash
sudo apt install python3-pyqt5.qtmultimedia
```


Others
------
- Matplotlib
```
sudo apt-get install python3-matplotlib
```

- others
```Bash
sudo pip install PyYAML==5.3.1 --ignore-installed
```
```Bash
sudo pip install keras==2.3.1 
```
```Bash
sudo pip install pyusb click xlsxwriter tqdm imutils qdarkstyle
```
```Bash
sudo pip install pandas==1.1.4 seaborn==0.11.0
```

- Anonymous FTP server
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

- Browser
```Bash
sudo apt install firefox
```

- Kolourpaint4
```Bash
sudo apt install kolourpaint4
```

- exFAT_driver
```Bash
sudo add-apt-repository universe
sudo apt install exfat-fuse exfat-utils
```

- Java
###### JRE
```Bash
sudo apt install default-jre
```
###### JDK
```Bash
sudo apt install default-jdk
```

- CloneSDcard
> [dd command1](https://blog.gtwang.org/linux/dd-command-examples/)

> [dd command2](https://blog.xuite.net/mb1016.flying/linux/28346040)
###### find your SD card
```Bash
sudo fdisk -l
```
###### clone SD card to img
```Bash
sudo dd if=/dev/sdb conv=sync,noerror bs=4M status=progress | gzip -c > ~/image.img.gz
```
###### clone img to SD card
```Bash
sudo gunzip -c ~/image.img.gz | sudo dd of=/dev/sdb bs=4M status=progress
```
