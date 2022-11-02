**Step-by-step guide to setting up and using TensorFlow's Object
Detection API**

This is a step-by-step tutorial/guide to setting up and using
**TensorFlow's Object Detection API** to perform **object detection in
images/video** on Windows. This guide explains, how to setup your
Windows environment correctly and how to install Tensorflow and the
Tensorflow Object Detection API

**Main Source**:
<https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html>

**Target Software versions**

OS: Windows, Linux

Python:
3.9 [[1]{.ul}](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html#id3)

(any Python 3.x version should work, although this is not been tested)

TensorFlow: 2.5.0

MSVC (Mircosoft Visual Studio Compiler) 2019

For GPU support (see later in GPU Support (Optional Step)):

CUDA Toolkit: 11.2

CuDNN: 8.1.0

Or one of the combinations shown on the official TensorFlow side:
<https://www.tensorflow.org/install/source_windows>

# **Steps overview** {#steps-overview .TOC-Heading}

[1 Create Virtual Environment (optional) 1](#_Toc118288347)

[2 TensorFlow Installation 3](#tensorflow-installation)

[3 Enable GPU Support (optional) 4](#enable-gpu-support-optional)

[4 TensorFlow Object Detection API Installation
7](#tensorflow-object-detection-api-installation)

[5 Install the TensorFlow Object Detection API
8](#install-the-tensorflow-object-detection-api)

[6 Copy TFOD API scripts for model training, evaluation and exporting in
custom script folder
9](#copy-tfod-api-scripts-for-model-training-evaluation-and-exporting-in-custom-script-folder)

[7 Install LabelImg 10](#install-labelimg)

[]{#\_Toc118288347 .anchor}

# Create Virtual Environment (optional)

**Advantages:**

- Isolates all python and library dependencies needed for the TFOD

  > model

- Ensures clean working environment

- Does not conflict with all other already installed libraries and
  > dependencies no dependency conflicts

> "Create separated room to work in"

1.  **Create new folder** \[CUSTOM_MODEL_NAME\]\\venv \*\*for the virtual

    > environment\*\*

2.  **Create the new virtual environment in the repo with:**

> cd \[CUSTOM_MODEL_NAME\]\\venv
>
> python -m venv tfod-sticky-notes
>
> Virtual environment has it's own isolated packages under
> Lib/site-packages
>
> ![Ein Bild, das Text enthält. Automatisch generierte
Beschreibung](../../readme-media/preprocessing/image1.png){width="2.8843930446194226in"
> height="1.081648075240595in"}

3.  \*\*To activate the virtual environment on windows, run the windows
    > batch script in\*\* Scripts/activate

.\\tfod-sticky-notes\\Scripts\\activate

> ![](../../readme-media/preprocessing/image2.png){width="2.35838145231846in"
> height="1.2460269028871391in"}
>
> Now (tfod-sticky-notes) is visible and pip list only shows isolated
> packages (Lib/site-packages) in your environment
>
> ![Ein Bild, das Text enthält. Automatisch generierte
Beschreibung](../../readme-media/preprocessing/image3.png){width="4.571428258967629in"
> height="0.7442880577427822in"}
>
> Deactivate environment with deactivate

4.  **Update pip libraries and dependencies**

> Ensure that we have latest resolvers and upgraded pip install app
>
> In virtual environment run:
>
> python -m pip install \--upgrade pip

# TensorFlow Installation

Tensorflow is the core deep learning library behind all object
detenction functionality

For it to work we need to install some dependencies

## Install TensorFlow PIP package

In Tutorial: pip install \--ignore-installed \--upgrade
tensorflow==2.5.0

(ERROR: Could not find a version that satisfies the requirement
tensorflow==2.5.0) Therefore install tensorflow without the version

**pip install tensorflow**

![](../../readme-media/preprocessing/image4.png){width="2.7708333333333335in"
height="1.9972222222222222in"}

### Possible LongPaths problem on windows

**ERROR:** Could not install packages due to an OSError

**HINT:** This error might have occurred since this system

does not have Windows Long Path support enabled.

**FIX**: Enable LongPaths on Windows

**SOURCE:**
<https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/>

## Install Visual C++ Build Tools:

Needed from tensorflow in order to run

<https://visualstudio.microsoft.com/vs/community/>

## Verify your Installation

python -c \"import tensorflow as
tf;print(tf.reduce_sum(tf.random.normal(\[1000, 1000\])))\"

Once above is run, follow print-out similar to one bellow should be
seen:

2020-06-22 19:20:32.614181: W
tensorflow/stream_executor/platform/default/dso_loader.cc:55\] **Could
not load dynamic library \'cudart64_101.dll\'; dlerror: cudart64_101.dll
not found**

2020-06-22 19:20:32.620571: I
tensorflow/stream_executor/cuda/cudart_stub.cc:29\] Ignore above cudart
dlerror **if** you do **not** have a GPU set up on your machine.

2020-06-22 19:20:35.146285: W
tensorflow/core/common_runtime/gpu/gpu_device.cc:1598\] Cannot dlopen
some GPU libraries. Please make sure the missing libraries mentioned
above are installed properly **if** you would like to use GPU. Follow
the guide at https://www.tensorflow.org/install/gpu **for** how to
download **and** setup the required libraries **for** your platform.

Skipping registering GPU devices\...

2020-06-22 19:20:35.196815: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1108\]

tf.Tensor(1620.5817, shape=(), dtype=float32)

# Enable GPU Support (optional)

Enables significant faster model training and evaluating

If machine is equipped with compatible CUDA-enabled GPU, it is
recommended to install relevant libraries necessary to enable TensorFlow
to make use of your GPU, because computational gains from GPU are
substantial.

By default, when TensorFlow is run it will attempt to register
compatible GPU devices.

If this fails, TensorFlow will resort to running on platform's CPU (can
be observed in printout of console log)

If GPU is not found or something didn't work out - number of messages
report missing library files
(e.g. Could not load dynamic library \'cudart64_101.dll\'; dlerror: cudart64_101.dll not found).

Install Appropriate CUDA and cuDNN version

Only necessary if you have GPU and you want to use GPU accelerated
training

Enables GPU based acceleration when training TOD models

Reason for using GPU: it is exponentially faster to train OD model using
GPU versus just using raw CPU and raw memory

**Source for all proper versions and configurations of dependencies
provided by Tensorflow:**

<https://www.tensorflow.org/install/source_windows>

![](../../readme-media/preprocessing/image5.png){width="2.7881944444444446in"
height="1.01875in"}**For TensorFlow to run on your GPU, following
requirements must be met:**

Nvidia GPU (GTX 650 or newer)

Installed CUDA Toolkit v11.2

Installed CuDNN 8.1.0

**IMPORTANT:**

Important that tensorflow version matches cuDNN and CUDA version

If it don't match, it will still run but won't leverage your GPU

(MSVC = Mircosoft Visual Studio Compiler)

Extract the contents of the zip file (i.e. the folder named cuda)
inside \<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\,
where \<INSTALL_PATH\> points to the installation directory specified
during the installation of the CUDA Toolkit. By
default \<INSTALL_PATH\> = C:\\Program Files.

![](../../readme-media/preprocessing/image6.png){width="3.2515332458442696in"
height="1.5569739720034996in"}

#### {#section .list-paragraph}

## Environment Setup for GPU Support

**Add the following paths to environment variables:**

\<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin

\<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp

\<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include

\<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\extras\\CUPTI\\lib64

\<INSTALL_PATH\>\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\cuda\\bin
**(does not exist)**

### Possible Python Path Error

Adjust pyvenv.cfg python path inside virtual environment

![](../../readme-media/preprocessing/image7.png){width="3.6412128171478564in"
height="3.1716415135608047in"}

## Verify the GPU support

Run the following command in a **NEW** *Terminal* window:

python -c \"import tensorflow as
tf;print(tf.reduce_sum(tf.random.normal(\[1000, 1000\])))\"

Once the above is run, you should see a print-out like the one bellow:

2021-06-08 18:28:38.452128: I
tensorflow/stream_executor/platform/default/dso_loader.cc:53\]
Successfully opened dynamic library cudart64_110.dll

\...

2021-06-08 18:28:40.973992: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1733\] Found device 0
**with** properties:

pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1

coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB
deviceMemoryBandwidth: 238.66GiB/s

2021-06-08 18:28:40.974115: I
tensorflow/stream_executor/platform/default/dso_loader.cc:53\]
Successfully opened dynamic library cudart64_110.dll

2021-06-08 18:28:41.001094: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1871\] Adding visible
gpu devices: 0

2021-06-08 18:28:41.001651: I
tensorflow/core/platform/cpu_feature_guard.cc:142\] This TensorFlow
binary **is** optimized **with** oneAPI Deep Neural Network Library
(oneDNN) to use the following CPU instructions **in**
performance-critical operations: AVX AVX2

To enable them **in** other operations, rebuild TensorFlow **with** the
appropriate compiler flags.

2021-06-08 18:28:41.003095: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1733\] Found device 0
**with** properties:

pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1

coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB
deviceMemoryBandwidth: 238.66GiB/s

2021-06-08 18:28:41.003244: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1871\] Adding visible
gpu devices: 0

2021-06-08 18:28:42.072538: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1258\] Device
interconnect StreamExecutor **with** strength 1 edge matrix:

2021-06-08 18:28:42.072630: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1264\] 0

2021-06-08 18:28:42.072886: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1277\] 0: N

2021-06-08 18:28:42.075566: I
tensorflow/core/common_runtime/gpu/gpu_device.cc:1418\] Created
TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 **with**
6613 MB memory) -\> physical GPU (device: 0, name: GeForce GTX 1070 Ti,
pci bus id: 0000:02:00.0, compute capability: 6.1)

tf.Tensor(641.5694, shape=(), dtype=float32)

Notice from the lines highlighted above that the library files are
now Successfully opened and a debugging message is presented to confirm
that TensorFlow has successfully Created TensorFlow device.

![](../../readme-media/preprocessing/image8.png){width="6.531944444444444in"
height="0.4388888888888889in"}

# TensorFlow Object Detection API Installation

## Downloading the TensorFlow Model Garden:

- Create new folder under e.g. Documents named TensorFlow.
  (e.g. C:\\Users\\sglvladi\\Documents\\TensorFlow).

- To download the models clone [TensorFlow Models
  repository](https://github.com/tensorflow/models) inside
  the TensorFlow folder

- You should now have a single folder named models under
  your TensorFlow folder, which contains another 4 folders as such:

TensorFlow/

└─ models/

├─ community/

├─ official/

├─ orbit/

├─ research/

└── \...

## Install and compile protobuf

Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, Protobuf
libraries must be downloaded and compiled.

Head to [protoc releases
page](https://github.com/google/protobuf/releases)

Download latest protoc-\*-\*.zip release
(e.g. protoc-3.12.3-win64.zip for 64-bit Windows)

![](../../readme-media/preprocessing/image9.png){width="5.194029965004375in"
height="0.66209208223972in"}

Extract contents of the downloaded protoc-\*-\*.zip in
directory \<PATH_TO_PB\> of your choice
(e.g. C:\\Program Files\\Google Protobuf)

Add C:\\Program Files\\Google Protobuf\\bin to your Path environment
variable

In new *Terminal* , cd into TensorFlow/models/research/ directory and
run the following command:

_\# From within TensorFlow/models/research/_

cd
C:\\\\\_WORK\\GitHub\\\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research

protoc object_detection/protos/\*.proto \--python_out=.

No feedback -- assuming no error

# Install the TensorFlow Object Detection API

Installation of the Object Detection API is achieved by installing
the object_detection package.

This is done by running the following commands from
within Tensorflow\\models\\research:

_\# From within TensorFlow/models/research/ copy the setup.py file
inside the folder_

copy
C:\\\\\_WORK\\GitHub\\\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research\\object_detection\\packages\\tf2
C:\\\\\_WORK\\GitHub\\\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research

_\# Run the setup.py file from within TensorFlow/models/research/ to
automatically install all necessary packages and libraires for the
TensorFlow Object Detection API_

cd
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research

python -m pip install \--use-feature=2020-resolver .

![](../../readme-media/preprocessing/image10.png){width="3.8541666666666665in"
height="2.2647222222222223in"}

## Possible Problem with protobuffer and the default setup.py from the package

**ERROR:**

ERROR: pip\'s dependency resolver does not currently take into account
all the packages that are installed. This behaviour is the source of the
following dependency conflicts.

tensorflow 2.9.2 requires protobuf\<3.20,\>=3.9.2

tensorflow-metadata 1.10.0 requires protobuf\<4,\>=3.13

tensorboard 2.9.1 requires protobuf\<3.20,\>=3.9.2

apache-beam 2.41.0 requires protobuf\<4,\>=3.12.2

but you have protobuf 4.21.6 which is incompatible.

**SUGGESTED FIX:**

Downgrade protobuf version

pip install \--upgrade protobuf==3.20.0

![](../../readme-media/preprocessing/image11.png){width="3.922222222222222in"
height="0.7902777777777777in"}![](../../readme-media/preprocessing/image12.png){width="3.6597222222222223in"
height="0.8839971566054243in"}

## Test your Installation

_\# From within TensorFlow/models/research/_

cd
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research

python object_detection/builders/model_builder_tf2_test.py

Printout should show "OK" and look like the one below:

\...

\[ OK \] ModelBuilderTF2Test.test_create_ssd_models_from_config

\[ RUN \] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update

\...

INFO:tensorflow:time(\_\_main\_\_.ModelBuilderTF2Test.test_unknown-ssd_feature_extractor):
0.0s

I0608 18:49:13.197239 29296 test_util.py:2102\]
time(\_\_main\_\_.ModelBuilderTF2Test.test_unknown-ssd_feature_extractor):
0.0s

\[ OK \] ModelBuilderTF2Test.test_unknown-ssd_feature_extractor

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

Ran 24 tests **in** 29.980s

**OK (skipped=1)**

# Copy TFOD API scripts for model training, evaluation and exporting in custom script folder

Create folder inside TensorFlow/scripts named tfod-api-scripts

Copy the TensorFlow Object Detection API scripts model_main_tf2.py from
and exporter_main_v2.py from
TensorFlow\\models\\research\\object_detection\\ into the new created
folder TensorFlow\\scripts\\tfod-api-scripts

model_main_tf2.py is used for the training and evaluation of the model
(necessary)

exporter_main_v2.py is used for the exporting of the model (optional)

_\# From within TensorFlow/models/research/_

copy
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research\\object_detection\\**model_main_tf2.py**
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\scripts\\tfod-api-scripts

_\# From within TensorFlow/models/research/_

copy
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\tensorflow-model-garden\\research\\object_detection\\**exporter_main_v2.py**
C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\scripts\\tfod-api-scripts

![](../../readme-media/preprocessing/image13.png){width="3.1346555118110238in"
height="1.4566929133858268in"}
![](../../readme-media/preprocessing/image14.png){width="3.153856080489939in"
height="1.4566929133858268in"}

# Install LabelImg

Recommended way with PIP Package Manager does not work properly with
python 3.10

Instead use the "Build from source" Way:

## **Clone labelImg repo from GitHub**

- Inside TensorFlow folder, create new directory nameed addons and
  navigate into it

- Clone the labelImg repo with Git inside
  the TensorFlow\\addons folder

cd C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\addons

git clone https://github.com/heartexlabs/labelImg.git

TensorFlow/

├─ scripts

│ └─ own-scripts/

│ └─ tfod-api-scripts/

│ └─ tfod-tutorial-scripts/

**├─ addons**

**│ └─ labelImg/**

└─ tensorflow-model-garden/

├─ community/

├─ official/

├─ orbit/

├─ research/

└─ \...

## **Install dependencies and compiling package**

- Open new *Terminal* and activate virtual environment

- Navigate into TensorFlow/addons/labelImg and run the following
  commands:

cd C:\\\_WORK\\GitHub\\\_data-science\\TensorFlow\\addons\\labelImg

pyrcc5 -o libs/resources.py resources.qrc

## **Test your installation**

_\# From within Tensorflow/addons/labelImg_

python labelImg.py

_\# or_

python labelImg.py \[IMAGE_PATH\] \[PRE-DEFINED CLASS FILE\]
