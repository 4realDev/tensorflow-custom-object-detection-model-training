r""" Changing something on the config please re-run all the scripts to create a new workspace for the new model

    0.	Activate virtual environment
    cd C:\_WORK\GitHub\_data-science\TensorFlow\venv
    .\tfod-sticky-notes\Scripts\activate

    1.	Adjust the “config.py” file in C:\\_WORK\GitHub\\_data-science\TensorFlow\scripts\own-scripts\preprocessing according to your needs

    2.	Run the “create_model_workspace.py” script to setup the folder structure, create the label map, download and extract the pre-trained model, copy the pipeline.config file from the pre-trained model into the custom model and adjust the pipeline.config
    python C:\_WORK\GitHub\_data-science\TensorFlow\scripts\own-scripts\preprocessing\create_model_workspace.py

    3.	Collect images and label them with labelImg to create their XML file
    python C:\_WORK\GitHub\_data-science\TensorFlow\addons\labelImg\labelImg.py

    4.	Run the “prepare_image_data_for_training.py” script to partition the image data into test and train and to create the tf.records for test.record and train.record out of the labeled image data
    python C:\\_WORK\GitHub\\_data-science\TensorFlow\scripts\own-scripts\preprocessing\prepare_image_data_for_training.py

    5.	Run the “train_custom_model.py” script to train the model
    python C:\\_WORK\GitHub\\_data-science\TensorFlow\scripts\own-scripts\preprocessing\train_custom_model.py

    6.	Run the “eval_custom_model.py” script to eval the model
    python C:\\_WORK\GitHub\\_data-science\TensorFlow\scripts\own-scripts\preprocessing\eval_custom_model.py

    7.	Run the “run_tensorboard_on_custom_model.py” script to start TensorBoard servor for custom model
    python C:\\_WORK\GitHub\\_data-science\TensorFlow\scripts\own-scripts\preprocessing\run_tensorboard_on_custom_model.py

    8.	Run the “export_custom_model.py” script to export the custom model into the export folder within the custom models workspace
    python C:\\_WORK\GitHub\_data-science\TensorFlow\scripts\own-scripts\preprocessing\export_custom_model.py


    MAIN SCRIPTS
    python C:\_WORK\GitHub\_data-science\TensorFlow\scripts\own-scripts\sticky-notes-detection\miro-sticky-notes-sync.py
"""

import os
# labels the custom model detects
labels = [{'name': 'yellow_sticky_note', 'id': 1, 'color': 'yellow'},
          {'name': 'blue_sticky_note', 'id': 2, 'color': 'blue'},
          {'name': 'pink_sticky_note', 'id': 3, 'color': 'pink'},
          {'name': 'green_sticky_note', 'id': 4, 'color': 'green'}]


# vars necessary for pipeline.config adjustments and for model training
num_classes = len(labels)
batch_size = 4
num_steps = 25000
fine_tune_checkpoint_type = "detection"  # "classification" or "detection"


# vars for PADDLE OCR model
# for german use 'german' -> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/quickstart_en.md
ocr_model_language = "en"
ocr_confidence_threshold = 0.50


# vars for real time object detection
min_score_thresh = 0.9
bounding_box_and_label_line_thickness = 10


# file names
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
CUSTOM_MODEL_NAME_SUFFIX = 'my_sticky_notes'
CUSTOM_MODEL_NAME = f'{CUSTOM_MODEL_NAME_SUFFIX}_{PRETRAINED_MODEL_NAME}_{str(num_steps)}'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'
LABEL_MAP_NAME = 'label_map.pbtxt'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
PARTITION_DATASET_SCRIPT_NAME = 'partition_dataset.py'
PIPELINE_CONFIG_NAME = 'pipeline.config'

BASE_PATH = 'C:\\_WORK\GitHub\\_data-science'   # necessary for cmd commands

# paths to folders
paths = {
    'OWN_SCRIPTS_PREPROCESSING_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'scripts', 'own-scripts', 'preprocessing'),
    'OWN_SCRIPTS_STICKY_NOTES_DETECTION_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'scripts', 'own-scripts', 'sticky-notes-detection'),
    'TFOD_TUTORIAL_SCRIPTS_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'scripts', 'tfod-tutorial-scripts'),
    'TFOD_API_SCRIPTS_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'scripts', 'tfod-api-scripts'),
    'APIMODEL_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'tensorflow-model-garden'),
    'EXECUTABLES_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'executables'),
    'MIRO_TIMEFRAME_SNAPSHOTS': os.path.join(BASE_PATH, 'Tensorflow', 'miro-timeframe-snapshots'),
    'ANNOTATION_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'annotations'),
    'IMAGE_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'images'),
    'LABELIMG_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'addons', 'labelimg'),
    'MODEL_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'pre-trained-models'),
    'CUSTOM_MODEL_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME),
    'CUSTOM_MODEL_EXPORT_PATH': os.path.join(BASE_PATH, 'Tensorflow', 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME, 'export'),
}

# paths to files
files = {
    'CHROME_WEB_DRIVER': os.path.join(paths['EXECUTABLES_PATH'], 'chromedriver.exe'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'PRETRAINED_MODEL_PIPELINE_CONFIG': os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, PIPELINE_CONFIG_NAME),
    'CUSTOM_MODEL_PIPELINE_CONFIG': os.path.join(paths['CUSTOM_MODEL_PATH'], PIPELINE_CONFIG_NAME),
    'TF_RECORD_SCRIPT': os.path.join(paths['TFOD_TUTORIAL_SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'PARTITION_DATASET_SCRIPT': os.path.join(paths['TFOD_TUTORIAL_SCRIPTS_PATH'], PARTITION_DATASET_SCRIPT_NAME),
    'TRAINING_AND_EVAL_SCRIPT': os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py'),
    'EXPORTING_SCRIPT': os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py')
}
