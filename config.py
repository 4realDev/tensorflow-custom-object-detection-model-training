r""" Changing something on the config please re-run all the scripts to create a new workspace for the new model

    0.	Activate virtual environment
    cd [ROOT_PATH]\\venv
    .\tfod-sticky-notes\Scripts\activate

    1.	Adjust the “config.py” file in [ROOT_PATH] according to your needs

    2.	Run the “create_model_workspace.py” script to setup the folder structure, create the label map, download and extract the pre-trained model, copy the pipeline.config file from the pre-trained model into the custom model and adjust the pipeline.config
    python [ROOT_PATH]\create_model_workspace.py

    3.	Collect images and label them with labelImg to create their XML file
    python [ROOT_PATH]\addons\labelImg\labelImg.py

    4.	Run the “prepare_image_data_for_training.py” script to partition the image data into test and train and to create the tf.records for test.record and train.record out of the labeled image data
    python [ROOT_PATH]\prepare_image_data_for_training.py

    5.	Run the “train_custom_model.py” script to train the model
    python [ROOT_PATH]\train_custom_model.py

    6.	Run the “eval_custom_model.py” script to eval the model
    python [ROOT_PATH]\eval_custom_model.py

    7.	Run the “run_tensorboard_on_custom_model.py” script to start TensorBoard servor for custom model
    python [ROOT_PATH]\run_tensorboard_on_custom_model.py

    8.	Run the “export_custom_model.py” script to export the custom model into the export folder within the custom models workspace
    python [ROOT_PATH]\export_custom_model.py
    
"""

import os
# labels the custom model detects
# labels = [{'name': 'sticky_note', 'id': 1}]

labels = [{'name': 'yellow_sticky_note', 'id': 1, 'color': 'yellow'},
          {'name': 'blue_sticky_note', 'id': 2, 'color': 'blue'},
          {'name': 'pink_sticky_note', 'id': 3, 'color': 'red'},
          {'name': 'green_sticky_note', 'id': 4, 'color': 'green'}]


# vars necessary for pipeline.config adjustments and for model training
num_classes = len(labels)
batch_size = 4
# num_steps = 2500
# num_steps = 10000
num_steps = 25000
fine_tune_checkpoint_type = "detection"  # "classification" or "detection"

# vars for real time object detection
min_score_thresh = 0.75
bounding_box_and_label_line_thickness = 5

# file names
# PRETRAINED_MODEL_NAME = 'efficientdet_d7_coco17_tpu-32'
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
# CUSTOM_MODEL_NAME_SUFFIX = 'my_sticky_notes_no_color'
# CUSTOM_MODEL_NAME_SUFFIX = 'updated_basic_data_set'
# CUSTOM_MODEL_NAME_SUFFIX = '2_own_min_data_no_color'
CUSTOM_MODEL_NAME_SUFFIX = '3_own_min_data'
CUSTOM_MODEL_NAME = f'{CUSTOM_MODEL_NAME_SUFFIX}_{PRETRAINED_MODEL_NAME}_{str(num_steps)}'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'
LABEL_MAP_NAME = 'label_map.pbtxt'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
PARTITION_DATASET_SCRIPT_NAME = 'partition_dataset.py'
PIPELINE_CONFIG_NAME = 'pipeline.config'

BASE_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)))  # necessary for cmd commands

# paths to folders
paths = {
    'TFOD_TUTORIAL_SCRIPTS_PATH':       os.path.join(BASE_PATH, 'scripts', 'tfod-tutorial-scripts'),
    'TFOD_API_SCRIPTS_PATH':            os.path.join(BASE_PATH, 'scripts', 'tfod-api-scripts'),
    'APIMODEL_PATH':                    os.path.join(BASE_PATH, 'tensorflow-model-garden'),
    'MIRO_TIMEFRAME_SNAPSHOTS':         os.path.join(BASE_PATH, 'miro-timeframe-snapshots'),
    'ANNOTATION_PATH':                  os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'annotations'),
    'IMAGE_PATH':                       os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'images'),
    'PRETRAINED_MODEL_PATH':            os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'pre-trained-models'),
    'CUSTOM_MODEL_PATH':                os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME),
    'CUSTOM_MODEL_EXPORT_PATH':         os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME, 'export'),
    'CUSTOM_MODEL_TFJS_EXPORT_PATH':    os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
}

# paths to files
files = {
    'LABELMAP':                         os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'PRETRAINED_MODEL_PIPELINE_CONFIG': os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, PIPELINE_CONFIG_NAME),
    'CUSTOM_MODEL_PIPELINE_CONFIG':     os.path.join(paths['CUSTOM_MODEL_PATH'], PIPELINE_CONFIG_NAME),
    'TF_RECORD_SCRIPT':                 os.path.join(paths['TFOD_TUTORIAL_SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'PARTITION_DATASET_SCRIPT':         os.path.join(paths['TFOD_TUTORIAL_SCRIPTS_PATH'], PARTITION_DATASET_SCRIPT_NAME),
    'TRAINING_AND_EVAL_SCRIPT':         os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py'),
    'EXPORTING_SCRIPT':                 os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py'),
}
