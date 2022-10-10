import os   # Import Operating System
import wget  # To download tensorflow pretrained model from github repo
import tensorflow as tf

# text_format and pipeline_pb2 needed for overwritting the pipeline_config file
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

import config

paths = config.paths
files = config.files
labels = config.labels
PRETRAINED_MODEL_NAME = config.PRETRAINED_MODEL_NAME
PRETRAINED_MODEL_URL = config.PRETRAINED_MODEL_URL

num_classes = config.num_classes
batch_size = config.batch_size
num_steps = config.num_steps
fine_tune_checkpoint_type = config.fine_tune_checkpoint_type


def setup_workspace_folders():
    for path in paths.values():
        if not os.path.exists(path) and "workspace" in path:
            if os.name == 'posix':
                os.makedirs(path)
            if os.name == 'nt':
                os.makedirs(path)


def partition_dataset_into_test_and_train():
    os.system(
        f"python        {files['PARTITION_DATASET_SCRIPT']} " +
        f"--xml " +
        f"--imageDir    {paths['IMAGE_PATH']} " +
        f"--ratio       {0.1}")


def create_label_map():
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tid: {}\n'.format(label['id']))
            f.write('\tname: \'{}\'\n'.format(label['name']))
            f.write('}\n\n')
        print("Successfully created the label_map.pbtxt file")


def download_and_extract_pretrained_model():
    if os.name == 'posix':
        print(
            f"Downloading and extracting {PRETRAINED_MODEL_NAME + '.tar.gz'} file into {paths['PRETRAINED_MODEL_PATH']}.")
        os.system(f"wget f{PRETRAINED_MODEL_URL}")
        os.system(
            f"mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
        os.system(
            f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")
        print(
            f"Successfully downloaded and extracted {PRETRAINED_MODEL_NAME + '.tar.gz'} file into {paths['PRETRAINED_MODEL_PATH']}.")

    if os.name == 'nt':
        print(
            f"Downloading and extracting {PRETRAINED_MODEL_NAME + '.tar.gz'} file into {paths['PRETRAINED_MODEL_PATH']}.")
        wget.download(PRETRAINED_MODEL_URL)
        os.system(
            f"move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
        os.system(
            f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")
        print(
            f"Successfully downloaded and extracted {PRETRAINED_MODEL_NAME + '.tar.gz'} file into {paths['PRETRAINED_MODEL_PATH']}.")


def copy_source_file_into_dest_folder(source: str, dest: str):
    if os.name == 'posix':
        if not os.path.exists(dest):
            os.makedirs(dest)
        os.system(
            f"cp {source} {dest}")
        print(
            f"Successfully copied {source} file into {dest}.")

    if os.name == 'nt':
        if not os.path.exists(dest):
            os.makedirs(dest)
        os.system(
            f"copy {source} {dest}")
        print(
            f"Successfully copied {source} file into {dest}.")


def adjust_pipeline_config_for_custom_model():
    # 1. Get pipeline_config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['CUSTOM_MODEL_PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # 2. Adjust pipeline_config
    # Set number of classes.
    pipeline_config.model.ssd.num_classes = num_classes

    # Set batch size
    # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
    # !!! in Tensorflow Guide: 8 !!!
    pipeline_config.train_config.batch_size = batch_size

    # Set training steps
    # !!! in Tensorflow Guide: 25000 !!!
    pipeline_config.train_config.num_steps = num_steps

    # Set train tf-record file path
    # train_tf_record_file_path = re.escape(
    #     os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'train.record')]

    # Set test tf-record file path
    # test_tf_record_file_path = re.escape(
    #     os.path.join(paths['ANNOTATION_PATH'], 'test.record'))
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    # Set fine_tune_checkpoint path
    # fine_tune_checkpoint = re.escape(os.path.join(
    #     paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0'))
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
        paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')

    # Set fine_tune_checkpoint_type to "detection"
    # Use "detection" since we want to be training the full detection model
    pipeline_config.train_config.fine_tune_checkpoint_type = fine_tune_checkpoint_type

    # Set labelmap path
    # labelmap_path = re.escape(files['LABELMAP'])
    pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']

    config_text = text_format.MessageToString(pipeline_config)

    # 3. Overwrite pipeline_config file with adjusted version
    with tf.io.gfile.GFile(files['CUSTOM_MODEL_PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

    print(
        f"Successfully adjusted {files['CUSTOM_MODEL_PIPELINE_CONFIG']} file in {paths['CUSTOM_MODEL_PATH']}.")


def main():
    setup_workspace_folders()

    create_label_map()

    download_and_extract_pretrained_model()

    # copy pretrained model pipeline config into custom model to adjust it for custom training
    copy_source_file_into_dest_folder(
        files['PRETRAINED_MODEL_PIPELINE_CONFIG'], paths['CUSTOM_MODEL_PATH'])

    adjust_pipeline_config_for_custom_model()

    # copy TensorFlow\models\research\object_detection\model_main_tf2.py into TensorFlow\scripts for model training and evaluation
    copy_source_file_into_dest_folder(
        files['TRAINING_AND_EVAL_SCRIPT'], paths['TFOD_API_SCRIPTS_PATH'])
    
    # copy TensorFlow\models\research\object_detection\export_main_tf2.py into TensorFlow\scripts for model export
    copy_source_file_into_dest_folder(
        files['EXPORTING_SCRIPT'], paths['TFOD_API_SCRIPTS_PATH'])


if __name__ == '__main__':
    main()
