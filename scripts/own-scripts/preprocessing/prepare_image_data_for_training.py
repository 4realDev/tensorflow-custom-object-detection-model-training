import os
import config

files = config.files
paths = config.paths


def partition_dataset_into_test_and_train():
    os.system(
        f"python        {files['PARTITION_DATASET_SCRIPT']} " +
        f"--xml " +
        f"--imageDir    {paths['IMAGE_PATH']} " +
        f"--ratio       {0.1}")


def create_tf_records():
    # Create train data:
    os.system(
        f"python        {files['TF_RECORD_SCRIPT']} " +
        f"--xml_dir     {os.path.join(paths['IMAGE_PATH'], 'train')} " +
        f"--labels_path {files['LABELMAP']} " +
        f"--output_path {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} ")

    # Create test data:
    os.system(
        f"python        {files['TF_RECORD_SCRIPT']} " +
        f"--xml_dir     {os.path.join(paths['IMAGE_PATH'], 'test')} " +
        f"--labels_path {files['LABELMAP']} " +
        f"--output_path {os.path.join(paths['ANNOTATION_PATH'], 'test.record')} ")


def main():
    partition_dataset_into_test_and_train()
    create_tf_records()


if __name__ == '__main__':
    main()
