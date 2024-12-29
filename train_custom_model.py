import os   # Import Operating System
import config
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Physical Devices: {len(physical_devices)}")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


paths = config.paths
files = config.files


def train_custom_model():
    os.system(
        f"python                    {files['TRAINING_AND_EVAL_SCRIPT']} " +
        f"--model_dir               {paths['CUSTOM_MODEL_PATH']} " +
        f"--pipeline_config_path    {files['CUSTOM_MODEL_PIPELINE_CONFIG']} ")


def main():
    train_custom_model()


if __name__ == '__main__':
    main()
