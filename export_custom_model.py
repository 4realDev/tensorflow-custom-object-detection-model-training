import os   # Import Operating System
import config

paths = config.paths
files = config.files


def export_custom_model():
    os.system(
        f"python                    {files['EXPORTING_SCRIPT']} " +
        f"--input_type              image_tensor  " +
        f"--pipeline_config_path    {files['CUSTOM_MODEL_PIPELINE_CONFIG']} " +
        f"--trained_checkpoint_dir  {paths['CUSTOM_MODEL_PATH']} " +
        f"--output_directory        {paths['CUSTOM_MODEL_EXPORT_PATH']}")


def main():
    export_custom_model()


if __name__ == '__main__':
    main()
