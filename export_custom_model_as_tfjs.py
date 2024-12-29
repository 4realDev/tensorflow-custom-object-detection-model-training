# needs the execution of export_custom_model.py first

import os   # Import Operating System
import config

paths = config.paths
files = config.files


def export_custom_model():
    # os.system(
    #     f"python                    {files['EXPORTING_SCRIPT']} " +
    #     f"--input_type              image_tensor  " +
    #     f"--pipeline_config_path    {files['CUSTOM_MODEL_PIPELINE_CONFIG']} " +
    #     f"--trained_checkpoint_dir  {paths['CUSTOM_MODEL_PATH']} " +
    #     f"--output_directory        {paths['CUSTOM_MODEL_EXPORT_PATH']}")

    os.system(
        f"tensorflowjs_converter \
            --input_format=tf_saved_model \
            --output_format=tfjs_graph_model \
            --signature_name=serving_default \
            {os.path.join(paths['CUSTOM_MODEL_EXPORT_PATH'], 'saved_model')} \
            {paths['CUSTOM_MODEL_TFJS_EXPORT_PATH']}")

    # --output_node_names= \
    #     'detection_boxes, \
    #     detection_classes, \
    #     detection_features, \
    #     detection_multiclass_scores, \
    #     detection_scores, \
    #     num_detections, \
    #     raw_detection_boxes, \
    #     raw_detection_scores' \


def main():
    export_custom_model()


if __name__ == '__main__':
    main()
