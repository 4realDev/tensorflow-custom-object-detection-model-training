import os   # Import Operating System
import config

paths = config.paths
files = config.files

def open_tensorboard():
    os.system(
        f"tensorboard " +
        f"--logdir {paths['CUSTOM_MODEL_PATH']}")

def main():
    open_tensorboard()

if __name__ == '__main__':
    main()
