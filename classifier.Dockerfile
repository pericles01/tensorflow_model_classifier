FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04:20220516.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/tensorflow-2.4
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8.3 pip=20.2.4 cudatoolkit=10.2 cudnn=7.6.5 imgaug=0.4.0 pandas=1.1.3 scikit-learn=0.23.2

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'matplotlib==3.3.2' \
                'albumentations==0.5.2' \
                'numpy==1.20' \
                'tensorboard==2.4.0' \
                'tensorflow-gpu==2.4.1' \
                'protobuf==3.20.0'\
                'azureml-sdk==1.29.0' \
                'imagehash==4.2.1' \
                'image-classifiers==1.0.0' \
                'tqdm==4.62.2' \
                'opencv-python==4.2.0.34' \
                'tensorflow-addons==0.12.1'\
                'tf-keras-vis' \
                'optuna'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH