###### *Note: We follow the guideline provided by [MeshTransformer/docs/DOWNLOAD.md](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md)*

# Download

## Getting Started

1.Download SMPL and MANO models

    To run our code smoothly, please visit the following websites to download SMPL and MANO models.
    - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${DIR}/src/modeling/data`.

    Please put the downloaded files under the `${DIR}/src/modeling/data` directory. The data structure should follow the hierarchy below. 
    ```
    ${DIR}  
    |-- models
    |-- src  
    |   |-- modeling
    |   |   |-- data
    |   |   |   |-- MANO_RIGHT.pkl
    |-- datasets
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
    Please check [/src/modeling/data/README.md](../src/modeling/data/README.md) for further details.


2.Download datasets and pseudo labels for training.

    We recommend to download large files with **AzCopy** for faster speed.
    AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy). Decompress the azcopy tar file and put the executable in any path. 

    To download the annotation files, please use the following command.
    ```bash
    cd $DIR
    path/to/azcopy copy 'https://datarelease.blob.core.windows.net/metro/datasets/filename.tar' /path/to/your/folder/filename.tar
    tar xvf filename.tar  
    ```
    `filename.tar` could be `Tax-H36m-coco40k-Muco-UP-Mpii.tar`, `human3.6m.tar`, `coco_smpl.tar`, `muco.tar`, `up3d.tar`, `mpii.tar`, `3dpw.tar`, `freihand.tar`. Total file size is about 200 GB. 

    The datasets and pseudo ground truth labels are provided by [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE). We only reorganize the data format to better fit our training pipeline. We suggest to download the orignal image files from the offical dataset websites.

    The `datasets` directory structure should follow the below hierarchy.
    ```
    ${ROOT}
    |-- src
    |-- datasets
    |   |-- freihand
    |   |   |-- train.yaml
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- train.label.tsv
    |   |   |-- train.linelist.tsv
    |   |   |-- test.yaml
    |   |   |-- test.img.tsv  
    |   |   |-- test.hw.tsv   
    |   |   |-- test.label.tsv
    |   |   |-- test.linelist.tsv
    |-- README.md 
    |-- ... 
    |-- ... 

    ```
