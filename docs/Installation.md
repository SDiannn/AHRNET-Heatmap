###### *Note: We follow the guideline provided by [MeshTransformer/docs/INSTALL.md](https://github.com/microsoft/MeshTransformer/blob/main/docs/INSTALL.md)*

# Installation


- Python 3.8
- Pytorch 1.12.0
- torchvision 0.13.0
- cuda 11.3

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name AHRNET python=3.8
conda activate fastmetro
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install requirements
pip install -r requirements.txt

# Install manopth
pip install ./manopth/.
```

---