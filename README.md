# Criminal Detection

Our system is built to detect criminals and identify them using a pretrained yolo model and the Siamese netowrk. 

## Design Choices:
- The yolo algorithm used is a pretrained YOLOv8 model for face detection.
- Siamese network with ResNet18 as its main core architecture implemented using pyTorch.


## Setting up the dataset.
The expected format for both the training and validation dataset is the same. Image containing a certain person should be placed in a folder specified for that perosn. The folders for every person are then to be placed within a common root directory (which will be passed to the trainined and evaluation scripts) and it has to be called "archive (2)". The folder structure is also explained below:
```
|--archive (2)
  |--Person1
    |-Image1
    |-Image2
    .
    .
    .
    |-ImageN
  |--Person2
  |--Person3
  .
  .
  .
  |--PersonN
```
[Click here to see the dataset used](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) 


## Setting up environment.
The provided setup instructions assume that anaconda is already installed on the system. To set up the environment for this repository, run the following commands to create and activate an environment named 'test'.:
```
conda create -n test python=3.9
conda activate test
```


## Run the website:
To run the website locally , run the following commands:
```
pip install -r requirements.txt
streamlit run main_page.py
```


