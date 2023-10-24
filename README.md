# Dual Cross-current Neural Networks (DCNN)

## Install

First, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training and test
### Training
To train DCNN: All the results were evaluated on a system with a NVIDIA GeForce RTX 4090 GPU with 24 GB of memory, running on Ubuntu20.01 (64-bit).

Firstly, modify the --data-path parameter in run_train_dcnn.sh to the root path of ImageNet1k
```
--data-path /path/to/imagenet/
```
Then run the follow scripts to train DCNN.
```
bash run_train_dcnn.sh
```
The trained models will be saved in the OUTPUT folder, OUTPUT is defined in run_train_dcnn.sh

### Testing
Modify the --data-path parameter in run_test_dcnn.sh to the root path of ImageNet1k
```
--data-path /path/to/imagenet/
```
Then test DCNN on ImageNet on a single gpu run:
```
bash run_test_dcnn.sh
```

### Few-Shot Image Classification
We provide four datasets to finetune the base model, [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Food101](https://www.kaggle.com/datasets/dansbecker/food-101/download?datasetVersionNumber=1), [oxfordFlowers](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/download?datasetVersionNumber=3) and [StanfordDogs](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/download?datasetVersionNumber=2), dowdload them and put them in anywhere like /path/to/dataset

Run the following commands separately.
```
bash run_dcnn_CUB200.sh
bash run_dcnn_Food101.sh
bash run_dcnn_oxfordFlowers.sh
bash run_dcnn_StanfordDogs.sh
```
Before running, modify the --data-set parameter to your dataset path.
```
--data-path /path/to/dataset/
```

For training, comment out the code of the Train section.

For validation, comment out the code of the Inference section.



# Proposed model evaluation on ImageNet1k
|  Model  | Parameters | MACs   | Top-1 Acc | Link |
| ------- | ---------- | ------ | --------- | ---- |
|  DCNN   |  52.68 M   | 11.0 G |   82.3 %  | release soon |


# DCNN Test Results on fine-grained benchmarks
|       Datasets    | Top-1 Acc | Top-5 Acc |
| ------------------| ----------| --------- |
| Oxford Flowers102 |   97.3%   |   99.1%   |
|       Food101     |   84.4%   |   96.2%   |
|    Stanford Dogs  |   87.4%   |   98.7%   |
|     CUB200-2011   |   70.5%   |   91.3%   |