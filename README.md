# Diff-HazeNet
# Dataset
We use datasets in our experiments:
(1)I-Haze:[click here..](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)

(2)O-Haze:[click here..](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)

(3)BeDDE:[click here..](https://github.com/xiaofeng94/BeDDE-for-defogging)

(4)RTTS:[click here..](https://utexas.app.box.com/s/2yekra41udg9rgyzi3ysi513cps621qz)

## Dataset format

    datasets      
    │─── datasets-name
        │─── train
            │─── clear
                │─── 1.jpg
            │─── haze
        │─── test
            │─── clear
            │─── haze
        │─── train_list.txt


# Environment
Python 3.10 and PyTorch 2.2.2 installed, and is configured with CUDA 12.6 to support GPU acceleration. A single NVIDIA RTX 4090 graphics card is utilized in the experiment.
