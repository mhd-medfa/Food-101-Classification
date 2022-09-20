# Food-101-Classification
Our aim is to build and train a Food-101 deep
learning classifier using Convolutional Neural Networks (CNN)
with a reasonable set of Augmentations. This project explores
food image classification with convolutional neural networks
(CNNs) for better image labeling by dish, which in turn may
improve the recommendation and search flows for a better digital food user experience overall. Specifically, the goal of the project is to, given an image of a dish as the input to the model; output the correct label categorization of the food image.

![](https://i.ibb.co/jbrBxr6/image.png)

A total of 101,000 images from 101 classes of food were
used from the Food-101 dataset, with 1000 images for each
class. Of the 1000 images for each class, 250 were manually
reviewed test images, and 750 were intentionally noisy training
images, for a total training data size of 75,750 training images
and 25,250 test images.

![](https://i.ibb.co/DV1rMt0/image.png)

### Results
![](https://i.ibb.co/phhBJrg/image.png)

### How To Run

To run this solution:

**Conda**
1. `$ conda create --name <envname> --file requirements.txt`
2. `$ source activate <envname>`
3. `$ python predict.py`

**Docker**
1. `$ docker build -t food_101_cls .`
2. `$ docker run --name food_101_cls -d food_101_cls`

### List of Contributors:
- Mohamad Al Mdfaa   m.almdfaa@innopolis.university
- Amer Al Badr       a.albadr@innopolis.university
