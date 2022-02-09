# Image Colourization using Conditional Generative Adversarial Networks

## Image Colourization
* Final project for the [Machine Learning](https://www.rug.nl/ocasys/fwn/vak/show?code=WMAI010-05) Master's course at University of Groningen
* The focus is on Colourization task using Conditional Generative Adversarial Networks
* Using pre-trained ResNet and UNet

## Instructions to run the code
* For running the train script use. Use `--help` to list all the commandline options.
```
python3 src/train.py --help
```
* After training, use the following script to compute evaluation metrics on the test set. Use `--help` to list all the commandline options.
* After training, use the following script to colourize the images using the Generator network. Use `--help` to list all the commandline options.
```
python3 src/generate_results.py --help
```
* Use notebook [src/plot_train_losses.ipynb](src/plot_train_losses.ipynb) to visualize training losses
* Use notebook [src/visualize_results.ipynb](src/visualize_results.ipynb) to visualize the generated results from Generator and compare it with the original images.

## Weights file
* [The weights file is available](https://drive.google.com/drive/folders/1i_MPfA2EB_BibkrmkHiHErcznkiZ6BA8)

## Dependencies
* The dependencies are available in the [requirements.txt](requirements.txt)

## Reference
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004v3)
