# MLMCU

From the course: Machine Learning on Microcontrollers at ETH Zurich

The code follows the description of [Maxim Integrated AI Development](https://github.com/MaximIntegratedAI/ai8x-training.git) for implementing a convolutional neural network on the MAX78000FTHR from Maxim Integrated. The dataset is based on the Intel Scene Classification Challenge which was released by Analytics Vidhya in collaboration with Intel. The dataset includes 25’000 Images with 6 Categories: Buildings, Forest, Glacier, Mountain, Sea, Street and can be downloaded from [kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). 


<p align="center">
  <img src=example_images.png alt= “dataset_samples” width="50%" height="30%">
</p>


Included in the repository is:


| File Name               	| Description                                                               	|
|-------------------------	|---------------------------------------------------------------------------	|
| 'ASCII_TO_IMG.ipynb'    	| Creates a 2D image in color based on ASCII characters                     	|
| 'Create_Sample.ipynb'   	| Picks an image from the dataset and create a HEX image needed for the MCU 	|
| 'Sample_Creator.py'     	| Helper function for 'Create_Sample.ipynb'                                 	|
| 'Keras_Network.ipynb'   	| Built and tested the network using Keras                                  	|
| 'Pytorch_Network.ipynb' 	| Built and tested the network using Pytorch                                	|
| 'Samples'               	| 6 images and their corresponding HEX images                               	|
| 'ai8x-training'         	| Necessary training files for the MAX8700 FEATHER BOARD                    	|
| 'ai8x-synthesis'        	| Necessary synthesis files for the MAX8700 FEATHER BOARD                   	|
| 'intelnet'              	| C-code files that are loaded on the MCU                                   	|

## Notes

The C-code files in 'intelnet' also includes code to use the camera on the board. Alternatively, a HEX image can be loaded directly without using the camera
