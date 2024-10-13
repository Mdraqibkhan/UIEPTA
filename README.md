# UIEPTA

Dataset Preparation

Download [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) and [EUVP dataset](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3)



# Folder Structure



- **`uw_data/`**: This is the main folder that contains all the underwater image datasets.

- **`train/`**: This folder is where all the training data is stored.

  - **`a/`**: This folder holds the input images, which are the degraded underwater images used to train your model.

  - **`b/`**: This folder contains the reference images, also known as ground truth images. These images are used to evaluate how well the model performs during training.

- **`test/`**: This folder contains all the testing data.

  - **`a/`**: This folder has the input images that will be used to test the trained model.

  - **`b/`**: This folder holds the reference images that correspond to the test input images. These images are used to check the quality of the model's outputs. 



# Training Instructions

To train the model, execute the following command in your terminal:

python train.py


## Citation
If you found this work helpful, please cite it using the following reference:

[UIEPTA](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=RdE9ayMAAAAJ&citation_for_view=RdE9ayMAAAAJ:u5HHmVD_uO8C)
```
@inproceedings{khan2023underwater, title={Underwater Image Enhancement with Phase Transfer and Attention}, author={Khan, MD Raqib and Kulkarni, Ashutosh and Phutke, Shruti S and Murala, Subrahmanyam}, booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, pages={1--8}, year={2023}, organization={IEEE}})
```
For any inquiries or further information, please contact me at srk689800@gmail.com.
