# UIEPTA

Dataset Preparation

Download [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) and [EUVP dataset](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3)



# Folder Structure

uw_data/: This is the main directory containing all underwater image datasets.

train/: This subdirectory holds all training data.

a/: This folder contains the input images (degraded underwater images) used for training your model.
b/: This folder contains the reference images (ground truth images) that correspond to the input images, used to evaluate the model's performance during training.
test/: This subdirectory holds all testing data.

a/: This folder contains the input images used for testing the trained model.
b/: This folder contains the reference images that correspond to the test input images, used to assess the quality of the model's outputs.


# Training Instructions

To train the model, execute the following command in your terminal:

python train.py


## Citation
If you found this work helpful, please cite it using the following reference:

[UIEPTA](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=RdE9ayMAAAAJ&citation_for_view=RdE9ayMAAAAJ:u5HHmVD_uO8C)

@inproceedings{khan2023underwater, title={Underwater Image Enhancement with Phase Transfer and Attention}, author={Khan, MD Raqib and Kulkarni, Ashutosh and Phutke, Shruti S and Murala, Subrahmanyam}, booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, pages={1--8}, year={2023}, organization={IEEE}})

For any inquiries or further information, please contact me at srk689800@gmail.com.
