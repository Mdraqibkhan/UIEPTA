# UIEPTA

Dataset Preparation

Download [UIEB Dataset]([https://drive.google.com/drive/folders/1Ulp-6R91zggImoEdk2X-SQpmk07oIlyZ?usp=sharing](https://li-chongyi.github.io/proj_benchmark.html)) and [EUVP dataset](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3)

Our folder structure is as follows:
uw_data/
│
├── train/
│   ├── a/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── b/
│       ├── image1_gt.jpg
│       ├── image2_gt.jpg
│       └── ...
│
└── test/
    ├── a/
    │   ├── image101.jpg
    │   ├── image102.jpg
    │   └── ...
    └── b/
        ├── image101_gt.jpg
        ├── image102_gt.jpg
        └── ...

Description
train/a/: Contains training input images.
train/b/: Contains corresponding ground truth images for training.
test/Aa: Contains testing input images.
test/b/: Contains corresponding ground truth images for testing.
