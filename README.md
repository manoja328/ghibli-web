# ghibli-web
ghibli web app with scraper to scraper from twitter using a chrome extension

Code for chrome extension is used from this repo:
https://github.com/faisalsayed10/no-ghibli

## Contributions:

- Model in pytorch
- onnx support


## Training the Model

1. Create a dataset directory structure:
```
dataset/
├── Ghibli/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── Non_Ghibli/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

2. Place your training images in the appropriate directories:
   - Put Ghibli-style images in the `dataset/Ghibli` directory
   - Put non-Ghibli images in the `dataset/Non_Ghibli` directory

3. Train the model:
```bash
python train.py
```
