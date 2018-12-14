# IVD-Net: Intervertebral disc localization and segmentation in MRI with a multi-modal UNet

Repository containing the source code of the IVD-Net segmentation network that we proposed for the MICCAI 2018 IVD segmentation challenge. This architecture was used to segment the inetervertebral disc in multi-modal MRI images. Nevertheless, it can be used to segment any other structure/s in a multi-modal image setting.

## Requirements

- The code has been written in Python (3.5) and pyTorch.
- You should also have installed torchvision and scipy

## Running the code

### Pre-processing
If you just want to run the IVD dataset with this code, you simply need to convert the 3D volumes to 2D slices/images. Then, the structure to save the images should be (note that val and test folders will have the same structure than train):

 .
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...
    
MainFolder/
----| train/
--------| Fat/
------------| ImgName_xxxx0.png
------------| ImgName_xxxx1.png
------------| ....
--------| GT/
------------| ImgName_xxxx0.png
------------| ImgName_xxxx1.png
------------| ....
--------| Inn/
------------| ImgName_xxxx0.png
------------| ImgName_xxxx1.png
------------| ....
--------| Opp/
------------| ImgName_xxxx0.png
------------| ImgName_xxxx1.png
------------| ....
--------| Wat/
------------| ImgName_xxxx0.png
------------| ImgName_xxxx1.png
------------| ....
----| Val/
----| Test/

Then you simply have to write in the command line:

```
python main.py 
```

## Some results from our paper

<br>
<img src="https://github.com/josedolz/IVD-Net/Images/IVD_Results.png" />
<br>


<br>
<img src="https://github.com/josedolz/IVD-Net/Images/3D.png"/>
<br>
If you use this code for your research, please consider citing the original paper:

- Dolz J, Desrosiers C, Ben Ayed I. [IVD-Net: Intervertebral disc localization and segmentation in MRI with a multi-modal UNet.](https://arxiv.org/abs/1811.08305) arXiv preprint arXiv:1811.08305. 2018 Nov 19.
