# LISK pytorch version

## Enviroment

- python 3.6
- opencv 3.4.2
- Pytorch 1.0.1
- scipy <=1.2
- scikit-image 0.14.3
- tensorboard
- tensorboardX
- tqdm 
- pyyaml 


## Codes

This repository provides a pytorch-based implementation of [LISK] introduced by Yang et. al.
The code built based on [EdgeConnect].

If you have any questions about the code logic, you can refer to the source code for more detailed information.

| File | logic |
| ------ | ------ |
| src/dataset.py | Dataloader: RGB image(3); mask(1); layout(1); layout instance(100) |
| src/config.py | Configuration logic. |
| src/SIInpainting.py | Model operation: train; eval; test; sample... |
| src/model.py | Network operation: inference; backpropogation...|
| src/network.py | Build the neural network architecture. |
| src/loss.py | Loss function defination.|
| src/ops.py | Convolution layer; attention modul... |
| src/resnet.py | Resnet.|
| src/util.py | Some image processing/visualization  tools(refer to [EdgeConnect]). |
| src/scripts.py | Weight conversion script between Tensorflow and PyTorch.|
| config.yml.example | Configuration template file. |
| main.py | Operation interface. |



## Testing/training

The testing can be performed directly by executing the following commands or define the testing dataset path in `config.yaml`.
```sh
python main.py --checkpoint <checkpoint_dir> --input <input dir or file> --mask <mask dir or file> --output <output dir> --dubug <optional>
```

If you want train/finetune models, You need to modify the `main.py` (Uncomment the comment part).
The training process following the setting of 'config.yml.example'., you need to set the path of the training dataset and the relevant hyperparameters.
Or you can execute the training command first and then stop instantly, then modify `config.yaml` in the checkpoint folder(automatically created after executing the training command), finally execute the command to continue training.

```sh
python main.py --checkpoint <checkpoint_dir>
```

## Pretrained model
Download link:
https://drive.google.com/drive/folders/1DwqL-aoPJsx8aLuREzXV_glDSP5dNCHD?usp=sharing

Place according to the location of the folder.

[//]: # ()

   [EdgeConnect]: <https://github.com/knazeri/edge-connect>
   [LISK]: <https://github.com/YoungGod/sturcture-inpainting>
   [SEAN]: <https://github.com/ZPdesu/SEAN>
   [Partial convolution]: <https://github.com/NVIDIA/partialconv>
   [Structured3D]: <https://github.com/bertjiazheng/Structured3D>
   [HorizonNet]: <https://github.com/sunset1995/HorizonNet>
