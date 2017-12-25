
有两种办法运行代码：使用1.已训练好的模型或者使用2.自己训练的模型

1.用 [Baidu Drive](https://pan.baidu.com/s/1i4GTS4d) 提供的模型，将模型文件下载到项目文件夹中后，在项目文件夹中打开命令窗口，运行以下代码即可生成目标输出图片

```
python eval.py --model_file <your path to wave.ckpt-done> --image_file img/test.jpg
```

输出结果的位置在控制台输出的print语句里显示有

2.我还没试，按照下面对应位置的教程去操作即可

快速风格迁移的速度确实很快，但实际上并没有解决风格迁移的核心目的，只是通过用现有的模型避开“训练”这一步来减少时间。不过速度确实变快了n倍，从接近10分钟减到1秒多一点，而且损失值不大，效果很好。

# fast-neural-style-tensorflow

A tensorflow implementation for [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

This code is based on [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/slim) and [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style).

## Samples:

| configuration | style | sample |
| :---: | :----: | :----: |
| [wave.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_wave.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/wave.jpg)  |
| [cubist.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/cubist.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_cubist.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/cubist.jpg)  |
| [denoised_starry.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/denoised_starry.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_denoised_starry.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/denoised_starry.jpg)  |
| [mosaic.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/mosaic.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_mosaic.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/mosaic.jpg)  |
| [scream.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/scream.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_scream.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/scream.jpg)  |
| [feathers.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/feathers.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_feathers.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/feathers.jpg)  |
| [udnie.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/udnie.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_udnie.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/udnie.jpg)  |

## Requirements and Prerequisites:
- Python 2.7.x
- <b>Now support Tensorflow >= 1.0</b>

<b>Attention: This code also supports Tensorflow == 0.11. If it is your version, use the commit 5309a2a (git reset --hard 5309a2a).</b>

And make sure you installed pyyaml:
```
pip install pyyaml
```

## Use Trained Models:

You can download all the 7 trained models from [Baidu Drive](https://pan.baidu.com/s/1i4GTS4d).

To generate a sample from the model "wave.ckpt-done", run:

```
python eval.py --model_file <your path to wave.ckpt-done> --image_file img/test.jpg
```

Then check out generated/res.jpg.

## Train a Model:
To train a model from scratch, you should first download [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) from Tensorflow Slim. Extract the file vgg_16.ckpt. Then copy it to the folder pretrained/ :
```
cd <this repo>
mkdir pretrained
cp <your path to vgg_16.ckpt>  pretrained/
```

Then download the [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). Please unzip it, and you will have a folder named "train2014" with many raw images in it. Then create a symbol link to it:
```
cd <this repo>
ln -s <your path to the folder "train2014"> train2014
```

Train the model of "wave":
```
python train.py -c conf/wave.yml
```

(Optional) Use tensorboard:
```
tensorboard --logdir models/wave/
```

Checkpoints will be written to "models/wave/".

View the [configuration file](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) for details.
