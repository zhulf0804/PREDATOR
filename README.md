## [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)

An inofficial PyTorch implementation of PREDATOR based on KPConv. 

The code has been tested on Ubuntu 16.4, Python 3.7, PyTorch (1.7.1+cu101),
torchvision (0.8.2+cu101), GCC 5.4.0 and Open3D (0.9 or 0.13).

All experiments were run on a Tesla V100 GPU with an Intel 6133CPU at 2.50GHz CPU.

## Download 3DMatch

We adopted the 3DMatch provided from PREDATOR, and download it from [here](https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip) [**5.17G**].
Unzip it, then we should get the following directories structure:

``` 
| -- indoor
    | -- train (#82, cats: #54)
        | -- 7-scenes-chess
        | -- 7-scenes-fire
        | -- ...
        | -- sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_4
    | -- test (#8, cats: #8)
        | -- 7-scenes-redkitchen
        | -- sun3d-home_md-home_md_scan9_2012_sep_30
        | -- ...
        | -- sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

## Compile python bindings and Reconfigure

```
# Compile

cd PREDATOR/cpp_wrappers
sh compile_wrappers.sh
cd ..


# Reconfigure configs/threedmatch.yaml by updating the following values based on your dataset.

exp_dir: your_saved_path for checkpoints and summary.
checkpoint: your_ckpt_path; it's just required during evaluating and visualizing.
root: your_data_path for the indoor.
```

## Train

```
cd PREDATOR
python train.py
```

## (Optional) Download pretrained weights

Download pretrained weights [[baidu disk](https://pan.baidu.com/s/199PFbEEeCLAwa7QCN93iIQ), 28.36M] with password `0zfl` for the following evaluation and visualization.
 
## Evaluate

```
cd PREDATOR
python evaluate.py
```

## Visualize

```
cd PREDATOR
python vis.py
```

## Results on 3DMatch

| npoints | Inlier Ratio | Feature Match Recall | Registration Recall | Weighted Registration Recall |
| :---: | :---: | :---: | :---: | :---: |
| 5000 | 0.519 | 0.964 | 0.903 | 0.929 |
| 1000 | 0.518 | 0.962 | 0.898 | 0.918 |

**Note**: We calculate `Registration Recall` and `Weighted Registration Recall` based on equation (3) in [PREDATOR Supplementary](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Huang_Predator_Registration_of_CVPR_2021_supplemental.pdf). It's a little different from implementation in [OverlapPredator](https://github.com/overlappredator/OverlapPredator), which is reported in the paper.

## Acknowledgements

Thanks for the open source code [OverlapPredator](https://github.com/overlappredator/OverlapPredator), [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) and [KPConv.pytorch](https://github.com/XuyangBai/KPConv.pytorch).
