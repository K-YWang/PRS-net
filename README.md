# Reproducing PRS-Net

本项目是 `PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models` 论文的代码复现工作，原论文请参考：[PRS-Net](http://geometrylearning.com/prs-net/)。依据文中的给出的网络结构，数据以及参数信息，部分了参考原作者仓库[GithubRepo](https://github.com/IGLICT/PRS-Net)。

## Installation

本项目推荐 Python 3.10 环境，并建议使用 `conda` 创建虚拟环境。

1. **克隆仓库：**

   ```bash
   git clone https://github.com/K-YWang/PRS-net.git
   cd PRS-net
   ```

2. **创建并激活 Python 环境：**

   ```bash
   # 使用 conda
   conda create -n prs python=3.10
   conda activate prs
   ```

3. **安装 PyTorch 和其他依赖：**
   **请访问 PyTorch 官网 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) 获取适合您 CUDA 版本的安装命令。** 例如（CUDA 11.8）：

   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   ```

   然后安装其他依赖：

   ```bash
   pip install -r requirements.txt
   ```

   **注意：** 如果遇到网络问题，可以尝试使用镜像源，例如：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`



## Data

训练使用了 `ShapeNetCore V2` ，原数据集可到huggingface进行下载[[ShapeNet](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)]，但是由于网络，数据集大小(~25G)还有申请下载等原因，获取相对麻烦耗时。我使用的是已经处理好的SNC V2的PointCloud版本，仅0.98G，然后经过处理得到需要的数据形式[[SN-POINTCLOUD](https://github.com/antao97/PointCloudDatasets.git)]。

`data/process_SNC_hdf5.py` 文件实现了处理上述数据集的代码，最终保存为pt文件，会有所需要的体素，label，最近网格等等信息，并且会自动处理使2048个点云涌fps算法均匀处理为1000个point，然后针对没有4k个样本的类别进行随机旋转增强得到4k样本，最后8:2划分为train，test进行存储。

**下载完成之后执行下列命令：**

```bash
unzip shapenetcorev2_hdf5_2048.zip # 可修改解压后的名字
mv data_name data/
cd data && python process_SNC_hdf5.py

# 按照自己修改文件路径
# process_SNC_hdf5.py(
# h5_dir = './ShapeNetCoreV2'
# out_root = './SNC_valid'
# )
```

`data/download_SNC.sh` 中有下载原数据的所有命令，有需要可直接使用，数据处理方式可参考原作者仓库数据处理模块，处理所需的结果请参考上述代码。



## Train

### Config.yaml

针对训练的所有设置都在`./config.yaml`文件中，更改配置请直接修改。

```yaml
# config.yaml
dataset:
  dataroot: ./data/SNC_valid
  train_split: train
  test_split: test

model:
  input_nc: 1
  output_nc: 4
  conv_layers: 5
  num_plane: 3
  num_quat: 3
  use_bn: true
  activation: lrelu

loss:
  grid_size: 32
  grid_bound: 0.5
  ref_weight: 1.0
  rot_weight: 1.0
  reg_plane_weight: 1.0
  reg_rot_weight: 1.0
  w_sym_reg: 25

training:
  batch_size: 32
  epochs: 40
  learning_rate: 0.01
  seed: 42
  device: cuda

output:
  save_dir: ./checkpoints
  log_dir: ./logs

testing:
  checkpoint: ./checkpoints/20250731-214831/00099_net_PRSNet.pth
```

### Training and Inference

```python
(PRS-net/)
python train.py
python test.py
```

所有的日志文件会保存在`logs/`，所有的权重会保存在`checkpoints/`，若两个文件夹不存在也无需额外创建，代码运行时会自动得到。

