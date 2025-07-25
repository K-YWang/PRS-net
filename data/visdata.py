import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_voxel_from_mat(mat_path, save_path=None):
    # 加载 .mat 文件
    mat = scipy.io.loadmat(mat_path)
    volume = mat['Volume']  # 体素网格 [32, 32, 32]

    # 获取非零体素的位置
    xs, ys, zs = np.nonzero(volume > 0)

    # 创建图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='s', alpha=0.6, s=20, c='black')

    ax.set_title(f"Voxel Visualization of {os.path.basename(mat_path)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    plt.tight_layout()

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()

# 示例调用：指定 .mat 文件路径和保存路径
visualize_voxel_from_mat(
    'converted_mat_data/test0_pc_00000.mat',
    save_path='voxel_visualization.png'
)
