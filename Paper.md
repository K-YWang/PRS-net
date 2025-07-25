Abstract—
In geometry processing, symmetry is a universal type of high-level structural information of 3D models and benefits many geometry processing tasks including shape segmentation, alignment, matching, and completion. Thus it is an important problem to analyze various symmetry forms of 3D shapes. Planar reflective symmetry is the most fundamental one. Traditional methods based on spatial sampling can be time-consuming and may not be able to identify all the symmetry planes. In this paper, we present a novel learning framework to automatically discover global planar reflective symmetry of a 3D shape. Our framework trains an unsupervised 3D convolutional neural network to extract global model features and then outputs possible global symmetry parameters, where input shapes are represented using voxels. We introduce a dedicated symmetry distance loss along with a regularization loss to avoid generating duplicated symmetry planes. Our network can also identify generalized cylinders by predicting their rotation axes. We further provide a method to remove invalid and duplicated planes and axes. We demonstrate that our method is able to produce reliable and accurate results. Our neural network based method is hundreds of times faster than the state-of-the-art methods, which are based on sampling. Our method is also robust even with noisy or incomplete input surfaces.

Index Terms— Deep Learning, Symmetry Detection, 3D Models, Planar Reflective Symmetry

摘要—
在几何处理领域，对称性是一种广泛存在的高层次结构信息，能够促进多种几何处理任务的发展，例如形状分割、对齐、匹配和补全。因此，分析三维形状的各种对称形式是一个重要问题。其中，平面反射对称是最基本的一类。传统基于空间采样的方法通常较为耗时，且可能无法识别出所有的对称平面。本文提出了一种新颖的学习框架，用于自动发现三维形状的全局平面反射对称性。该框架采用无监督的三维卷积神经网络来提取全局模型特征，并输出可能的全局对称参数，其中输入形状以体素形式表示。我们引入了一个专门的对称距离损失函数以及一个正则化损失函数，以避免生成重复的对称平面。我们的网络还可以通过预测旋转轴来识别广义圆柱体。我们进一步提出了一种方法来移除无效和重复的平面及轴。实验表明，我们的方法能够生成可靠且精确的结果。与当前基于采样的先进方法相比，我们基于神经网络的方法在速度上提高了数百倍，并且在输入表面存在噪声或不完整的情况下仍具有良好的鲁棒性。

关键词— 深度学习，对称检测，三维模型，平面反射对称

