---
layout: post
title: "Types of Machine Learning Systems (ongoing)"
subtitle:
excerpt_image: /assets/images/ml/types-machine-learning.png
author: Hakuna
categories: [Module]
tags: [machine learning]
top: 
sidebar: []
datatable: true
---

机器学习系统有很多不同的类型，根据以下几个标准可以将它们分成几大类：

- **是否在人类的监督下进行训练**：这包括了监督学习（supervised）、无监督学习（unsupervised）、半监督学习（semisupervised）和强化学习（Reinforcement Learning）。简单来说，监督学习是指我们给模型提供了输入和期望的输出，让模型学会如何从输入映射到输出；无监督学习则是在没有明确输出的情况下让模型自我学习，寻找数据的内在结构；半监督学习介于监督学习和无监督学习之间，使用的是部分标记的数据；而强化学习是让模型通过试错的方式自我学习，根据行为的结果来调整行为策略。

<div align="center">

|supervised|unsupervised|
| ---------| -----------|
|k-Nearest Neighbors|K-Means|
|Linear Regression|DBSCAN|
|Logistic Regression|Hierarchical Cluster Analysis (HCA)|
|Support Vector Machines (SVMs)|One-class SVM|
|Decision Trees and Random Forests|Isolation Forest|
|Neural networks|Principal Component Analysis (PCA)|
||Kernel PCA|
||Locally Linear Embedding (LLE)|
||t-Distributed Stochastic Neighbor Embedding (t-SNE)|
||Apriori|
||Eclat|
</div>

- **是否能够即时增量学习**：这指的是在线学习（online）与批量学习（batch）。在线学习指的是模型能够连续学习，逐步接收数据流进行训练和调整；而批量学习则是指模型在接收到所有训练数据后进行一次性学习。在线学习适用于数据量巨大或持续变化的情况，而批量学习适合于一次性处理静态数据集。

- **工作方式是比较新数据点与已知数据点，还是通过检测训练数据中的模式并构建预测模型**：这涉及到基于实例的学习（instance-based）与基于模型的学习（model-based）。基于实例的学习简单来说就是通过比较新数据点与已知数据点来做决策或预测，而基于模型的学习则是通过探索训练数据中的模式，建立一个预测模型，就像科学家们做实验一样。

通过这些分类，我们可以更好地理解不同机器学习系统的工作原理和适用场景，从而选择最适合特定问题的方法。
