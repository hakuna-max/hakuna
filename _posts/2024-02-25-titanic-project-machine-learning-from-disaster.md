---
layout: post
title: "Titanic project: Machine learning from disaster (ongoing)"
subtitle:
excerpt_image: /assets/images/python/titanic-project.jpg
author: Hakuna
categories: [Python, Project]
tags: [machine learning, classification]
top: 
sidebar: []
---

## Titanic 项目介绍

大家好！今天，我想带你们走进一个非常有趣的机器学习项目——Kaggle上的Titanic生还预测挑战。这个项目的目标是使用Titanic号乘客的数据来预测哪些乘客在这场历史性的灾难中幸存下来（即，分类问题）。这个项目不仅是一个绝佳的机会来实践和理解机器学习的基本流程，而且也是一个向所有对商务智能与机器学习感兴趣的同学们展示如何从实际数据中提取洞见的绝佳案例。

项目开始于对数据集的介绍——我们有乘客的各种信息，如年龄、性别、票价和乘客在船上的等级，这些都可能影响他们的生还机会。理解这些特征及其与目标变量之间的关系是我们任务的第一步。

接下来，我们会进行数据预处理，包括处理缺失值、异常值和特征编码，为建模准备数据。然后是探索性数据分析，或称EDA，它帮助我们通过可视化和数据摘要来揭示数据的内在模式和特征关系。

特征工程阶段，我们会选择最有影响的特征，并可能创造新特征来帮助模型更好地理解数据。紧接着，我们将探索和比较不同的机器学习模型，比如逻辑回归、随机森林、支持向量机、朴素贝叶斯、决策树等，以找到最适合我们数据的模型。

通过训练模型和使用交叉验证等技术评估其性能后，我们将选择一个最终模型。然后，我们将深入分析模型的结果，理解哪些因素对生还预测最为重要，这不仅加深了我们对数据的理解，也让我们学习到了如何解释机器学习模型的预测。

总体上，希望通过该项目实验，同学们不仅学习了机器学习的整个流程，还获得了宝贵的实践经验。

探索机器学习的奇妙世界，解锁数据的潜力，为未来铺平道路。

## 项目的前期准备

- 获取数据集，请从Kaggle网站上下载相关数据集，链接：<https://www.kaggle.com/c/titanic/data>
- [了解titanic数据集](https://www.kaggle.com/c/titanic/data)
- 初始化项目，初始化后的项目目录结构大致如下：

```plaintext
titanic/
│
├── pyproject.toml                    # Poetry configuration file with dependencies and project information
├── README.md                         # Project README for an overview and setup instructions
├── .gitignore                        # Specifies intentionally untracked files to ignore
│
├── data/                             # Data directory
│   ├── raw/                          # Unprocessed, raw Titanic dataset
│   └── processed/                    # Cleaned and processed dataset
│
├── notebooks/                        # Jupyter notebooks for exploratory data analysis and prototyping
│
├── titanic/                          # Source code for the project
│   ├── __init__.py                   # Initializes Python package
│   ├── data_preprocessing.py         # Script for data preprocessing
│   ├── model.py                      # Model definition and training script
│   └── evaluation.py                 # Script for model evaluation and testing
│
├── tests/                            # Test directory
│   ├── __init__.py                   # Initializes Python package
│   └── test_data_preprocessing.py    # Tests for data preprocessing
│
└── configs/                          # Configuration files directory
    └── model_config.yaml             # Model parameters and configuration options
```

**Note**：

- 以上项目目录结构为初始化状态，后续根据项目需要，文件夹和其中的文件会有所增加。
- `pyproject.toml`, `README.md`, `titanic/ ` 以及 `test/` 文件和文件夹是通过 `poetry new titanic` 初始化后生成，具体过程可以参考[Initializing a project with Poetry and VS Code editor]({% post_url 2024-02-22-project-init-with-poetry %})。
- 其他文件以及文件夹为自己创建。
- 为了加速依赖库的安装过程，项目使用 `poetry source add tsinghua https://pypi.tuna.tsinghua.edu.cn/simple/` 命令添加了[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。
- 初始化后的 `pyproject.toml` 内容如下：

```toml
[tool.poetry]
name = "titanic"
version = "0.1.0"
description = ""
authors = ["hak@mac <zlp@upc.edu.cn>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"
numpy = "^1.26.4"
pandas = "^2.2.1"
scikit-learn = "^1.4.1.post1"
matplotlib = "^3.8.3"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.1"


[[tool.poetry.source]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
priority = "primary"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```
