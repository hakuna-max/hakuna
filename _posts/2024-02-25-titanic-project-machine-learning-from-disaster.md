---
layout: post
title: "Titanic project: Machine learning from disaster"
subtitle:
excerpt_image: /assets/images/python/titanic-project.jpg
author: Hakuna
categories: [Python, Project]
tags: [machine learning, classification]
top: 
sidebar: []
---

## 项目目录结构

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

[[tool.poetry.source]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
priority = "default"


[tool.poetry.group.dev.dependencies]
pytest = "^8.0.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```


```python
def test_highlighter():
    print("is that right")
```
