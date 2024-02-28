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
titanic_project/
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
├── src/                              # Source code for the project
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

**Note**：以上项目目录结构为初始化状态，后续根据项目需要，文件夹和其中的文件会有所增加