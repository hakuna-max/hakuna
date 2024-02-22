---
layout: post
title: Initializing a project with Poetry and VS Code editor
subtitle:
excerpt_image: /assets/images/python/poetry-init.jpg
author: Hakuna
categories: Python
tags: environment
top: 
sidebar: []
---

初始化项目使用到的工具如下：
- [Poetry](https://python-poetry.org/): 依赖管理
- [VS Code](https://code.visualstudio.com/)：代码编辑
- [Pytest](https://docs.pytest.org/en/8.0.x/)：代码测试
- [Windows Terminal](https://github.com/microsoft/terminal)：执行相关命令，或者使用 VS Code 内嵌的 Terminal（在 VS Code 中按下`` Ctrl + ` ``）工具。以下代码块中右上角如有显示`POWERSHELL`，则对应代码均表示将在Terminal中执行。


在开始之前，请确认已经在您的电脑上安装好了 [Poetry](https://python-poetry.org/) 以及 [VS Code](https://code.visualstudio.com/)。如未配置好 Python 的开发环境， 请参考[Setting Up a Modern Python Development Environment with pyenv-win and Poetry](/2024-02-08-environment-for-development-with-python.md)以及其中的参考资料。

## Step 1: 进入到项目目标文件夹
假设我们希望将项目放在电脑的 D 盘，可以执行以下代码，进入到目标盘：
```powershell
cd D:
```

然后我们可以使用 Poetry 来初始化我们的项目（以下步骤假设项目文件夹未存在）：
```powershell
poetry new project_name
```

这将在我们目标盘下初始化一个名为 `project_name` 的文件夹，其中的目录结构如下：
```text
project_name
├── pyproject.toml
├── README.md
├── project_name
│   └── __init__.py
└── tests
    └── __init__.py
```