---
layout: post
title: Setting up a modern python development environment with pyenv-win and Poetry
subtitle:
excerpt_image: /assets/images/python/poetry-install.gif
author: Hakuna
categories: Python
tags: [environment]
sidebar: []
---

## 简介

构建和维护一个高效、可靠的 Python 开发环境是至关重要的。为了实现这一目标，开发者可以选择多种工具来搭建自己的开发环境，包括但不限于 virtualenv、venv、conda、pipenv、poetry 以及 pyenv。每种工具都有其独特的特点和优势，但本文档将重点介绍在 Windows 系统下，如何利用 pyenv-win 和 Poetry 这两个强大的工具来构建一个现代化的 Python 开发环境。

[pyenv-win](https://github.com/pyenv-win/pyenv-win) 是 pyenv（一个流行的多版本 Python 管理工具）的 Windows 版本。它提供了一个简单而强大的解决方案，使开发者能够在同一个系统中安装、管理，并轻松切换多个 Python 版本。这一功能尤其对于同时进行多个项目，且每个项目需要不同 Python 版本的开发者来说，是极其有用的。pyenv-win 的引入，极大地简化了在 Windows 系统上管理多个 Python 版本的复杂度，从而使开发者能够专注于编码和项目构建，而非环境配置。

[Poetry](https://python-poetry.org/)是一个 Python 依赖管理和打包工具。 Poetry 使用 `pyproject.toml` 文件来定义项目的依赖和配置，该方法提供了比传统 `setup.py` 文件更清晰、更直观的依赖声明方式。此外，Poetry 自带的依赖解析器能够自动解决依赖冲突，确保项目依赖的一致性和项目的可重复构建。

## 安装指南

### 前提条件

- Windows 7 或更高版本。
- PowerShell 5 或更高版本（推荐使用 PowerShell 7）。
- [git](https://git-scm.com/downloads)。

### 安装 pyenv-win

1. 打开 PowerShell：以管理员身份运行 PowerShell。

2. 安装 pyenv-win：使用 git 克隆 pyenv-win 到 '~/.pyenv' 目录。
```powershell
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
```

3. 配置环境变量：在 PowerShell 中运行以下命令，添加 pyenv 相关的环境变量。
```powershell
[System.Environment]::SetEnvironmentVariable('PYENV', "$HOME/.pyenv/pyenv-win/", [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable('Path', "$env:Path;$HOME/.pyenv/pyenv-win/bin;$HOME/.pyenv/pyenv-win/shims", [System.EnvironmentVariableTarget]::User)
```

4. 重启 PowerShell：关闭并重新打开 PowerShell 以应用更改。

5. 验证安装：运行以下命令，如果安装成功，将显示 pyenv 版本。
```shell
pyenv --version
```

**Note**: 由于网络原因，建议设置`PYTHON_BUILD_MIRROR_URL`为镜像站点，如<https://mirrors.huaweicloud.com/python/>。也可以设置`PYTHON_BUILD_ARIA2_OPTS`为建议值：`-x 10 -k 1M`。在设置该值前，请确保安装了[aria2](https://aria2.github.io/)。


### 安装 Poetry

1. 打开 PowerShell：不需要以管理员身份运行。

2. 运行安装脚本：使用 Poetry 官方提供的安装命令。
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

2. 配置环境变量（如果安装脚本未自动完成）：确保 Poetry 的安装路径添加到您的环境变量中。通常，Poetry 会被安装在 `C:\Users\用户名\AppData\Roaming\Python\Scripts` 或 `C:\Users\用户名\.poetry\bin`。

3. 验证安装：重新打开 PowerShell 并运行以下命令来验证 Poetry 安装成功：
```powershell
poetry --version
```

### 使用 pyenv-win 管理 Python 版本

1. 列出可用的 Python 版本
```powershell
pyenv install --list
```

2. 安装特定版本的 Python
```powershell
pyenv install 3.8.10
```

3. 设置全局 Python 版本
```powershell
pyenv global 3.8.10
```

### 使用 Poetry 管理项目依赖

1. 创建新项目
```powershell
poetry new my_project
```

2. 添加依赖
```powershell
cd my_project
poetry add requests
```

3. 安装依赖
```powershell
poetry install
```

### 参考资料

- [pyenv-win GitHub 仓库](https://github.com/pyenv-win/pyenv-win)
- [Poetry 官方文档](https://python-poetry.org/docs/)
- [Modern Python Environments - dependency and workspace management](https://testdriven.io/blog/python-environments/)
- [Python Virtual Environments tutorial using Virtualenv and Poetry](https://serpapi.com/blog/python-virtual-environments-using-virtualenv-and-poetry/)
- [How to Create and Use Virtual Environments in Python With Poetry](https://www.youtube.com/watch?v=0f3moPe_bhk&t=494s)


