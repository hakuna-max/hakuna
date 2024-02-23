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


在开始之前，请确认已经在您的电脑上安装好了 [Poetry](https://python-poetry.org/) 以及 [VS Code](https://code.visualstudio.com/)。如未配置好 Python 的开发环境， 请参考 [Setting Up a Modern Python Development Environment with pyenv-win and Poetry]({% post_url 2024-02-08-environment-for-development-with-python %}) 以及其中的参考资料。

## Step 1: 进入到项目目标文件夹

假设我们希望将项目放在电脑的 D 盘，可以执行以下代码，进入到目标盘：
```powershell
cd D:
```

## Step 2: 初始化项目

在开始执行以下 Poetry 命令前，请确认 Poetry 的 `virtualenvs.in-project = true` （个人习惯，也可以不设置，但建议）。

然后我们可以使用 Poetry 来初始化我们的项目（以下步骤假设项目文件夹未存在）：
```powershell
poetry new project_name
```

这将在我们目标盘下初始化一个名为 `project_name` 的文件夹，其中的目录结构如下：
```plaintext
project_name
├── pyproject.toml
├── README.md
├── project_name
│   └── __init__.py
└── tests
    └── __init__.py
```

- `project_name/project_name/`文件夹主要用于存放项目主要代码，
- `project_name/texts/`文件夹将主要用于存放测试代码。
- `pyproject.toml`文件为项目依赖的配置文件。

由 `poetry new project_name` 初始化的 `pyproject.toml` 文件内容类似于：
```toml
[tool.poetry]
name = "project-name"
version = "0.1.0"
description = ""
authors = ["hakuna@thinkstation <hakuna@thinkstation.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

*Note*: 在刚开始不熟悉的情况下，建议不要手动修改该文件，可以通过相关 Poetry 命令来添加相关配置。 

## Step 3: 添加项目所需依赖

比如，该项目的主要目的是数据分析，那么我们常用的数据分析依赖库有[numpy](https://numpy.org/doc/stable/), [matplotplib](https://matplotlib.org/stable/users/index), [pandas](https://pandas.pydata.org/docs/)等。此外，在数据分析的过程中，我们也可能运用 [Pytest](https://docs.pytest.org/en/8.0.x/) 依赖库来完成自动化测试工作。那么，我们可以在包含 `pyproject.toml` 文件的顶层目录下，执行以下 Poetory 命令，添加相关依赖。
```powershell
poetry add numpy matplotlib pandas
```

执行完以上 Poetry 命令后，我们的 `pyproject.toml` 文件将呈现如下内容：
```toml
[tool.poetry]
name = "project-name"
version = "0.1.0"
description = ""
authors = ["hakuna@thinkstation <hakuna@thinkstation.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"
numpy = "^1.26.4"
matplotlib = "^3.8.3"
pandas = "^2.2.0"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

对于 [Pytest](https://docs.pytest.org/en/8.0.x/)，由于该依赖库主要代码测试，并不是项目的主要依赖。因此通常情况下，我们希望将其作为开发环境中使用。为了实现该目的，我们需要给 `poetry add` 命令添加 `--group=dev` 参数，如下
```powershell
poetry add --group=dev pytest
```

由此，我们的 `pyproject.toml` 文件将呈现如下内容：
```toml
[tool.poetry]
name = "project-name"
version = "0.1.0"
description = ""
authors = ["hakuna@thinkstation <hakuna@thinkstation.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"
numpy = "^1.26.4"
matplotlib = "^3.8.3"
pandas = "^2.2.0"


[tool.poetry.group.dev.dependencies]
pytest = "^8.0.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

新的 `pyproject.toml` 文件与之前的最大区别在于将 `pytest` 依赖放在了单独的 `[tool.poetry.group.dev.dependencies]`.

假设以上依赖是该项目的需要的所有库，到此，初始化项目的工作到一阶段。

## Step 4：在 VS Code 中设置 Python 解释器

通常情况下，VS Code 足够聪明，当我们第一次使用 `poetry add` 命令后，在VS Code中的右下角会提示 “发现新的虚拟环境”，只要点击yes后，VS Code会自动设置好项目的虚拟环境。

![](/assets/images/python/python-environment-prompt.png)

如果没有，那我们需要手动设置 Python 解释器。首先，我们需要找到虚拟环境中的 Python 解释器路径。这可以通过如下命令实现
```powershell
poetry env info
```

该 Poetry 命令将显示电脑上可用的 Python 解释器，如：
```plaintext
Virtualenv
Python:         3.12.1
Implementation: CPython
Path:           D:\git\project_name\.venv
Executable:     D:\git\project_name\.venv\Scripts\python.exe
Valid:          True

System
Platform:   win32
OS:         nt
Python:     3.12.1
Path:       C:\Users\hakuna-o\.pyenv\pyenv-win\versions\3.12.1
Executable: C:\Users\hakuna-o\.pyenv\pyenv-win\versions\3.12.1\python.exe
```

可以发现，我的电脑上有两个 Python 解释器，一个显示为 Virtualenv，一个显示为 System。我们需要的是 Virtualenv 虚拟环境中的 Python 解释器。那么我们可以复制其中的 Executable 路径，此处为：`D:\git\project_name\.venv\Scripts\python.exe`。然后回到 VS Code。在 VS Code 中调出 Command Palette （`ctrl + shift + p`），输入python：select interpreter后回车会出现一个列表，如下：

![](/assets/images/python/python_interpreter.png)

从上图可以看见，VS Code其实已经识别了我们电脑上可用的 Python 解释器。我们可以直接选择对应的 Python 解释器。此项目的解释器为 `Python 3.12.1 ('.venv': Poetry)`，该解释器也是VS Code默认推荐的。如果此处没有列出相应的 Python 解释器。则，我们可以选择 `Enter interpreter path...`，然后将刚才复制的路径填进去。按照如上操作，当我们在 `project_name/project_name/` 文件夹下新建一个 `.py`文件后，VS Code的状态栏的右边会显示相应的 Python 解释器信息，如 `Python 3.12.1 ('.venv': Poetry)`。

## Step 5: 在 VS Code 中设置 Pytest

在 VS Code 中调出 Command Palette （`ctrl + shift + p`），输入 `Python：Configiure Tests` 后回车会出现一个列表:

![](/assets/images/python/python_config_tests.png)

这里出现了两个单元测试框架，由于我们前期选择Pytest，所以在这，我们选择第二个作为项目的测试框架，即 pytest framework。然后，会出现类似与下图所显示的内容：

![](/assets/images/python/python_config_tests_folder.png)

此处应该选择我们放置tests文件的文件夹，也即 `tests`。

完成以上设置后，VS Code会在 `project_name/` 文件夹下生成 `.vscode/settings.json` 和 `.pytest_cache`文件夹。在 `tests/` 文件夹下生成 `__pychace__` 和 `.pytest_cache` 文件夹。我们需要适当关注 `.vscode/settings.json`。默认生成的配置文件内容类似如下：

```json
{
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
}
```

这样，我们可以在 `project_name/` 下执行如下命令，执行测试工作：
```powershell
poetry run pytest
```

假设在 `tests/` 下有一个名为 `test_exmaple.py` 的测试文件，
```python
def test_sum():
    assert 1 + 1 == 2
```

执行 `poetry run pytest`后会在Terminal中显示：
```plaintext
==================================== test session starts ====================================
platform win32 -- Python 3.12.1, pytest-8.0.1, pluggy-1.4.0
rootdir: D:\git\project_name
collected 1 item                                                                              

tests\test_test.py .                                                                   [100%] 

===================================== 1 passed in 0.01s =====================================
```

关于在 VS Code 中实现单元测试的更为详细的介绍，请参考如下资料

- [Python testing in Visual Studio Code](https://code.visualstudio.com/docs/python/testing)
- Okken, B., 2022. Python Testing with pytest: Simple, Rapid, Effective, and Scalable, 2nd ed. Pragmatic Bookshelf.

