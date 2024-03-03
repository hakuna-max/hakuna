---
layout: post
title: "Titanic project: Machine learning from disaster (ongoing)"
subtitle:
excerpt_image: /assets/images/python/titanic-project.jpg
author: Hakuna
categories: [Python, Project, Module]
tags: [machine learning, classification]
top:
sidebar: []
---

## Titanic 项目介绍

大家好！今天，我想带你们走进一个非常有趣的机器学习项目——Kaggle 上的 Titanic 生还预测挑战。这个项目的目标是使用 Titanic 号乘客的数据来预测哪些乘客在这场历史性的灾难中幸存下来（即，分类问题）。这个项目不仅是一个绝佳的机会来实践和理解机器学习的基本流程，而且也是一个向所有对商务智能与机器学习感兴趣的同学们展示如何从实际数据中提取洞见的绝佳案例。

项目开始于对数据集的介绍——我们有乘客的各种信息，如年龄、性别、票价和乘客在船上的等级，这些都可能影响他们的生还机会。理解这些特征及其与目标变量之间的关系是我们任务的第一步。

接下来，我们会进行数据预处理，包括处理缺失值、异常值和特征编码，为建模准备数据。然后是探索性数据分析，或称 EDA，它帮助我们通过可视化和数据摘要来揭示数据的内在模式和特征关系。

特征工程阶段，我们会选择最有影响的特征，并可能创造新特征来帮助模型更好地理解数据。紧接着，我们将探索和比较不同的机器学习模型，比如逻辑回归、随机森林、支持向量机、朴素贝叶斯、决策树等，以找到最适合我们数据的模型。

通过训练模型和使用交叉验证等技术评估其性能后，我们将选择一个最终模型。然后，我们将深入分析模型的结果，理解哪些因素对生还预测最为重要，这不仅加深了我们对数据的理解，也让我们学习到了如何解释机器学习模型的预测。

总体上，希望通过该项目实验，同学们不仅学习了机器学习的整个流程，还获得了宝贵的实践经验。

探索机器学习的奇妙世界，解锁数据的潜力，为未来铺平道路。

## 项目的前期准备

- 获取数据集，请从 Kaggle 网站上下载相关数据集，链接：<https://www.kaggle.com/c/titanic/data>。也可以通过点击 [data link](/assets/downloadables/ml/data/titanic.zip) 下载。
- [了解 titanic 数据集](https://www.kaggle.com/c/titanic/data)
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
- `pyproject.toml`, `README.md`, `titanic/` 以及 `test/` 文件和文件夹是通过 `poetry new titanic` 初始化后生成，具体过程可以参考[Initializing a project with Poetry and VS Code editor]({% post_url 2024-02-22-project-init-with-poetry %})。
- 其他文件以及文件夹为自己创建。
- 为了加速依赖库的安装过程，项目使用 `poetry source add tsinghua https://pypi.tuna.tsinghua.edu.cn/simple/` 命令添加了[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。
- 通过 `poetry add package_name` 添加必要的项目依赖。初始化后的 `pyproject.toml` 内容如下：

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
jupyter = "^1.0.0"

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

## 项目步骤

整体上，该项目遵循传统的机器学习步骤，如下图所示

![](/assets/images/ml/end-to-end-machine-learning-project.svg)

假设同学们已经对 titanic 项目有了基本的了解，并且已经获到了相关数据。现在，我们可以开始 EDA 的相关分析工作。

## EDA 及其可视化

### 了解数据集基本情况

为了更为直观的呈现分析过程，我们可以借助于 jupyter 项目中的 [notebook](https://jupyter-notebook.readthedocs.io/en/latest/) 或者 [jupyterlab](https://jupyterlab.readthedocs.io/en/latest/) 工具来做 EDA 及其可视化。在 [VS Code](https://code.visualstudio.com/) 中，我们可以通过安装 jupyter 插件来实现相关功能。如何在 [VS Code](https://code.visualstudio.com/) 中使用 jupyter notebook，请参考[Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)。

假设您已经按照相关说明配置好项目环境，接下来，我们可以在 `notebooks/` 文件夹下新建一个名为 `eda_vis.ipynb` 的 notebook。

`eda_vis.ipynb` 第一个 cell 通常用来导入相关依赖，设置相关环境变量等。例如，我们需要 `pandas` 中的 `read_csv` 方法来加载相关数据，那么我们需要在其中写上如下代码

```python
# import packages
import pandas as pd
```

然后，我们就可以使用 `pd.read_csv(file_path)` 的方式将相关数据加载到工作空间中，例如，在新的 cell 中我们可以按照如下方式加载数据

```python
# load raw data
train_data = pd.read_csv("../data/raw/train.csv")    # train data
test_data = pd.read_csv("../data/raw/test.csv")      # test data
```

如果存放数据的路径正确，那么我们的数据已经被加载到工作空间了。接下来，可以鸟瞰下数据：

```python
print(train_data.head())
```

默认参数设置下，`print(train_data.head())` 会打印出数据框的前 5 行，所有列的数据，例如：

```plaintext
   PassengerId  Survived  Pclass  \
0            1         0       3
1            2         1       1
2            3         1       3
3            4         1       1
4            5         0       3

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1
2                             Heikkinen, Miss. Laina  female  26.0      0
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1
4                           Allen, Mr. William Henry    male  35.0      0

   Parch            Ticket     Fare Cabin Embarked
0      0         A/5 21171   7.2500   NaN        S
1      0          PC 17599  71.2833   C85        C
2      0  STON/O2. 3101282   7.9250   NaN        S
3      0            113803  53.1000  C123        S
4      0            373450   8.0500   NaN        S
```

可以初略地看出，每一行代表一个乘客的信息，列包括乘客的不同特征。结合 [Kaggle 对 titanic 数据集](https://www.kaggle.com/c/titanic/data)的介绍，可以大致地分析每个特征在机器学习模型中的潜在重要性：

1. **PassengerId**: 乘客 ID，唯一标识每个乘客。这个特征对于模型的预测通常没有直接作用，主要用于索引和排序。

2. **Survived**: 生存状态，是目标变量（即我们想要预测的变量）。0 表示未生存，1 表示生存。

3. **Pclass**: 乘客舱等级，是一个社会经济地位的指标，有 1、2、3 三个值。通常第一等舱的乘客生存率更高。

4. **Name**: 乘客姓名。虽然姓名本身对预测可能没有直接影响，但可以从中提取出有用的特征，如头衔，可能会反映乘客的社会地位。

5. **Sex**: 性别，是一个重要的特征，因为历史数据表明女性乘客的生存率高于男性。

6. **Age**: 年龄，可能会影响生存率。例如，小孩和老人可能在撤离时获得优先权。

7. **SibSp**: 兄弟姐妹和配偶的数量。家庭成员的数量可能会影响乘客的生存率。

8. **Parch**: 父母和孩子的数量。同样，家庭大小可能是一个重要因素。

9. **Ticket**: 票号。这个特征可能不会直接影响生存率，但有可能包含一些有用的信息，例如团体旅行可能有相同的票号前缀。

10. **Fare**: 票价，可能反映了乘客的社会经济地位和舱位。

11. **Cabin**: 船舱号。这个特征有很多缺失值 (需要考虑如何处理)，但对于有记录的船舱号，它可能反映了乘客的位置，进而可能影响到他们的生存率。

12. **Embarked**: 登船港口。有三个可能的值 S、C、Q，分别代表南安普顿（Southampton）、瑟堡（Cherbourg）和皇后镇（Queenstown），这可能是生存率的一个因素。

此外，我们还可以通过 `info` 方法进一步查看数据集的的其他基本信息，例如总的行数、列数、每列的数据类型和非空值的数量，甚至内存的使用情况等

```python
train_data.info()
```

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

从以上输出，我们可以得知以下有用信息：

1. **行数和列数**：训练数据集一共有 891 行（乘客数），总共有 12 列（特征数），每列代表不同的特征。
2. **非空值计数**：这部分信息显示了每列非空（非缺失）值的数量。例如，`Age`列只有 714 个非空值，意味着有 177 个缺失值（891 - 714）。`Cabin`列只有 204 个非空值，这表明大多数乘客的船舱信息是缺失的。
3. **数据类型**：
   - `int64`：整数类型，如`PassengerId`, `Survived`, `Pclass`, `SibSp`, `Parch`。
   - `float64`：浮点数类型，如`Age`和`Fare`。
   - `object`：通常是字符串类型，用于文本或混合数据类型，如`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`。
4. **内存使用**：数据集大约占用 83.7 KB 内存。

进一步分析，

- **`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`**: 这些都是分类特征，但处理方式可能不同。例如，`Name`可能需要从中提取称谓，`Cabin`的缺失值需要特别处理。
- **`Age`, `SibSp`, `Parch`, `Fare`**: 这些是数值特征，可以直接用于模型，但可能需要处理缺失值和标准化或归一化。

考虑缺失值处理策略：

- **`Age`**: 缺失值可以通过中位数、均值或基于其他相关特征的预测模型来填充。
- **`Cabin`**: 由于缺失值较多，可以考虑转换为有船舱信息和无船舱信息两类，或者直接忽略此特征。
- **`Embarked`**: 只有两个缺失值，可以填充为最常见的值，或者根据其他特征来推断。

通过以上信息，可以对数据有一个基本的了解，为后续的数据清洗、特征工程和建模工作打下基础。

### 单因素分析

单变量分析，即分别分析每个变量，是进行 EDA 的一个很好的起点。单变量分析可以帮助我们了解数据的分布和结构，为后续的多变量分析和模型建立提供基础。针对 Titanic 数据集可以进行以下单变量分析：

1. 分析目标变量（`Survived`）。这是我们需要预测的变量，因此，我们打算首先分析该变量。可以分析生存和未生存乘客的比例并使用条形图展示生存状态的分布：
2. 数值型变量分析（`Age`, `SibSp`, `Parch`, `Fare`）。比如可以绘制直方图或核密度估计图来查看 `Age` 和 `Fare` 的分布；计算这些数值型变量的基本统计量，如均值、中位数、标准差等；对于 `SibSp` 和 `Parch`，可以计算不同值的乘客数量，并使用条形图展示。
3. 类别型变量分析（`Pclass`,`Sex`, `Ticket`, `Cabin`, `Embarked`）。对于类别型变量，计算每个类别的乘客数量；使用条形图展示这些类别变量的分布；特别是对于 `Cabin` ，由于存在大量缺失值，需要决定如何处理这些缺失值
4. 缺失值分析(Age, Cabin, Embarked)。深入分析这些缺失值，例如，它们是否随机出现，是否存在某种模式；决定如何处理这些缺失值，比如填充、删除或其他方法。
5. 额外的单变量分析。可以考虑更深入地分析 `Name` 和 `Ticket`。例如，从 `Name` 中提取头衔，并分析不同头衔的乘客分布；对 Ticket 进行类似的分析，看是否可以从中提取有用的信息。

通过这些单变量分析，你可以对数据集有一个全面的了解，这将为后续的多变量分析和数据预处理提供坚实的基础。在完成这些分析后，可以根据发现的洞见进行多变量分析，探索变量之间的关系，尤其是与目标变量 Survived 之间的关系。

为了分析目标变量，基于如下代码：

```python
# 计算生存和未生存的乘客数量
survival_counts = train_data['Survived'].value_counts()

# 打印生存和未生存的乘客数量
print(f"Survival counts: {survival_counts}")

# 计算生存和未生存的乘客比例
survival_rates = (train_data['Survived'].value_counts(normalize=True) * 100).round(2)

# 打印生存和未生存的乘客比例
print(f"\nSurvival rates (%): {survival_rates}")

# 绘制条形图
plt.figure(figsize=(8, 6))
bars = survival_counts.plot(kind='bar')
plt.title('Survival Count in Titanic Dataset')
plt.xlabel('Survived (1 = Survived, 0 = Not Survived)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 在每个条形上添加数值标签
for bar in bars.patches:
    # 获取条形的位置信息和高度
    y_value = bar.get_height()
    x_value = bar.get_x() + bar.get_width() / 2

    # 设置标签显示的数值
    label = f"{y_value:.0f}"

    # 在条形上方显示标签
    plt.text(x_value, y_value, label, ha='center', va='bottom')

plt.show()
```

可以得到如下结果：

```plaintext
Survival counts: Survived
0    549
1    342
Name: count, dtype: int64

Survival rates (%): Survived
0    61.62
1    38.38
Name: proportion, dtype: float64
```

![](/assets/images/ml/titianic_factor_survived.png)

结合数据分析结果，可以发现超过半数的乘客（61.62%）在 Titanic 事件中未能生存，仅有约三分之一的乘客在事故中生存下来。

对于数值型变量，我们首先借助于直方图和核密度估计图来查看 `Age` 和 `Fare` 的分布。[seaborn](https://seaborn.pydata.org/tutorial.html) 中的 `histplot` 方法对此提供了非常好的支持。由于前期我们在初始化中并没有添加该依赖，可以通过 `poetry add seaborn` 的方式将该依赖添加到我们的项目环境中。记得通过 `import searbon as sns` 将该包导入到工作区。分析的代码如下：

```python
# 绘制 Age 的分布图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1行2列的第一个
sns.histplot(train_data['Age'].dropna(), kde=True, bins=30)
plt.title('Distribution of Age')

# 绘制 Fare 的分布图
plt.subplot(1, 2, 2)  # 1行2列的第二个
sns.histplot(train_data['Fare'].dropna(), kde=True, bins=30)
plt.title('Distribution of Fare')

plt.tight_layout()
plt.show()
```

其结果如下：

![](/assets/images/ml/titianic_factor_age_fare_dist.png)

当然，我们也可以借助于 `describe()` 方法来查看 `Age` 和 `Fare` 的基本统计量

```python
age_fare_stats = train_data[['Age', 'Fare']].describe()
print(age_fare_stats)
```

基本统计量结果如下：

```plaintext
              Age        Fare
count  714.000000  891.000000
mean    29.699118   32.204208
std     14.526497   49.693429
min      0.420000    0.000000
25%     20.125000    7.910400
50%     28.000000   14.454200
75%     38.000000   31.000000
max     80.000000  512.329200
```
