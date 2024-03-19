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

- [Titanic 项目介绍](#titanic-项目介绍)
- [项目的前期准备](#项目的前期准备)
- [项目步骤](#项目步骤)
- [EDA 及其可视化](#eda-及其可视化)
  - [了解数据集基本情况](#了解数据集基本情况)
  - [单因素分析](#单因素分析)
  - [双因素或多因素分析](#双因素或多因素分析)
- [特征工程](#特征工程)
  - [基线模型的构建](#基线模型的构建)
  - [变量：`Age`](#变量age)
  - [变量：`SibSp` 和 `Parch`](#变量sibsp-和-parch)
  - [变量：`Ticket`](#变量ticket)
  - [变量：`Fare`](#变量fare)
  - [变量：`Cabin`](#变量cabin)
  - [变量：`Embarked`](#变量embarked)
  - [考虑组合特征](#考虑组合特征)
  - [模型训练与评估流程图](#模型训练与评估流程图)

<hr/>

## Titanic 项目介绍

大家好！今天，我想带你们走进一个非常有趣的机器学习项目——Kaggle 上的 Titanic 生还预测挑战。这个项目的目标是使用 Titanic 号乘客的数据来预测哪些乘客在这场历史性的灾难中幸存下来（即，分类问题）。这个项目不仅是一个绝佳的机会来实践和理解机器学习的基本流程，而且也是一个向所有对商务智能与机器学习感兴趣的同学们展示如何从实际数据中提取洞见的绝佳案例。

项目开始于对数据集的介绍——我们有乘客的各种信息，如年龄、性别、票价和乘客在船上的等级，这些都可能影响他们的生还机会。理解这些特征及其与目标变量之间的关系是我们任务的第一步。

接下来，我们会进行探索性数据分析，或称 EDA，它帮助我们通过可视化和数据摘要来揭示数据的内在模式和特征关系。

特征工程阶段，我们会选择最有影响的特征，并可能创造新特征来帮助模型更好地理解数据。紧接着，我们将探索和比较不同的机器学习模型，比如逻辑回归、随机森林、支持向量机、朴素贝叶斯、决策树等，以找到最适合我们数据的模型。

通过训练模型和使用交叉验证等技术评估其性能后，我们将选择一个最终模型。然后，我们将深入分析模型的结果，理解哪些因素对生还预测最为重要，这不仅加深了我们对数据的理解，也让我们学习到了如何解释机器学习模型的预测。

总体上，希望通过该项目实验，同学们不仅学习了机器学习的整个流程，还获得了宝贵的实践经验。

探索机器学习的奇妙世界，解锁数据的潜力，为未来铺平道路。

<hr/>

## 项目的前期准备

- 获取数据集，请从 Kaggle 网站上下载相关数据集，链接：<https://www.kaggle.com/c/titanic/data>。也可以通过点击 [data link](/assets/downloadables/ml/data/titanic.zip) 下载。
- [了解 titanic 数据集](https://www.kaggle.com/c/titanic/data)，建议结合[Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)。
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

<hr/>

## 项目步骤

整体上，该项目遵循传统的机器学习步骤，如下图所示

![](/assets/images/ml/end-to-end-machine-learning-project.svg)

假设同学们已经对 titanic 项目有了基本的了解，并且已经获到了相关数据。现在，我们可以开始 EDA 的相关分析工作。

<hr/>

## EDA 及其可视化

<hr/>

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

<hr/>

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

从直方图可以看出

- 年龄分布：约呈右偏态，较多的乘客集中在年轻的年龄段。大多数乘客的年龄在 20 到 40 岁之间。
- 票价分布：呈现出极度的右偏，表明大多数乘客支付的票价较低。

该分布特征也表明，我们需要考虑特征的极端值对模型训练和性能的影响，导致预测偏向数据的主体部分，而忽视尾部的重要信息。

针对分布不均匀的数据集，在后期数据处理时，可能会考虑采用不同策略对其进行处理，比如**数据转换**，**分箱（Binning）**，**剔除极端值**，甚至考虑使用对偏态分布不敏感的**非线性模型**（如，随机森林，梯度提升树等）。具体采用何种方法将取决与数据的具体情况和模型的需求。一般建议在数据转换或处理前后，都进行可视化，以此评估转换或处理的效果。在应用任何转换或者处理方法之前，最好在原始数据上训练模型，以便有一个基准性能进行比较。

对于 `SibSp` 和 `Parch`，实现其不同值的乘客数量的条形图的代码如下：

```python
# 绘制 SibSp 的分布
plt.subplot(1, 2, 1)  # 1行2列的第一个
sns.countplot(x='SibSp', data=train_data)
plt.title("Distribution of SibSp")
plt.ylabel("Number of Passengers")
plt.xlabel("SibSp")

# 绘制 Parch 的分布
plt.subplot(1, 2, 2)  # 1行2列的第一个
sns.countplot(x='Parch', data=train_data)
plt.title('Distribution of Parch')
plt.ylabel('Number of Passengers')
plt.xlabel('Parch')

plt.tight_layout()
plt.show()
```

结果：

![](/assets/images/ml/titianic_factor_sibsp_parch_dist.png)

从条形图中，我们可以观察到以下几点关于 `SibSp`（兄弟姐妹/配偶数量）和 `Parch`（父母/孩子数量）的分布：

- **`SibSp` 分布**：大多数乘客没有兄弟姐妹或配偶同行（ `SibSp` 为 0）。有一些乘客有一个兄弟姐妹或配偶（ `SibSp` 为 1），而有两个或更多兄弟姐妹或配偶同行的乘客数量较少。

- **`Parch` 分布**：与 `SibSp` 类似，大多数乘客没有携带父母或孩子（ `Parch` 为 0）。少数乘客有一到三个父母或孩子同行，而更多的父母或孩子同行的情况则更为罕见。

针对以上结果，我们在后期分析中，可能需要注意以下几点：

- **特征组合**：考虑将 `SibSp` 和 `Parch` 合并为一个新特征，如家庭成员总数，这可能有助于揭示家庭规模与生存率之间的关系。
- **模型选择**：选择对分类数据敏感度低的模型，如随机森林或梯度提升树，可能在处理这类特征时表现更好。
- **数据预处理**：对于 `SibSp` 和 `Parch` 值较大的少数样本，可以考虑进行分组或其他形式的处理，以防止它们对模型产生不成比例的影响。

针对类别型变量（ `Pclass`, `Sex`, `Ticket`, `Cabin`, `Embarked`），可以计算每个类别的乘客数量，并绘制条形图。该分析的代码如下：

```python
plt.figure()

# Pclass 分布
plt.subplot(1, 2, 1)
sns.countplot(x='Pclass', data=train_data)
plt.title('Distribution of Pclass')
plt.ylabel('Number of Passengers')
plt.xlabel('Pclass')

# Sex 分布
plt.subplot(1, 2, 2)
sns.countplot(x='Sex', data=train_data)
plt.title('Distribution of Sex')
plt.ylabel('Number of Passengers')
plt.xlabel('Sex')

plt.figure()
# Embarked 分布
plt.subplot(1, 2, 1)
sns.countplot(x='Embarked', data=train_data)
plt.title('Distribution of Embarked')
plt.ylabel('Number of Passengers')
plt.xlabel('Embarked')

# Cabin 缺失值情况
# 计算缺失值比例
cabin_null_percentage = train_data['Cabin'].isnull().sum() / len(train_data) * 100
cabin_not_null_percentage = 100 - cabin_null_percentage

# 绘制 Cabin 缺失值情况的条形图
plt.subplot(1, 2, 2)
plt.bar(['Missing', 'Present'], [cabin_null_percentage, cabin_not_null_percentage])
plt.title('Cabin Missing Value Percentage')
plt.ylabel('Percentage')
plt.xlabel('Cabin Value Status')

plt.tight_layout()
plt.show()

# 输出 Cabin 缺失值的具体比例
print(f"Carbin null percentage (%): {cabin_null_percentage:.2f}")
```

我们首先来看看各个特征的分布情况：

![](/assets/images/ml/titianic_factor_cate_dist_1.png)

![](/assets/images/ml/titianic_factor_cate_dist_2.png)

可以观察到以下几点关于类别型变量的分布：

- **`Pclass`（船舱等级）**：不同等级的船舱乘客数量分布显示，第三等舱乘客最多，其次是第一等舱和第二等舱。
- **`Sex`（性别）**：男性乘客数量多于女性乘客。
- **`Embarked`（登船港口）**：大多数乘客从 S 港口登船，其次是 C 港口，最少的是 Q 港口。

由于**`Cabin`（船舱号）**的数据确实情况严重，我们可以重点关注下：

- `Cabin` 特征有约 77.1%的缺失值，这是一个非常高的比例，对于这种情况，我们需要决定如何处理这些大量的缺失值。
- 对于如此高比例的缺失值，直接删除这个特征可能是一个选择，因为它可能包含的信息太少，无法对模型构建有实质性的帮助。
- 另一种策略是将 `Cabin` 是否缺失作为一个特征，即转换为一个二元特征，表示船舱号是否已知。
- 如果要利用 `Cabin` 信息，也可以考虑将所有缺失值归为一个新类别，例如用一个特殊值表示。

在后续的分析和模型训练中，需要根据上述观察和 `Cabin` 的处理策略来决定如何利用这些类别型变量。对于 `Cabin`，特别是要决定是直接舍弃这个特征，还是通过某种方式尝试利用它，这将取决于这个特征对模型预测能力的影响。

我们进一步考虑 `Name` 和 `Ticket`，如前所述，可能在里面能发现一些有用信息。

对于乘客的名字，分析原始数据发现，除了 Mr., Mrs., Miss 等，还有像 Capt., Sir. 等，可能是标识乘客身份的单词，这些身份单词反映了乘客的社会经济地位、年龄、性别，甚至是与他人关系的信息。比如:

- "Braund, Mr. Owen Harris" 中的 "Mr." 表示 Owen Harris Braund 是一位成年男性。
- "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" 中的 "Mrs." 表示 Florence Briggs Thayer（Cumings）是已婚女性，她的丈夫名为 John Bradley Cumings。
- "Heikkinen, Miss. Laina" 中的 "Miss." 表示 Laina Heikkinen 是一位未婚女性。
- "Palsson, Master. Gosta Leonard" 中的 "Master." 表示 Gosta Leonard Palsson 是一位年幼的男孩，这个头衔通常用于表示未成年男性。

可以看出，这些头衔不仅反映了乘客的性别和婚姻状况，还可能间接反映了他们的年龄和社会地位。例如，"Master" 通常用于较年轻的男性，而 "Mr."、"Mrs." 和 "Miss." 则用于成年人，其中 "Mrs." 通常暗示该女性已婚，这在当时可能也与她的社会地位相关。

从名字中提取这些头衔可以作为一个有用的特征，因为它们可能与乘客的生存率相关。历史数据表明，妇女和儿童在灾难中的生存机会通常高于成年男性，因此这些头衔可能帮助我们预测乘客的生存概率。通过将这些头衔作为模型的一个特征，我们可以更准确地预测乘客的生存情况。这种类型的特征工程是在构建预测模型时常见且有价值的步骤。

我们可以通过分割名字字符串，提取出每个乘客的头衔，并分析不同头衔的分布情况。由于乘客名字具有显著的特征，第一个 `,` 前面是姓，跟着有一个空格，然后就是头衔，头衔后更正 `.`。这使分割名字字符串就比较容易了，主要运用字符串的 `split()` 方法就可以了。具体操作如下：

```python
# 提取头衔
train_data['Title'] = train_data['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])

# 分析头衔分布
title_counts = train_data['Title'].value_counts()

# 关联头衔和生存率
title_survival_rates = (train_data.groupby('Title')['Survived'].mean().sort_values(ascending=False)).round(2)

# 对罕见头衔进行分组
rare_titles = title_counts[title_counts < 10].index  # 假设少于10个乘客的头衔为罕见头衔
train_data['Title_Grouped'] = train_data['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)

# 再次计算分组后的头衔和生存率关系
title_grouped_survival_rates = (train_data.groupby('Title_Grouped')['Survived'].mean().sort_values(ascending=False)).round(2)


# 打印头衔分布
print(f"title counts: {title_counts}")

# 打印头衔和生存率的关系
print(f"title survival rates: {title_survival_rates}")

# 打印分组后的头衔和生存率的关系
print(f"title grouped survival rates: {title_grouped_survival_rates}")

# 绘制头衔分布的条形图
plt.figure()
sns.barplot(x=title_counts.index, y=title_counts.values)
plt.title('Distribution of Titles')
plt.ylabel('Number of Passengers')
plt.xlabel('Title')
plt.xticks(rotation=90)

# 绘制头衔和生存率的关系条形图
plt.figure()
sns.barplot(x=title_survival_rates.index, y=title_survival_rates.values)
plt.title('Survival Rate by Title')
plt.ylabel('Survival Rate')
plt.xlabel('Title')
plt.xticks(rotation=90)

# 绘制分组后的头衔和生存率的关系条形图
plt.figure()
sns.barplot(x=title_grouped_survival_rates.index, y=title_grouped_survival_rates.values)
plt.title('Survival Rate by Grouped Title')
plt.ylabel('Survival Rate')
plt.xlabel('Grouped Title')
plt.xticks(rotation=90)

# 调整子图间距
plt.show()
```

其结果如下：

![](/assets/images/ml/titianic_factor_title_dist_1.png)

![](/assets/images/ml/titianic_factor_title_dist_2.png)

![](/assets/images/ml/titianic_factor_title_dist_3.png)

借助于以上分析结果，我们可以发现

- **Mr**：最常见的头衔，有 517 名乘客，表示已婚或成年男性。
- **Miss**：第二常见，有 182 名未婚女性。
- **Mrs**：有 125 名已婚女性。
- **Master**：有 40 名年幼的男孩。
- 其他头衔如 **Dr**、**Rev** 等出现的次数较少，表示这些头衔的乘客在样本中较为罕见。

从头衔与生存率（Title Survival Rates）的关系上，我们可以发现

- 一些罕见头衔（如 **the Countess**、**Mlle**、**Sir**、**Ms**、**Lady**、**Mme**）的生存率是 100%，但这可能是由于样本量较小，不足以作出统计上的一般性结论。
- **Mrs**（已婚女性）和 **Miss**（未婚女性）的生存率较高，分别为 79%和 70%，这与“妇女和儿童优先”的救生原则相吻合。
- **Master**（年幼男孩）的生存率也相对较高，为 57%。
- **Mr**（成年男性）的生存率最低，仅为 16%，反映了成年男性在灾难中的生存几率较低。
- 有些头衔如 **Rev**（牧师）、**Don**、**Jonkheer**、**Capt**（船长）的生存率为 0%，但这可能是由于样本量太小，不能确定这是否具有统计意义。

从分组后的头衔与生存率（Title Grouped Survival Rates）的关系上，我们可以进一步发现

- 将罕见头衔归类为 **Rare** 后，可以看到 **Mrs**、**Miss**、**Master** 的生存率仍然较高。
- **Rare** 类别的生存率为 44%，高于 **Mr**，但由于包含多种不同的头衔，这个数字可能不够具体。
- **Mr** 的生存率依然是最低的，为 16%。

通过这些分析，我们可以看到头衔确实是一个强有力的特征，因为它在很大程度上反映了乘客的性别、社会地位和年龄，这些因素显然影响了乘客的生存率。

下面，对 `Ticket` 特征进行分类并提取有用信息，我们可以探索票号的结构，看看是否可以从中识别出任何模式或分类。票号可能包含字母和数字，其中字母可能表示票的种类或发行地点，而数字可能是序列号。我们可以尝试将票号分解为前缀和数字两部分，以查看是否存在与生存率相关的模式。

1. **提取票号前缀**：如果票号中包含字母，我们可以将这些字母作为票号的前缀。如果票号只包含数字，我们可以将其前缀设为"None"或一个特殊标记。
2. **分析票号前缀的分布**：统计不同前缀的频率，看看哪些前缀最常见，哪些较为罕见。
3. **关联票号前缀和生存率**：分析不同票号前缀的乘客生存率，看看是否有特定的前缀与较高或较低的生存率相关。
4. **票号长度**：考虑分析票号长度是否与生存率有关。不同的票号长度可能反映了不同的票务系统或发行批次。

代码示例如下：

```python
# 提取票号前缀
train_data['Ticket_Prefix'] = train_data['Ticket'].apply(lambda x: ''.join(filter(str.isalpha, x.split(' ')[0])) if not x.isdigit() else 'None')

# 分析票号前缀的分布
ticket_prefix_counts = train_data['Ticket_Prefix'].value_counts()

print(f"Ticket Prefix Counts: {ticket_prefix_counts}")

# 关联票号前缀和生存率
ticket_prefix_survival_rates = (train_data.groupby('Ticket_Prefix')['Survived'].mean().sort_values(ascending=False)).round(2)

print(f"Ticket Prefix Survival Rates: {ticket_prefix_survival_rates}")

# 可视化票号前缀分布
plt.figure()
plt.subplot(1, 2, 1)
sns.barplot(x=ticket_prefix_counts.index, y=ticket_prefix_counts.values)
plt.title('Distribution of Ticket Prefixes')
plt.ylabel('Frequency')
plt.xlabel('Ticket Prefix')
plt.xticks(rotation=90)

# 可视化票号前缀和生存率的关系
plt.subplot(1, 2, 2)
sns.barplot(x=ticket_prefix_survival_rates.index, y=ticket_prefix_survival_rates.values)
plt.title('Survival Rate by Ticket Prefix')
plt.ylabel('Survival Rate')
plt.xlabel('Ticket Prefix')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
```

结果如下：

![](/assets/images/ml/titianic_factor_ticket_dist.png)

从 Ticket Prefix Counts 图中可以看出

- **None** 是最常见的“前缀”，表示没有明显的字母前缀的票号，有 661 张票。
- **PC**, **CA**, **A**, **STONO** 等是接下来最常见的票号前缀。
- 其他前缀如 **SC**, **SWPP**, **FCC** 等出现的次数相对较少。

对于 Ticket Prefix Survival Rates

- **SC** 和 **SWPP** 前缀的票号有最高的生存率（1.00），但由于样本量可能较小，这些高生存率可能不够稳健。
- **FCC** 和 **PP** 前缀的生存率较高，分别为 0.80 和 0.67。
- **PC** 前缀的票有较高的生存率（0.65），这可能表明持有这种票的乘客处于较高的船舱等级或有其他生存优势。
- **None** 前缀的票，即没有明显前缀的票，生存率为 0.38，这可能是最具代表性的一般情况。
- 有些前缀如 **SCOW**, **SOP**, **SOPP**, **SOTONO**, **AS**, **SP**, **Fa**, **FC**, **CASOTON**, **SCA** 的生存率为 0，但这些数据可能由于样本量较小而不具代表性。

可以看出，票号前缀与乘客的生存率之间存在一定的关联。一些前缀似乎与较高的生存率相关联，这可能反映了乘客的船舱等级、乘客类型或票务渠道等因素； 高频前缀（如 **None**, **PC**, **CA**）可能代表了更常见的票务类别，而与之关联的生存率可能更具有一般性的指示意义；罕见前缀的生存率可能受到随机波动的影响较大，因此在对这些数据进行解释时需要更加谨慎。

在构建预测模型时，考虑将票号前缀作为一个特征可能有助于提高模型的准确性，特别是那些与生存率有明显相关性的前缀。然而，对于样本量较小的前缀类别，可能需要谨慎处理，以避免模型过度拟合这些可能由于随机因素而出现的生存率模式。综合来说，我们可能由此发现票号前缀或长度与乘客生存率之间的相关性，这可以为我们提供额外的特征，用于改进预测模型。

最后，我们再回过头来看看缺失值的情况。从上面的分析可以发现，`train_data` 中的 `Age`，`Cabin` 和 `Embarked` 三个特征存在缺失值。其中 `Cabin` 的缺失值占比极高（有 687 个缺失值，占总数据的约 77.10%），可以遵循前面所述，采用直接删除该特征或转换为是否缺失的二元特征的方式可能更为合理。对于 `Age`，有 177 个缺失值，占总数据的约 19.87%。这个比例相对较小，可以考虑使用统计方法（如中位数或根据其他特征分组的中位数）或模型预测方法来填充这些缺失值。对于 `Embarked`，仅有 2 个缺失值，占总数据的约 0.22%。由于数量非常少，可以用出现最频繁的港口来填充这些缺失值，或者基于与 Embarked 最相关的特征（如 `Fare` 或 `Pclass`）来推断可能的登船港口。

对于有缺失值的特征的处理策略选择问题，主要还是得检查它们是否与其他特征相关，特别是与目标值的关系，从而判断缺失值出现的随机性或是否存在某种模式。如果存在某种模式，直接删除可能并不是一个明智的选择。考察不同特征的关系就落脚到多因素分析上面了，接下来我们开始多因素分析。

<hr/>

### 双因素或多因素分析

该部分分析的主要目的是分析特征之间的相关性，探索特征与目标变量（生存情况）之间的关系，帮助我们理解特征之间的相互作用以及它们是如何共同影响泰坦尼克号乘客生存率。可以采用相关系数矩阵、热力图、点图和箱型图等方式呈现分析结果。其实，在分析 `Name` 和 `Ticket` 时，我们已经分析了其与目标变量的关系。下面试着从如下几个方面做更为深入的分析。

1. **性别和船舱等级（Sex and Pclass）**：
   - 分析不同船舱等级中男性和女性的生存率。
   - 探讨船舱等级是否影响性别与生存率之间的关系。
2. **年龄、性别和生存率（Age, Sex, and Survived）**：
   - 分析不同性别和年龄组的生存率。
   - 研究儿童（如定义为 16 岁以下）与成人在不同性别下的生存率差异。
3. **票价、船舱等级和生存率（Fare, Pclass, and Survived）**：
   - 探讨票价和船舱等级如何共同影响生存率。
   - 检查高票价是否与高生存率相关，以及这种关系是否在所有船舱等级中都成立。
4. **头衔、性别和年龄（Title, Sex, and Age）**：
   - 分析不同头衔对应的年龄分布和性别比例。
   - 研究不同头衔的乘客生存率是否受性别和年龄的影响。
5. **家庭规模、性别和生存率（Family Size, Sex, and Survived）**：
   - 创建家庭规模变量（SibSp + Parch），分析家庭规模对生存率的影响。
   - 研究家庭规模是否对男性和女性乘客的生存率有不同的影响。
6. **登船港口、船舱等级和生存率（Embarked, Pclass, and Survived）**：
   - 分析不同登船港口的乘客在不同船舱等级下的生存率。
   - 探讨登船港口是否与船舱等级和生存率之间存在交互作用。
7. **票号前缀、船舱等级和生存率（Ticket Prefix, Pclass, and Survived）**：
   - 分析不同票号前缀的乘客在不同船舱等级下的生存率。
   - 探讨票号前缀是否为船舱等级和生存率之间的关系提供了额外的信息。

对于性别和船舱等级对生存率的影响分析，我们可以使用分组、汇总，当然可视化必不可少，来探索数据。以下是详细步骤和相应的示例代码：

1. **分组数据**：首先，我们可以按性别（`Sex`）和船舱等级（`Pclass`）分组，然后计算每组的平均生存率。
2. **数据汇总**：使用分组数据创建一个汇总表，显示每个性别和船舱等级组合的生存率。
3. **数据可视化**：通过可视化手段展示性别和船舱等级如何共同影响生存率，可以更直观地理解这些变量之间的关系。

```python
# 分组并计算生存率
survival_rates = train_data.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()

# 数据可视化
sns.heatmap(survival_rates, annot=True, fmt=".2f")
plt.title('Survival Rates by Sex and Pclass')
plt.ylabel('Sex')
plt.xlabel('Pclass')
plt.show()
```

这段代码首先按性别和船舱等级分组计算生存率，然后使用热图展示这些分组的生存率。在热图中，每个格子的颜色深浅表示生存率的高低，数值则给出了具体的生存率。我们期望能从该分析中得知

- 在不同船舱等级中，男性和女性的生存率分别是多少？
- 船舱等级是否对男性和女性的生存率差异产生了影响？

一般而言，我们可能会发现女性的生存率普遍高于男性，且头等舱（Pclass 1）的乘客生存率高于二等舱和三等舱。

结果如下：

![](/assets/images/ml/titanic_sex_pclass_survival_rates.png)

这个结果表明了船舱等级（Pclass）和性别（Sex）对生存率的共同影响：

1. **女性乘客的生存率**：

   - 头等舱女性乘客的生存率最高，接近 96.81%。
   - 二等舱女性的生存率也很高，达到 92.11%。
   - 三等舱女性的生存率明显降低，为 50%。尽管降低，这个比率仍然显著高于所有类别的男性乘客。

2. **男性乘客的生存率**：
   - 头等舱男性的生存率为 36.89%，在男性中是最高的。
   - 二等舱男性的生存率下降到 15.74%。
   - 三等舱男性的生存率最低，仅为 13.54%。

大致可以得出如下结论：

- **性别影响**：在所有船舱等级中，女性的生存率都显著高于男性。这可能反映了“妇女和儿童优先”政策的实施以及社会对性别角色的期望。
- **船舱等级影响**：对于两性，生存率都随着船舱等级的提高而增加。头等舱乘客的生存率显著高于其他船舱等级，这可能反映了社会经济地位在紧急情况下的生存机会中的作用。
- **性别与船舱等级的交互影响**：虽然女性在所有等级的船舱中生存率都较高，但三等舱的女性乘客生存率与一等舱和二等舱相比有显著下降。这可能表明，尽管性别是一个强有力的生存预测因子，船舱等级也在生存机会中扮演了重要角色。

这些分析结果为我们提供了关于 Titanic 上不同群体乘客生存机会的深入见解，并且强调了在灾难情况下性别和社会经济地位的重要性。

对于年龄、性别与生存率之间的关系，我们可以将年龄分成几个组来观察不同年龄段的乘客生存率如何受性别的影响，并特别注意儿童与成人的生存率差异。具体步骤如下：

1. **年龄分组**：首先，我们需要将年龄分成几个组，例如：儿童（0-16 岁）、青少年（16-25 岁）、成年人（25-60 岁）、老年人（60 岁以上）。
2. **计算生存率**：对每个年龄组和性别的组合计算生存率。
3. **数据可视化**：使用图表展示不同年龄组和性别的生存率，以便直观比较。
4. **儿童与成人的比较**：特别关注儿童（16 岁以下）与成人在不同性别下的生存率差异。

示例代码：

```python
# 定义年龄分组函数
def age_group(age):
    if age <= 16:
        return 'Child'
    elif age <= 25:
        return 'Youth'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# 应用年龄分组
train_data['Age_Group'] = train_data['Age'].apply(age_group)

# 计算每个年龄组和性别组合的生存率
age_sex_survival = train_data.groupby(['Age_Group', 'Sex'])['Survived'].mean().unstack()

# 数据可视化
plt.figure()
age_sex_survival.plot(kind='bar')
plt.title('Survival Rates by Age Group and Sex')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.legend(title='Sex')

plt.show()
```

这段代码将首先为每位乘客分配一个年龄组，然后计算不同年龄组和性别组合的生存率，并通过条形图展示结果。这样，我们可以清晰地看到不同年龄段乘客的生存率是如何受到性别影响的，特别是儿童与成人之间的差异。其结果如下：

![](/assets/images/ml/titanic_age_sex_survival_rates.png)

可以发现:

1. **成年人（Adult）**：
   - 女性的生存率约为 78.68%，远高于男性的 21.18%。
   - 这表明成年女性的生存机会比成年男性高得多，可能反映了救援时“妇女优先”的原则。
2. **儿童（Child）**：
   - 儿童中，女性的生存率为 67.35%，男性为 43.14%。
   - 男性儿童的生存率显著高于成年男性，这可能反映了“儿童优先”的救援原则。
3. **老年人（Senior）**：
   - 老年女性的生存率为 69.64%，而老年男性仅为 12.59%。
   - 老年女性的生存率仍然显著高于老年男性，尽管他们的生存率比成年人和儿童低。
4. **青年（Youth）**：
   - 青年女性的生存率为 73.97%，而青年男性为 11.72%。
   - 青年女性的生存率高于所有年龄组的男性，但略低于成年女性。

大致可以得出如下结论：

- **性别影响**：在所有年龄组中，女性的生存率都显著高于男性，这一结果与整体的 Titanic 生存数据一致，再次强调了性别在生存机会上的重要性。
- **年龄影响**：儿童的生存率普遍高于其他年龄组，尤其是男性儿童，这可能是因为救生时给予儿童优先考虑。老年人的生存率普遍较低，这可能是由于在紧急情况下，老年人的身体状况可能不利于生存。
- **性别与年龄的交互影响**：虽然所有年龄组的女性生存率都高于男性，但不同年龄组间的生存率差异也值得关注。特别是，男性儿童与成年男性相比有较大的生存率提升，而青年和成年女性的生存率差异较小。

对于票价（`Fare`）、船舱等级（`Pclass`）和生存率（`Survived`）之间的关系，我们可以进行以下分析：

1. **票价与生存率的关系**：我们可以分析票价和生存率之间的关系，看看是否高票价的乘客有更高的生存率。
2. **船舱等级与生存率的关系**：分析不同船舱等级的乘客生存率，确定船舱等级是否是影响生存率的重要因素。
3. **票价、船舱等级和生存率的综合分析**：我们将同时考虑票价和船舱等级对生存率的影响，看看这两个因素如何共同作用。
4. **分组和可视化**：我们可以将票价分成几个区间，然后计算每个区间内不同船舱等级乘客的生存率，最后通过图表进行可视化。

示例代码：

```python
# 创建票价区间
train_data['Fare_Bin'] = pd.qcut(train_data['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# 分析票价区间、船舱等级与生存率的关系
fare_pclass_survival = train_data.groupby(['Fare_Bin', 'Pclass'])['Survived'].mean().unstack()

# 可视化
plt.figure()
sns.heatmap(fare_pclass_survival, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Survival Rates by Fare Bin and Pclass')
plt.ylabel('Fare Bin')
plt.xlabel('Pclass')

plt.show()
```

这段代码首先将票价分成四个区间，然后计算每个票价区间和船舱等级组合的平均生存率，并通过热图展示结果。这样我们可以直观地看到票价和船舱等级是如何共同影响生存率的，并验证是否高票价总是对应更高的生存率，以及这种趋势是否在所有船舱等级中都成立。

结果如下：
![](/assets/images/ml/titanic_fare_pclass_survival_rates.png)

结果表明：

1. **低票价区间（Low Fare Bin）**：
   - 一等舱和二等舱的生存率为 0%，这可能是样本数量不足或者这个票价区间确实没有一等舱和二等舱的乘客。
   - 三等舱的生存率为 20.85%，是低票价区间中唯一有生存率数据的船舱等级。
2. **中等票价区间（Medium Fare Bin）**：
   - 二等舱的生存率为 38.37%，三等舱为 25.36%。这表明即使票价较低，二等舱的乘客生存率也高于三等舱。
   - 一等舱在这个票价区间没有数据（NaN），可能是没有一等舱的票价在这个区间内。
3. **高票价区间（High Fare Bin）**：
   - 一等舱的生存率显著上升到 52.94%，二等舱为 60%，三等舱为 31.68%。可以看出，在高票价区间，船舱等级对生存率的影响仍然显著。
4. **非常高的票价区间（Very High Fare Bin）**：
   - 一等舱的生存率进一步上升到 68.55%，二等舱为 54.55%。
   - 三等舱的生存率下降到 19.51%，这可能是因为在非常高的票价区间，购买三等舱的乘客较少，或者这个区间内的三等舱乘客特殊情况影响了生存率。

大致可以得出如下结论：

- **票价与生存率**：高票价区间的乘客通常有更高的生存率，特别是在一等舱和二等舱中。这可能反映了经济状况较好的乘客有更多资源和更好的机会在紧急情况下生存下来。
- **船舱等级与生存率**：在所有票价区间中，一等舱乘客的生存率普遍高于二等舱和三等舱，这强调了船舱等级作为影响生存率的重要因素。
- **票价与船舱等级的交互作用**：虽然高票价通常意味着更高的生存率，但三等舱在非常高的票价区间的生存率反而下降，这表明单一因素（如票价）并不能完全决定生存率，船舱等级和其他因素也起着重要作用。

对于头衔、性别和年龄，我们需要关注头衔对应的年龄分布、性别比例以及如何影响生存率。具体分析步骤如下：

1. **头衔与年龄分布**：我们可以分析不同头衔对应的年龄分布，了解各个头衔年龄范围的差异。
2. **头衔与性别比例**：分析不同头衔的性别比例，这有助于我们理解头衔与性别的关系。
3. **头衔、性别和生存率**：我们将分析不同头衔的乘客在不同性别和年龄组下的生存率，看看这些因素是如何交互影响生存率的。
4. **数据可视化**：使用图表来可视化上述分析，帮助直观理解不同头衔的年龄分布、性别比例以及生存率情况。

示例代码：

```python
# 头衔与年龄分布
sns.boxplot(x='Title', y='Age', data=train_data)
plt.title('Age Distribution by Title')
plt.xticks(rotation=90)

# 头衔与性别比例
title_sex_count = train_data.groupby('Title')['Sex'].value_counts().unstack().fillna(0)
title_sex_count.plot(kind='bar', stacked=True)
plt.title('Sex Proportion by Title')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=90)

# 头衔、性别和生存率
sns.barplot(x='Title', y='Survived', hue='Sex', data=train_data)
plt.title('Survival Rates by Title and Sex')
plt.xticks(rotation=90)
plt.legend(loc="upper left", title='Sex')

plt.show()
```

这段代码首先绘制了不同头衔对应的年龄分布盒图，然后绘制了头衔与性别比例的堆叠条形图，最后展示了不同头衔和性别下的生存率条形图。通过这些图表，我们可以详细了解不同头衔的年龄分布、性别比例以及它们如何影响生存率。

可以得到：

![](/assets/images/ml/titanic_age_title.png)

![](/assets/images/ml/titanic_sex_title.png)

![](/assets/images/ml/titanic_sex_age_title.png)

从以上结果（结合 `train_data.groupby('Title').describe()['Age']`），我们可以大致发现：

1. **头衔与年龄分布**：
   - 箱型图显示了每个头衔对应的年龄分布，包括中位数、四分位数和异常值。从图中可以看出，不同头衔对应的年龄分布差异显著。例如，拥有**Master** 头衔的乘客通常很年轻，平均年龄约为 4.5 岁，这符合这些头衔通常用于孩子的预期。
   - 相比之下 **Mr** ， **Mrs**， **Rev**， **Dr** 头衔的平均年龄较老，分别约为 32 岁，35 岁，43 岁和 42 岁。
2. **头衔与性别比例**：
   - 头衔与性别的堆叠条形图揭示了不同头衔中男性和女性的比例。一些头衔是专门男性或女性使用的。例如， **Mr** 专属于男性，而 **Miss** 和**Mrs** 则专属于女性 （显而易见:scream_cat:）。
   - 显然，**Mr** 将是男性占绝大多数的头衔，而 **Mrs** 和 **Miss** 则主要是女性。**Master** 头衔是专门用于男孩的，**Dr** 在这个数据集中主要是男性。
3. **头衔、性别和生存率**：
   - Survival Rates by Title and Sex 条形图展示了不同头衔和性别组合的生存率。不同头衔和性别的生存率有所不同。例如，**Miss** 和 **Mrs** 的生存率相对较高，表明这些头衔的女性更有可能生存下来。
   - 拥有 **Master** 头衔的年轻男性也比同头衔的其他年龄组有更高的生存率，这表明在救生时儿童被给予了优先考虑。

进而得出如下结论：

- **年龄分布**：不同头衔的年龄分布可以反映乘客的年龄结构，有助于我们理解特定头衔群体的特点。
- **性别比例**：头衔与性别的关系揭示了社会角色和乘客身份，不同头衔的性别比例有助于我们进一步分析生存率。
- **生存率**：头衔、性别和生存率之间的关系可以帮助我们理解在灾难中社会地位、性别和年龄是如何影响个人生存机会的。

对于家庭规模、性别和生存率之间的关系，我们可以通过添加 `SibSp` 和 `Parch` 创建一个家庭规模变量 `FamilySize`，然后分析其对生存率的影响，注意，家庭规模中，别忘记添加本人。示例代码如下：

```python
# 通过添加 SibSp 和 Parch 创建一个家庭规模变量，然后分析其对生存率的影响
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1  # 加1是为了包括乘客本人

# 分析家庭规模对生存率的影响
family_survival_rate = train_data.groupby('FamilySize')['Survived'].mean()

# 分析家庭规模和性别对生存率的共同影响
family_sex_survival_rate = train_data.groupby(['FamilySize', 'Sex'])['Survived'].mean().unstack()

# 显示分析结果
family_survival_rate, family_sex_survival_rate

# 结果的可视化
fig, axes = plt.subplots(1, 2)

sns.barplot(x=family_survival_rate.index, y=family_survival_rate.values, ax=axes[0])
axes[0].set_title("Survival Rate by Family Size")
axes[0].set_ylabel("Survival Rate")
axes[0].set_xlabel("Family Size")

family_sex_survival_rate.plot(kind='bar', ax=axes[1])
axes[1].set_title("Survival Rate by Family Size and Sex")
axes[1].set_ylabel("Survival Rate")
axes[1].set_xlabel("Family Size")
axes[1].tick_params(axis='x', labelrotation=0)

plt.tight_layout()
plt.show()
```

结果如下：

![](/assets/images/ml/titanic_family_size_survival_analysis.png)

可以发现：

1. **家庭规模对生存率的影响**：
   - 家庭规模为 6 的乘客生存率最低，仅为 13.64%。
   - 相比之下，家庭规模在 2 到 4 人之间的乘客生存率较高，特别是当家庭规模为 4 时，生存率最高，达到 72.41%。
   - 独行乘客（家庭规模为 1）的生存率为 30.35%，虽然不是最低，但相对较低。
   - 随着家庭规模增加到 5 人以上，生存率显著下降，特别是当家庭规模为 8 人和 11 人时，生存率为 0%。
2. **家庭规模和性别对生存率的共同影响**：
   - 在所有家庭规模中，女性的生存率普遍高于男性。
   - 独行的女性乘客（家庭规模为 1）的生存率约为 78.57%，而独行的男性乘客生存率只有 15.57%。
   - 对于家庭规模在 2 到 4 人的乘客，女性的生存率继续保持较高水平（81.61%至 84.21%），而在这个范围内，男性的生存率也有所提高，尤其是当家庭规模为 4 时，男性的生存率达到 50%。
   - 家庭规模大于 4 人时，男女乘客的生存率都有所下降，尤其是男性，家庭规模为 5 人及以上时几乎没有生还者。

对于登船港口、船舱等级和生存率之间的关系，我们任然可以聚焦于以下两点的分析：

- 分析不同登船港口的乘客在不同船舱等级下的生存率。
- 探讨登船港口是否与船舱等级和生存率之间存在交互作用。

示例代码如下：

```python
# 分析不同登船口岸对生存率的影响
embarked_pclass_survival_rate = train_data.groupby(['Embarked', 'Pclass'])['Survived'].mean().unstack()

print(f"Embarked Pclass Survival Rate:\n {embarked_pclass_survival_rate}")

# 结果可视化
plt.figure()
sns.heatmap(embarked_pclass_survival_rate, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Survival Rate by Embarkation Port and Pclass')
plt.ylabel('Embarkation Port')
plt.xlabel('Pclass')
plt.yticks(rotation=0)

plt.show()
```

结果如下：

![](/assets/images/ml/titanic_embarked_pclass_survival_rate.png)

可以发现：

1. **不同登船港口和船舱等级的乘客生存率**：
   - 对于从 C 港口（Cherbourg）登船的乘客，一等舱的生存率最高，为 69.41%。二等舱和三等舱的生存率分别为 52.94%和 37.88%。
   - 从 Q 港口（Queenstown）登船的乘客中，二等舱的生存率最高，为 66.67%。一等舱和三等舱的生存率分别为 50.00%和 37.50%。
   - 对于从 S 港口（Southampton）登船的乘客，一等舱的生存率为 58.27%，二等舱为 46.34%，三等舱最低，为 18.98%。
2. **登船港口与船舱等级的交互作用对生存率的影响**：

- C 港的一等舱乘客有最高的生存率，这可能反映了经济地位较高的乘客更多选择从该港口登船，且更倾向于购买高等级船舱。
- Q 港的数据显示，尽管乘客数量可能较少，但二等舱乘客的生存率出奇地高，可能是由于特定的社会经济因素或该港口乘客的特殊组成。
- S 港为主要的登船港口，其所有船舱等级的生存率普遍低于从 Cherbourg 登船的乘客，特别是三等舱，生存率明显较低。

最后，我们借助于热图或堆叠条形图分析票号前缀、船舱等级和生存率之间的关系，分析不同票号前缀的乘客在不同船舱等级下的生存率以及探讨票号前缀是否为船舱等级和生存率之间的关系提供了额外的信息。

示例代码：

```python
# 分析票号前缀与Plclass对生存率的影响
ticket_prefix_pclass_survival_rate = train_data.groupby(['Ticket_Prefix', 'Pclass'])['Survived'].mean().unstack().fillna(0)

print(f"ticket prefix pclass survival rate: {ticket_prefix_pclass_survival_rate}")

sns.heatmap(ticket_prefix_pclass_survival_rate, annot=True, fmt=".2f")
plt.title('Survival Rate by Ticket Prefix and Pclass')
plt.ylabel('Ticket Prefix')
plt.xlabel('Pclass')

plt.show()
```

结果如下：

![](/assets/images/ml/titanic_ticket_prefix_pclass_survival_rate.png)

可以发现：

1. **不同票号前缀和船舱等级的乘客生存率**：

   - 票号前缀为 **None**（即没有前缀，只有数字的票号）的乘客在一等舱有较高的生存率（约 62.18%），二等舱和三等舱的生存率分别为 46.99%和 24.15%。
   - **PC** 前缀的票在一等舱有相对较高的生存率（65%），而在二、三等舱的生存率为 0%。
   - 某些前缀如 **SC** 在二等舱有 100%的生存率，但样本可能较小，需要谨慎解读。
   - **LINE** 前缀的票在三等舱有 25%的生存率。
   - 其他票号前缀如 **C**、**CA**、**Fa**、**PP** 在特定船舱等级中的生存率差异显著，可能反映了不同票务类别或船票购买方式与乘客生存率的关系。

2. 票号前缀对船舱等级和生存率关系的影响：
   - 票号前缀似乎为船舱等级和生存率之间的关系提供了额外信息。特定的票号前缀可能与乘客的生存率有关，这可能反映了不同的票务类别、服务或船舱位置等因素。
   - 例如，**None** 前缀可能代表标准票务流程，而特定的字母前缀如 **PC** 可能代表更高端的服务或特殊的船舱位置。

双变量和多变量分析到一段落，结合 EDA 分析结果，我们再回过头来考虑下缺失值处理策略。

1. **年龄（Age）**：从上面分析可以发现，年龄是一个重要因素，对生存率有明显影响。考虑到头衔与年龄分布有关，我们可以根据乘客的头衔来估算缺失的年龄值。例如，可以使用具有相同头衔乘客的年龄中位数来填充对应乘客的缺失年龄。
2. **船舱号（Cabin）**：船舱号缺失较多，但从船舱号中可能提取的船舱等级信息对生存率有影响。如果直接删除可能会丢失大量数据。一种策略是将缺失的船舱号视为一个单独的类别，或者根据票价（Fare）和船舱等级（Pclass）来推断可能的船舱区域。
3. **登船港口（Embarked）**：登船港口的缺失值相对较少。考虑到不同登船港口的乘客生存率存在差异，可以用最常见的登船港口来填充缺失值，或者根据票价和船舱等级进行更细致的分析来推断缺失的登船港口。
4. **票号前缀（Ticket Prefix）**：票号前缀反映了票的类型或购买方式，可能与生存率有关。如果票号前缀缺失，可以考虑将其归类为一个特殊类别，或根据相关特征如船舱等级和票价来推断。

通过结合多变量分析结果来指导缺失值的处理，我们可以更合理地填补缺失值，同时保留数据中的重要信息，从而提高后续模型分析的准确性和可靠性。在实际操作中，每种策略的选择都应基于对数据的深入理解和详细分析。

<hr/>

## 特征工程

完成探索性数据分析后，现在可以开始进行特征工程（Feature Engineering）。特征工程是利用数据中的信息来创建新特征或修改现有特征以提高模型的性能的过程。这是模型构建过程中至关重要的一步，因为它直接影响模型的学习能力和预测性能。在特征工程阶段，我们将重点考虑以下几个方向：

1. **特征创建**：基于对现有数据的理解，可以创建新的特征。例如，根据家庭成员的数量创建是否独行这一特征，或者结合 `SibSp` 和 `Parch` 来形成一个新的家庭规模特征。
2. **特征转换**：对某些特征进行转换，例如将连续变量分箱（Binning）、进行对数转换、归一化或标准化等，以改善模型的性能。
3. **特征编码**：对分类数据进行编码，如使用独热编码（One-Hot Encoding）、标签编码（Label Encoding）或目标编码（Target Encoding）等方法。
4. **缺失值处理**：根据 EDA 阶段的发现，采用合适的方法填补缺失值，如使用中位数、平均数、众数或基于模型的填充方法。
5. **特征选择**：选择与目标变量最相关的特征，并移除不相关或冗余的特征，可以使用各种特征选择技术，如基于统计的方法、模型特征重要性评估等。
6. **交互特征**：考虑不同特征之间的交互，例如，可能需要结合 `Pclass` 和 `Age` 来创建新的交互特征，反映不同舱位年龄组的乘客生存率差异。

完成特征工程后，期望能拥有一组更能代表数据本质、更适合机器学习模型的特征集，这将可能提升模型的预测能力。特征工程是一个迭代的过程，可能需要多次调整和验证来找到最优的特征组合。

为了给特征工程提供必要的参考，有必要在进行特征工程之前构建一个基线模型。基线模型提供了一个参考点，可以帮助我们评估后续特征工程和模型调优的效果。这种方法可以让我们明确地看到任何改变（不管是添加新特征、特征转换还是特征选择）是否真正带来了性能的提升。具体来说，基线模型大致有如下几点的作用：

1. **性能基准**：基线模型为后续改进提供了一个基准。通过比较基线模型和改进后模型的性能，我们可以判断新特征或模型调整是否有效。
2. **快速反馈**：基线模型通常应该简单快速，它允许我们迅速获得关于数据和模型选择的初步反馈。
3. **排除问题**：如果基线模型的性能异常差，那可能意味着数据中存在问题，如数据质量问题、错误的标签或数据泄露等。这有助于早期发现和修正这些问题。

构建基线模型的步骤：

1. **选择模型**：选择一个适合问题类型的简单模型。对于本问题，由于是一个典型的分类问题，可以使用逻辑回归或决策树。
2. **使用原始特征**：先不进行复杂的特征工程，使用原始特征训练模型。
3. **性能评估**：使用适当的评估指标（如准确率、AUC、均方误差等）评估模型性能。
4. **记录结果**：记录基线模型的性能结果，以便后续与改进后的模型进行比较。

进行特征工程和进一步的模型调优后，我们可以将新模型的性能与这个基线模型进行比较，这有助于量化特征工程的效果并指导后续的优化方向。

<hr/>

### 基线模型的构建

这里我们采用 OOP 的方式呈现相关示例代码 (当然，采用过程的方式也是可以的，OOP的方式为我们呈现了清晰的结构，每个部分的职责都明确分离，便于维护和扩展)。

首先我们需要在 `titanic/titanic/` 文件夹下新建几个 `.py` 文件，如下：

```plaintext
titanic/
│
├── pyproject.toml              # Poetry 依赖文件
├── README.md                   # 项目说明文件
│
├── titanic/                    # 主项目包
│   ├── __init__.py
│   ├── data_processing.py      # 数据处理模块，包含数据预处理的类或函数，例如清洗数据、填充缺失值、特征编码等。
│   ├── model.py                # 模型相关模块，包含模型训练和评估的类，例如上面定义的 BaseModel 类。
│   └── main.py                 # 主执行脚本，用于组织数据处理流程和模型训练评估流程。
```

现在我们来设计下 `main.py` 中的代码，这是我们的主执行脚本，主要用于组织数据处理流程和模型训练评估流程。因此，这里的代码应该是读取并处理相关数据，将处理好的数据输入到构建好的模型中进行训练，当然最后应该输出相关的模型训练评估结果。假设在 `data_processing.py` 中存在一个名为 `DataProcessor` 数据处理类，其中包括相关的处理方法，比如 `preprocess()`；在 `model.py` 中构建好了一个名为 `BaseModel` 的模型类，同样，在该类下包含相关的训练和评估方法 `train()` 和 `evaluate()`。此外，由于目前建立的是一个基线模型。因此，我们不需要考虑太多的特征，也不需要太复杂的数据处理和模型，保持数据和模型尽可能简单，清晰。假设，我们当前仅考虑 `Pclass`，`Sex` 和 `Age`；分类模型采用 Logistic Regression 模型；模型评估采用准确率。这样，我们的 `main.py` 中示例代码就可以呈现为如下内容：

```python
# titanic/titanic/main.py
import pandas as pd
from tools import load_data
from data_preprocessing import DataPreprocessor
from model import BaseModel


def load_and_preprocess_data(data_path, columns):
    data = load_data(data_path)
    processor = DataPreprocessor(data, columns=columns)
    processed_data, features = processor.preprocess()
    return processed_data, features


def train_and_evaluate_model(data, features, target):
    model = BaseModel()
    model.train(data[features], data[target])
    accuracy = model.evaluate()
    return accuracy


def main():
    def main():
    data_path = "./data/raw/train.csv"
    processed_data, features = load_and_preprocess_data(data_path)
    # print(features)
    target = "Survived"

    train_and_evaluate_model(processed_data, features, target)

    print(f"Baseline Model Accuracy: {accuracy:.04f}")


if __name__ == "__main__":
    main()

```

到目前为止，我们还没有构建 `DataProcessor` 和 `BaseModel` 类及其方法，因此，还不能运行 `main.py`。接下来，根据前面的假设，我们继续完善`DataProcessor` 和 `BaseModel` 类及其方法。

对于 `BaseModel` 类及其方法， 其示例代码如下：

```python
# titanic/titanic/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)


class BaseModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=0)
        self.evaluator = None  # 在训练时设置

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.evaluator = ModelEvaluator(
            self.model, X_test, y_test
        )  # 在训练后创建评估器

    def evaluate(self):
        if self.evaluator:
            return self.evaluator.evaluate()
        else:
            raise ValueError("The model needs to be trained before evaluation.")
```

对于 `DataProcessor`，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class BaseProcessor:
    def __init__(self, data):
        self.data = data
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

     def one_hot_encode(self, column):
        encoded = self.encoder.fit_transform(self.data[[column]])
        new_cols = [f"{column}_{cat}" for cat in self.encoder.categories_[0]]
        self.data = self.data.drop(column, axis=1)
        self.data[new_cols] = pd.DataFrame(encoded, index=self.data.index)
        return self.data, new_cols

class PclassProcessor(BaseProcessor):
    def process_pclass(self):
        new_features = ["Pclass"]
        return self.data, new_features

class SexProcessor(BaseProcessor):
    def sex_one_hot_encode(self):
        self.data, new_features = super().one_hot_encode("Sex")
        return self.data, new_features

class AgeProcessor(BaseProcessor):
    def fill_missing_values(self):
        new_feature = ["AgeFillMedian"]
        self.data[new_feature[0]] = self.data["Age"].fillna(self.data["Age"].median())
        return self.data, new_feature

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.features = []

     def preprocess(self):
        """数据处理逻辑， 返回处理后的数据集以及应该考虑的特征"""

        plcass_processor = PclassProcessor(self.data)
        self.data, new_features_plcass = plcass_processor.process_pclass()
        self.features.extend(new_features_plcass)

        age_processor = AgeProcessor(self.data)
        self.data, new_features_age = age_processor.fill_missing_values()
        self.features.extend(new_features_age)

        sex_processor = SexProcessor(self.data)
        self.data, new_features_sex = sex_processor.sex_one_hot_encode()
        self.features.extend(new_features_sex)

        return self.data, self.features
```

有两点值得说明：

- 关于 `Age` 的处理。结合 EDA 分析，我们是知道该特征有缺失值，从分布上看，也存在异常值。因此，这里用了该列的中位数来填充（可能有更好的处理方式，后面再讨论）。中位数对于异常值不敏感，相对更加稳定。
- 关于 `Pclass` 特征。在数据处理中，我们并没有预处理该特征，主要是考虑到 `Pclass` 中的数值（1， 2， 3）能够直接反应生存概率的顺序关系（即1级舱生存概率最高，然后是2级，最后是3级）。Logistic Regression 模型可以直接处理这种有序的数值特征。
- 关于 `Sex` 特征。由于该特征的原始值为 `male` 和 `female`。不能直接输入到逻辑回归模型中，需要对其进行编码转换，在这里我们选择了 One-Hot 的方式。考虑到后续还要用该方法对其他类别型数据进行处理，所以，我们在基类中构建了一个专门处理One-Hot编码的方法，此方法接受一个列名。此外，为了方便在 `main` 函数中增加处理后的特征，特意在该类中 `one_hot_encoder` 方法中返回了新构建的特征名称。

到此，我们的基线模型就构建好了，运行 `main.py`， 可以得出如下结果：
```plaintext
Baseline Model Accuracy: 0.810056
```

<hr/>

### 变量：`Age`

接下来，我们试着对 `Age` 进一步处理，考虑运用不同头衔的 `Age` 中位数来填充其缺失值， 并查看对模型训练效果的影响。

从 EDA 的分析结果来看，不同年龄对生存率存在明显影响，对于该特征的缺失值，我们是否可以考虑运用不同头衔的中位数来填充会更好？基于该想法，我们适当修改 `data_preprocessing.py` 中的相关类及其方法，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class TitleProcessor(BaseProcessor):
    def extract_title(self):
        self.data["Title"] = self.data["Name"].apply(
            lambda x: x.split(", ")[1].split(". ")[0]
        )
        return self

    def group_titles(self):
        title_counts = self.data["Title"].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        self.data["Title_Grouped"] = self.data["Title"].apply(
            lambda x: "Rare" if x in rare_titles else x
        )
        return self


class AgeProcessor(BaseProcessor):
    # 其他代码保持不变

    def fill_age_by_title_group(self, title_grouped_column="Title_Grouped"):
        if title_grouped_column not in self.data.columns:
            raise ValueError(f"{title_grouped_column} column is missing in the data")

        new_feature = ["AgeFillTitleGrouped"]
        self.data[new_feature[0]] = self.data["Age"].copy()

        for title_group, group in self.data.groupby(title_grouped_column):
            median_age = group["Age"].median()
            self.data.loc[
                (self.data["Age"].isnull())
                & (self.data[title_grouped_column] == title_group),
                new_feature[0],
            ] = median_age

        # 如果仍有缺失值（例如，某个 Title_Grouped 分组内所有 Age 值都是缺失的），用总体中位数填充
        self.data[new_feature[0]] = self.data[new_feature[0]].fillna(
            self.data["Age"].median()
        )

        return self.data, new_feature


class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        title_processor = TitleProcessor(self.data)
        self.data = title_processor.extract_title().group_titles()
        age_processor = AgeProcessor(self.data)
        self.data, new_features_age = age_processor.fill_age_by_title_group()
        self.features.extend(new_features_age)

        return self.data, self.features
```

值得说明的是：

- 我们在 `AgeProcessor` 类中增加了一个新方法 `fill_age_by_title_group`，该方法实现一下目的：按不同头衔的年龄中位数来填充 `Age` 列中的对应的缺失值。
- 我们适当修改了原始的 `DataProcessor` 类中的 `preprocess` 方法。主要是用 `fill_age_by_title_group` 方法替代了之前的 `fill_missing_values` 方法。
- 考虑到我们需要先确认不同头衔，因此，针对 `Name` 特征构建了一个 `TitleProcessor` 类。在使用 `fill_age_by_title_group` 方法之前，我们先运用 `TitleProcessor` 类对数据进行了处理。
- 由于我们只修改了 `data_preprocessing.py`, 训练模型不变，由此我们并不需要修改 `model.py` 以及 `main.py`。

重新运行 `main.py`，可以得出相应的训练准确率的结果：

```plaintext
Model Accuracy (fill age by title group): 0.810056
```

:disappointed: 不要怀疑以上结果，你没看错。一通操作猛如虎，模型的准确率并没有实质性变化，与基线模型的准确度完全一致。

这种情况可能发生在几种情况下：

1. **数据特性**：如果 `Age` 列对于模型的预测影响不大，或者不同填充策略之间的差异对最终结果没有显著影响，那么准确率可能会保持一致。
2. **模型不敏感**：逻辑回归模型可能对这种细微的数据变化不太敏感（:question:），一般情况下，基于树的算法，如随机森林或梯度提升树，在一定程度上对缺失值的处理方法不太敏感。
3. **数据其他特征的影响较大**：如果数据集中还有其他特征对目标变量有强烈的预测作用，那么 `Age` 特征的变化可能不会对整体模型准确率产生显著影响。
4. **结果偶然相同**：在某些情况下，两种不同的处理方法可能恰好导致模型具有相同的准确率，这可能是偶然事件，特别是在数据集较小或模型训练过程中存在随机性的情况下。

有没有办法进一步验证为什么两种方法得到相同的准确率，当然，可以尝试以下方法：

- **混淆矩阵**：查看每种填充方法的混淆矩阵，可能会揭示不同的错误模式。
- **交叉验证**：通过交叉验证可以获得更稳健的性能估计，可能会揭示准确率的差异。
- **特征重要性**：查看模型中 `Age` 特征的重要性，以判断其对模型的影响程度。
- **其他评估指标**：除了准确率外，还可以考虑使用其他指标（如F1分数、ROC曲线下面积等）来评估模型性能的差异。

现在试着增加模型评估指标，进一步评估采用了按头衔分类后的中位数填补 `Age` 缺失值后的模型训练情况。这里主要涉及到 `evaluation.py` 文件的增加，将所有评估指标均放入到该文件中，示例代码如下：

```python
# titanic/titian/evaluation.py
class ModelEvaluator:
    def __init__(self, model, X_test, y_test, results_file):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.results_file = results_file
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                self.results = json.load(f)
        else:
            self.results = []

    def calculate_metrics(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]  # 获取正类的概率

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, average="binary"),
            "Recall": recall_score(self.y_test, y_pred, average="binary"),
            "F1 Score": f1_score(self.y_test, y_pred, average="binary"),
            "ROC AUC": roc_auc_score(self.y_test, y_proba),
        }
        return y_pred, y_proba, metrics

    def print_metrics(self, metrics):
        print("Evaluation Metrics:")
        print(pd.DataFrame([metrics], index=["Values"]))

    def print_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(
            pd.DataFrame(
                conf_matrix,
                columns=["Predicted Negative", "Predicted Positive"],
                index=["Actual Negative", "Actual Positive"],
            )
        )

    def perform_cross_validation(self, cv):
        if cv > 1:
            cross_val_accuracy = np.mean(
                cross_val_score(
                    self.model, self.X_test, self.y_test, cv=cv, scoring="accuracy"
                )
            )
            print(f"\nCross-validated Accuracy ({cv}-fold): {cross_val_accuracy:.6f}")
            return cross_val_accuracy
        return None

    def plot_roc_curve(self, y_proba):
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.6f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig("./fig/ROC.png", bbox_inches="tight")
        # plt.show()

    def save_results(self, metrics):
        try:
            with open(self.results_file, "w") as f:
                json.dump(self.results + [metrics], f)
        except IOError as e:
            print(f"Error saving results: {e}")

    def evaluate(self, cv=5):
        y_pred, y_proba, metrics = self.calculate_metrics()
        self.print_metrics(metrics)
        self.print_confusion_matrix(y_pred)
        cv_score = self.perform_cross_validation(cv)
        if cv_score is not None:
            metrics["Cross-validated Accuracy"] = cv_score
        self.plot_roc_curve(y_proba)
        self.save_results(metrics)
        return metrics
```

添加如上文件后，我们需要将 `model.py` 涉及到模型评估的代码删除，并且导入 `ModelEvaluator`。

`main.py` 中的代码可以不用修改。但由于我们在 `ModelEvaluator` 包含了评估指标结果输出。因此，可以适当修改 `main.py`，去掉之前的打印结果的部分，如下：

```python
# titanic/titanic/main.py
# 其他代码保持不变
def train_and_evaluate_model(data, features, target, results_file):
    model = BaseModel(results_file=results_file)
    model.train(data[features], data[target])
    model.evaluate()
    return 1


def main():
    data_path = "./data/raw/train.csv"
    results_file = "./data/evaluation/evaluation_results.json"
    processed_data, features = load_and_preprocess_data(data_path)
    # print(features)
    target = "Survived"

    train_and_evaluate_model(processed_data, features, target, results_file)

# 其他代码保持不变
```

可能需要说明的是，为了模型训练过程的展示方便，在模型评估类中，添加了相关的保存训练结果的方法，因此，该文件中，我们给出了相应的参数，比如 `results_file`。该参数与本项目的实际训练过程无关。现在，我们可以查看采用不同的缺失值处理策略后的模型训练评估结果：

对于采用 `Age` 列中位数填补的策略：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillMedian']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.810056   0.794118  0.72973  0.760563  0.872008

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.827143
```

对于采用 `Age` 列按头衔分类后的中位数填补的策略：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGrouped']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.810056   0.794118  0.72973  0.760563  0.881982

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.849365
```

从以上评估结果上看，两种不同的 `Age` 填补策略在单次评估中得到了相同的准确度、精确度、召回率和F1分数。而且混淆矩阵也完全一致，这表明两种策略在预测真正例、假正例、真负例和假负例的数量上没有差异。这也表明在这次测试集上，两种填补策略对模型性能的影响相同。不过，我们也发现，使用 `Age` 列按头衔分类后的中位数填补的策略在5折交叉验证的平均准确度上高于使用 `Age` 列整体中位数填补的策略（0.849365 vs. 0.827143）。这表明虽然在单个测试集上两种策略的性能相同，但在更广泛的数据上考虑，按头衔分类填补 `Age` 的策略可能更为稳健，能够提供更高的平均准确度。它们在交叉验证的准确度上有所不同。因此，后面我们将考虑采用<strong style="color:#c21d03">按头衔分类后的中位数填补 `Age` 策略</strong>，该种策略可能对未见数据具有更好的泛化能力。

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

考虑到逻辑回归会受到特征尺度的影响（在逻辑回归的情况下，模型是基于数据的线性组合），因此，现在我们尝试将 `Age` 特征标准化/归一化，然后评估模型效果。值得注意的是，标准化/归一化方法很多，比如常用的 Z 得分标准化，最小-最大归一化等。数据的不同分布将影响我们选择不同标准化/归一化的方法。比如，如果数据接近正态分布， Z 得分标准化可能是一个更好的选择。而如果数据的范围更为重要，而数据分布不是正太分布，可能最小-最大更为合适。我们先来看看经过缺失值填补后的 `Age` 分布情况 (这部分代码可以参考EDA分析中[单因素分析](#单因素分析)中的年龄可视化示例代码)。

![](/assets/images/ml/titanic_distribution_age_fill_title_group.png)

从左图可以发现，`Age` 数据似乎不是严格的正态分布，但也没有特别极端的偏斜。但是，右图中显示，存在少部分异常值。为此，我们需要一种更为稳健的标准化/归一化方法，以确保这些异常值不会对整体标准化结果产生过大影响。在此，我们计划采用 `RobustScaler` 来对 `Age` 进行处理。[`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) 通过去除中位数并按四分位范围（IQR）缩放数据，可以降低异常值的影响力。

现在，我们需要回到 `data_preprocessing.py` 文件，添加标准化/归一化的代码。显然，我们需要在已经填补上缺失值的数据上进行相关操作。由于前期我们构建了 `AgeProcessor` 类，因此，我们只需要在 `DataPreprocessor` 中涉及到年龄处理的部分添加上一个标准化操作（考虑到后期其他特征可能也需要标准化/归一化等数据转换操作，现在假设基类中有一个方法，名为：`scaling_robust`）就行。示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        title_processor = TitleProcessor(self.data)
        self.data = title_processor.extract_title().group_titles()
        age_processor = AgeProcessor(self.data)
        self.data, new_features_age = age_processor.fill_age_by_title_group()

        base_processor = BaseProcessor(self.data)
        self.data, new_features_age_robost = base_processor.scaling_robust(
            new_features_age[0]
        )
        self.features.extend(new_features_age_robost)

        # 其他代码保持不变
        return self.data, new_columns
```

现在的问题是我们应该如何实现这个标准化操作。考虑到代码的模块化问题，且该方法可能也会被其他特征使用。因此，我们可以在基类中实现该方法，然后在各个需要使用的特征类中调用。具体实现过程如下：

```python
# titanic/titanic/data_preprocessing.py
class BaseProcessor:
    def __init__(self, data):
        self.data = data

    def scaling_robust(self, column):
        scaler = RobustScaler()
        column_name = column + "RobustScaler"
        self.data[column_name] = scaler.fit_transform(self.data[[column]])
        return self.data, [column_name]
```

现在我们就可以直接运行 `main.py`，评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedRobustScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.881853

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.849365
```

与未标准化时的评估指标结果对比，我们发现，除了混淆矩阵和交叉验证的结果没有变化外，其他评估指标均有不同程度的降低。这好像不是我们所期望的。下面我们可以根据以上逻辑，试试其他的常用标准化方法是否对评估指标有所影响。例如，我们采用 `Min-Max` 的方式对 `Age` 特征进行标准化，其结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedMinMaxScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.793296   0.768116  0.716216  0.741259  0.879408

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  89                  16
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.810317
```
🤣，所有指标都降低了，看来 `Min-Max` 也不是一个好选择。选择 `Z-Score`，继续测试，结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.810056   0.794118  0.72973  0.760563  0.881853

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.84936
```

🤩，采用 `Z-Score` 后的结果居然和未标准化的一致。有些意外。逻辑回归模型似乎对 `Age` 特征的的标准化过程有较为敏感的返回。`Z-Score` 的评估结果优于其他两种方法的原因可能是由于 `Age` 特征在未标准化时已经相对集中，倾向于正态分布。而 `Z-Score` 恰恰适合于该类分布。虽然 `Z-Score` 并没有增强模型的能力，但似乎也没有什么坏处，考虑到后期我们可能会选择其他分类模型，<strong style="color:#c21d03">暂时保留 `Z-Score` 对 `Age` 特征的标准化</strong>。<a id=basemodel>后续特征工程将以此为基准进行对比选择</a>

<hr/>

###  变量：`SibSp` 和 `Parch`

基于 EDA 分析，不同家庭成员数量似乎对生存率存在影响，因此，这里我们计划进一步将该因素融入到上面的模型中。先来看看分别考虑`SibSp` 和 `Parch` 特征会不会对模型训练效果产生影响。由于这两特征没有缺失值，我们可以暂时直接加入到特征中。为了保持代码的一致性，我们对 `SibSp` 和 `Parch` 特征分别构建两个处理类，示例代码如下：

```python
# titanic/titanci/data.preprocessing.py
class SibSpProcessor(BaseProcessor):
    def sibsp_process(self):
        new_feature = ["SibSp"]
        return self.data, new_feature


class ParchProcessor(BaseProcessor):
    def sibsp_process(self):
        new_feature = ["Parch"]
        return self.data, new_feature
```

然后，我们可以在 `DataPreprocessor` 的 `preprocess` 方法添加如下代码：

```python
# titanic/titanci/data.preprocessing.py
class DataPreprocessor:
    def preprocess(self):
        # 其他代码保持不变

        sibsp_processor = SibSpProcessor(self.data)
        self.data, new_features_sibsp = sibsp_processor.sibsp_process()
        self.features.extend(new_features_sibsp)

        return self.data, self.features
```

重新运行 `main.py`，可以得到如下结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.797101  0.743243  0.769231  0.892342

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.866190
```

同理，单独加入 `Parch` 的结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'Parch']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.798883   0.787879  0.702703  0.742857  0.883655

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  22                  52

Cross-validated Accuracy (5-fold): 0.855079
```

同时考虑 `SibSp` 和 `Parch` 特征的结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229   0.808824  0.743243  0.774648  0.893372

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.866190
```

分析这三组逻辑回归模型评估结果，我们可以发现：

1. **考虑 `SibSp` 特征时**：引入 `SibSp` 后，模型的准确率、精确度、召回率、F1分数有所提高，ROC AUC 显著增加。这表明 `SibSp` 是一个有价值的特征，能提高模型的预测性能。
2. **考虑 `Parch` 特征时**：引入 `Parch` 后，准确率略有下降，精确度和召回率也有所变化，但 ROC AUC 略有提高。这表明 `Parch` 对模型的影响不如 `SibSp` 明显，但仍提供了一定的信息增益。
3. **同时考虑 `SibSp` 和 `Parch` 特征时**：同时考虑这两个特征时，模型在所有评估指标上都有所提升，尤其是准确率和ROC AUC，表明这两个特征的组合提供了比单独使用时更多的信息。

综上所述，`SibSp` 和 `Parch` 特征对模型有正面影响，尤其是当它们同时使用时，能显著提高模型的预测性能。这可能是因为这些特征能够反映家庭结构对乘客生存率的影响，这是模型在没有这些信息时无法捕捉到的。因此，后面，<strong style="color:#c21d03"> 计划考虑将 `SibSp` 和 `Parch` 作为特征构建模型 </strong>，以提高预测的准确性和模型的泛化能力。但是，考虑到 `Parch` 的添加可能对模型的影响有限，我们计划进一步处理特征。

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

由于 `SibSp` 和 `Parch` 特征都是表示家庭成员结构。因此，接下来，我们考虑下，是否将其组合成新的**家庭成员数量**特征，会对模型训练效果有所提升。由于我们需要构建新的特征，这就需要我们在 `data_preprocessing.py` 中添加一个新类 `FamilySizeProcessor`。然后在 `DataPreprocessor` 中实例化，其他保持不变就可以了，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class FamilySizeProcessor(BaseProcessor):
    def process_family_size(self):
        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        return self

# 其他代码保持不变

class DataPreprocessor:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def preprocess(self):
        # 其他代码保持不变

        # sibsp_processor = SibSpProcessor(self.data)
        # self.data, new_features_sibsp = sibsp_processor.sibsp_process()
        # self.features.extend(new_features_sibsp)

        # parch_processor = ParchProcessor(self.data)
        # self.data, new_features_parch = parch_processor.parch_process()
        # self.features.extend(new_features_parch)

        family_processor = FamilySizeProcessor(self.data)
        self.data, new_features_family = family_processor.family_size_process()
        self.feature.extend(new_features_family)
        # 其他代码保持不变
        return self.data, new_columns
```

重新运行 `main.py`，我们将得到如下结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'FamilySize']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.815385  0.716216   0.76259  0.890541

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.855079
```

从上述结果中，我们可以看到，当考虑 `SibSp`和`Parch` 特征时，模型的准确率、精确度、召回率、F1分数和ROC AUC值分别为0.821229、0.808824、0.743243、0.774648和0.893372。在这种情况下，模型表现较好，特别是在ROC AUC值上，显示出良好的分类能力。然而，当将 `SibSp` 和 `Parch` 合并为 `FamilySize` 特征时，各项指标有所下降，准确率为0.815642，精确度为0.815385，召回率为0.716216，F1分数为0.76259，ROC AUC值为0.890541。虽然精确度略有提升，但其他指标，尤其是召回率和F1分数，都有所下降。这种变化表明，尽管合并 `SibSp` 和 `Parch` 为单一的 `FamilySize` 特征简化了模型，并可能有助于减少过拟合的风险，但它也可能损失了一些重要的信息，从而影响了模型的整体性能。特别是，召回率的下降表明，在合并特征后，模型识别出的实际正类数量减少了。此外，交叉验证的准确率从0.866190下降到0.855079，也显示了模型在合并特征后在不同数据子集上的泛化能力略有下降。总的来说，这种变化强调了特征工程决策对模型性能的重要影响，以及在决定合并特征之前需要仔细考虑的权衡。在实际应用中，最佳的特征工程策略取决于特定问题的上下文以及对不同类型错误的容忍程度。

还记得前面 EDA 分析中发现，大多数乘客没有兄弟姐妹、配偶、父母或孩子同行吗？这个特征可能会导致数据倾斜，从而对模型产生不成比例的影响，为了缓解这种影响，我们可能需要进一步处理 `FamilySize` 这个新特征。下面提供了几种策略:

1. **二值化处理**：将 `FamilySize` 转换为二元特征，例如，将独自一人的乘客标记为0，有家庭成员的乘客标记为1。这样的处理可以突出是否有家庭成员这一信息，而不是家庭成员的具体数量。
2. **分段（分箱）**：将 `FamilySize` 进行分段（或称为分箱），例如，将家庭大小划分为"无家庭成员"、"小家庭"和"大家庭"等几个类别。这样可以在保留一定家庭大小信息的同时，减少异常值的影响。   
3. **归一化或标准化**：虽然 `FamilySize` 已经是数值型特征，但如果其分布非常偏斜（的确），可以考虑对其进行归一化或标准化处理，使其在更合适的数值范围内，这可能对于基于梯度的模型特别有用。
4. **考虑与其他特征的交互**：可以进一步探索 `FamilySize` 与其他特征的交互，例如，家庭大小可能与船舱等级（`Pclass`）或票价（`Fare`）有关联。这种交互特征可能会揭示更多的信息。
5. **特征选择**：如果通过模型评估发现 `FamilySize` 对模型性能的贡献有限，可以考虑不将其包括在最终模型中，或者使用特征选择算法来确定其重要性。

根据以上策略，我们先来完成前两种，针对不同策略，建立新变量，如对二值化，我们构建一个 `IsAlone` 的新变量；对分段，我们构建一个 `FamilySizeGroup`，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class FamilySizeProcessor(BaseProcessor):
    # 其他代码保持不变

    def is_alone_family(self):
        new_feature = ["IsAlone"]
        self.data[new_feature[0]] = (self.data["FamilySize"] == 1).astype(int)
        return self.data, new_feature

    def categorize_family_size(self):
        new_feature = ["FamilySizeGroup"]
        self.data[new_feature[0]] = pd.cut(
            self.data["FamilySize"],
            bins=[0, 1, 4, 11],
            labels=["Solo", "SmallFamily", "LargeFamily"],
        )
        return self.data, new_feature


class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        family_processor = FamilySizeProcessor(self.data)
        self.data, _ = family_processor.family_size_process()
        self.data, new_feature_isalone = family_processor.is_alone_family()
        self.features.extend(new_feature_isalone)

        return self.data, self.features
```
注意，以上处理中， `categorize_family_size` 方法只是将 `FamilySize` 分成了 `Solo`, `SmallFamily`, `LargeFamily`。这样的类别数据需要经过处理后才能输入到逻辑回归模型中。后续在处理时，我们可以采用处理 `Sex` 特征时的策略。

重新运行 `main.py` 得到二值化后的模型评估结果，如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'IsAlone']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.883012

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.849365
```

同理，对于分段，我们可以在以上基础上添加对 `FamilySize` 进行One-Hot编码，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class FamilySizeProcessor(BaseProcessor):
    # 其他代码保持不变

    def family_one_hot_encode(self):
        self.data, new_features = super().one_hot_encode("FamilySizeGroup")
        return self.data, new_features

# 其他代码保持不变

class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        family_processor = FamilySizeProcessor(self.data)
        self.data, _ = family_processor.family_size_process()
        self.data, _ = family_processor.family_size_categorize()
        self.data, new_features_family_one_hot = (
            family_processor.family_one_hot_encode()
        )
        self.features.extend(new_features_family_one_hot)
        # 其他代码保持不变
        return self.data, self.features
```

重新运行 `main.py` 得到分段后的模型评估结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'FamilySizeGroup_LargeFamily', 'FamilySizeGroup_SmallFamily', 'FamilySizeGroup_Solo']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.810056    0.80303  0.716216  0.757143  0.894788

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.849365
```

对于`FamilySize` 特征的标准化/归一化，由于前期在数据处理的基类中，我们已经构建了相应方法，在这，我们仍然可以复用前面的方法。。在进行标准化/归一化之前，我们先来分析下 `FamilySize` 特征的分布情况，如下：

![](/assets/images/ml/titanic_distribution_family_size.png)

显然，`RobustScaler` 似乎是一个更为明智的选择。那么只需要在 `DataPreprocessor` 类中调用基类的 `scaling_robust` 对 `FamilySize` 进行标准化就行，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        family_processor = FamilySizeProcessor(self.data)
        self.data, new_features_family = family_processor.family_size_process()
        self.data, new_features_family_robust = base_processor.scaling_robust(
            new_features_family[0]
        )
        self.features.extend(new_features_family_robust)

        return self.data, self.features
```

运行 `main.py`，得到 `FamilySize` 标准化后的评估结果，如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'FamilySizeRobustScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.815385  0.716216   0.76259  0.890541

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.855079
```

当然，我们也可以试着用其他标准化方法，查看评估结果的变化：

采用 Min-Max:

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'FamilySizeMinMaxScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.889511

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.849365
```

采用 Z-Score：
```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'FamilySizeStandardScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.815385  0.716216   0.76259  0.890541

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.855079
```

对新构建的`FamilySize`特征，我们采用了多种不同的处理方法，并分析了这些方法对逻辑回归模型评估指标的影响，发现：

1. **未进一步处理**：未对 `FamilySize` 进行任何额外处理时，模型表现中等，准确率、精确度、召回率、F1分数和ROC AUC分别为0.815642、0.815385、0.716216、0.76259和0.890541。
2. **是否独自分类处理**：将 `FamilySize` 简单分为独自一人和非独自两类时，各项指标略有下降，特别是准确率和F1分数，这表明过于简化的分类可能损失了部分信息。
3. **多分类处理**：对 `FamilySize` 进行更细致的分类处理后，模型的准确率、F1分数和ROC AUC略有改善，表明适度的分类可以提供额外的信息，有助于改善模型的性能。
4. **RobustScaler标准化**：使用 `RobustScaler` 处理后，模型的表现与未处理时相当，说明这种标准化方法保持了原始数据的分布特征，对模型性能影响不大。
5. **Min-Max标准化**：使用 `Min-Max` 标准化后，模型的准确率和F1分数有所下降，但ROC AUC值与其他方法相近，说明 `Min-Max` 可能导致某些信息的压缩。
6. **Z-Score标准化**：采用 `Z-Score` 处理后，模型的各项指标与 `RobustScaler` 处理相当，说明 `Z-Score` 标准化在这种情况下维持了数据的分布特性。

综合来看，`FamilySize` 特征的不同处理方法对模型性能有一定影响。其中，`RobustScaler` 和 `Z-Score` 标准化方法在保持数据分布特性的同时，维持了较高的模型评估指标。而简单的是否独自分类处理则可能由于信息损失导致性能下降。这些结果强调了特征处理方法选择的重要性，以及它们对模型性能的潜在影响。在实际应用中，选择适当的特征处理策略对于优化模型表现至关重要。因此，如果需要对 `FamilySize` 特征进一步处理的话，后续计划与处理 `Age` 类似，<strong style="color:#c21d03"> 保留通过`RobustScaler` 或 `Z-Score` 标准化方法对 `FamilySize` 特征的处理</strong>。但是，<strong style="color:#c21d03">当我们继续对比同时考虑 `SibSp` 和 `Parch` 特征时，各个评估指标均有所下降。从这个角度，可能不用构建新特征，对逻辑回归分类模型效果最好。因此，下面的模型将基于同时考虑 `SibSp` 和 `Parch` 特征，而不考虑由此特征衍生出来的新特征</strong>

<hr/>

### 变量：`Ticket`

从 EDA 分析可以看出，票号前缀与乘客的生存率之间存在一定的关联。票号前缀大致可以分为高频前缀（如 None, PC, CA）和低频前缀。高频前缀（如 None, PC, CA）可能代表了更常见的票务类别，而与之关联的生存率可能更具有一般性的指示意义；低频前缀的生存率可能受到随机波动的影响较大，这给我们如何处理票号前缀提出了一定的挑战。

我们计划先提取出票号前缀，并采用 One-Hot 的方式对其进行编码（主要是考虑到，票号前缀并没有顺序性）。显然，我们需要在 `data_preprocessing.py` 中，添加一个专门处理票号的类 `TicketProcessor`。示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class TicketProcessor(BaseProcessor):
    def ticket_process(self):
        new_feature = ["TicketPrefix"]
        self.data[new_feature[0]] = self.data["Ticket"].apply(
            lambda x: (
                "".join(filter(str.isalpha, x.split(" ")[0]))
                if not x.isdigit()
                else "None"
            )
        )
        return self.data, new_feature

    def ticket_one_hot_encode(self, column):
        self.data, new_feature = super().one_hot_encode(column)
        return self.data, new_feature
# 其他代码保持不变

class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变

        ticket_processor = TicketProcessor(self.data)
        self.data, _ = ticket_processor.ticket_process()
        self.data, new_features_ticket = ticket_processor.ticket_one_hot_encode("TicketPrefix")
        self.features.extend(new_features_ticket)

        return self.data, self.features
```

以上处理逻辑是，如果有前缀，则提取其前缀，如果没有，则将其前缀命名为 `None`。对提取出的前缀进行 One-Hot 编码也很简单，仍然可以请参考 `Sex` 的处理，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        AgeProcessor(self.data).fill_age_by_title_group().scaling_z_score("Age")
        TicketProcessor(self.data).process_ticket() # 添加对Ticket前缀的处理
        # 其他代码保持不变
        return self.data, new_columns
```

重新运行 `main.py`，评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'TicketPrefix_A', 'TicketPrefix_AS', 'TicketPrefix_C', 'TicketPrefix_CA', 'TicketPrefix_CASOTON', 'TicketPrefix_FC', 'TicketPrefix_FCC', 'TicketPrefix_Fa', 'TicketPrefix_LINE', 'TicketPrefix_None', 'TicketPrefix_PC', 'TicketPrefix_PP', 'TicketPrefix_PPP', 'TicketPrefix_SC', 'TicketPrefix_SCA', 'TicketPrefix_SCAH', 'TicketPrefix_SCOW', 'TicketPrefix_SCPARIS', 'TicketPrefix_SCParis', 'TicketPrefix_SOC', 'TicketPrefix_SOP', 'TicketPrefix_SOPP', 'TicketPrefix_SOTONO', 'TicketPrefix_SOTONOQ', 'TicketPrefix_SP', 'TicketPrefix_STONO', 'TicketPrefix_SWPP', 'TicketPrefix_WC', 'TicketPrefix_WEP']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.888803

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.843810
```

注意：由于对 `Ticket_Prefix` 进行了One-Hot编码，因此，增加了很多特征。以上结果是将新生成了所有特征都纳入到前面的模型中的各个评估指标的结果：

对比不考虑票号前缀，可以注意到以下几点差异：
1. **准确率(Accuracy)**: 考虑票号前缀的模型准确率略高于不考虑票号前缀的模型（0.826816 vs 0.821229）。这表明加入票号前缀特征后，模型在整体上能更准确地预测乘客的生存状态。
2. **精确度(Precision)**: 考虑票号前缀的模型比不考虑票号前缀的模型精确度稍高（0.820896 vs 0.808824）。这意味着在预测乘客生存时，加入票号前缀可以稍微减少将未生存的乘客错误分类为生存的情况。
3. **召回率(Recall)**: 两组结果相当（0.743243），表明在实际生存的乘客中，两个模型都有相同比例的乘客被正确预测为生存。
4. **F1分数(F1 Score)**: 考虑票号前缀的模型F1分数略高（0.780142 vs 0.774648），表示考虑票号前缀的模型在精确度和召回率之间取得了较好的平衡。
5. **ROC AUC**: 不考虑票号前缀的模型ROC AUC略高（0.893372 vs 0.888803），这表明在区分乘客生存与否的能力上，不考虑票号前缀的模型整体表现略优。
6. **交叉验证准确率(Cross-validated Accuracy)**: 不考虑票号前缀的模型在交叉验证中表现更好（0.866190 vs 0.843810），说明其泛化能力可能更强。

总体来看，加入票号前缀特征后，模型在准确率、精确度、F1分数上有所提高，但在ROC AUC和交叉验证准确率上有轻微下降。这可能意味着票号前缀提供了有用的信息，但也可能引入了一些噪声，影响了模型的泛化能力。在实际应用中，是否加入这类特征需要根据具体情况和模型表现来决定。

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

由于考虑票号前缀时引入了较多的新特征。下面，我们计划首先从降维的角度思考如何处理 `Ticket_Prefix`。降维可以减少特征空间的维度，同时尽量保留原始数据中的重要信息。下面列出了部分常用的降维方法：

1. **主成分分析（PCA）**: PCA 是一种非常流行的降维技术，可以将特征转换到一个新的坐标系统中，并按照方差大小排序，保留最重要的几个主成分。对于One-Hot编码后的特征，PCA可以帮助识别哪些变量捕获了大部分信息。
2. **截断奇异值分解（Truncated SVD）**: 与PCA类似，截断SVD适用于稀疏数据（例如，One-Hot编码后的数据）。它可以减少特征的维度，同时保留数据的关键信息。
3. **线性判别分析（LDA）**: LDA是一种监督学习的降维技术，旨在找到一个能够最大化类别间分离的特征子空间。特别是在分类项目中，LDA可以帮助提升模型的分类能力。
4. **t-SNE 或 UMAP**: t-SNE（t-distributed Stochastic Neighbor Embedding）和UMAP（Uniform Manifold Approximation and Projection）是两种流行的非线性降维技术，可以帮助揭示高维数据的内在结构。这些方法尤其擅长于保留局部邻域结构，因此它们在可视化聚类或组间差异方面通常具有出色的表现。t-SNE和UMAP通常用于探索性数据分析，以帮助理解数据集中可能存在的模式或聚类。虽然它们主要用于可视化，但在某些情况下，降维后的数据也可以用于训练模型，特别是在原始数据维度非常高时。不过，需要注意的是，这两种方法可能会增强数据中的噪声，所以在解释降维结果时应当谨慎。
5. **自编码器（Autoencoders）**: 自编码器是一种基于神经网络的降维技术，特别适合于非线性降维。通过训练一个将输入数据编码成低维表示，然后再解码回原始空间的网络，自编码器可以学习到数据的有效低维表示。
6. **特征选择**: 除了降维，还可以考虑特征选择方法来减少特征数量。基于树的方法（例如随机森林或XGBoost）可以提供特征重要性评分，帮助我们识别并选择最重要的特征。

降维是一个需要实验和评估的过程。我们可能需要尝试不同的降维方法和参数设置，然后根据模型的性能和复杂度来选择最适合咱们数据的方法。

考虑到当前项目的特征，我们暂时选择前两种降维技术，对 `Ticket_Prefix` One-Hot 编码后的数据进行处理，并评估其对模型的影响。从代码组织上，我们计划在 `data_preprocessing.py` 中针对数据降维构建一个新的类。新的降维类示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
from sklearn.decomposition import PCA, TruncatedSVD


# 其他代码保持不变
class DimensionalityReducer:
    def __init__(
        self,
        data,
        method="PCA",
        n_components=0.95,
        random_state=None,
        feature_prefix="TicketPrefix_",
    ):
        self.data = data
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.feature_prefix = feature_prefix
        self.model = None
        if self.method == "PCA":
            self.model = PCA(n_components=self.n_components)
        elif self.method == "SVD":
            self.model = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            )

    def apply_reduction(self):
        if not self.model:
            raise ValueError("Invalid dimensionality reduction method")

        features = [
            col for col in self.data.columns if col.startswith(self.feature_prefix)
        ]
        reduced_data = self.model.fit_transform(self.data[features])

        n_components = (
            self.model.n_components_
            if self.method == "PCA"
            else self.model.n_components
        )
        new_feature_names = [
            f"{self.method}_{self.feature_prefix}_{i+1}" for i in range(n_components)
        ]

        self.data.drop(columns=features, inplace=True)
        for i, feature_name in enumerate(new_feature_names):
            self.data[feature_name] = reduced_data[:, i]

        print(f"{self.method} reduced the features to {n_components} components.")
        return self.data, new_feature_names
```

在添加了降维处理类后，我们还需要在 `DataPreprocessor` 实例化该类，注意在使用降维类之前，应该将 `TicketPrefix` 先进行编码。示例代码如下

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class DataPreprocessor:
        # 其他代码保持不变

        ticket_processor = TicketProcessor(self.data)
        self.data, _ = ticket_processor.ticket_process()
        self.data, _ = ticket_processor.ticket_one_hot_encode()
        reducer = DimensionalityReducer(
            self.data, method="PCA", n_components=0.95, random_state=None
        )
        self.data, new_feature_names_ticket_reduced = reducer.apply_reduction()
        self.features.extend(new_feature_names_ticket_reduced)

        return self.data, self.features
```

运行新的 `main.py`，得到如下结果：

```plaintext
PCA reduced the TicketPrefix_ features to 15 components.
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'PCA_TicketPrefix__1', 'PCA_TicketPrefix__2', 'PCA_TicketPrefix__3', 'PCA_TicketPrefix__4', 'PCA_TicketPrefix__5', 'PCA_TicketPrefix__6', 'PCA_TicketPrefix__7', 'PCA_TicketPrefix__8', 'PCA_TicketPrefix__9', 'PCA_TicketPrefix__10', 'PCA_TicketPrefix__11', 'PCA_TicketPrefix__12', 'PCA_TicketPrefix__13', 'PCA_TicketPrefix__14', 'PCA_TicketPrefix__15']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.887773

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.843810
```

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

相较于 `PCA` 来说，运用 `SVD` 技术对特征进行降维稍微会复杂些。复杂的点主要是在确认 `n_components` 上。`SVD` 并没有类似于 `PCA` 方差阈值的参数可以设置。在降维过程中，需要我们根据经验来判断 `n_components` 值的合理性。为了更为直观，我们先探索性分析，不同 `n_components` 下的解释方差比以及累计方差，如下（这部分代码请参考项目文件中的 `notebook/feature_ead.ipynb` 文件）：

![](/assets/images/ml/titanic_svd_num_comp.png)

可以看出，当 `n_components=16` 时，累计方差达到约95%。这意味着保留前 16 个 SVD 组件就可以解释原始数据约 95% 的方差，这通常是一个选择组件数量的好标准，因为它确保了大部分信息被保留，同时也减少了特征数量，降低了模型的复杂度。但是，查看解释方差比，发现在 `n_components = 5` 左右时出现 elbow of the curve。哪到底 `n_components` 如何选择？有的时候（比如从降低计算成本的角度）根据解释方差比来选择可能更好。不过，这里我们可以都试试，检查下对模型评估指标的影响。

由于在 `DimensionalityReducer` 类中，我们已经构建了 `SVD` 的相关计算过程，因此，我们只需要在 `DataPreprocessor` 类中修改 `DimensionalityReducer` 的参数 `method`, `n_components`, `random_state`即可。

当设置 `n_components=16` 时，模型的评估指标如下：

```plaintext
SVD reduced the TicketPrefix_ features to 16 components.
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'SVD_TicketPrefix__1', 'SVD_TicketPrefix__2', 'SVD_TicketPrefix__3', 'SVD_TicketPrefix__4', 'SVD_TicketPrefix__5', 'SVD_TicketPrefix__6', 'SVD_TicketPrefix__7', 'SVD_TicketPrefix__8', 'SVD_TicketPrefix__9', 'SVD_TicketPrefix__10', 'SVD_TicketPrefix__11', 'SVD_TicketPrefix__12', 'SVD_TicketPrefix__13', 'SVD_TicketPrefix__14', 'SVD_TicketPrefix__15', 'SVD_TicketPrefix__16']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.888417

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.843810
```

当 `n_components=5` 时，模型的评估指标如下：

```plaintext
SVD reduced the TicketPrefix_ features to 5 components.
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'SVD_TicketPrefix__1', 'SVD_TicketPrefix__2', 'SVD_TicketPrefix__3', 'SVD_TicketPrefix__4', 'SVD_TicketPrefix__5']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.832402   0.833333  0.743243  0.785714  0.883398

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  94                  11
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

对比这两情景下，当 `n_components=5` 时 只在 `ROC AUC` 有了些许降低，其他评估指标均优于 `n_components=16` 。从这个结论上说，`n_components=5` 可能是一个明智的选择。

与前面的结果进行对比，发现：

1. **未进行降维处理的情况**：模型的准确度为 0.826816，精确度为 0.820896，召回率为 0.743243，F1 分数为 0.780142，ROC AUC 为 0.888803。在不降维的情况下，模型可能面临特征过多而导致的维度灾难，但在这个结果中表现出相对稳健的性能。
2. **PCA 降维处理的情况**：PCA 将 `TicketPrefix` 特征降至15个主成分，准确度、精确度、召回率、F1 分数与未降维的情况几乎一致，但 ROC AUC 略有下降。这表明 PCA 降维在保持模型性能的同时，减少了模型的复杂度，有助于模型的解释性和计算效率。
3. **SVD 降维处理的情况**：SVD 降维至5个组件后，模型的准确度略有提高至 0.832402，精确度也有所提升，但 ROC AUC 略有下降。SVD 降维较 PCA 更为激进，只保留了5个组件，可能导致了一些信息的损失，从而影响了ROC AUC的表现。

从上述结果可以看出，降维技术在减少模型复杂度和计算资源消耗的同时，对模型的整体性能影响较小，特别是在准确度和 F1 分数上。降维后模型的 ROC AUC 有轻微下降，这可能是因为降维过程中丢失了一些有助于模型区分正负类别的信息。在实际应用中，选择是否进行降维以及使用哪种降维技术，需要根据具体问题和数据集的特性综合考虑，同时要考虑到模型性能、计算效率和可解释性之间的平衡。<strong style="color:#c21d03"> 因此，如果计算资源充足，且对模型的可解释性要求不高，可以考虑不进行降维。如果需要减少计算资源消耗或提高模型的可解释性，可以选择 PCA 或 SVD 等降维方法。</strong> 不过，具体选择哪种降维方法，还需要根据实际应用场景和模型性能的需求来决定。

同时，我们与不考虑 `Ticket` 特征时的评估结果对比，发现，虽然考虑 `Ticket` 特征时，在部分指标（如准确率，精确率，F1 得分）上对逻辑回归模型训练效果有所提升，但，其在ROC AUC, 交叉验证上均有一定程度的下降。虽然下降不是很明显，但提升的效果也是有限的。<strong style="color:#c21d03"> 因此，后续的模型训练中，我们暂时不考虑 `Ticket` 特征。 </strong>

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

虽然我们在后续的模型训练中暂时不纳入 `Ticket` 特征。但是，以上是采用相关降维技术对 One-Hot 编码后的特征进行处理。从上面的分析过程可以发现，我们在处理 `Ticket` 特征时，其实并没有应用 EDA 分析得出的一些有用信息。比如说，在单变量分析中，我们发现，大多数票都没有前缀，**PC**, **CA**, **A**, **STONO** 等是接下来最常见的票号前缀。其他前缀如 **SC**, **SWPP**, **FCC** 等出现的次数相对较少。结合生存率来看，**SC** 和 **SWPP** 前缀的票号有最高的生存率（1.00），没有明显前缀的票，生存率仅为 0.38。同时也发现，生存率较高的对应的票数较少，而普通的票最多的，生存率仅约1/3。这给我们一个启示，能否在对 `Ticket` 的前缀 One-Hot 编码前进行处理。以减少 One-Hot 编码后的特征，从而达到降维的要求。同时，这也可以帮助我们更为细致的控制如何处理这些前缀，特别是某些罕见的前缀。

很明显，如果仅考虑 `TicketPrefix` 时，我们可以将其简单的分成常见和罕见两类，从而大大减少了特征编码后的变量。这里的难点是如何定义常见和罕见？可能有同学会想到，我们可以设定一个阈值，比如选择覆盖约80%-90%的数据的前缀为常见前缀，其余的为罕见。当然是可以的。如果想更为细致的分类，我们还可以选择多个阈值区间？

除此之外，我们还可以结合生存率来对 `TicketPrefix` 分类。比如按照生存率的 \([0, 0.2)\), \([0.2, 0.4)\)，等等来对前缀进行分类。但是这里有个问题，在训练集上进行该种分类确实可行，但在预测集上如何应用相同的分组？因为预测集上我们没有生存率这个指标。这里也有大致的解决方案，比如**不直接根据生存率来分组，而是找到与生存率相关的其他特征**，比如船票价格、船舱等级等，这些在预测集上也是可用的。如果确实要使用生存率来辅助分组，可以考虑以下方法：

1. **分组依据仅用于降维**：在训练阶段，使用生存率信息帮助确定 `TicketPrefix` 的分组，然后进行 One-Hot 编码和降维。在预测阶段，只需根据训练阶段确定的前缀分组对新数据进行相同的 One-Hot 编码和降维处理。这种方法的前提是能够确保新数据中的 `TicketPrefix` 在训练集中已有相应的处理逻辑。
2. **创建预测时也能获取的特征**：如果依据生存率对 `TicketPrefix` 进行分组，可以尝试创建一个新特征，比如`TicketPrefixGroupSur`，这个特征在预测时也能够根据 `TicketPrefix` 直接获得，即使没有生存率信息。例如，如果在训练阶段发现某些前缀与高生存率相关，就可以将这些前缀归为一个组，预测时只需检查 `TicketPrefix` 是否属于这个组即可。

下面我们试着从最简单（即，按照 `TicketPrefix` 的频率和其累计频率）的分类开始，看看以上想法是否对逻辑回归模型训练效果有所影响。

`Ticket_Prefix` 的频率和其累计频率如下图所示：

![](/assets/images/ml/titanic_ticket_prefix_frequency_cumulative.png)

从上图可以看出来，`None` 这一类是最多的，大约占了 75%，其次是 `PC`， `CA`， `A`， `STONO`，与 `None` 一起占了大约 90%。因此，我们想将 `None` 单独成一类，而 `PC`， `CA`， `A`， `STONO`一起作为 `Common` 类，其余的为 `Rare`。但是，考虑到我们需要确保模型处理新数据时更加鲁棒，我们可以按照累计频次来分类。与前面的硬编码不同，这里我们需要确定分类的阈值，比如，我们可以设置该阈值为 85%。为了实现该分类，我们可以在 `TicketProcessor` 类中增加一个分类方法 `categorize_ticket_prefix`，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class TicketProcessor(BaseProcessor):
    # 其他代码保持不变

    def categorize_ticket_prefix(self, threshold=0.85):
        prefix_freq = self.data["TicketPrefix"].value_counts(normalize=True)
        prefix_cumsum = prefix_freq.cumsum()
        common_prefixes = prefix_cumsum[prefix_cumsum <= threshold].index.tolist()

        new_feature = ["TicketPrefixCategorized"]
        self.data[new_feature[0]] = self.data["TicketPrefix"].apply(
            lambda x: (
                "None"
                if x == "None"
                else ("Common" if x in common_prefixes else "Rare")
            )
        )
        return self.data, new_feature
```

按照如上构建，我们只需要再修改 `DataPreprocessor` 类，示例代码如下：

```python
class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变

        ticket_processor = TicketProcessor(self.data)
        self.data, _ = ticket_processor.ticket_process()
        self.data, _ = ticket_processor.categorize_ticket_prefix()
        self.data, new_feature_ticket_freq_grouped = (
            ticket_processor.ticket_one_hot_encode("TicketPrefixCategorized")
        )
        # reducer = DimensionalityReducer(
        #     self.data, method="SVD", n_components=5, random_state=42
        # )
        # self.data, new_feature_names_ticket_reduced = reducer.apply_reduction()
        self.features.extend(new_feature_ticket_freq_grouped)

        return self.data, self.features
```

重新运行 `main.py`，得到如下结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'TicketPrefixCategorized_Common', 'TicketPrefixCategorized_None', 'TicketPrefixCategorized_Rare']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.888031

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

与前面的结果进行对比，发现：

1. **与未进行特别处理相比**：按累计频率分类后，仅在 ROC AUC 值方面，分类处理后略有降低，可能是因为减少了特征的细节程度，影响了模型区分正负样本的能力。模型的其他指标均一致。
2. **与 PCA 降维相比**：按累计频率分类后的结果在 ROC AUC 值和交叉验证上略高，且其他指标保持一致。这表明累计频率分类明显优于 PCA 处理方式。
3. **与 SVD 降维相比**：SVD 降维后的模型在准确率，精确率，以及 F1 得分上略高于分类处理，但优势不明显，在 ROC AUC 上略低。这可能说明SVD在减少特征维度时保留了更多对模型预测有用的信息，但同时也可能引入了一些噪声或过拟合的风险。

因此，<strong style="color:#c21d03">如果要考虑对 `TicketPrefix` 进行降维处理，相对于 PCA 和 SVD 来说，按照频率的分类可能是一个较好的选择。</strong>

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

现在我们试着用其他与生存率相关性较强的特征对票号前缀进行分类。要实现这个目的，我们首先需要确定不同特征与生存率之间的相关性。由于原始数据集中既包括数值型变量，也包括类别型数据，这就出现了如何评估类别型数据与数值型变量之间的相关性问题。以下是几种处理数值型和类别型数据相关性分析的方法：
1. **对于数值型变量之间**：可以使用皮尔逊相关系数（Pearson correlation coefficient）来衡量它们之间的线性相关性。
2. **对于类别型变量之间**：（1）使用卡方检验（Chi-squared test）来测试两个类别型变量之间的独立性；（2）计算 Cramér's V 统计量（Cramér's V statistic），它是基于卡方统计的一种衡量类别型变量之间相关性的方法，取值范围从0到1。
3. **对于数值型和类别型变量之间**：（1）点双列相关（Point biserial correlation）：当一个变量是二元类别型，另一个是数值型时，可以使用点双列相关来衡量它们之间的相关性[^1]；（2）以数值变量为基础的分组分析：例如，可以根据类别型变量的分组计算数值型变量的均值，并通过 ANOVA（方差分析）检验不同组之间是否存在显著差异；（3）编码：将类别型变量转换为数值型（例如，独热编码），然后计算这些新变量与其他数值型变量之间的相关性。

具体到本项目的数据，由于目标变量是类别型变量。那么，我们应该主要考虑采用**类别型变量之间**以及**对于数值型和类别型变量之间**两类方法来进行相关性分析。其结果如下图所示（相关代码请查看 `feature_eda.ipynb`）:

![](/assets/images/ml/titanic_feature_correlations.png)

由于我们本意是结合生存率来对 `TicketPrefix` 分类。因此，按照上图所示中的相关性，可能选择 `Title_Grouped` 来对 `TicketPrefix` 更为合适（`Title` 与生存率之间存在最大的正相关）。具体来说，我们可以选择查看每个 `Title_Grouped` 类别下，`TicketPrefix` 的出现频率，然后根据这个频率来分类，示例代码如下：

```python
# 其他代码保持不变
class TicketProcessor(BaseProcessor):
    # 其他代码保持不变

    def categorize_ticket_prefix_using_title(self):
        if "Title_Grouped" not in self.data.columns:
            self.data = TitleProcessor(self.data).extract_title().group_titles()

        new_feature = ["TicketPrefixCategorized"]
        self.data[new_feature[0]] = "Others"

        for title_group in self.data["Title_Grouped"].unique():
            prefix_freq = self.data[self.data["Title_Grouped"] == title_group][
                "TicketPrefix"
            ].value_counts(normalize=True)
            threshold = 0.1
            significant_prefixes = prefix_freq[prefix_freq >= threshold].index.tolist()

            for prefix in significant_prefixes:
                self.data.loc[
                    (self.data["Title_Grouped"] == title_group)
                    & (self.data["TicketPrefix"] == prefix),
                    new_feature[0],
                ] = prefix

        # 将未分类的 "Others" 重新赋值，以确保所有的票据前缀都被分类
        self.data[new_feature[0]] = self.data[new_feature[0]].where(
            self.data[new_feature[0]] != "Others", other="Rare"
        )

        return self.data, new_feature
# 其他代码保持不变

class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变

        ticket_processor = TicketProcessor(self.data)
        self.data, _ = ticket_processor.ticket_process()
        self.data, _ = ticket_processor.categorize_ticket_prefix_using_title()
        self.data, new_feature_ticket_title = ticket_processor.ticket_one_hot_encode(
            "TicketPrefixCategorized"
        )
        # reducer = DimensionalityReducer(
        #     self.data, method="SVD", n_components=5, random_state=42
        # )
        # self.data, new_feature_names_ticket_reduced = reducer.apply_reduction()
        self.features.extend(new_feature_ticket_title)

        return self.data, self.features
```

重新运行 `main.py`，评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'TicketPrefixCategorized_CA', 'TicketPrefixCategorized_None', 'TicketPrefixCategorized_PC', 'TicketPrefixCategorized_Rare']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.810056   0.785714  0.743243  0.763889  0.888546

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  90                  15
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.843810
```

与按照 `TicketPrefix` 的频率和其累计频率分类的评估结果对比，这种复杂的分类方法并没有产生太多正面的影响。除了在 ROC AUC 有了些许上升外，其他指标要不持平，要不下降。因此，<strong style="color:#c21d03">可以得出，该种分类方法对于本项目的逻辑回归模型训练可能并不合适。</strong>

同学们可以测试下余下的其他分类策略是否能对逻辑回归模型训练效果有所影响。对于  `Ticket` 特征的处理暂告一段落。 

[^1]: 使用点双列相关通常需要满足一些前提假设，例如**正态分布假设**，**线性关系**，**样本容量**等。但在实际应用中，这些假设可以有一定的灵活性。比如关于**正态分布假设**，确实，理想情况下，连续变量应接近正态分布。但在实践中，特别是对于大样本数据，中心极限定理保证了即使数据不完全正态，相关性测试结果也是可靠的。在 Titanic 数据集中，连续变量可能不完全符合正态分布，但仍可计算点双列相关系数以得到大致的相关趋势。对于**线性关系假设**，点双列相关系数度量的是变量之间的线性关系。即使实际关系不是完全线性的，该系数也可以提供一个关系强度的估计。对于 Titanic 数据集，可以先通过可视化（如散点图）初步探索生存率与数值变量之间的关系，判断是否存在大致的线性趋势。因此，尽管 Titanic 数据集中的数值型特征可能不完全符合点双列相关系数的所有理论假设，该方法仍然是探索生存特征与其他数值型特征相关性的有用工具。

<hr/>

###  变量：`Fare`

在此，我们继续进一步单独考虑 `Fare` 特征。 由于是数值型数据，该特征处理过程相对简单。由 EDA 分析可知， `Fare` 特征分布呈现出极度的右偏，表明大多数乘客支付的票价较低。这种情况下，我们至少需要对其进行数据转换。从代码角度上来说，也是比较简单的，由于我们前期在 `BaseProcessor` 基础类中已经构建了相关数据转换方法。由此，下面只需要在 `DataPreprocessor` 中添加对该特征的处理就成,示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class FareProcessor(BaseProcessor):
    def fare_process(self):
        new_feature = ["Fare"]
        return self.data, new_feature


class DataPreprocessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变

        fare_processor = FareProcessor(self.data)
        self.data, _ = fare_processor.fare_process()
        self.data, new_features_fare = base_processor.scaling_robust("Fare")
        self.features.extend(new_features_fare)

        return self.data, self.features
```

当选择不同数据转换方法时，重新运行 `main.py` 后的结果如下：

经过 `RobustScaler` 转换后的结果：
```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareRobustScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.896139

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.860635
```

经过 `Min-Max` 转换后的结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.797101  0.743243  0.769231  0.895882

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.866190
```
经过 `Z-Score` 转换后的结果：
```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareStandardScaler']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.896139

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.860635
```

不经过转换时的评估结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'Fare']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score  ROC AUC
Values  0.804469   0.791045  0.716216  0.751773  0.89601

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.855079
```

对比这些结果，我们可以注意到几个关键点：

- **考虑 `Fare` 特征前后**：在考虑 `Fare` 特征之前的模型表现在多数指标上稍优于考虑 `Fare` 特征后的模型。准确率最高，而且交叉验证的准确率也较高。
- **不同的 `Fare` 转换方法**：**RobustScaler** 和 **Z-Score** 转换后的结果相同，显示了相对较低的准确率和交叉验证的准确率。**Min-Max** 转换提供了较好的准确率和交叉验证的准确率，与不考虑 `Fare` 特征的模型表现相当。未经转换的 `Fare` 特征结果表现在准确率和交叉验证的准确率上略低于 **Min-Max** 转换和原始模型。

因此，在**考虑 `Fare` 特征**后，模型的表现在某些度量上略有下降，这可能表明 `Fare` 特征没有提供额外的有用信息，或者模型无法有效地利用这一信息。而在**特征转换的影响**上，特征的转换方法对模型的性能有明显的影响。在这种情况下，**Min-Max** 转换表现得最好，而 **RobustScaler** 和 **Z-Score** 转换没有带来预期的性能提升。<strong style="color:#c21d03"> 所以，我们大致可以得出，在逻辑回归模型下，如果要考虑 `Fare` 特征，应该采用 **Min-Max** 对原始数据进一步处理。</strong>

<hr/>

### 变量：`Cabin`

`Cabin` 特征的缺失值较多，占了约 80%。按照之前的分析，处理这类特征的策略可以多种多样，比如：

1. **缺失值标记法**：可以创建一个新的特征来表示 `Cabin` 是否缺失。这种方法不会填补缺失值，而是将缺失的存在转化为一个信息特征，因为缺失本身可能就携带着一些信息。例如，`Cabin_Missing` 特征，它可以是 1（如果 `Cabin` 缺失）或 0（如果 `Cabin` 非缺失）。
2. **填充缺失值**：如果决定填充缺失的 `Cabin` 数据，可以选择一种统一的填充方式，例如使用一个特殊字符或字符串，比如 "Unknown"。这样可以保留 `Cabin` 的信息，同时处理缺失值。
3. **利用 `Cabin` 的首字母**：如果 `Cabin` 值不缺失，它通常以字母开头，这个字母可能表示船舱所在的甲板。因此，可以提取这个首字母作为一个新特征，用于模型训练。对于缺失值，同样可以用 "Unknown" 或其他特殊字符标记。
4. **分组处理**：根据已有的 `Cabin` 数据，可以尝试将其分为不同的组别。比如，基于船舱号码的数字部分或首字母，将乘客分为不同的组。这可能需要一些对数据的了解和预处理工作。
5. **丢弃特征**：如果经过探索性分析发现 `Cabin` 特征与生存情况关系不大，或者缺失值太多以至于填充或转换后的信息可信度不高，可以考虑直接丢弃这个特征。

在处理完 `Cabin` 特征后，记得通过模型的交叉验证来检查特征处理的效果，选择最有助于提高模型性能的方法。`Cabin` 特征的以上部分处理策略的实现的示例代码如下：

```python
# titanic/titanci/data_preprocessing.py
class CabinProcessor(BaseProcessor):
    def add_missing_indicator(self):
        new_feature = ["CabinMissing"]
        self.data[new_feature[0]] = self.data["Cabin"].isnull().astype(int)
        return self.data, new_feature

    def fill_missing(self, fill_value="Unknown"):
        new_feature = ["CabinMissingFill"]
        self.data[new_feature[0]] = self.data["Cabin"].fillna(fill_value)
        return self.data, new_feature

    def extract_first_letter(self):
        new_feature = ["CabinFirstLetter"]
        self.data[new_feature[0]] = self.data["Cabin"].apply(
            lambda x: x[0] if pd.notnull(x) else "U"
        )
        return self.data, new_feature
```

采用缺失值标记法的评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'CabinMissing']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229   0.808824  0.743243  0.774648  0.896525

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

采用填充缺失值（未经过降维处理）的评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'CabinMissingFill_A10', 'CabinMissingFill_A14', 'CabinMissingFill_A16', 'CabinMissingFill_A19', 'CabinMissingFill_A20', 'CabinMissingFill_A23', 'CabinMissingFill_A24', 'CabinMissingFill_A26', 'CabinMissingFill_A31', 'CabinMissingFill_A32', 'CabinMissingFill_A34', 'CabinMissingFill_A36', 'CabinMissingFill_A5', 'CabinMissingFill_A6', 'CabinMissingFill_A7', 'CabinMissingFill_B101', 'CabinMissingFill_B102', 'CabinMissingFill_B18', 'CabinMissingFill_B19', 'CabinMissingFill_B20', 'CabinMissingFill_B22', 'CabinMissingFill_B28', 'CabinMissingFill_B3', 'CabinMissingFill_B30', 'CabinMissingFill_B35', 'CabinMissingFill_B37', 'CabinMissingFill_B38', 'CabinMissingFill_B39', 'CabinMissingFill_B4', 'CabinMissingFill_B41', 'CabinMissingFill_B42', 'CabinMissingFill_B49', 'CabinMissingFill_B5', 'CabinMissingFill_B50', 'CabinMissingFill_B51 B53 B55', 'CabinMissingFill_B57 B59 B63 B66', 'CabinMissingFill_B58 B60', 'CabinMissingFill_B69', 'CabinMissingFill_B71', 'CabinMissingFill_B73', 'CabinMissingFill_B77', 'CabinMissingFill_B78', 'CabinMissingFill_B79', 'CabinMissingFill_B80', 'CabinMissingFill_B82 B84', 'CabinMissingFill_B86', 'CabinMissingFill_B94', 'CabinMissingFill_B96 B98', 'CabinMissingFill_C101', 'CabinMissingFill_C103', 'CabinMissingFill_C104', 'CabinMissingFill_C106', 'CabinMissingFill_C110', 'CabinMissingFill_C111', 'CabinMissingFill_C118', 'CabinMissingFill_C123', 'CabinMissingFill_C124', 'CabinMissingFill_C125', 'CabinMissingFill_C126', 'CabinMissingFill_C128', 'CabinMissingFill_C148', 'CabinMissingFill_C2', 'CabinMissingFill_C22 C26', 'CabinMissingFill_C23 C25 C27', 'CabinMissingFill_C30', 'CabinMissingFill_C32', 'CabinMissingFill_C45', 'CabinMissingFill_C46', 'CabinMissingFill_C47', 'CabinMissingFill_C49', 'CabinMissingFill_C50', 'CabinMissingFill_C52', 'CabinMissingFill_C54', 'CabinMissingFill_C62 C64', 'CabinMissingFill_C65', 'CabinMissingFill_C68', 'CabinMissingFill_C7', 'CabinMissingFill_C70', 'CabinMissingFill_C78', 'CabinMissingFill_C82', 'CabinMissingFill_C83', 'CabinMissingFill_C85', 'CabinMissingFill_C86', 'CabinMissingFill_C87', 'CabinMissingFill_C90', 'CabinMissingFill_C91', 'CabinMissingFill_C92', 'CabinMissingFill_C93', 'CabinMissingFill_C95', 'CabinMissingFill_C99', 'CabinMissingFill_D', 'CabinMissingFill_D10 D12', 'CabinMissingFill_D11', 'CabinMissingFill_D15', 'CabinMissingFill_D17', 'CabinMissingFill_D19', 'CabinMissingFill_D20', 'CabinMissingFill_D21', 'CabinMissingFill_D26', 'CabinMissingFill_D28', 'CabinMissingFill_D30', 'CabinMissingFill_D33', 'CabinMissingFill_D35', 'CabinMissingFill_D36', 'CabinMissingFill_D37', 'CabinMissingFill_D45', 'CabinMissingFill_D46', 'CabinMissingFill_D47', 'CabinMissingFill_D48', 'CabinMissingFill_D49', 'CabinMissingFill_D50', 'CabinMissingFill_D56', 'CabinMissingFill_D6', 'CabinMissingFill_D7', 'CabinMissingFill_D9', 'CabinMissingFill_E10', 'CabinMissingFill_E101', 'CabinMissingFill_E12', 'CabinMissingFill_E121', 'CabinMissingFill_E17', 'CabinMissingFill_E24', 'CabinMissingFill_E25', 'CabinMissingFill_E31', 'CabinMissingFill_E33', 'CabinMissingFill_E34', 'CabinMissingFill_E36', 'CabinMissingFill_E38', 'CabinMissingFill_E40', 'CabinMissingFill_E44', 'CabinMissingFill_E46', 'CabinMissingFill_E49', 'CabinMissingFill_E50', 'CabinMissingFill_E58', 'CabinMissingFill_E63', 'CabinMissingFill_E67', 'CabinMissingFill_E68', 'CabinMissingFill_E77', 'CabinMissingFill_E8', 'CabinMissingFill_F E69', 'CabinMissingFill_F G63', 'CabinMissingFill_F G73', 'CabinMissingFill_F2', 'CabinMissingFill_F33', 'CabinMissingFill_F38', 'CabinMissingFill_F4', 'CabinMissingFill_G6', 'CabinMissingFill_T', 'CabinMissingFill_Unknown']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.797101  0.743243  0.769231  0.897426

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

采用填充缺失值（经过 PCA 降维处理）的评估结果如下：

```plaintext
PCA reduced the CabinMissingFill features to 129 components.
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'PCA_CabinMissingFill_1', 'PCA_CabinMissingFill_2', 'PCA_CabinMissingFill_3', 'PCA_CabinMissingFill_4', 'PCA_CabinMissingFill_5', 'PCA_CabinMissingFill_6', 'PCA_CabinMissingFill_7', 'PCA_CabinMissingFill_8', 'PCA_CabinMissingFill_9', 'PCA_CabinMissingFill_10', 'PCA_CabinMissingFill_11', 'PCA_CabinMissingFill_12', 'PCA_CabinMissingFill_13', 'PCA_CabinMissingFill_14', 'PCA_CabinMissingFill_15', 'PCA_CabinMissingFill_16', 'PCA_CabinMissingFill_17', 'PCA_CabinMissingFill_18', 'PCA_CabinMissingFill_19', 'PCA_CabinMissingFill_20', 'PCA_CabinMissingFill_21', 'PCA_CabinMissingFill_22', 'PCA_CabinMissingFill_23', 'PCA_CabinMissingFill_24', 'PCA_CabinMissingFill_25', 'PCA_CabinMissingFill_26', 'PCA_CabinMissingFill_27', 'PCA_CabinMissingFill_28', 'PCA_CabinMissingFill_29', 'PCA_CabinMissingFill_30', 'PCA_CabinMissingFill_31', 'PCA_CabinMissingFill_32', 'PCA_CabinMissingFill_33', 'PCA_CabinMissingFill_34', 'PCA_CabinMissingFill_35', 'PCA_CabinMissingFill_36', 'PCA_CabinMissingFill_37', 'PCA_CabinMissingFill_38', 'PCA_CabinMissingFill_39', 'PCA_CabinMissingFill_40', 'PCA_CabinMissingFill_41', 'PCA_CabinMissingFill_42', 'PCA_CabinMissingFill_43', 'PCA_CabinMissingFill_44', 'PCA_CabinMissingFill_45', 'PCA_CabinMissingFill_46', 'PCA_CabinMissingFill_47', 'PCA_CabinMissingFill_48', 'PCA_CabinMissingFill_49', 'PCA_CabinMissingFill_50', 'PCA_CabinMissingFill_51', 'PCA_CabinMissingFill_52', 'PCA_CabinMissingFill_53', 'PCA_CabinMissingFill_54', 'PCA_CabinMissingFill_55', 'PCA_CabinMissingFill_56', 'PCA_CabinMissingFill_57', 'PCA_CabinMissingFill_58', 'PCA_CabinMissingFill_59', 'PCA_CabinMissingFill_60', 'PCA_CabinMissingFill_61', 'PCA_CabinMissingFill_62', 'PCA_CabinMissingFill_63', 'PCA_CabinMissingFill_64', 'PCA_CabinMissingFill_65', 'PCA_CabinMissingFill_66', 'PCA_CabinMissingFill_67', 'PCA_CabinMissingFill_68', 'PCA_CabinMissingFill_69', 'PCA_CabinMissingFill_70', 'PCA_CabinMissingFill_71', 'PCA_CabinMissingFill_72', 'PCA_CabinMissingFill_73', 'PCA_CabinMissingFill_74', 'PCA_CabinMissingFill_75', 'PCA_CabinMissingFill_76', 'PCA_CabinMissingFill_77', 'PCA_CabinMissingFill_78', 'PCA_CabinMissingFill_79', 'PCA_CabinMissingFill_80', 'PCA_CabinMissingFill_81', 'PCA_CabinMissingFill_82', 'PCA_CabinMissingFill_83', 'PCA_CabinMissingFill_84', 'PCA_CabinMissingFill_85', 'PCA_CabinMissingFill_86', 'PCA_CabinMissingFill_87', 'PCA_CabinMissingFill_88', 'PCA_CabinMissingFill_89', 'PCA_CabinMissingFill_90', 'PCA_CabinMissingFill_91', 'PCA_CabinMissingFill_92', 'PCA_CabinMissingFill_93', 'PCA_CabinMissingFill_94', 'PCA_CabinMissingFill_95', 'PCA_CabinMissingFill_96', 'PCA_CabinMissingFill_97', 'PCA_CabinMissingFill_98', 'PCA_CabinMissingFill_99', 'PCA_CabinMissingFill_100', 'PCA_CabinMissingFill_101', 'PCA_CabinMissingFill_102', 'PCA_CabinMissingFill_103', 'PCA_CabinMissingFill_104', 'PCA_CabinMissingFill_105', 'PCA_CabinMissingFill_106', 'PCA_CabinMissingFill_107', 'PCA_CabinMissingFill_108', 'PCA_CabinMissingFill_109', 'PCA_CabinMissingFill_110', 'PCA_CabinMissingFill_111', 'PCA_CabinMissingFill_112', 'PCA_CabinMissingFill_113', 'PCA_CabinMissingFill_114', 'PCA_CabinMissingFill_115', 'PCA_CabinMissingFill_116', 'PCA_CabinMissingFill_117', 'PCA_CabinMissingFill_118', 'PCA_CabinMissingFill_119', 'PCA_CabinMissingFill_120', 'PCA_CabinMissingFill_121', 'PCA_CabinMissingFill_122', 'PCA_CabinMissingFill_123', 'PCA_CabinMissingFill_124', 'PCA_CabinMissingFill_125', 'PCA_CabinMissingFill_126', 'PCA_CabinMissingFill_127', 'PCA_CabinMissingFill_128', 'PCA_CabinMissingFill_129']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score  ROC AUC
Values  0.815642   0.797101  0.743243  0.769231  0.89704

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```
利用 `Cabin` 的首字母 (未经过 PCA 降维处理)

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'CabinFirstLetter_A', 'CabinFirstLetter_B', 'CabinFirstLetter_C', 'CabinFirstLetter_D', 'CabinFirstLetter_E', 'CabinFirstLetter_F', 'CabinFirstLetter_G', 'CabinFirstLetter_T', 'CabinFirstLetter_U']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.797101  0.743243  0.769231  0.895624

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

利用 `Cabin` 的首字母 (经过 PCA 降维处理)

```plaintext
PCA reduced the CabinFirstLetter features to 6 components.
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'PCA_CabinFirstLetter_1', 'PCA_CabinFirstLetter_2', 'PCA_CabinFirstLetter_3', 'PCA_CabinFirstLetter_4', 'PCA_CabinFirstLetter_5', 'PCA_CabinFirstLetter_6']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229   0.808824  0.743243  0.774648  0.896396

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.849365
```

比较不同处理 `Cabin` 特征策略下的逻辑回归模型评估指标结果，可以发现：

1. **缺失值标记法** 和 **填充缺失值** 方法的模型在准确率、精确率、召回率、F1分数和ROC AUC方面的表现都相近，但缺失值标记法在准确率和 F1 分数上略有优势。
2. **PCA降维** 处理后的模型与未进行降维处理的模型相比，在大多数评估指标上差异不大。这表明降维可能没有对模型的性能产生显著影响。
3. **利用 `Cabin` 首字母** 的方法，在经过 PCA 降维处理后，模型在准确率和 F1 分数上略有提升，但整体表现与其他处理方法相比没有显著差异。
4. 与**不考虑 `Cabin` 特征**的模型相比，考虑 `Cabin` 特征的模型在准确率和 ROC AUC 上有轻微的提升，但整体差异不显著。

因此，<strong style="color:#c21d03">考虑 `Cabin` 特征确实对模型性能有一定的影响，尽管这种影响并不是非常显著。当考虑将其纳入模型时，采用缺失值标记法或利用 `Cabin` 的首字母作为特征的策略可能比较合适。</strong>

<hr/>

### 变量：`Embarked`

对于 `Embarked` 特征，由 EDA 分析可知，其缺失值较少，仅有两个，因此，计划将其缺失值**填充为最常见的值**，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
class EmbarkedProcessor(BaseProcessor):
    def fill_missing_with_most_common(self):
        most_common_value = self.data["Embarked"].mode()[0]
        new_feature = ["EmbarkedFillCommon"]
        self.data[new_feature[0]] = self.data["Embarked"].fillna(most_common_value)
        return self.data, new_feature


class DataProcessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        embarked_processor = EmbarkedProcessor(self.data)
        self.data, new_features_embarked = (
            embarked_processor.fill_missing_with_most_common()
        )
        self.data, new_features_embarked = base_processor.one_hot_encode(
            new_features_embarked[0]
        )
        self.features.extend(new_features_embarked)
        return self.data, self.features
```

运行 `main.py` 后的逻辑回归模型评估指标结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SibSp', 'Parch', 'FareMinMaxScaler', 'CabinMissing', 'EmbarkedFillCommon_C', 'EmbarkedFillCommon_Q', 'EmbarkedFillCommon_S']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229        0.8  0.756757  0.777778  0.892921

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  18                  56

Cross-validated Accuracy (5-fold): 0.843651
```

与基线模型中的评估结果相比，所有评估指标均有所提升（交叉验证保持一致），但效果不明显。因此，可以说，在训练逻辑回归模型时可以将 `Embarked` 特征纳入其中。

<hr/>

到此，我们基本上对每一个特征进行了相关分析。但是，我们基本上是单独考虑各个特征。由前面的多变量分析可知，不同变量的组合对生存情况有组合效应。例如：虽然女性在所有等级的船舱中生存率都较高，但三等舱的女性乘客生存率与一等舱和二等舱相比有显著下降。这可能表明，尽管性别是一个强有力的生存预测因子，船舱等级也在生存机会中扮演了重要角色。因此，接下来，我们进一步根据多变量分析的启示，构建组合特征，并探讨其对基线模型的训练效果的影响。

<hr/>

### 考虑组合特征

特征之间的组合需要考虑数据类型特征。不同数据类型之间的组合需要选择不同的处理方式。下面简要介绍下常见的不同数据类型可以采用的组合策略。

1. **数值型与数值型组合**：这种数据类型处理起来比较简单，根据项目目的，可以采用加减乘除甚至是幂运算等常见的数学运算的方式。比如前期在构建 `FamilySize` 时，我们是将 `SibSp` 和 `Parch` 这两相关的特征进行了加和处理。
2. **类别型与类别型组合**：该类数据常用的有两种策略，一种是连接组合，即将两个类别型特征的字符串值连接起来，形成新的类别；一种是交叉组合，即基于两个类别型特征的所有唯一值对生成新的特征，通常用于高维特征的生成，如在特征哈希或嵌入技术中。
3. **类别型与数值型组合**：该类数据常用的也有两种策略。一种是交互项组合，基于类别型特征的不同类别，为数值型特征生成不同的数值列；一种是分组统计，根据类别型特征的类别分组，计算数值型特征的统计量（如均值、中位数、标准差等）。其实我们在处理 `Age` 的缺失值时，也采用了这种策略。
4. **时间型与数值型组合**：在遇到该种数据组合式，可以将时间型特征分解（如年、月、日、小时等），再与与数值型特征相结合，这种处理方式也叫时间分解处理；其次，我们可以计算两个时间点之间的间隔，再结合数值型特征，构建新特征，即时间间隔组合策略。
5. **时间型与类别型组合**：遇到该种数据类型时，可以采用类似于**时间型与数值型组合**的处理方式，将时间型特征分解，再与类别型特征进行交叉组合。
6. **文本与其他类型组合**：该种组合一般较少见，但有时候有用。可以结合文本长度（或其他文本衍生特征）和数值型特征。甚至可以根据文本内容的关键字或主题与类别型特征组合。

在构造这些特征时，需要注意特征的**解释性和实际意义**，以及可能导致的**数据维度诅咒**[^2]。不是所有组合都会对模型带来好处，有时候可能会引入噪声，因此在引入新的特征组合后，进行适当的特征选择和模型验证很重要。

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

分析各个特征，该项目主要涉及到以下几类组合：类别型数据之间（如，性别和船舱等级、登船港口与船舱等级、性别与头衔、票号前缀与船舱等级等）；数值型与类别型数据之间（如，票价与船舱等级、头衔、性别与年龄、家庭规模与性别等）；数值型数据之间（如，兄弟姐妹和配偶的数量和父母和孩子的数量）。此外，还可以考虑家庭总数人与其他类别（如船舱等级）的组合，或者根据已创建的“是否独自一人与其他类别特征的组合。类似的，还可以将票价分成不同的区间，将其转换为类别型特征，再与其他特征进行组合。

我们先来处理类别型数据之间的组合，这里我们选择简单的连接组合策略。在代码组织上，该种策略主要是将两个类别型特征的字符串连接起来，应该具有一定的通用性，因此，我们试着构建一个较为通用的方法，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class FeatureInteractionProcessor(BaseProcessor):
    def add_interaction_feature(self, feature1, feature2, separator=""):
        new_feature_name = f"{feature1}{separator}{feature2}"
        self.data[new_feature_name] = (
            self.data[feature1].astype(str)
            + separator
            + self.data[feature2].astype(str)
        )
        return self.data, [new_feature_name]
```

由此，我们可以在 `DatPreprocessor` 类中添加对类别型特征的组合，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        cate_interaction_processor = FeatureInteractionProcessor(self.data)
        self.data, new_feature_sex_pclass = (
            cate_interaction_processor.add_interaction_feature("Sex", "Pclass")
        )
        self.data, new_feature_sex_pclass = base_processor.one_hot_encode(
            new_feature_sex_pclass[0]
        )
        self.features.extend(new_feature_sex_pclass)

        return self.data, self.features
```

这里我们首先考虑了**性别与船舱等级**特征之间的组合。运行 `main.py` 得到如下结果：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SexPclass_female1', 'SexPclass_female2', 'SexPclass_female3', 'SexPclass_male1', 'SexPclass_male2', 'SexPclass_male3']
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229       0.85  0.689189  0.761194  0.890862

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  96                   9
Actual Positive                  23                  51

Cross-validated Accuracy (5-fold): 0.849365
```

与[采用 `Z-Score` 方式对 `Age`](#basemodel)进行处理后的基线模型评估结果对比，可以发现：

1. **准确率 (Accuracy)**: 加入性别与船舱等级交互特征后，模型的准确率从0.810056提高到了0.821229。这表明在考虑了性别与船舱等级的相互作用之后，模型在整体上对数据的预测更加准确。
2. **精确率 (Precision)**: 精确率从0.794118增加到了0.85。这意味着在预测乘客幸存的情况下，模型的错误率降低了，预测为正类（幸存）的乘客中，实际为正类的比例更高。
3. **召回率 (Recall)**: 召回率略有下降，从0.72973降到0.689189。这表示在所有实际为正类的乘客中，被模型正确识别的比例略有降低。
4. **F1 分数 (F1 Score)**: F1 分数从0.760563上升到0.761194，F1分数是精确率和召回率的调和平均，这里的轻微提高表明加入交互特征后，模型在精确率和召回率之间保持了较好的平衡。
5. **ROC AUC**: 模型的 ROC AUC 从0.881853提高到了0.890862，表明模型区分正负类的能力有所提高。
6. **混淆矩阵 (Confusion Matrix)**: 加入交互特征后，模型正确预测幸存者的数量有所减少（从54降到51），但同时，将幸存者错误预测为死亡的情况也减少了（从20降到23），且将死亡者正确预测为死亡的情况增加（从91增到96）。
7. **交叉验证准确率 (Cross-validated Accuracy)**: 两种模型在交叉验证的准确率几乎相同，表明模型的稳定性和泛化能力相近。

整体上，加入性别与船舱等级的交互特征后，模型在准确率、精确率、F1分数和ROC AUC上有所提升，但召回率略有下降。这表明<strong style="color:#c21d03">性别与船舱等级的交互特征有助于提高模型的整体预测性能，尤其是在确定乘客是否幸存的任务上更为精确，但同时可能略微牺牲了将所有实际幸存者识别出来的能力</strong>。

<hr style="border-top: dashed #8fbf9f; border-bottom: none; background-color: transparent"/>

考虑**登船港口与船舱等级**之间的组合效应，示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        embarked_processor = EmbarkedProcessor(self.data)
        self.data, new_features_embarked = (
            embarked_processor.fill_missing_with_most_common()
        )

        cate_interaction_processor = FeatureInteractionProcessor(self.data)
        self.data, new_features_embarked_pclass = (
            cate_interaction_processor.add_interaction_feature(
                new_features_embarked[0], "Pclass"
            )
        )
        self.data, new_features_embarked_pclass = base_processor.one_hot_encode(
            new_features_embarked_pclass[0]
        )
        self.features.extend(new_features_embarked_pclass)

        return self.data, self.features
```

评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'EmbarkedFillCommonPclass_C1', 'EmbarkedFillCommonPclass_C2', 'EmbarkedFillCommonPclass_C3', 'EmbarkedFillCommonPclass_Q1', 'EmbarkedFillCommonPclass_Q2', 'EmbarkedFillCommonPclass_Q3', 'EmbarkedFillCommonPclass_S1', 'EmbarkedFillCommonPclass_S2', 'EmbarkedFillCommonPclass_S3']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.804469   0.782609  0.72973  0.755245  0.881145

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  90                  15
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.838254
```

同理，对比发现：
1. **准确率 (Accuracy)**: 在加入了与 `Embarked` 和 `Pclass` 相关的交互特征后，模型的准确率从0.810056下降到了0.804469。这表明在考虑了这些交互特征后，模型在整体上的预测准确度有所下降。
2. **精确率 (Precision)**: 精确率从0.794118减少到了0.782609。这意味着在加入交互特征后，模型在预测乘客幸存的情况下，准确性有所降低。
3. **召回率 (Recall)**: 召回率保持不变，仍然是0.72973。这表示在所有实际为正类的乘客中，模型正确识别的比例没有改变。
4. **F1 分数 (F1 Score)**: F1分数从0.760563下降到了0.755245。F1分数是精确率和召回率的调和平均数，这里的下降表明在精确率和召回率之间的平衡略有下降。
5. **ROC AUC**: 模型的ROC AUC从0.881853微降至0.881145，显示模型区分正负类的能力略有下降。
6. **混淆矩阵 (Confusion Matrix)**: 加入交互特征后，将幸存者预测为死亡的情况略微增加（从14增至15），将死亡者预测为死亡的情况略有下降（从91降至90）。
7. **交叉验证准确率 (Cross-validated Accuracy)**: 加入交互特征后，交叉验证的准确率从0.84936降至0.838254，表明模型的泛化能力有所下降。

整体上，加入 `Embarked` 和 `Pclass` 的交互特征后，模型在多数性能指标上有所下降，尤其是在准确率、精确率、F1分数和交叉验证准确率上更为明显。这可能<strong style="color:#c21d03">表明 `Embarked` 和 `Pclass` 的交互特征并未为模型提供有用的信息，反而增加了模型的复杂度，导致性能略有下降。</strong>

<hr style="border-top: dashed #8fbf9f; border-bottom: none; background-color: transparent"/>

考虑**性别与头衔**之间的组合效应，该部分示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        title_processor = TitleProcessor(self.data)
        self.data = title_processor.extract_title().group_titles()

        cate_interaction_processor = FeatureInteractionProcessor(self.data)
        self.data, new_features_sex_title = (
            cate_interaction_processor.add_interaction_feature("Sex", "Title_Grouped")
        )
        self.data, new_features_sex_title = base_processor.one_hot_encode(
            new_features_sex_title[0]
        )
        self.features.extend(new_features_sex_title)

        return self.data, self.features
```

评估结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'SexTitle_Grouped_femaleMiss', 'SexTitle_Grouped_femaleMrs', 'SexTitle_Grouped_femaleRare', 'SexTitle_Grouped_maleMaster', 'SexTitle_Grouped_maleMr', 'SexTitle_Grouped_maleRare']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.798883   0.771429  0.72973      0.75  0.877928

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  89                  16
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.832698
```

同理，对比分析后可以得出结论：加入性别与头衔的组合特征后，模型在多数性能指标上有所下降，尤其是在准确率、精确率、F1分数和交叉验证准确率上更为明显。这可能<strong style="color:#c21d03">表明性别与头衔的组合特征并未为模型提供额外的有用信息，反而增加了模型的复杂度，导致性能下降。</strong>

<hr style="border-top: dashed #8fbf9f; border-bottom: none; background-color: transparent"/>

考虑**票号前缀与船舱等级**之间的组合效应,，该部分示例代码如下：

```python
# titanic/titanic/data_preprocessing.py
# 其他代码保持不变
class DataPreprocessor:
    # 其他代码保持不变
    def preprocess(self):
        # 其他代码保持不变
        ticket_processor = TicketProcessor(self.data)
        self.data, _ = ticket_processor.ticket_process()
        self.data, new_feature_ticket_title = (
            ticket_processor.categorize_ticket_prefix_using_title()
        )

        cate_interaction_processor = FeatureInteractionProcessor(self.data)
        self.data, new_feature_ticket_pclass = (
            cate_interaction_processor.add_interaction_feature(
                new_feature_ticket_title[0], "Pclass"
            )
        )
        self.data, new_feature_ticket_pclass = base_processor.one_hot_encode(
            new_feature_ticket_pclass[0]
        )
        self.features.extend(new_feature_ticket_pclass)

        return self.data, self.features
```

结果如下：

```plaintext
Features considered in the model: ['Pclass', 'Sex_female', 'Sex_male', 'AgeFillTitleGroupedStandardScaler', 'TicketPrefixCategorizedPclass_CA2', 'TicketPrefixCategorizedPclass_CA3', 'TicketPrefixCategorizedPclass_None1', 'TicketPrefixCategorizedPclass_None2', 'TicketPrefixCategorizedPclass_None3', 'TicketPrefixCategorizedPclass_PC1', 'TicketPrefixCategorizedPclass_Rare1', 'TicketPrefixCategorizedPclass_Rare2', 'TicketPrefixCategorizedPclass_Rare3']
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.804469   0.782609  0.72973  0.755245  0.881467

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  90                  15
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.849365
```

同理，对比分析后可以得出结论：加入票号前缀与船舱等级的组合特征后，模型在准确率、精确率、F1分数上有所下降，而召回率保持不变，ROC AUC略有下降。这与考虑性别与头衔之间的组合效应时的效果一样，<strong style="color:#c21d03">表明该组合特征并未为模型提供额外的有用信息，反而增加了模型的复杂度，导致性能下降。</strong>

<hr style="border-top: dashed #E7D1BB; border-bottom: none; background-color: transparent"/>

类别型数据之间的组合特征构建暂告一段落，同学们可以试着构建其他的类别型变量之间的组合特征。现在我们试着构建数值型和类别型之间的组合特征。


[^2]: 维度诅咒（curse of dimensionality），或者称为维度爆炸，维度灾难，是指随着数据集的特征数量增加，模型所需的数据量呈指数级增长的现象。在高维空间中，数据的表现和我们在低维空间直观感受到的性质有很大不同，这对数据分析和机器学习模型的建立和性能有着深远的影响。具体体现在以下几个方面：1) **空间稀疏性**：随着维度的增加，数据点在空间中越来越稀疏，大部分数据点都远离彼此。这意味着为了准确地学习数据间的关系，需要指数级别增长的数据量。2) **距离度量失效**：在高维空间中，常用的距离度量（如欧氏距离）变得不再有效。不同点之间的距离差异变得非常小，这使得基于距离的算法（如k-最近邻）性能下降。3) **模型过拟合**：随着特征数量的增加，模型复杂度增加，使得模型容易在训练数据上过拟合，即在训练集上表现很好，但在未见过的测试数据上表现不佳。4) **计算复杂性增加**：随着特征维度的增加，模型的计算复杂性也会增加，这不仅增加了训练模型所需的时间，也增加了存储和计算资源的需求。5) **降维困难**：虽然可以通过降维技术（如PCA、t-SNE）来减少特征的数量，但在极高维度下这些技术的效果可能会下降，而且降维本身也可能丢失一些重要信息。

<hr>

### 模型训练与评估流程图

![](/assets/images/ml/titianic_model_training_evaluation_workflow.svg)

