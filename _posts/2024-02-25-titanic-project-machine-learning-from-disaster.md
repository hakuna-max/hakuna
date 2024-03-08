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
  - [第一次尝试（缺失值的处理：不同头衔的 `Age` 中位数）](#第一次尝试缺失值的处理不同头衔的-age-中位数)
  - [第二次尝试（`Age` 特征标准化/归一化）](#第二次尝试age-特征标准化归一化)
  - [第三次尝试（考虑 `SibSp` 和 `Parch` 特征)](#第三次尝试考虑-sibsp-和-parch-特征)
  - [第四次尝试（考虑 `Ticket` 特征）](#第四次尝试考虑-ticket-特征)

## Titanic 项目介绍

大家好！今天，我想带你们走进一个非常有趣的机器学习项目——Kaggle 上的 Titanic 生还预测挑战。这个项目的目标是使用 Titanic 号乘客的数据来预测哪些乘客在这场历史性的灾难中幸存下来（即，分类问题）。这个项目不仅是一个绝佳的机会来实践和理解机器学习的基本流程，而且也是一个向所有对商务智能与机器学习感兴趣的同学们展示如何从实际数据中提取洞见的绝佳案例。

项目开始于对数据集的介绍——我们有乘客的各种信息，如年龄、性别、票价和乘客在船上的等级，这些都可能影响他们的生还机会。理解这些特征及其与目标变量之间的关系是我们任务的第一步。

接下来，我们会进行探索性数据分析，或称 EDA，它帮助我们通过可视化和数据摘要来揭示数据的内在模式和特征关系。

特征工程阶段，我们会选择最有影响的特征，并可能创造新特征来帮助模型更好地理解数据。紧接着，我们将探索和比较不同的机器学习模型，比如逻辑回归、随机森林、支持向量机、朴素贝叶斯、决策树等，以找到最适合我们数据的模型。

通过训练模型和使用交叉验证等技术评估其性能后，我们将选择一个最终模型。然后，我们将深入分析模型的结果，理解哪些因素对生还预测最为重要，这不仅加深了我们对数据的理解，也让我们学习到了如何解释机器学习模型的预测。

总体上，希望通过该项目实验，同学们不仅学习了机器学习的整个流程，还获得了宝贵的实践经验。

探索机器学习的奇妙世界，解锁数据的潜力，为未来铺平道路。

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
from data_preprocessing import DataProcessor
from model import BaseModel


def load_and_preprocess_data(data_path):
    """加载数据并进行预处理"""
    processor = DataProcessor(data_path)
    data = processor.preprocess()
    return data


def train_and_evaluate_model(data, features, target):
    """训练模型并进行评估"""
    model = BaseModel()
    model.train(data[features], data[target])
    accuracy = model.evaluate()
    return accuracy


def main():
    # 设置数据路径
    data_path = "./data/raw/train.csv"

    # 加载和预处理数据
    data = load_and_preprocess_data(data_path)

    # 模型训练与评估
    features = ["Pclass", "Sex", "Age"]
    target = "Survived"
    accuracy = train_and_evaluate_model(data, features, target)

    print(f"Baseline Model Accuracy: {accuracy:.04f}")


if __name__ == "__main__":
    main()

```

到目前为止，我们还没有构建 `DataProcessor` 和 `BaseModel` 类及其方法，因此，还不能运行 `main.py`。接下来，根据前面的假设，我们继续完善`DataProcessor` 和 `BaseModel` 类及其方法。

对于 `BaseModel` 类及其方法， 其示例代码如下：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self):
        self.model = LogisticRegression()
        
    def train(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
```

对于 `DataProcessor`，示例代码如下：

```python
class DataProcessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
    
    def preprocess(self):
        # 填充缺失值
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        
        # 特征编码
        label_encoder = LabelEncoder()
        self.data['Sex'] = label_encoder.fit_transform(self.data['Sex'])
        
        return self.data
```

有两点值得说明：

- 关于 `Age` 的处理。结合EDA分析，我们是知道该特征有缺失值，从分布上看，也存在异常值。因此，这里用了该列的中位数来填充（可能有更好的处理方式，后面再讨论）。中位数对于异常值不敏感，相对更加稳定。
- 关于 `Pclass` 特征。在数据处理中，我们并没有预处理该特征，主要是考虑到 `Pclass` 中的数值（1， 2， 3）能够直接反应生存概率的顺序关系（即1级舱生存概率最高，然后是2级，最后是3级）。Logistic Regression 模型可以直接处理这种有序的数值特征。 

到此，我们的基线模型就构建好了，运行 `main.py`， 可以得出如下结果：
```plaintext
Baseline Model Accuracy: 0.8101
```

### 第一次尝试（缺失值的处理：不同头衔的 `Age` 中位数）

接下来，我们试着对 `Age` 进一步处理，查看对模型训练效果。

从 EDA 的分析结果来看，不同年龄对生产率存在明显影响，对于该特征的缺失值，我们是否可以考虑运用不同头衔的中位数来填充会更好？基于该想法，我们适当修改 `data_processing.py` 中的相关类及其方法，示例代码如下：

```python
class DataProcessor:
    def __init__(self, data_path=None):
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = None

    def preprocess(self):
        self.age_preprocess()
        self.sex_preprocess()

        return self.data

    def age_preprocess(self):
        assert self.data is not None, "Data is not set before preprocessing."
        # 填充缺失值
        self.data["Age"] = self.data["Age"].fillna(self.data["Age"].median())
        return self.data

    def sex_preprocess(self):
        assert self.data is not None, "Data is not set before preprocessing."
        # 特征编码
        label_encoder = LabelEncoder()
        self.data["Sex"] = label_encoder.fit_transform(self.data["Sex"])
        return self.data


class AdvancedDataProcessor(DataProcessor):
    def __init__(self, data_path=None):
        super().__init__(data_path if data_path is not None else "")

    def preprocess(self):
        super().preprocess()
        self.age_preprocess()
        return self.data

    def age_preprocess(self):
        assert self.data is not None, "Data is not set."
        self.fill_age_by_title_group()

    def fill_age_by_title_group(self):
        # 提取头衔
        self.data["Title"] = self.data["Name"].apply(
            lambda x: x.split(", ")[1].split(". ")[0]
        )

        # 对罕见头衔进行分组
        title_counts = self.data["Title"].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        self.data["Title_Grouped"] = self.data["Title"].apply(
            lambda x: "Rare" if x in rare_titles else x
        )

        # 对每个分组填充年龄
        for title_group, group in self.data.groupby("Title_Grouped"):
            median_age = group["Age"].median()
            self.data.loc[
                (self.data["Age"].isnull())
                & (self.data["Title_Grouped"] == title_group),
                "Age",
            ] = median_age
```

值得说明的是：

- 我们适当修改了原始的 `DataProcessor` 类。主要是对 `Age` 和 `Sex` 特征的处理分开了，以便于后期继承该类的部分功能，减少代码的重复。
- `AdvancedDataProcessor` 类是新定义的，其目的是用不同头衔的年龄中位数来填充 `Age` 列中的对应的缺失值，该部分主要工作由其中的 `fill_age_by_title_group` 方法完成。
- 由于我们只修改了 `data_processing.py`, 训练模型不变，由此我们并不需要修改 `model.py`。但我们需要适当修改下 `main.py`,使其能使用新的处理后的数据。修改后的 `main.py` 如下：

```python
from data_preprocessing import DataProcessor, AdvancedDataProcessor # 导入新的类
from model import BaseModel


def load_and_preprocess_data(data_path, Processor):  # 添加了Processor这个参数
    """加载数据并进行预处理"""
    processor = Processor(data_path)                 # DataProcessor --> Processor
    data = processor.preprocess()
    return data


def train_and_evaluate_model(data, features, target):
    """训练模型并进行评估"""
    model = BaseModel()
    model.train(data[features], data[target])
    accuracy = model.evaluate()
    return accuracy


def main():
    # 设置数据路径
    data_path = "./data/raw/train.csv"

    # 加载和预处理数据
    data = load_and_preprocess_data(data_path, AdvancedDataProcessor) # 使用 AdvancedDataProcessor 来处理相关数据

    # 模型训练与评估
    features = ["Pclass", "Sex", "Age"]
    target = "Survived"
    accuracy = train_and_evaluate_model(data, features, target)

    print(f"Model Accuracy (fill age by title group): {accuracy}") # 增加结果区分度


if __name__ == "__main__":
    main()
```

需要修改的地方，均在以上代码块中进行了备注，其他暂时不需要修改。

运行新的 `main.py`，可以得出相应的训练准确率的结果：

```plaintext
Model Accuracy (fill age by title group): 0.8101
```

:disappointed: 不要怀疑以上结果，你没看错。一通操作猛如虎，模型的准确率并没有实质性变化，与基线模型的准确度完全一致。

这种情况可能发生在几种情况下：

1. **数据特性**：如果 `Age` 列对于模型的预测影响不大，或者不同填充策略之间的差异对最终结果没有显著影响，那么准确率可能会保持一致。
2. **模型不敏感**：逻辑回归模型可能对这种细微的数据变化不太敏感（:question:），另外好像如果模型是基于树的算法，如随机森林或梯度提升树，在一定程度上对缺失值的处理方法不太敏感。
3. **数据其他特征的影响较大**：如果数据集中还有其他特征对目标变量有强烈的预测作用，那么 `Age` 特征的变化可能不会对整体模型准确率产生显著影响。
4. **结果偶然相同**：在某些情况下，两种不同的处理方法可能恰好导致模型具有相同的准确率，这可能是偶然事件，特别是在数据集较小或模型训练过程中存在随机性的情况下。

有没有办法进一步验证为什么两种方法得到相同的准确率，当然，可以尝试以下方法：

- **混淆矩阵**：查看每种填充方法的混淆矩阵，可能会揭示不同的错误模式。
- **交叉验证**：通过交叉验证可以获得更稳健的性能估计，可能会揭示准确率的差异。
- **特征重要性**：查看模型中 `Age` 特征的重要性，以判断其对模型的影响程度。
- **其他评估指标**：除了准确率外，还可以考虑使用其他指标（如F1分数、ROC曲线下面积等）来评估模型性能的差异。

现在试着使用以上方法，进一步评估采用了按头衔分类后的中位数填补 `Age` 缺失值后的模型训练情况。这里主要涉及到 `model.py` 文件的修改，如下：

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)  # 导入更多的评估指标
from sklearn.model_selection import train_test_split, cross_val_score # 天骄交叉验证


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, cv=5):
        y_pred = self.model.predict(self.X_test)
        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, average="binary"),
            "Recall": recall_score(self.y_test, y_pred, average="binary"),
            "F1 Score": f1_score(self.y_test, y_pred, average="binary"),
        }

        # 打印评估指标
        print("Evaluation Metrics:")
        print(pd.DataFrame([metrics], index=["Values"]))

        # 打印混淆矩阵
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(
            pd.DataFrame(
                conf_matrix,
                columns=["Predicted Negative", "Predicted Positive"],
                index=["Actual Negative", "Actual Positive"],
            )
        )

        # 交叉验证
        if cv > 1:
            cross_val_accuracy = np.mean(
                cross_val_score(
                    self.model, self.X_test, self.y_test, cv=cv, scoring="accuracy"
                )
            )
            print(f"\nCross-validated Accuracy ({cv}-fold): {cross_val_accuracy:.6f}")

        return metrics


class BaseModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.evaluator = None  # 在训练时设置

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.evaluator = ModelEvaluator(
            self.model, X_test, y_test
        )  # 在训练后创建评估器

    def evaluate(self, cv=5):
        if self.evaluator:
            return self.evaluator.evaluate(cv=cv)
        else:
            raise ValueError("The model needs to be trained before evaluation.")
```

`main.py` 中的代码可以不用修改。但由于我们在 `ModelEvaluator` 包含了评估指标结果输出。因此，可以适当修改 `main.py`，去掉之前的打印结果的部分，如下：

```python
def train_and_evaluate_model(data, features, target):
    """训练模型并进行评估"""
    model = BaseModel()
    model.train(data[features], data[target])
    model.evaluate()
    return 1

def main():
    # 设置数据路径
    data_path = "./data/raw/train.csv"

    # 加载和预处理数据
    data = load_and_preprocess_data(data_path, AdvancedDataProcessor)

    # 模型训练与评估
    features = ["Pclass", "Sex", "Age"]
    target = "Survived"
    train_and_evaluate_model(data, features, target)
```

现在，我们可以查看采用不同的缺失值处理策略后的模型训练评估结果：

对于采用 `Age` 列中位数填补的策略：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score
Values  0.810056   0.794118  0.72973  0.760563

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.826825
```

对于采用 `Age` 列按头衔分类后的中位数填补的策略：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score
Values  0.810056   0.794118  0.72973  0.760563

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.855079
```

从以上评估结果上看，两种不同的 `Age` 填补策略在单次评估中得到了相同的准确度、精确度、召回率和F1分数。而且混淆矩阵也完全一致，这表明两种策略在预测真正例、假正例、真负例和假负例的数量上没有差异。这表明在这次测试集上，两种填补策略对模型性能的影响相同。不过，我们也发现，使用 `Age` 列按头衔分类后的中位数填补的策略在5折交叉验证的平均准确度上高于使用 `Age` 列整体中位数填补的策略（0.855079 vs. 0.826825）。这表明虽然在单个测试集上两种策略的性能相同，但在更广泛的数据上考虑，按头衔分类填补 `Age` 的策略可能更为稳健，能够提供更高的平均准确度。它们在交叉验证的准确度上有所不同。因此，后面我们将考虑采用**按头衔分类后的中位数填补 `Age` 策略**，该种策略可能对未见数据具有更好的泛化能力。


### 第二次尝试（`Age` 特征标准化/归一化）

考虑到逻辑回归会受到特征尺度的影响（在逻辑回归的情况下，模型是基于数据的线性组合），因此，现在我们尝试将 `Age` 特征标准化/归一化，然后评估模型效果。值得注意的是，标准化/归一化方法很多，比如常用的 Z 得分标准化，最小-最大归一化等。数据的不同分布将影响我们选择不同标准化/归一化的方法。比如，如果数据接近正态分布， Z 得分标准化可能是一个更好的选择。而如果数据的范围更为重要，而数据分布不是正太分布，可能最小-最大更为合适。我们先来看看经过缺失值填补后的 `Age` 分布情况 (这部分代码可以参考EDA分析中[单因素分析](#单因素分析)中的年龄可视化示例代码)。

![](/assets/images/ml/titanic_distribution_age_fill_title_group.png)

从左图可以发现，`Age` 数据似乎不是严格的正态分布，但也没有特别极端的偏斜。但是，右图中显示，存在少部分异常值。为此，我们需要一种更为稳健的标准化/归一化方法，以确保这些异常值不会对整体标准化结果产生过大影响。在此，我们计划采用 `RobustScaler` 来对 `Age` 进行处理。[`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) 通过去除中位数并按四分位范围（IQR）缩放数据，可以降低异常值的影响力。

现在，我们需要回到 `data_preprocessing.py` 文件，添加标准化/归一化的代码。显然，我们需要在已经填补上缺失值的数据上进行相关操作，那么，我们应该修改 `AdvancedDataProcessor` 类中的 `age_preprocess` 方法就可以了，如下：

```python
from sklearn.preprocessing import LabelEncoder, RobustScaler


# 其他代码保持不变

class AdvancedDataProcessor(DataProcessor):

    # 其他代码保持不变
    def age_preprocess(self):
        assert self.data is not None, "Data is not set."

        self.fill_age_by_title_group()

        # 标准化
        scaler = RobustScaler()
        self.data["Age"] = scaler.fit_transform(self.data["Age"])

    # 其他代码保持不变
```

然后运行 `main.py`，评估结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score
Values  0.810056   0.794118  0.72973  0.760563

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.855079
```

:rofl: 又没有变化。我们还是得考虑**模型的鲁棒性**，**数据本身的特性**， **评估指标的选择**等方面对训练效果的影响。逻辑回归模型可能对 `Age` 特征的尺度不太敏感，尤其是当其他特征对预测结果有更强烈影响时（思考下如何确认？）。因此，即使进行了标准化，模型的表现也没有显著变化。从年龄的分布上看（查看该节前面的 Distribution of Age 图）， `Age` 数据在未标准化时好像已经相对集中（即使存在少数异常值），标准化可能不会对数据的相对关系产生重大影响，因此模型性能保持不变。此外，当下使用的评估指标（准确率、精确率、召回率、F1分数和混淆矩阵）可能没有捕捉到标准化带来的细微变化。我们可以再添加分类问题中常用的其他指标，比如AUC指标。关于前面提到的**特征的影响程度**问题，基于 EDA 分析，我们相信，`Age` 特征应该是对生产率有明显影响的，因此，从这个角度上，我们暂时不打算放弃`Age` 特征。

现在，我们试着增加AUC指标。为此，我们只需要在 `model.py` 中的 `ModelEvaluator` 类添加相关代码就成，如下：

```python
# 其他保持不变
import matplotlib.pyplot as plt
from sklearn.metrics import (
    # 其他保持不变
    roc_auc_score,
    roc_curve,
    auc,
)


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, cv=5):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]  # 获取正类的概率

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, average="binary"),
            "Recall": recall_score(self.y_test, y_pred, average="binary"),
            "F1 Score": f1_score(self.y_test, y_pred, average="binary"),
            "ROC AUC": roc_auc_score(self.y_test, y_proba),  # 计算ROC AUC
        }

        # 打印评估指标
        print("Evaluation Metrics:")
        print(pd.DataFrame([metrics], index=["Values"]))

        # 打印混淆矩阵
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(
            pd.DataFrame(
                conf_matrix,
                columns=["Predicted Negative", "Predicted Positive"],
                index=["Actual Negative", "Actual Positive"],
            )
        )

        # 交叉验证
        if cv > 1:
            cross_val_accuracy = np.mean(
                cross_val_score(
                    self.model, self.X_test, self.y_test, cv=cv, scoring="accuracy"
                )
            )
            print(f"\nCross-validated Accuracy ({cv}-fold): {cross_val_accuracy:.6f}")

        # 绘制ROC曲线
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
        plt.title("Receiver Operating Characteristic") # 建议修改该title，使图片信息更直观
        plt.legend(loc="lower right")
        plt.savefig("fig/ROC.png", bbox_inches="tight")
        # plt.show()

        return metrics

# 其他保持不变
```

重新运行 `main.py`，不对 `Age` 进行标准化的结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.810056   0.794118  0.72973  0.760563  0.882239

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.855079
```
![](/assets/images/ml/titanic_ROC_age_no_scaling.png)

标准化后的结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.810056   0.794118  0.72973  0.760563  0.882239

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.855079
```
![](/assets/images/ml/titanic_ROC_age_scaling.png)

好吧，从结果上看，我们只能得出，在逻辑回归模型下，是否对 `Age` 进行标准化，暂时并不能对其训练效果产生明显好的影响。但是，好像也没什么坏处。那么，考虑到后期，我们可能还会选择其他分类模型，暂时保留采用 `RobustScaler` 的方式对 `Age` 进行的标准化。

### 第三次尝试（考虑 `SibSp` 和 `Parch` 特征)

基于 EDA 分析，不同家庭成员数量似乎对生存率存在影响，因此，这里我们计划进一步将该因素融入到上面的模型中。先来看看分别考虑`SibSp` 和 `Parch` 特征会不会对模型训练效果产生影响。由于这两特征没有缺失值，我们可以暂时直接加入到特征中。这样，我们只需要将其添加到 `main.py` 的 `main` 函数中的 `feature`，如下：

```python
def main():
    # 设置数据路径
    data_path = "./data/raw/train.csv"

    # 加载和预处理数据
    data = load_and_preprocess_data(data_path, AdvancedDataProcessor)

    # 模型训练与评估
    features = ["Pclass", "Sex", "Age", "SibSp"]  # 添加 "SibSp" 新特征
    target = "Survived"
    train_and_evaluate_model(data, features, target)
```

重新运行 `main.py`，可以得到如下结果：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229   0.808824  0.743243  0.774648  0.893115

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.860635
```

![](/assets/images/ml/titanic_ROC_sibsp.png)

同样，加入 `Parch` 的结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.798883   0.787879  0.702703  0.742857  0.883784

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  22                  52

Cross-validated Accuracy (5-fold): 0.855079
```

![](/assets/images/ml/titanic_ROC_parch.png)

同时考虑`SibSp` 和 `Parch` 特征的结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.894273

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.860635
```

![](/assets/images/ml/titanic_ROC_sibsp_parch.png)

分析这三组逻辑回归模型评估结果，我们可以发现：

1. **准确率（Accuracy）**:
   - 当考虑 `Pclass`, `Sex`, `Age`, `SibSp` 特征时，准确率为0.821229。
   - 当考虑 `Pclass`, `Sex`, `Age`, `Parch` 特征时，准确率为0.798883，比只考虑 `SibSp` 时低。
   - 当同时考虑 `Pclass`, `Sex`, `Age`, `SibSp`, `Parch` 特征时，准确率为0.826816，这是三个模型中最高的。
2. **精确度（Precision）、召回率（Recall）和 F1 分数**:
   - 精确度和召回率的最高值出现在同时考虑 `SibSp` 和 `Parch` 的情况下（精确度0.820896，召回率0.743243）。但是，召回率在同时考虑 `SibSp` 和 `Parch` 的情况下和只考虑 `SibSp` 特征的情况下是一样的，并且，对比只考虑 `Parch` 的情况下的召回率（三种情况下最低，且低于仅考虑 `Pclass`, `Sex`, `Age`时的评估结果，0.702703），这可能意味着在识别实际为正类的乘客方面，`Parch` 的添加对模型的影响有限。
   - F1 分数是精确度和召回率的调和，最高（0.780142）也是在同时考虑 `SibSp` 和 `Parch` 时，说明模型在这种情况下平衡了精确度和召回率。
3. **ROC AUC**:
   - ROC AUC最高（0.894273）也是在同时考虑 `SibSp` 和 `Parch` 时，表明该模型具有较好的区分正负样本的能力。
4. **混淆矩阵**:
   - 在考虑 `SibSp` 和 `Parch` 时，模型预测正类和负类的能力最强，即预测为正类（生存）和负类（未生存）的数量均最多。
5. **交叉验证准确率**:
   - 交叉验证准确率最高（0.860635，且高于仅考虑 `Pclass`, `Sex`, `Age` 时的结果）也是在考虑所有特征时，这表明该模型具有较好的泛化能力。

总体来说，无论是在单次评估还是交叉验证中，在考虑 `SibSp` 和 `Parch` 时，模型的表现最佳。这可能表明 `SibSp` 和 `Parch` 特征与目标变量（生存与否）之间存在一定的相关性（印证了EDA结论），且这两个特征一起使用时能提供更多关于乘客生存概率的信息。因此，后面，计划考虑将 `SibSp` 和 `Parch` 作为特征构建模型，以提高预测的准确性和模型的泛化能力。但是，考虑到 `Parch` 的添加可能对模型的影响有限，我们计划进一步处理特征。

由于 `SibSp` 和 `Parch` 特征都是表示家庭成员结构。因此，接下来，我们考虑下，是否将其组合成新的**家庭成员数量**特征，会对模型训练效果有所提升。由于我们需要构建新的特征，这就需要我们在 `data_preprocessing.py` 中添加相应代码。由于这块数据处理代码似乎并不影响前面的模型，且比较简单，因此，我们计划将其放在我们的基础模块中，即 `DataProcessor`，其他保持不变就可以了，如下：

```python
class DataProcessor:
    # 其他不变

    def preprocess(self):
        self.age_preprocess()
        self.sex_preprocess()
        self.family_size_preprocess()

        return self.data

    # 其他不变

    def family_size_preprocess():
        self.data["Family_Size"] = self.data["SibSp"] + self.data["Parch"] + 1

# 其他不变
```

现在回到 `main.py` 中，将 `Family_Size` 纳入到 `features` 变量中， 如下：


```python
# 其他不变
    features = ["Pclass", "Sex", "Age", "Family_Size"]
# 其他不变
```

重新运行 `main.py`，我们将得到如下结果：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.826816   0.820896  0.743243  0.780142  0.894015

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  19                  55

Cross-validated Accuracy (5-fold): 0.866190
```

![](/assets/images/ml/titanic_ROC_family_size.png)

有趣了，对比只考虑 `Pclass`, `Sex`, `Age`, `SibSp`和`Parch` 特征时的结果，我们可以发现：

1. **准确率（Accuracy）**：
   - 在添加 `Family_Size` 后，准确率保持不变（0.826816），这表明添加 `Family_Size` 并没有改变模型在整体测试集上的预测准确性。
2. **精确度（Precision）、召回率（Recall）和 F1 分数**：
   - 这些指标同样保持不变，这意味着模型对正类和负类的预测能力并没有因为添加 `Family_Size` 而受到显著影响。
3. **ROC AUC**：
   - ROC AUC略有下降（从0.894273降至0.894015），但变化非常微小，可能在实际应用中并不显著。
4. **混淆矩阵**：
   - 混淆矩阵结果与之前相同，进一步确认了模型对特定类别的预测能力未受到新增特征的影响。
5. **交叉验证准确率（Cross-validated Accuracy）**：
   - 交叉验证的准确率从0.860635提高到0.866190，这肯定是一个积极的信号，表明在不同子集的数据上模型的泛化能力有所提升。

因此，尽管在单次测试集评估中，添加 `Family_Size` 并没有显著改变模型的性能，但在交叉验证中观察到一定程度的准确率提升，这表明 `Family_Size` 可能增强了模型对不同数据分布的适应性和泛化能力。精确度、召回率和F1分数的稳定性表明，`Family_Size` 特征的加入并未对模型预测正负类产生不利影响，而且在一定程度上有助于提高模型的稳健性。所以大致可以得出，`Family_Size` 是一个有价值的特征，可以保留在模型中以期进一步提升模型的准确性和泛化能力。

还记得前面 EDA 分析中发现，大多数乘客没有兄弟姐妹、配偶、父母或孩子同行吗？这个特征可能会导致数据倾斜，从而对模型产生不成比例的影响，为了缓解这种影响，我们可能需要进一步处理 `Family_Size` 这个新特征。下面提供了几种策略:

1. **二值化处理**：将 `Family_Size` 转换为二元特征，例如，将独自一人的乘客标记为0，有家庭成员的乘客标记为1。这样的处理可以突出是否有家庭成员这一信息，而不是家庭成员的具体数量。
2. **分段（分箱）**：将 `Family_Size` 进行分段（或称为分箱），例如，将家庭大小划分为"无家庭成员"、"小家庭"和"大家庭"等几个类别。这样可以在保留一定家庭大小信息的同时，减少异常值的影响。   
3. **归一化或标准化**：虽然 `Family_Size` 已经是数值型特征，但如果其分布非常偏斜（的确），可以考虑对其进行归一化或标准化处理，使其在更合适的数值范围内，这可能对于基于梯度的模型特别有用。
4. **考虑与其他特征的交互**：可以进一步探索 `Family_Size` 与其他特征的交互，例如，家庭大小可能与船舱等级（`Pclass`）或票价（`Fare`）有关联。这种交互特征可能会揭示更多的信息。
5. **特征选择**：如果通过模型评估发现 `Family_Size` 对模型性能的贡献有限，可以考虑不将其包括在最终模型中，或者使用特征选择算法来确定其重要性。

根据以上策略，我们先来完成前三种，针对不同策略，建立新变量，如对二值化，我们构建一个 `Is_Alone` 的新变量；对分段，我们构建一个 `Family_Size_Group`，对于标准化，我们构建一个 `Family_Size_Scaling`。为此，我们需要回到刚才在 `DataProcessor` 中新建立的 `family_size_preprocess` 方法，对其修改，示例代码如下：

```python
class DataProcessor:
    # 其他代码保持不变

    def family_size_preprocess(self):
        self.data["Family_Size"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data['Is_Alone'] = (self.data['FamilySize'] == 1).astype(int)
        self.data['Family_Size_Group'] = pd.cut(self.data['FamilySize'], bins=[0, 1, 4, 11], labels=['Solo', 'SmallFamily', 'LargeFamily'])

# 其他代码保持不变
```

同上，我们在标准化/归一化的过程中，需要确认选择何种标准化/归一化方法。这就需要我们查看下 `Family_Size` 特征的分布情况，如下：

![](/assets/images/ml/titanic_distribution_family_size.png)

显然，`RobustScaler` 可能是一个更为明智的选择。将标准化/归一化的代码添加到上面的函数中，如下：

```python
class DataProcessor:
    # 其他代码保持不变

    def family_size_preprocess(self):
        self.data["Family_Size"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data["Is_Alone"] = (self.data["Family_Size"] == 1).astype(
            int
        )  # 二值化处理
        
        scaler = RobustScaler()
        self.data["Family_Size_Scaling"] = scaler.fit_transform(
            self.data[["Family_Size"]]
        )  # 标准化/归一化处理

        self.data["Family_Size_Group"] = pd.cut(
            self.data["Family_Size"],
            bins=[0, 1, 4, 11],
            labels=["Solo", "SmallFamily", "LargeFamily"],
        )  # 分段（分箱）处理
        self.data = pd.get_dummies(self.data, columns=["Family_Size_Group"])
```

注意，在上面的代码中，我们使用了 `One-Hot` 的方式将分段处理后的 `Family_Size_Group` 进行了转换。`get_dummies` 会创建新的列来表示 `Family_Size_Group` 中的每个类别，列名为 `Family_Size_Group_<类别名>`。如 `Family_Size_Group_Solo`, `Family_Size_Group_SmallFamily`, `Family_Size_Group_LargeFamily`。这些列可以直接用于训练逻辑回归模型。

我们回到 `main.py` 中，其他代码可以保持不变，我们只需要将新构建的特征添加到 `features` 中就可以了，考虑不同处理策略下的结果如下：

**二值化处理**后结果：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.798883   0.787879  0.702703  0.742857  0.883269

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  91                  14
Actual Positive                  22                  52

Cross-validated Accuracy (5-fold): 0.843810
```

**分段**处理后结果：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.810056    0.80303  0.716216  0.757143  0.895302

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  92                  13
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.838254
```

**标准化/归一化**处理后结果：
```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.815642   0.815385  0.716216   0.76259  0.891055

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  93                  12
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.855079
```

结合原始处理（即直接使用 `Family_Size` 特征，而不处理），对比分析这些模型评估结果，我们可以观察到对 `Family_Size` 特征采用不同处理策略后模型性能的变化：

1. **不处理`Family_Size`**：准确率最高，达到0.826816，且其他评估指标如精确度、召回率、F1分数和ROC AUC也相对较高。混淆矩阵显示预测正确的数量最多，且交叉验证准确率（0.866190）同样是最高的。
2. **二值化处理后**：准确率和其他指标普遍下降，准确率降到0.798883，交叉验证准确率下降到0.843810，这表明二值化处理可能损失了部分重要信息，导致模型性能下降。
3. **分段处理后**：模型的准确率提高到0.810056，但相比不处理 `Family_Size` 的结果仍有所下降。尽管ROC AUC稍有提高，但交叉验证准确率降低到0.838254，表明模型的泛化能力减弱。
4. **标准化/归一化处理后**：模型准确率为0.815642，虽然高于二值化和分段处理，但低于不进行任何处理的情况。交叉验证准确率也有所下降，表明在这种情况下标准化/归一化处理并没有带来预期的性能提升。

整体来说，在这种情况下，**不对 `Family_Size` 进行任何处理似乎是最佳选择**，因为它为模型提供了最高的准确率和最好的泛化能力。二值化和分段处理虽然简化了特征，但同时也可能导致信息损失，影响模型性能。标准化/归一化处理在这里并没有显著提高模型性能，可能是因为 `Family_Size` 的原始分布已经足够适合模型使用。

### 第四次尝试（考虑 `Ticket` 特征）

从 EDA 分析可以看出，票号前缀与乘客的生存率之间存在一定的关联。票号前缀大致可以分为高频前缀（如 None, PC, CA）和低频前缀。高频前缀（如 None, PC, CA）可能代表了更常见的票务类别，而与之关联的生存率可能更具有一般性的指示意义；低频前缀的生存率可能受到随机波动的影响较大，这给我们如何处理票号前缀提出了一定的挑战。

我们计划先提取出票号前缀，并采用One-Hot的方式对其进行编码（主要是考虑到，票号前缀并没有顺序性）。显然，我们需要在 `data_preprocessing.py` 中，合适的位置添加上相关的处理。同 `Family_Size` 的处理，可以直接在 `DataProcessor` 类中添加相关方法，如下：

```python
class DataProcessor:
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        self.ticket_preprocess()
        return self.data

    def ticket_preprocess(self):
        self.data["Ticket_Prefix"] = self.data["Ticket"].apply(
            lambda x: (
                "".join(filter(str.isalpha, x.split(" ")[0]))
                if not x.isdigit()
                else "None"
            )
        )
        self.data = pd.get_dummies(self.data, columns=["Ticket_Prefix"])
```

由于对 `Ticket_Prefix` 进行了One-Hot编码，因此，增加了很多特征。我们先将新生成了所有特征都纳入到前面的模型中，并评估其对模型的影响。为了考虑所有的新变量，我们对 `main.py` 进行适当修改，如下：

```python
def main():
    # 设置数据路径
    data_path = "./data/raw/train.csv"

    # 加载和预处理数据
    data = load_and_preprocess_data(data_path, AdvancedDataProcessor)

    # 模型训练与评估
    features = [
        "Pclass",
        "Sex",
        "Age",
        "Family_Size",
    ]
    ticket_prefix_features = [col for col in data.columns if "Ticket_Prefix_" in col]  # 添加所有新的ticket_prefix_

    all_features = features + ticket_prefix_features
    target = "Survived"

    train_and_evaluate_model(data, all_features, target)
```

可能需要解释的是，由于 One-Hot 对 `Ticket_Prefix` 编码产生了较多的新变量，但考虑到其对新变量的命名方式有一定的共同特征（`Ticket_Prefix_<类别名>`），因此，在此，我们先用列表生成式，提取出所有的新变量，然后与之前的变量进行组合，形成新的所有的变量（`all_feature`）。

重新运行 `main.py`， 评估结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision    Recall  F1 Score   ROC AUC
Values  0.821229   0.828125  0.716216  0.768116  0.884427

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  94                  11
Actual Positive                  21                  53

Cross-validated Accuracy (5-fold): 0.843810
```

与仅考虑 `Pclass`，`Sex`，`Age`，`Family_Size` 四个特征的情况对比，我们可以发现：
1. **准确率（Accuracy）**：考虑 `ticket_prefix_` 特征后，模型准确率略微下降，从0.826816降到了0.821229。这表明添加这些新特征可能引入了一些噪声，对模型的整体预测准确性产生了轻微的负面影响。
2. **精确度（Precision）和召回率（Recall）**：精确度有所提高（从0.820896增加到0.828125），这表明在考虑 `ticket_prefix_` 特征后，模型在预测乘客生存状态为正类时更加准确。但召回率下降（从0.743243降低到0.716216），意味着模型可能错过了更多实际为正类的预测。
3. **F1分数（F1 Score）**：F1分数略有下降，从0.780142降到了0.768116，显示在精确度的提高和召回率的下降之间，模型失去了一些平衡。
4. **ROC AUC**：ROC AUC也有所下降（从0.894015降至0.884427），表明考虑 `ticket_prefix_` 特征后，模型区分正负类的能力有所减弱。
5. **混淆矩阵（Confusion Matrix）**和**交叉验证准确率**：考虑 `ticket_prefix_` 特征后，混淆矩阵显示虽然模型正确预测正类的数量减少了，但是正类的误判数量也有所减少。交叉验证准确率下降（从0.866190降至0.843810），进一步证实了在包含 `ticket_prefix_` 特征后模型泛化能力的轻微下降。

所以说，考虑 `ticket_prefix_` 特征可能为模型提供了一些额外信息，但同时也可能引入了一些不太相关的噪声，导致模型在某些方面的性能略有下降。此外，添加大量新特征后，可能也会导致模型变得更复杂，增加了过拟合的风险，同时可能会掩盖一些更重要特征的效用。这也意味着，我们需要进一步处理 `Ticket_Prefix`。


下面，我们计划首先从降维的角度思考如何处理 `Ticket_Prefix`。降维可以减少特征空间的维度，同时尽量保留原始数据中的重要信息。下面列出了部分常用的降维方法：

1. **主成分分析（PCA）**: PCA 是一种非常流行的降维技术，可以将特征转换到一个新的坐标系统中，并按照方差大小排序，保留最重要的几个主成分。对于One-Hot编码后的特征，PCA可以帮助识别哪些变量捕获了大部分信息。
2. **截断奇异值分解（Truncated SVD）**: 与PCA类似，截断SVD适用于稀疏数据（例如，One-Hot编码后的数据）。它可以减少特征的维度，同时保留数据的关键信息。
3. **线性判别分析（LDA）**: LDA是一种监督学习的降维技术，旨在找到一个能够最大化类别间分离的特征子空间。特别是在分类项目中，LDA可以帮助提升模型的分类能力。
4. **t-SNE 或 UMAP**: t-SNE（t-distributed Stochastic Neighbor Embedding）和UMAP（Uniform Manifold Approximation and Projection）是两种流行的非线性降维技术，可以帮助揭示高维数据的内在结构。这些方法尤其擅长于保留局部邻域结构，因此它们在可视化聚类或组间差异方面通常具有出色的表现。t-SNE和UMAP通常用于探索性数据分析，以帮助理解数据集中可能存在的模式或聚类。虽然它们主要用于可视化，但在某些情况下，降维后的数据也可以用于训练模型，特别是在原始数据维度非常高时。不过，需要注意的是，这两种方法可能会增强数据中的噪声，所以在解释降维结果时应当谨慎。
5. **自编码器（Autoencoders）**: 自编码器是一种基于神经网络的降维技术，特别适合于非线性降维。通过训练一个将输入数据编码成低维表示，然后再解码回原始空间的网络，自编码器可以学习到数据的有效低维表示。
6. **特征选择**: 除了降维，还可以考虑特征选择方法来减少特征数量。基于树的方法（例如随机森林或XGBoost）可以提供特征重要性评分，帮助我们识别并选择最重要的特征。

降维是一个需要实验和评估的过程。我们可能需要尝试不同的降维方法和参数设置，然后根据模型的性能和复杂度来选择最适合咱们数据的方法。

考虑到当前项目的特征，我们暂时选择前两种降维技术，对 `Ticket_Prefix` One-Hot编码后的数据进行处理，并评估其对模型的影响。从代码组织上，我们计划将新的数据处理方法放在 `data_preprocessing.py` 中的 `AdvancedDataProcessor` 类中。之所以这么处理，主要是从模块化的角度考虑。这些降维处理函数是对 `ticket_preprocess` 方法的进一步扩展，而 `ticket_preprocess` 已经定义在 `DataProcessor` 类中。通过在 `AdvancedDataProcessor` 类中添加这些方法，我们可以保持基础的数据处理流程在 `DataProcessor` 类中，同时将更高级或特定的处理流程放在继承自它的 `AdvancedDataProcessor` 类中。这样的设计不仅保持了代码的组织性和可读性，还提供了灵活性，允许我们在不同的处理级别上扩展或修改数据处理流程，而不会影响到基础类的结构。因此，对 `AdvancedDataProcessor` 类进行扩展的示例代码如下：

```python
# 其他代码保持不变
from sklearn.decomposition import PCA


# 其他代码保持不变

class AdvancedDataProcessor(DataProcessor):
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        self.ticket_preprocess_with_pca()
        return self.data

    def ticket_preprocess_with_pca(self):
        self.ticket_preprocess()  # 先执行ticket_preprocess函数
        pca = PCA(n_components=0.95)  # 设置方差阈值，保留95%的方差
        ticket_prefix_features = [
            col for col in self.data.columns if "Ticket_Prefix_" in col
        ]
        pca_transformed = pca.fit_transform(self.data[ticket_prefix_features])

        # 确定PCA产生的特征数量
        n_components = pca.n_components_

        # 为PCA生成的特征创建新的列名
        new_feature_names = [f"PCA_Ticket_{i+1}" for i in range(n_components)]

        # 删除原有的ticket_prefix特征，避免冗余
        self.data.drop(columns=ticket_prefix_features, inplace=True)

        # 将PCA转换后的数据添加到data中
        for i, feature_name in enumerate(new_feature_names):
            self.data[feature_name] = pca_transformed[:, i]
```

以上 `ticket_preprocess_with_pca` 方法中，我们除了对 `Ticket_Prefix` 进行 PCA 的常规操作外，还对新特征进行了重新命名。主要是为了方便在 `main` 中添加相关新特征。现在回到 `main.py` 中，并将新特征纳入到模型训练中。其实我们只需要修改 `ticket_prefix_features = [col for col in data.columns if "Ticket_Prefix_" in col]` 这行代码就成。示例代码如下：

```python
def main():
    # 其他代码保持不变

    # 添加所有新的ticket_prefix_
    ticket_prefix_features = [col for col in data.columns if "PCA_Ticket_" in col]

    # 其他代码保持不变
```

运行新的 `main.py`，结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score  ROC AUC
Values  0.837989   0.857143  0.72973  0.788321  0.88713

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  96                   9
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.849365
```

对比前面的结果，我们可以发现：

1. **使用PCA降维的Ticket_Prefix特征**: 采用 PCA 对 `Ticket_Prefix` 的 One-hot 编码后的数据进行降维处理后，模型的准确率提升到了0.837989，ROC AUC为0.88713，交叉验证准确率为0.849365。这表明通过PCA降维能有效提升模型性能，可能是因为降维有助于减少噪声和冗余信息，使模型能更关注于重要的特征。

2. **不考虑Ticket特征**: 仅考虑 `Pclass`, `Sex`, `Age`, `Family_Size`时，模型的准确率为0.826816，ROC AUC为0.894015，交叉验证准确率为0.866190。这组结果显示了一个较强的基线，表明即使不考虑 `Ticket_Prefix` 特征，模型也能表现良好。

因此，使用 PCA 降维处理 `Ticket_Prefix` 特征后，模型在准确率和交叉验证准确率上都有所提升，说明 PCA 的确有助于提取有效的特征，增强模型的预测能力。但与不考虑 `Ticket` 特征相比，虽然使用 PCA 降维的模型准确率和 ROC AUC 略有提升，但交叉验证准确率略有下降。这可能意味着 `Ticket_Prefix` 特征提供了一些有价值的信息，但这些信息的贡献相对有限。

相较于 PCA 来说，运用 SVD 技术对特征进行降维稍微会复杂些。复杂的点主要是在确认 `n_components` 上。SVD 并没有类似于 PCA 方差阈值的参数可以设置。在降维过程中，需要我们根据经验来判断 `n_components` 值的合理性。为了更为直观，我们先探索性分析，不同 `n_components` 下的累计方差以及其对应的新特征输入到逻辑回归模型中的评估指标变化情况，如下（这部分代码请参考项目文件中的 `notebook/feature_ead.ipynb` 文件）：

![](/assets/images/ml/titanic_svd_num_comp.png)

可以看出，当 `n_components=15` 时，累计方差达到95%。这意味着保留前 15 个 SVD 组件就可以解释原始数据约 95% 的方差，这通常是一个选择组件数量的好标准，因为它确保了大部分信息被保留，同时也减少了特征数量，降低了模型的复杂度。

查看模型评估结果（右图），我们可以看到，当 `n_components=15` 时，模型的准确度、精确度、召回率、F1 得分以及 ROC AUC 都达到了较高的水平。尽管在 `n_components=19` 时，准确度和F1得分稍微有所提高，但考虑到累计方差和模型复杂性，`n_components=15` 可能是一个更加均衡和合理的选择。

现在，我们继续完善 `AdvancedDataProcessor` 如下：

```python
class AdvancedDataProcessor(DataProcessor):
    # 其他代码保持不变

    def preprocess(self):
        # 其他代码保持不变
        # self.ticket_preprocess_with_pca()
        self.ticket_preprocess_with_svd()
        return self.data

     def ticket_preprocess_with_svd(self):
        self.ticket_preprocess() 
        svd = TruncatedSVD(n_components=15)
        ticket_prefix_features = [
            col for col in self.data.columns if "Ticket_Prefix_" in col
        ]

        # 应用Truncated SVD变换
        svd_transformed = svd.fit_transform(self.data[ticket_prefix_features])

        # 创建新的特征名称
        new_feature_names = [f"SVD_Ticket_{i+1}" for i in range(15)]

        # 删除原有的ticket_prefix特征
        self.data.drop(columns=ticket_prefix_features, inplace=True)

        # 将SVD转换后的数据添加到data中
        for i, feature_name in enumerate(new_feature_names):
            self.data[feature_name] = svd_transformed[:, i]
```

同理，我们需要修改 `main` 函数中的 `ticket_prefix_features = [col for col in data.columns if "Ticket_Prefix_" in col]` 这行代码就成，如下：

```python
def main():
    # 其他代码保持不变

    # 添加所有新的ticket_prefix_
    ticket_prefix_features = [col for col in data.columns if "SVD_Ticket_" in col]

    # 其他代码保持不变
```

运行后的结果如下：

```plaintext
Evaluation Metrics:
        Accuracy  Precision   Recall  F1 Score   ROC AUC
Values  0.832402    0.84375  0.72973  0.782609  0.886486

Confusion Matrix:
                 Predicted Negative  Predicted Positive
Actual Negative                  95                  10
Actual Positive                  20                  54

Cross-validated Accuracy (5-fold): 0.849365
```

与前面的结果进行对比，我们可以发现：

1. **SVD降维后的模型**: 准确率为0.832402，精确度为0.84375，召回率为0.72973，F1得分为0.782609，ROC AUC为0.886486。这表明模型在预测正类时具有较好的准确性，但在识别所有正类（召回率）方面略显不足。
2. **PCA降维后的模型**: 准确率为0.837989，精确度为0.857143，召回率为0.72973，F1得分为0.788321，ROC AUC为0.88713。这组结果在准确率、精确度和F1得分上均略优于SVD降维后的模型，说明PCA降维可能更适合这个数据集。
3. **不降维的模型**：准确率为0.821229，精确度为0.828125，召回率为0.716216，F1得分为0.768116，ROC AUC为0.884427。这组结果均低于采用两种降维后的模型指标，说明如果要考虑 `ticket` 特征，降维处理是一个较好的选择。
4. **不考虑 `ticket` 特征的模型**: 准确率为0.826816，精确度为0.820896，召回率为0.743243，F1得分为0.780142，ROC AUC为0.894015。尽管该模型的准确率和精确度稍低，但它在召回率和ROC AUC上表现更佳，显示了更好的综合性能和对正类的识别能力。

整体来说，PCA 和 SVD 都是有效的降维方法，但在这个特定的数据集上，PCA 降维后的模型在多数评估指标上略胜一筹，可能是因为 PCA 更适合捕捉这些数据中的关键变异。虽然降维方法有助于提高模型的准确率和精确度，但不考虑 `Ticket` 特征的模型在召回率和 ROC AUC 上的表现更优。这可能意味着 `Ticket` 特征并不是非常关键的特征，或者这些特征的信息在降维过程中部分丢失。在交叉验证准确率方面，所有模型都相对一致，但不考虑 `Ticket` 特征的模型略高，这表明其泛化能力可能更强。
