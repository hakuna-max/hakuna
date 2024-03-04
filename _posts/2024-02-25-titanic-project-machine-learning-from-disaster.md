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

从直方图可以看出

- 年龄分布：约呈右偏态，较多的乘客集中在年轻的年龄段。大多数乘客的年龄在20到40岁之间。
- 票价分布：呈现出极度的右偏，表明大多数乘客支付的票价较低。

该分布特征也表明，我们需要考虑特征的极端值对模型训练和性能的影响，导致预测偏向数据的主体部分，而忽视尾部的重要信息。

针对分布不均匀的数据集，在后期数据处理时，可能会考虑采用不同策略对其进行处理，比如**数据转换**，**分箱（Binning）**，**剔除极端值**，甚至考虑使用对偏态分布不敏感的**非线性模型**（如，随机森林，梯度提升树等）。具体采用何种方法将取决与数据的具体情况和模型的需求。一般建议在数据转换或处理前后，都进行可视化，以此评估转换或处理的效果。在应用任何转换或者处理方法之前，最好在原始数据上训练模型，以便有一个基准性能进行比较。

对于 `SibSp` 和 `Parch`，实现其不同值的乘客数量的条形图的代码如下：

```python
# 绘制 SibSp 的分布
plt.figure(figsize=(12, 6))
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

- **`SibSp` 分布**：大多数乘客没有兄弟姐妹或配偶同行（ `SibSp` 为0）。有一些乘客有一个兄弟姐妹或配偶（ `SibSp` 为1），而有两个或更多兄弟姐妹或配偶同行的乘客数量较少。

- **`Parch` 分布**：与 `SibSp` 类似，大多数乘客没有携带父母或孩子（ `Parch` 为0）。少数乘客有一到三个父母或孩子同行，而更多的父母或孩子同行的情况则更为罕见。

针对以上结果，我们在后期分析中，可能需要注意以下几点：

- **特征组合**：考虑将 `SibSp` 和 `Parch` 合并为一个新特征，如家庭成员总数，这可能有助于揭示家庭规模与生存率之间的关系。
- **模型选择**：选择对分类数据敏感度低的模型，如随机森林或梯度提升树，可能在处理这类特征时表现更好。
- **数据预处理**：对于 `SibSp` 和 `Parch` 值较大的少数样本，可以考虑进行分组或其他形式的处理，以防止它们对模型产生不成比例的影响。


针对类别型变量（ `Pclass`, `Sex`, `Ticket`, `Cabin`, `Embarked`），可以计算每个类别的乘客数量，并绘制条形图。该分析的代码如下：

```python
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Pclass 分布
sns.countplot(x='Pclass', data=train_data, ax=ax[0, 0])
ax[0, 0].set_title('Distribution of Pclass')
ax[0, 0].set_ylabel('Number of Passengers')
ax[0, 0].set_xlabel('Pclass')

# Sex 分布
sns.countplot(x='Sex', data=train_data, ax=ax[0, 1])
ax[0, 1].set_title('Distribution of Sex')
ax[0, 1].set_ylabel('Number of Passengers')
ax[0, 1].set_xlabel('Sex')

# Embarked 分布
sns.countplot(x='Embarked', data=train_data, ax=ax[1, 0])
ax[1, 0].set_title('Distribution of Embarked')
ax[1, 0].set_ylabel('Number of Passengers')
ax[1, 0].set_xlabel('Embarked')

# Cabin 缺失值情况
# 计算缺失值比例
cabin_null_percentage = train_data['Cabin'].isnull().sum() / len(train_data) * 100
cabin_not_null_percentage = 100 - cabin_null_percentage

# 绘制 Cabin 缺失值情况的条形图
ax[1, 1].bar(['Missing', 'Present'], [cabin_null_percentage, cabin_not_null_percentage])
ax[1, 1].set_title('Cabin Missing Value Percentage')
ax[1, 1].set_ylabel('Percentage')
ax[1, 1].set_xlabel('Cabin Value Status')

plt.tight_layout()
plt.show()

# 输出 Cabin 缺失值的具体比例
print(f"Carbin null percentage (%): {cabin_null_percentage:.2f}")
```

我们首先来看看各个特征的分布情况：

![](/assets/images/ml/titianic_factor_cate_dist.png)

可以观察到以下几点关于类别型变量的分布：

- **`Pclass`（船舱等级）**：不同等级的船舱乘客数量分布显示，第三等舱乘客最多，其次是第一等舱和第二等舱。
- **`Sex`（性别）**：男性乘客数量多于女性乘客。
- **`Embarked`（登船港口）**：大多数乘客从S港口登船，其次是C港口，最少的是Q港口。

由于**`Cabin`（船舱号）**的数据确实情况严重，我们可以重点关注下：

- `Cabin` 特征有约77.1%的缺失值，这是一个非常高的比例，对于这种情况，我们需要决定如何处理这些大量的缺失值。
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

在数据分析和机器学习的上下文中，从名字中提取这些头衔可以作为一个有用的特征，因为它们可能与乘客的生存率相关。历史数据表明，妇女和儿童在灾难中的生存机会通常高于成年男性，因此这些头衔可能帮助我们预测乘客的生存概率。通过将这些头衔作为模型的一个特征，我们可以更准确地预测乘客的生存情况。这种类型的特征工程是在构建预测模型时常见且有价值的步骤。

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

# 设置画布和子图
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# 绘制头衔分布的条形图
sns.barplot(x=title_counts.index, y=title_counts.values, ax=axes[0])
axes[0].set_title('Distribution of Titles')
axes[0].set_ylabel('Number of Passengers')
axes[0].set_xlabel('Title')
axes[0].tick_params(axis='x', rotation=90)

# 绘制头衔和生存率的关系条形图
sns.barplot(x=title_survival_rates.index, y=title_survival_rates.values, ax=axes[1])
axes[1].set_title('Survival Rate by Title')
axes[1].set_ylabel('Survival Rate')
axes[1].set_xlabel('Title')
axes[1].tick_params(axis='x', rotation=90)

# 绘制分组后的头衔和生存率的关系条形图
sns.barplot(x=title_grouped_survival_rates.index, y=title_grouped_survival_rates.values, ax=axes[2])
axes[2].set_title('Survival Rate by Grouped Title')
axes[2].set_ylabel('Survival Rate')
axes[2].set_xlabel('Grouped Title')
axes[2].tick_params(axis='x', rotation=90)

# 调整子图间距
plt.tight_layout()
plt.show()
```

其结果如下：

![](/assets/images/ml/titianic_factor_title_dist.png)

借助于以上分析结果，我们可以发现

- **Mr**：最常见的头衔，有517名乘客，表示已婚或成年男性。
- **Miss**：第二常见，有182名未婚女性。
- **Mrs**：有125名已婚女性。
- **Master**：有40名年幼的男孩。
- 其他头衔如 **Dr**、**Rev** 等出现的次数较少，表示这些头衔的乘客在样本中较为罕见。

从头衔与生存率（Title Survival Rates）的关系上，我们可以发现
- 一些罕见头衔（如 **the Countess**、**Mlle**、**Sir**、**Ms**、**Lady**、**Mme**）的生存率是100%，但这可能是由于样本量较小，不足以作出统计上的一般性结论。
- **Mrs**（已婚女性）和 **Miss**（未婚女性）的生存率较高，分别为79%和70%，这与“妇女和儿童优先”的救生原则相吻合。
- **Master**（年幼男孩）的生存率也相对较高，为57%。
- **Mr**（成年男性）的生存率最低，仅为16%，反映了成年男性在灾难中的生存几率较低。
- 有些头衔如 **Rev**（牧师）、**Don**、**Jonkheer**、**Capt**（船长）的生存率为0%，但这可能是由于样本量太小，不能确定这是否具有统计意义。

从分组后的头衔与生存率（Title Grouped Survival Rates）的关系上，我们可以进一步发现
- 将罕见头衔归类为 **Rare** 后，可以看到 **Mrs**、**Miss**、**Master** 的生存率仍然较高。
- **Rare** 类别的生存率为44%，高于 **Mr**，但由于包含多种不同的头衔，这个数字可能不够具体。
- **Mr** 的生存率依然是最低的，为16%。

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
plt.figure(figsize=(18, 6))
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

最后，我们再回过头来看看缺失值的情况。从上面的分析可以发现，`train_data` 中的 `Age`，`Cabin` 和 `Embarked` 三个特征存在缺失值。其中 `Cabin` 的缺失值占比极高（有687个缺失值，占总数据的约77.10%），可以遵循前面所述，采用直接删除该特征或转换为是否缺失的二元特征的方式可能更为合理。对于 `Age`，有177个缺失值，占总数据的约19.87%。这个比例相对较小，可以考虑使用统计方法（如中位数或根据其他特征分组的中位数）或模型预测方法来填充这些缺失值。对于 `Embarked`，仅有2个缺失值，占总数据的约0.22%。由于数量非常少，可以用出现最频繁的港口来填充这些缺失值，或者基于与Embarked最相关的特征（如 `Fare` 或 `Pclass`）来推断可能的登船港口。

对于有缺失值的特征的处理策略选择问题，主要还是得检查它们是否与其他特征相关，特别是与目标值的关系，从而判断缺失值出现的随机性或是否存在某种模式。如果存在某种模式，直接删除可能并不是一个明智的选择。考察不同特征的关系就落脚到多因素分析上面了，接下来我们开始多因素分析。

### 双因素或多因素分析

该部分分析的主要目的是分析特征之间的相关性，探索特征与目标变量（生存情况）之间的关系，帮助我们理解特征之间的相互作用以及它们是如何共同影响泰坦尼克号乘客生存率。可以采用相关系数矩阵、热力图、点图和箱型图等方式呈现分析结果。其实，在分析 `Name` 和 `Ticket` 时，我们已经分析了其与目标变量的关系。下面试着从如下几个方面做更为深入的分析。

1. **性别和船舱等级（Sex and Pclass）**：
   - 分析不同船舱等级中男性和女性的生存率。
   - 探讨船舱等级是否影响性别与生存率之间的关系。
2. **年龄、性别和生存率（Age, Sex, and Survived）**：
   - 分析不同性别和年龄组的生存率。
   - 研究儿童（如定义为16岁以下）与成人在不同性别下的生存率差异。
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
plt.figure(figsize=(10, 6))
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

