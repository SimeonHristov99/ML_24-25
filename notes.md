# Table of Contents

- [Table of Contents](#table-of-contents)
- [Week 01 - Numpy, Pandas, Matplotlib \& Seaborn](#week-01---numpy-pandas-matplotlib--seaborn)
  - [Python Packages](#python-packages)
    - [Introduction](#introduction)
    - [Importing means executing `main.py`](#importing-means-executing-mainpy)
  - [NumPy](#numpy)
    - [Context](#context)
    - [Problems](#problems)
    - [Solution](#solution)
    - [Benefits](#benefits)
    - [Usage](#usage)
    - [2D NumPy Arrays](#2d-numpy-arrays)
    - [Basic Statistics](#basic-statistics)
    - [Generate data](#generate-data)
  - [Matplotlib](#matplotlib)
    - [Line plot](#line-plot)
    - [Scatter plot](#scatter-plot)
    - [Drawing multiple plots on one figure](#drawing-multiple-plots-on-one-figure)
    - [The logarithmic scale](#the-logarithmic-scale)
    - [Histogram](#histogram)
      - [Introduction](#introduction-1)
      - [In `matplotlib`](#in-matplotlib)
      - [Use cases](#use-cases)
    - [Checkpoint](#checkpoint)
    - [Customization](#customization)
      - [Axis labels](#axis-labels)
      - [Title](#title)
      - [Ticks](#ticks)
      - [Adding more data](#adding-more-data)
      - [`plt.tight_layout()`](#plttight_layout)
        - [Problem](#problem)
        - [Solution](#solution-1)
  - [Seaborn](#seaborn)
  - [Pandas](#pandas)
    - [Introduction](#introduction-2)
    - [DataFrame from Dictionary](#dataframe-from-dictionary)
    - [DataFrame from CSV file](#dataframe-from-csv-file)
    - [Indexing and selecting data](#indexing-and-selecting-data)
      - [Column Access using `[]`](#column-access-using-)
      - [Row Access using `[]`](#row-access-using-)
      - [Row Access using `loc`](#row-access-using-loc)
      - [Checkpoint](#checkpoint-1)
      - [`iloc`](#iloc)
    - [Filtering pandas DataFrames](#filtering-pandas-dataframes)
      - [Single condition](#single-condition)
      - [Multiple conditions](#multiple-conditions)
      - [Filtering and selecting columns](#filtering-and-selecting-columns)
    - [Looping over dataframes](#looping-over-dataframes)
  - [Random numbers](#random-numbers)
    - [Context](#context-1)
    - [Random generators](#random-generators)
  - [A note on code formatting](#a-note-on-code-formatting)
- [Week 02 - Machine learning with scikit-learn](#week-02---machine-learning-with-scikit-learn)
  - [What is machine learning?](#what-is-machine-learning)
  - [The `scikit-learn` syntax](#the-scikit-learn-syntax)
  - [The classification challenge](#the-classification-challenge)
  - [Measuring model performance](#measuring-model-performance)
  - [Model complexity (overfitting and underfitting)](#model-complexity-overfitting-and-underfitting)
  - [Hyperparameter optimization (tuning) / Model complexity curve](#hyperparameter-optimization-tuning--model-complexity-curve)
- [Week 03 - Regression](#week-03---regression)
  - [Which are the two main jobs connected to machine learning?](#which-are-the-two-main-jobs-connected-to-machine-learning)
    - [Example](#example)
  - [Loading and exploring the data](#loading-and-exploring-the-data)
  - [Data Preparation](#data-preparation)
  - [Modelling](#modelling)
  - [Diving Deep (Regression mechanics)](#diving-deep-regression-mechanics)
  - [Model evaluation](#model-evaluation)
    - [Using a metric](#using-a-metric)
      - [Adjusted $R^2$ ($R\_{adj}^2$)](#adjusted-r2-r_adj2)
    - [Using a loss function](#using-a-loss-function)
- [Week 04 - Cross Validation, Regularized Regression, Classification Metrics](#week-04---cross-validation-regularized-regression-classification-metrics)
  - [Cross Validation](#cross-validation)
  - [Regularized Regression](#regularized-regression)
    - [Regularization](#regularization)
    - [Ridge Regression](#ridge-regression)
    - [Hyperparameters](#hyperparameters)
    - [Lasso Regression](#lasso-regression)
    - [Feature Importance](#feature-importance)
    - [Lasso Regression and Feature Importance](#lasso-regression-and-feature-importance)
  - [Classification Metrics](#classification-metrics)
    - [A problem with using `accuracy` always](#a-problem-with-using-accuracy-always)
    - [Confusion matrix in scikit-learn](#confusion-matrix-in-scikit-learn)
- [Week 05 - Logistic Regression, The ROC curve, Hyperparameter optimization/tuning](#week-05---logistic-regression-the-roc-curve-hyperparameter-optimizationtuning)
  - [Logistic Regression](#logistic-regression)
    - [Introduction](#introduction-3)
    - [In `scikit-learn`](#in-scikit-learn)
  - [The receiver operating characteristic curve (`ROC` curve)](#the-receiver-operating-characteristic-curve-roc-curve)
    - [In `scikit-learn`](#in-scikit-learn-1)
    - [The Area Under the Curve (`AUC`)](#the-area-under-the-curve-auc)
    - [In `scikit-learn`](#in-scikit-learn-2)
  - [Hyperparameter optimization/tuning](#hyperparameter-optimizationtuning)
    - [Introduction](#introduction-4)
    - [Grid search cross-validation](#grid-search-cross-validation)
    - [In `scikit-learn`](#in-scikit-learn-3)
    - [Randomized search cross-validation](#randomized-search-cross-validation)
      - [Benefits](#benefits-1)
    - [Evaluating on the test set](#evaluating-on-the-test-set)
- [Week 06 - Preprocessing and Pipelines](#week-06---preprocessing-and-pipelines)
  - [Dealing with categorical features](#dealing-with-categorical-features)
    - [Dropping one of the categories per feature](#dropping-one-of-the-categories-per-feature)
    - [In `scikit-learn` and `pandas`](#in-scikit-learn-and-pandas)
  - [EDA with categorical feature](#eda-with-categorical-feature)
  - [Handling missing data](#handling-missing-data)
    - [Removing missing values](#removing-missing-values)
    - [Imputing missing values](#imputing-missing-values)
    - [Using pipelines](#using-pipelines)
  - [Centering and scaling](#centering-and-scaling)
  - [Definitions](#definitions)
  - [Scaling in `scikit-learn`](#scaling-in-scikit-learn)
  - [How do we decide which model to try out in the first place?](#how-do-we-decide-which-model-to-try-out-in-the-first-place)
    - [The size of our dataset](#the-size-of-our-dataset)
    - [Interpretability](#interpretability)
    - [Flexibility](#flexibility)
    - [Train several models and evaluate performance out of the box (i.e. without hyperparameter tuning)](#train-several-models-and-evaluate-performance-out-of-the-box-ie-without-hyperparameter-tuning)
    - [Scale the data](#scale-the-data)

# Week 01 - Numpy, Pandas, Matplotlib & Seaborn

## Python Packages

### Introduction

You write all of your code to one and the same Python script.

<details>

<summary>What are the problems that arise from that?</summary>

- Huge code base: messy;
- Lots of code you won't use;
- Maintenance problems.

</details>

<details>

<summary>How do we solve this problem?</summary>

We can split our code into libraries (or in the Python world - **packages**).

Packages are a directory of Python scripts.

Each such script is a so-called **module**.

Here's the hierarchy visualized:

![w01_packages_modules.png](./assets/w01_packages_modules.png "w01_packages_modules.png")

These modules specify functions, methods and new Python types aimed at solving particular problems. There are thousands of Python packages available from the Internet. Among them are packages for data science:

- there's **NumPy to efficiently work with arrays**;
- **Matplotlib for data visualization**;
- **scikit-learn for machine learning**.

</details>

Not all of them are available in Python by default, though. To use Python packages, you'll first have to install them on your own system, and then put code in your script to tell Python that you want to use these packages. Advice:

- always install packages in **virtual environments** (abstractions that hold packages for separate projects).
  - You can create a virtual environment by using the following code:

    ```console
    python3 -m venv .venv
    ```

    This will create a hidden folder, called `.venv`, that will store all packages you install for your current project (instead of installing them globally on your system).

  - If there is a `requirements.txt` file, use it to install the needed packages beforehand.
    - In the github repo, there is such a file - you can use it to install all the packages you'll need in the course. This can be done by using this command:

    ```console
    (if on Windows) > .venv\Scripts\activate
    (if on Linux) > source .venv/bin/activate
    (.venv) > pip install -r requirements.txt
    ```

Now that the package is installed, you can actually start using it in one of your Python scripts. To do this you should import the package, or a specific module of the package.

You can do this with the `import` statement. To import the entire `numpy` package, you can do `import numpy`. A commonly used function in NumPy is `array`. It takes a Python list as input and returns a [`NumPy array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) object as an output. The NumPy array is very useful to do data science, but more on that later. Calling the `array` function like this, though, will generate an error:

```python
import numpy
array([1, 2, 3])
```

```console
NameError: name `array` is not defined
```

To refer to the `array` function from the `numpy` package, you'll need this:

```python
import numpy
numpy.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time it works.

Using this `numpy.` prefix all the time can become pretty tiring, so you can also import the package and refer to it with a different name. You can do this by extending your `import` statement with `as`:

```python
import numpy as np
np.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

Now, instead of `numpy.array`, you'll have to use `np.array` to use NumPy's functions.

There are cases in which you only need one specific function of a package. Python allows you to make this explicit in your code.

Suppose that we ***only*** want to use the `array` function from the NumPy package. Instead of doing `import numpy`, you can instead do `from numpy import array`:

```python
from numpy import array
array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time, you can simply call the `array` function without `numpy.`.

This `from import` version to use specific parts of a package can be useful to limit the amount of coding, but you're also loosing some of the context. Suppose you're working in a long Python script. You import the array function from numpy at the very top, and way later, you actually use this array function. Somebody else who's reading your code might have forgotten that this array function is a specific NumPy function; it's not clear from the function call.

![w01_from_numpy.png](./assets/w01_from_numpy.png "w01_from_numpy.png")

^ using numpy, but not very clear

Thus, the more standard `import numpy as np` call is preferred: In this case, your function call is `np.array`, making it very clear that you're working with NumPy.

![w01_import_as_np.png](./assets/w01_import_as_np.png "w01_import_as_np.png")

- Suppose you want to use the function `inv()`, which is in the `linalg` subpackage of the `scipy` package. You want to be able to use this function as follows:

    ```python
    my_inv([[1,2], [3,4]])
    ```

    Which import statement will you need in order to run the above code without an error?

  - A. `import scipy`
  - B. `import scipy.linalg`
  - C. `from scipy.linalg import my_inv`
  - D. `from scipy.linalg import inv as my_inv`

    <details>

    <summary>Reveal answer:</summary>

    Answer: D

    </details>

### Importing means executing `main.py`

Remember that importing a package is equivalent to executing everything in the `main.py` module. Thus. you should always have `if __name__ == '__main__'` block of code and call your functions from there.

Run the scripts `test_script1.py` and `test_script2.py` to see the differences.

## NumPy

### Context

Python lists are pretty powerful:

- they can hold a collection of values with different types (heterogeneous data structure);
- easy to change, add, remove elements;
- many built-in functions and methods.

### Problems

This is wonderful, but one feature is missing, a feature that is super important for aspiring data scientists and machine learning engineers - carrying out mathematical operations **over entire collections of values** and doing it **fast**.

Let's take the heights and weights of your family and yourself. You end up with two lists, `height`, and `weight` - the first person is `1.73` meters tall and weighs `65.4` kilograms and so on.

```python
height = [1.73, 1.68, 1.71, 1.89, 1.79]
height
```

```console
[1.73, 1.68, 1.71, 1.89, 1.79]
```

```python
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
weight
```

```console
[65.4, 59.2, 63.6, 88.4, 68.7]
```

If you now want to calculate the Body Mass Index for each family member, you'd hope that this call can work, making the calculations element-wise. Unfortunately, Python throws an error, because it has no idea how to do calculations on lists. You could solve this by going through each list element one after the other, and calculating the BMI for each person separately, but this is terribly inefficient and tiresome to write.

### Solution

- `NumPy`, or Numeric Python;
- Provides an alternative to the regular Python list: the NumPy array;
- The NumPy array is pretty similar to the list, but has one additional feature: you can perform calculations over entire arrays;
- super-fast as it's based on C++
- Installation:
  - In the terminal: `pip install numpy`

### Benefits

Speed, speed, speed:

- Stackoverflow: <https://stackoverflow.com/questions/73060352/is-numpy-any-faster-than-default-python-when-iterating-over-a-list>
- Visual Comparison:

    ![w01_np_vs_list.png](./assets/w01_np_vs_list.png "w01_np_vs_list.png")

### Usage

```python
import numpy as np
np_height = np.array(height)
np_height
```

```console
array([1.73, 1.68, 1.71, 1.89, 1.79])
```

```python
import numpy as np
np_weight = np.array(weight)
np_weight
```

```console
array([65.4, 59.2, 63.6, 88.4, 68.7])
```

```python
# Calculations are performed element-wise.
# 
# The first person's BMI was calculated by dividing the first element in np_weight
# by the square of the first element in np_height,
# 
# the second person's BMI was calculated with the second height and weight elements, and so on.
bmi = np_weight / np_height ** 2
bmi
```

```console
array([21.851, 20.975, 21.750, 24.747, 21.441])
```

in comparison, the above will not work for Python lists:

```python
weight / height ** 2
```

```console
TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
```

You should still pay attention, though:

- `numpy` assumes that your array contains values **of a single type**;
- a NumPy array is simply a new kind of Python type, like the `float`, `str` and `list` types. This means that it comes with its own methods, which can behave differently than you'd expect.

    ```python
    python_list = [1, 2, 3]
    numpy_array = np.array([1, 2, 3])
    ```

    ```python
    python_list + python_list
    ```

    ```console
    [1, 2, 3, 1, 2, 3]
    ```

    ```python
    numpy_array + numpy_array
    ```

    ```console
    array([2, 4, 6])
    ```

- When you want to get elements from your array, for example, you can use square brackets as with Python lists. Suppose you want to get the bmi for the second person, so at index `1`. This will do the trick:

    ```python
    bmi
    ```

    ```console
    [21.851, 20.975, 21.750, 24.747, 21.441]
    ```

    ```python
    bmi[1]
    ```

    ```console
    20.975
    ```

- Specifically for NumPy, there's also another way to do list subsetting: using an array of booleans.

    Say you want to get all BMI values in the bmi array that are over `23`.

    A first step is using the greater than sign, like this: The result is a NumPy array containing booleans: `True` if the corresponding bmi is above `23`, `False` if it's below.

    ```python
    bmi > 23
    ```

    ```console
    array([False, False, False, True, False])
    ```

    Next, you can use this boolean array inside square brackets to do subsetting. Only the elements in `bmi` that are above `23`, so for which the corresponding boolean value is `True`, is selected. There's only one BMI that's above `23`, so we end up with a NumPy array with a single value, that specific BMI. Using the result of a comparison to make a selection of your data is a very common way to work with data.

    ```python
    bmi[bmi > 23]
    ```

    ```console
    array([24.747])
    ```

### 2D NumPy Arrays

If you ask for the type of these arrays, Python tells you that they are `numpy.ndarray`:

```python
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
type(np_height)
```

```console
numpy.ndarray
```

```python
type(np_weight)
```

```console
numpy.ndarray
```

`ndarray` stands for n-dimensional array. The arrays `np_height` and `np_weight` are one-dimensional arrays, but it's perfectly possible to create `2`-dimensional, `3`-dimensional and `n`-dimensional arrays.

You can create a 2D numpy array from a regular Python list of lists:

```python
np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
                  [65.4, 59.2, 63.6, 88.4, 68.7]])
np_2d
```

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

Each sublist in the list, corresponds to a row in the `2`-dimensional numpy array. Using `.shape`, you can see that we indeed have `2` rows and `5` columns:

```python
np_2d.shape
```

```console
(2, 5) # 2 rows, 5 columns
```

`shape` is a so-called **attribute** of the `np2d` array, that can give you more information about what the data structure looks like.

> **Note:** The syntax for accessing an attribute looks a bit like calling a method, but they are not the same! Remember that methods have round brackets (`()`) after them, but attributes do not.
>
> **Note:** For n-D arrays, the NumPy rule still applies: an array can only contain a single type.

You can think of the 2D numpy array as a faster-to-work-with list of lists: you can perform calculations and more advanced ways of subsetting.

Suppose you want the first row, and then the third element in that row - you can grab it like this:

```python
np_2d[0][2]
```

```console
1.71
```

or use an alternative way of subsetting, using single square brackets and a comma:

```python
np_2d[0, 2]
```

```console
1.71
```

The value before the comma specifies the row, the value after the comma specifies the column. The intersection of the rows and columns you specified, are returned. This is the syntax that's most popular.

Suppose you want to select the height and weight of the second and third family member from the following array.

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

<details>

<summary>How can this be achieved?</summary>

Answer: np_2d[:, [1, 2]]

</details>

### Basic Statistics

A typical first step in analyzing your data, is getting to know your data in the first place.

Imagine you conduct a city-wide survey where you ask `5000` adults about their height and weight. You end up with something like this: a 2D numpy array, that has `5000` rows, corresponding to the `5000` people, and `2` columns, corresponding to the height and the weight.

```python
np_city = ...
np_city
```

```console
array([[ 2.01, 64.33],
       [ 1.78, 67.56],
       [ 1.71, 49.04],
       ...,
       [ 1.73, 55.37],
       [ 1.72, 69.73],
       [ 1.85, 66.69]])
```

Simply staring at these numbers, though, won't give you any insights. What you can do is generate summarizing statistics about them.

- you can try to find out the average height of these people, with NumPy's `mean` function:

```python
np.mean(np_city[:, 0]) # alternative: np_city[:, 0].mean()
```

```console
1.7472
```

It appears that on average, people are `1.75` meters tall.

- What about the median height? This is the height of the middle person if you sort all persons from small to tall. Instead of writing complicated python code to figure this out, you can simply use NumPy's `median` function:

```python
np.median(np_city[:, 0]) # alternative: np_city[:, 0].median()
```

```console
1.75
```

You can do similar things for the `weight` column in `np_city`. Often, these summarizing statistics will provide you with a "sanity check" of your data. If you end up with a average weight of `2000` kilograms, your measurements are most likely incorrect. Apart from mean and median, there's also other functions, like:

```python
np.corrcoef(np_city[:, 0], np_city[:, 1])
```

```console
array([[1.       , 0.0082912],
       [0.0082912, 1.       ]])
```

```python
np.std(np_city[:, 0])
```

```console
np.float64(0.19759467357193614)
```

`sum()`, `sort()`, etc, etc. See all of them [here](https://numpy.org/doc/stable/reference/routines.statistics.html).

### Generate data

The data used above was generated using the following code. Two random distributions were sampled 5000 times to create the `height` and `weight` arrays, and then `column_stack` was used to paste them together as two columns.

```python
import numpy as np
height = np.round(np.random.normal(1.75, 0.2, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))
```

## Matplotlib

The better you understand your data, the better you'll be able to extract insights. And once you've found those insights, again, you'll need visualization to be able to share your valuable insights with other people.

![w01_matplotlib.png](./assets/w01_matplotlib.png "w01_matplotlib.png")

There are many visualization packages in python, but the mother of them all, is `matplotlib`. You will need its subpackage `pyplot`. By convention, this subpackage is imported as `plt`:

```python
import matplotlib.pyplot as plt
```

### Line plot

Let's try to gain some insights in the evolution of the world population. To plot data as a **line chart**, we call `plt.plot` and use our two lists as arguments. The first argument corresponds to the horizontal axis, and the second one to the vertical axis.

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.plot(year, pop)

# "plt.show" displays the plot
plt.show()
```

You'll have to call `plt.show()` explicitly because you might want to add some extra information to your plot before actually displaying it, such as titles and label customizations.

As a result we get:

![w01_matplotlib_result.png](./assets/w01_matplotlib_result.png "w01_matplotlib_result.png")

We see that:

- the years are indeed shown on the horizontal axis;
- the populations on the vertical axis;
- this type of plot is great for plotting a time scale along the x-axis and a numerical feature on the y-axis.

There are four data points, and Python draws a line between them.

![w01_matplotlib_edited.png](./assets/w01_matplotlib_edited.png "w01_matplotlib_edited.png")

In 1950, the world population was around 2.5 billion. In 2010, it was around 7 billion.

> **Insight:** The world population has almost tripled in sixty years.
>
> **Note:** If you pass only one argument to `plt.plot`, Python will know what to do and will use the index of the list to map onto the `x` axis, and the values in the list onto the `y` axis.

### Scatter plot

We can reuse the code from before and just swap `plt.plot(...)` with `plt.scatter(...)`:

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.scatter(year, pop)

# "plt.show" displays the plot
plt.show()
```

![w01_matplotlib_scatter.png](./assets/w01_matplotlib_scatter.png "w01_matplotlib_scatter.png")

The resulting scatter plot:

- plots the individual data points;
- dots aren't connected with a line;
- is great for plotting two numerical features (example: correlation analysis).

### Drawing multiple plots on one figure

This can be done by first instantiating the figure and two axis and the using each axis to plot the data. Example taken from [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Sharing Y axis')

ax1.plot(x, y)
ax2.scatter(x, y)

plt.show()
```

![w01_multiplot.png](./assets/w01_multiplot.png "w01_multiplot.png")

### The logarithmic scale

Sometimes the correlation analysis between two variables can be done easier when one or all of them is plotted on a logarithmic scale. This is because we would reduce the difference between large values as this scale "squashes" large numbers:

![w01_logscale.png](./assets/w01_logscale.png "w01_logscale.png")

In `matplotlib` we can use the [plt.xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xscale.html) function to change the scaling of an axis using `plt` or [ax.set_xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale) to set the scale of an axis of a subplot.

### Histogram

#### Introduction

The histogram is a plot that's useful to explore **distribution of numeric** data;

Imagine `12` values between `0` and `6`.

![w01_histogram_ex1.png](./assets/w01_histogram_ex1.png "w01_histogram_ex1.png")

To build a histogram for these values, you can divide the line into **equal chunks**, called **bins**. Suppose you go for `3` bins, that each have a width of `2`:

![w01_histogram_ex2.png](./assets/w01_histogram_ex2.png "w01_histogram_ex2.png")

Next, you count how many data points sit inside each bin. There's `4` data points in the first bin, `6` in the second bin and `2` in the third bin:

![w01_histogram_ex3.png](./assets/w01_histogram_ex3.png "w01_histogram_ex3.png")

Finally, you draw a bar for each bin. The height of the bar corresponds to the number of data points that fall in this bin. The result is a histogram, which gives us a nice overview on how the `12` values are **distributed**. Most values are in the middle, but there are more values below `2` than there are above `4`:

![w01_histogram_ex4.png](./assets/w01_histogram_ex4.png "w01_histogram_ex4.png")

#### In `matplotlib`

In `matplotlib` we can use the `.hist` function. In its documentation there're a bunch of arguments you can specify, but the first two are the most used ones:

- `x` should be a list of values you want to build a histogram for;
- `bins` is the number of bins the data should be divided into. Based on this number, `.hist` will automatically find appropriate boundaries for all bins, and calculate how may values are in each one. If you don't specify the bins argument, it will by `10` by default.

![w01_histogram_matplotlib.png](./assets/w01_histogram_matplotlib.png "w01_histogram_matplotlib.png")

The number of bins is important in the following way:

- too few bins will oversimplify reality and won't show you the details;
- too many bins will overcomplicate reality and won't show the bigger picture.

Experimenting with different numbers and/or creating multiple plots on the same canvas can alleviate that.

Here's the code that generated the above example:

```python
import matplotlib.pyplot as plt
xs = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(xs, bins=3)
plt.show()
```

and the result of running it:

![w01_histogram_matplotlib_code.png](./assets/w01_histogram_matplotlib_code.png "w01_histogram_matplotlib_code.png")

#### Use cases

Histograms are really useful to give a bigger picture. As an example, have a look at this so-called **population pyramid**. The age distribution is shown, for both males and females, in the European Union.

![w01_population_pyramid.png](./assets/w01_population_pyramid.png "w01_population_pyramid.png")

Notice that the histograms are flipped 90 degrees; the bins are horizontal now. The bins are largest for the ages `40` to `44`, where there are `20` million males and `20` million females. They are the so called baby boomers. These are figures of the year `2010`. What do you think will have changed in `2050`?

Let's have a look.

![w01_population_pyramid_full.png](./assets/w01_population_pyramid_full.png "w01_population_pyramid_full.png")

The distribution is flatter, and the baby boom generation has gotten older. **With the blink of an eye, you can easily see how demographics will be changing over time.** That's the true power of histograms at work here!

### Checkpoint

<details>

<summary>
You want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: C.

</details>

<details>

<summary>
You want to visually assess if longer answers on exam questions lead to higher grades. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: B.

</details>

### Customization

Creating a plot is one thing. Making the correct plot, that makes the message very clear - that's the real challenge.

For each visualization, you have many options:

- change colors;
- change shapes;
- change labels;
- change axes, etc., etc.

The choice depends on:

- the data you're plotting;
- the story you want to tell with this data.

Below are outlined best practices when it comes to creating an MVP plot.

If we run the script for creating a line plot, we already get a pretty nice plot:

![w01_plot_basic.png](./assets/w01_plot_basic.png "w01_plot_basic.png")

It shows that the population explosion that's going on will have slowed down by the end of the century.

But some things can be improved:

- **axis labels**;
- **title**;
- **ticks**.

#### Axis labels

The first thing you always need to do is label your axes. We can do this by using the `xlabel` and `ylabel` functions. As inputs, we pass strings that should be placed alongside the axes.

![w01_plot_axis_labels.png](./assets/w01_plot_axis_labels.png "w01_plot_axis_labels.png")

#### Title

We're also going to add a title to our plot, with the `title` function. We pass the actual title, `'World Population Projections'`, as an argument:

![w01_plot_title.png](./assets/w01_plot_title.png "w01_plot_title.png")

#### Ticks

Using `xlabel`, `ylabel` and `title`, we can give the reader more information about the data on the plot: now they can at least tell what the plot is about.

To put the population growth in perspective, the y-axis should start from `0`. This can be achieved by using the `yticks` function. The first input is a list, in this example with the numbers `0` up to `10`, with intervals of `2`:

![w01_plot_ticks.png](./assets/w01_plot_ticks.png "w01_plot_ticks.png")

Notice how the curve shifts up. Now it's clear that already in `1950`, there were already about `2.5` billion people on this planet.

Next, to make it clear we're talking about billions, we can add a second argument to the `yticks` function, which is a list with the display names of the ticks. This list should have the same length as the first list.

![w01_plot_tick_labels.png](./assets/w01_plot_tick_labels.png "w01_plot_tick_labels.png")

#### Adding more data

Finally, let's add some more historical data to accentuate the population explosion in the last `60` years. If we run the script once more, three data points are added to the graph, giving a more complete picture.

![w01_plot_more_data.png](./assets/w01_plot_more_data.png "w01_plot_more_data.png")

#### `plt.tight_layout()`

##### Problem

With the default Axes positioning, the axes title, axis labels, or tick labels can sometimes go outside the figure area, and thus get clipped.

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.show()
```

![w01_tight_layout_1.png](./assets/w01_tight_layout_1.png "w01_tight_layout_1.png")

##### Solution

To prevent this, the location of Axes needs to be adjusted. `plt.tight_layout()` does this automatically:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_2.png](./assets/w01_tight_layout_2.png "w01_tight_layout_2.png")

When you have multiple subplots, often you see labels of different Axes overlapping each other:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.show()
```

![w01_tight_layout_3.png](./assets/w01_tight_layout_3.png "w01_tight_layout_3.png")

`plt.tight_layout()` will also adjust spacing between subplots to minimize the overlaps:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_4.png](./assets/w01_tight_layout_4.png "w01_tight_layout_4.png")

## Seaborn

Seaborn is a Python data visualization library based on `matplotlib`. It provides a higher-level interface for drawing attractive and informative statistical graphics. Most of the time what you can do in `matplotlib` with several lines of code, you can do with `seaborn` with one line! Refer to the notebook `matplotlib_vs_seaborn.ipynb` in `Week_01` to see a comparison between `matplotlib` and `seaborn`.

In the course you can use both `matplotlib` and `seaborn`! **If you want to only work with `seaborn`, this is completely fine!** Use a plotting library of your choice for any plotting exercises in the course.

## Pandas

### Introduction

As a person working in the data field, you'll often be working with tons of data. The form of this data can vary greatly, but pretty often, you can boil it down to a tabular structure, that is, in the form of a table like in a spreadsheet.

Suppose you're working in a chemical plant and have temperature measurements to analyze. This data can come in the following form:

![w01_pandas_ex1.png](./assets/w01_pandas_ex1.png "w01_pandas_ex1.png")

- row = measurement = observation;
- column = variable. There are three variables above: temperature, date and time, and the location.

- To start working on this data in Python, you'll need some kind of rectangular data structure. That's easy, we already know one! The 2D `numpy` array, right?
    > It's an option, but not necessarily the best one. There are different data types and `numpy` arrays are not great at handling these.
        - it's possible that for a single column we have both a string variable and a float variable.

Your datasets will typically comprise different data types, so we need a tool that's better suited for the job. To easily and efficiently handle this data, there's the `pandas` package:

- a high level data manipulation tool;
- developed by [Wes McKinney](https://en.wikipedia.org/wiki/Wes_McKinney);
- built on the `numpy` package, meaning it's fast;
- Compared to `numpy`, it's more high level, making it very interesting for data scientists all over the world.

In `pandas`, we store the tabular data in an object called a **DataFrame**. Have a look at this [BRICS](https://en.wikipedia.org/wiki/BRICS) data:

![w01_pandas_brics.png](./assets/w01_pandas_brics.png "w01_pandas_brics.png")

We see a similar structure:

- the rows represent the observations;
- the columns represent the variables.

Notice also that:

- each row has a unique row label: `BR` for `Brazil`, `RU` for `Russia`, and so on;
- the columns have labels: `country`, `population`, and so on;
- the values in the different columns have different types.

### DataFrame from Dictionary

One way to build dataframes is using a dictionary:

- the keys are the column labels;
- the values are the corresponding columns, in list form.

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}

df_brics = pd.DataFrame(data)
df_brics
```

```console
        country    capital    area  poplulation
0        Brazil   Brasilia   8.516       200.40
1        Russia     Moscow  17.100       143.50
2         India  New Delhi   3.286      1252.00
3         China    Beijing   9.597      1357.00
4  South Africa   Pretoria   1.221        52.98
```

We're almost there:

- `pandas` assigned automatic row labels, `0` up to `4`;
- to specify them manually, you can set the `index` attribute of `df_brics` to a list with the correct labels.

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}
index = ['BR', 'RU', 'IN', 'CH', 'SA']

df_brics = pd.DataFrame(data, index)
df_brics
```

or

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}

df_brics = pd.DataFrame(data)
df_brics.index = ['BR', 'RU', 'IN', 'CH', 'SA']
df_brics
```

```console
         country    capital    area  poplulation
BR        Brazil   Brasilia   8.516       200.40
RU        Russia     Moscow  17.100       143.50
IN         India  New Delhi   3.286      1252.00
CH         China    Beijing   9.597      1357.00
SA  South Africa   Pretoria   1.221        52.98
```

The resulting `df_brics` is the same one as we saw before.

### DataFrame from CSV file

Using a dictionary approach is fine, but what if you're working with tons of data - you won't build the DataFrame manually, right? Yep - in this case, you import data from an external file that contains all this data.

A common file format used to store such data is the `CSV` file ([comma-separated file](https://en.wikipedia.org/wiki/Comma-separated_values)).

Here are the contents of our `brics.csv` file (you can find it in the `DATA` folder):

```console
,country,capital,area,poplulation
BR,Brazil,Brasilia,8.516,200.4
RU,Russia,Moscow,17.1,143.5
IN,India,New Delhi,3.286,1252.0
CH,China,Beijing,9.597,1357.0
SA,South Africa,Pretoria,1.221,52.98
```

Let's try to import this data into Python using the `read_csv` function:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv')
df_brics
```

```console
  Unnamed: 0       country    capital    area  poplulation
0         BR        Brazil   Brasilia   8.516       200.40
1         RU        Russia     Moscow  17.100       143.50
2         IN         India  New Delhi   3.286      1252.00
3         CH         China    Beijing   9.597      1357.00
4         SA  South Africa   Pretoria   1.221        52.98
```

There's something wrong:

- the row labels are seen as a column;
- to solve this, we'll have to tell the `read_csv` function that the first column contains the row indexes. This is done by setting the `index_col` argument.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics
```

```console
         country    capital    area  poplulation
BR        Brazil   Brasilia   8.516       200.40
RU        Russia     Moscow  17.100       143.50
IN         India  New Delhi   3.286      1252.00
CH         China    Beijing   9.597      1357.00
SA  South Africa   Pretoria   1.221        52.98
```

This time `df_brics` contains the `DataFrame` we started with. The `read_csv` function features many more arguments that allow you to customize your data import, so make sure to check out its [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

### Indexing and selecting data

#### Column Access using `[]`

Suppose you only want to select the `country` column from `df_brics`. How do we do this? We can put the column label - `country`, inside square brackets.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics['country']
```

```console
BR          Brazil
RU          Russia
IN           India
CH           China
SA    South Africa
Name: country, dtype: object
```

- we see as output the entire column, together with the row labels;
- the last line says `Name: country, dtype: object` - we're not dealing with a regular DataFrame here. Let's find out about the `type` of the object that gets returned.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
type(df_brics['country'])
```

```console
<class 'pandas.core.series.Series'>
```

- we're dealing with a `Pandas Series` here;
- you can think of the Series as a 1-dimensional NumPy array that can be labeled:
  - if you paste together a bunch of Series next to each other, you can create a DataFrame.

If you want to select the `country` column but keep the data in a DataFrame, you'll need double square brackets:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)

print(f"Type is {type(df_brics[['country']])}")

df_brics[['country']]
```

```console
Type is <class 'pandas.core.frame.DataFrame'>
         country
BR        Brazil
RU        Russia
IN         India
CH         China
SA  South Africa
```

- to select two columns, `country` and `capital`, add them to the inner list. Essentially, we're putting a list with column labels inside another set of square brackets, and end up with a 'sub DataFrame', containing only the wanted columns.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

#### Row Access using `[]`

You can also use the same square brackets to select rows from a DataFrame. The way to do it is by specifying a slice. To get the **second**, **third** and **fourth** rows of brics, we use the slice `1` colon `4` (the end of the slice is exclusive).

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[1:4]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

- these square brackets work, but it only offers limited functionality:
  - we'd want something similar to 2D NumPy arrays:
    - the index or slice before the comma referred to the rows;
    - the index or slice after the comma referred to the columns.
- if we want to do a similar thing with Pandas, we have to extend our toolbox with the `loc` and `iloc` functions.

`loc` is a technique to select parts of your data **based on labels**, `iloc` is position **based**.

#### Row Access using `loc`

Let's try to get the row for Russia. We put the label of the row of interest in square brackets after `loc`.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc['RU']
```

```console
country       Russia
capital       Moscow
area            17.1
population     143.5
Name: RU, dtype: object
```

Again, we get a Pandas Series, containing all the row's information (rather inconveniently shown on different lines).

To get a DataFrame, we have to put the `'RU'` string inside another pair of brackets.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU']]
```

```console
   country capital  area  population
RU  Russia  Moscow  17.1       143.5
```

We can also select multiple rows at the same time. Suppose you want to also include `India` and `China` - we can add some more row labels to the list.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU', 'IN', 'CH']]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

This was only selecting entire rows, that's something you could also do with the basic square brackets and integer slices.

**The difference here is that you can extend your selection with a comma and a specification of the columns of interest. This is used a lot!**

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU', 'IN', 'CH'], ['country', 'capital']]
```

```console
   country    capital
RU  Russia     Moscow
IN   India  New Delhi
CH   China    Beijing
```

You can also use `loc` to select all rows but only a specific number of columns - replace the first list that specifies the row labels with a colon (this is interpreted as a slice going from beginning to end). This time, the intersection spans all rows, but only two columns.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[:, ['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

Note that the above can be done, by simply omitting the slice which is usually done.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

#### Checkpoint

- Square brackets:
  - Column access `brics[['country', 'capital']]`;
  - Row access: only through slicing `brics[1:4]`.
- loc (label-based):
  - Row access `brics.loc[['RU', 'IN', 'CH']]`;
  - Column access `brics.loc[:, ['country', 'capital']]`;
  - Row and Column access `df_brics.loc[['RU', 'IN', 'CH'], ['country', 'capital']]` (**very powerful**).

#### `iloc`

In `loc`, you use the `'RU'` string in double square brackets, to get a DataFrame. In `iloc`, you use the index `1` instead of `RU`. The results are exactly the same.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)

print(df_brics.loc[['RU']])
print()
print(df_brics.iloc[[1]])
```

```console
   country capital  area  population
RU  Russia  Moscow  17.1       143.5

   country capital  area  population
RU  Russia  Moscow  17.1       143.5
```

To get the rows for `Russia`, `India` and `China`:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[[1, 2, 3]]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

To in addition only keep the `country` and `capital` column:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[[1, 2, 3], [0, 1]]
```

```console
   country    capital
RU  Russia     Moscow
IN   India  New Delhi
CH   China    Beijing
```

To keep all rows and just select columns:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[:, [0, 1]]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

### Filtering pandas DataFrames

#### Single condition

Suppose you now want to get the countries for which the area is greater than 8 million square kilometers. In this case, they are Brazil, Russia, and China.

```console
         country    capital    area  population
BR        Brazil   Brasilia   8.516      200.40
RU        Russia     Moscow  17.100      143.50
IN         India  New Delhi   3.286     1252.00
CH         China    Beijing   9.597     1357.00
SA  South Africa   Pretoria   1.221       52.98
```

There are three steps:

1. Get the `area` column.
2. Perform the comparison on this column and store its result.
3. Use this result to do the appropriate selection on the DataFrame.

- There are a number of ways to achieve step `1`. What's important is that we should get a Pandas Series, not a Pandas DataFrame. How can we achieve this?
    > `brics['area']` or `brics.loc[:, 'area']` or `brics.iloc[:, 2]`

Next, we actually perform the comparison. To see which rows have an area greater than `8`, we simply append `> 8` to the code:

```python
print(df_brics['area'])
print()
print(df_brics['area'] > 8)
```

```console
BR     8.516
RU    17.100
IN     3.286
CH     9.597
SA     1.221
Name: area, dtype: float64

BR     True
RU     True
IN    False
CH     True
SA    False
Name: area, dtype: bool
```

We get a Series containing booleans - areas with a value over `8` correspond to `True`, the others - to `False`.

The final step is using this boolean Series to subset the Pandas DataFrame. To do this, we put the result of the comparison inside square brackets.

```python
is_huge = df_brics['area'] > 8
df_brics[is_huge]
```

```console
   country   capital    area  population
BR  Brazil  Brasilia   8.516       200.4
RU  Russia    Moscow  17.100       143.5
CH   China   Beijing   9.597      1357.0
```

We can also write this in a one-liner. Here is the final full code:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[df_brics['area'] > 8]
```

```console
   country   capital    area  population
BR  Brazil  Brasilia   8.516       200.4
RU  Russia    Moscow  17.100       143.5
CH   China   Beijing   9.597      1357.0
```

#### Multiple conditions

Suppose you only want to keep the observations that have an area between `8` and `10` million square kilometers, inclusive. To do this, we'll have to place not one, but two conditions in the square brackets and connect them via **boolean logical operators**. The boolean logical operators we can use are:

- `&`: both conditions on each side have to evaluate to `True` (just like `&&` in `C++`);
- `|`: at least one of the conditions on each side have to evaluate to `True` (just like `||` in `C++`);
- `~`: unary operator, used for negation (just like `!` in `C++`).

And here is the Python code:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[(df_brics['area'] >= 8) & (df_brics['area'] <= 10)]
```

```console
   country   capital   area  population
BR  Brazil  Brasilia  8.516       200.4
CH   China   Beijing  9.597      1357.0
```

> **Note:** It is very important to remember that when we have multiple conditions each condition should be surrounded by round brackets `(`, `)`.

There is also quite a handy pandas function that can do this for us called [`between`](https://pandas.pydata.org/docs/reference/api/pandas.Series.between.html):

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[df_brics['area'].between(8, 10)]
```

```console
   country   capital   area  population
BR  Brazil  Brasilia  8.516       200.4
CH   China   Beijing  9.597      1357.0
```

#### Filtering and selecting columns

You can also put these filters in `loc` and select columns - this is very handy for performing both subsetting row-wise and column-wise. This is how we can get the **countries** that have an area between `8` and `10` million square kilometers:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[(df_brics['area'] >= 8) & (df_brics['area'] <= 10), 'country']
```

```console
BR    Brazil
CH     China
Name: country, dtype: object
```

### Looping over dataframes

If a Pandas DataFrame were to function the same way as a 2D NumPy array or a list with nested lists in it, then maybe a basic for loop like the following would print out each row. Let's see what the output is.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for val in df_brics:
    print(val)
```

```console
country
capital
area
population
```

Well, this was rather unexpected. We simply got the column names.

In Pandas, you have to mention explicitly that you want to iterate over the rows. You do this by calling the `iterrows` method - it looks at the DataFrame and on each iteration generates two pieces of data:

- the label of the row;
- the actual data in the row as a Pandas Series.

Let's change the for loop to reflect this change. We'll store:

- the row label as `lab`;
- the row data as `row`.

To understand what's happening, let's print `lab` and `row` seperately.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    print(f'Lab is:')
    print(lab)
    print(f'Row is:')
    print(row)
    print()
```

```console
Lab is:
BR
Row is:
country         Brazil 
capital       Brasilia 
area             8.516 
population       200.4 
Name: BR, dtype: object

Lab is:
RU
Row is:
country       Russia
capital       Moscow
area            17.1
population     143.5
Name: RU, dtype: object

Lab is:
IN
Row is:
country           India
capital       New Delhi
area              3.286
population       1252.0
Name: IN, dtype: object

Lab is:
CH
Row is:
country         China
capital       Beijing
area            9.597
population     1357.0
Name: CH, dtype: object

Lab is:
SA
Row is:
country       South Africa
capital           Pretoria
area                 1.221
population           52.98
Name: SA, dtype: object
```

Because this `row` variable on each iteration is a Series, you can easily select additional information from it using the subsetting techniques from earlier.

<details>

<summary>Suppose you only want to print out the capital on each iteration. How can this be done?
</summary>

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    print(f"{lab}: {row['capital']}")
```

```console
BR: Brasilia
RU: Moscow
IN: New Delhi
CH: Beijing
SA: Pretoria
```

</details>

<details>

<summary>Let's add a new column to the end, named `name_length`, containing the number of characters in the country's name. What steps could we follow to do this?
</summary>

Here's our plan of attack:

  1. The specification of the `for` loop can be the same, because we'll need both the row label and the row data;
  2. We'll calculate the length of each country name by selecting the country column from row, and then passing it to the `len()` function;
  3. We'll add this new information to a new column, `name_length`, at the appropriate location.
  4. To see whether we coded things correctly, we'll add a printout of the data after the `for` loop.

In Python we could implement the steps like this:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    df_brics.loc[lab, 'name_length'] = len(row['country'])

print(df_brics)
```

</details>

```console
         country    capital    area  population  name_length
BR        Brazil   Brasilia   8.516      200.40          6.0
RU        Russia     Moscow  17.100      143.50          6.0
IN         India  New Delhi   3.286     1252.00          5.0
CH         China    Beijing   9.597     1357.00          5.0
SA  South Africa   Pretoria   1.221       52.98         12.0
```

<details>

<summary>This is all nice, but it's not especially efficient.
</summary>

We are creating a Series object on every iteration.

A way better approach if we want to calculate an entire DataFrame column by applying a function on a particular column in an element-wise fashion, is `apply()`. In this case, we don't even need a for loop.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics['name_length'] = df_brics['country'].apply(len) # alternative: df_brics['country'].apply(lambda val: len(val))
print(df_brics)
```

```console
         country    capital    area  population  name_length
BR        Brazil   Brasilia   8.516      200.40          6.0
RU        Russia     Moscow  17.100      143.50          6.0
IN         India  New Delhi   3.286     1252.00          5.0
CH         China    Beijing   9.597     1357.00          5.0
SA  South Africa   Pretoria   1.221       52.98         12.0
```

You can think of it like this:

- the `country` column is selected out of the DataFrame;
- on it `apply` calls the `len` function with each value as input;
- `apply` produces a new `array`, that is stored as a new column, `"name_length"`.

This is way more efficient, and also easier to read.

</details>

## Random numbers

### Context

Imagine the following:

- you're walking up the empire state building and you're playing a game with a friend.
- You throw a die `100` times:
  - If it's `1` or `2` you'll go one step down.
  - If it's `3`, `4`, or `5`, you'll go one step up.
  - If you throw a `6`, you'll throw the die again and will walk up the resulting number of steps.
- also, you admit that you're a bit clumsy and have a chance of `0.1%` of falling down the stairs when you make a move. Falling down means that you have to start again from step `0`.

With all of this in mind, you bet with your friend that you'll reach `60` steps high. What is the chance that you will win this bet?

- one way to solve it would be to calculate the chance analytically using equations;
- another possible approach, is to simulate this process thousands of times, and see in what fraction of the simulations that you will reach `60` steps.

We're going to opt for the second approach.

### Random generators

We have to simulate the die. To do this, we can use random generators.

```python
import numpy as np
np.random.rand() # Pseudo-random numbers
```

```console
0.026360555982748446
```

We get a random number between `0` and `1`. This number is so-called pseudo-random. Those are random numbers that are generated using a mathematical formula, starting from a **random seed**.

This seed was chosen by Python when we called the `rand` function, but you can also set this manually. Suppose we set it to `123` and then call the `rand` function twice.

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

> **Note:** Set the seed in the global scope of the Python module (not in a function).

We get two random numbers, however, if call `rand` twice more ***from a new python session***, we get the exact same random numbers!

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

This is funky: you're generating random numbers, but for the same seed, you're generating the same random numbers. That's why it's called pseudo-random; **it's random but consistent between runs**; this is very useful, because this ensures ***"reproducibility"***. Other people can reproduce your analysis.

Suppose we want to simulate a coin toss.

- we set the seed;
- we use the `np.random.randint()` function: it will randomly generate either `0` or `1`. We'll pass two arguments to determine the range of the generated numbers - `0` and `2` (non-inclusive on the right side).

```python
import numpy as np
np.random.seed(123)
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
```

```console
0
1
0
```

We can extend the code with an `if-else` statement to improve user experience:

```python
import numpy as np
np.random.seed(123)
coin = np.random.randint(0, 2)
print(coin)
if coin == 0:
    print('heads')
else:
    print('tails')
```

```console
heads
```

## A note on code formatting

In this course we'll strive to learn how to develop scripts in Python. In general, good code in software engineering is one that is:

1. Easy to read.
2. Safe from bugs.
3. Ready for change.

This section focuses on the first point - how do we make our code easier to read? Here are some principles:

1. Use a linter/formatter.
2. Simple functions - every function should do one thing. This is the single responsibility principle.
3. Break up complex logic into multiple steps. In other words, prefer shorter lines instead of longer.
4. Do not do extended nesting. Instead of writing nested `if` clauses, prefer [`match`](https://docs.python.org/3/tutorial/controlflow.html#match-statements) or many `if` clauses on a single level.

You can automatically handle the first point - let's see how to install and use the `yapf` formatter extension in VS Code.

1. Open the `Extensions` tab, either by using the UI or by pressing `Ctrl + Shift + x`. You'll see somthing along the lines of:
  ![w01_yapf_on_vscode.png](./assets/w01_yapf_on_vscode.png "w01_yapf_on_vscode.png")
2. Search for `yapf`:
  ![w01_yapf_on_vscode_1.png](./assets/w01_yapf_on_vscode_1.png "w01_yapf_on_vscode_1.png")
3. Select and install it:
  ![w01_yapf_on_vscode_2.png](./assets/w01_yapf_on_vscode_2.png "w01_yapf_on_vscode_2.png")
4. After installing, please apply it on every Python file. To do so, press `F1` and type `Format Document`. The script would then be formatted accordingly.
  ![w01_yapf_on_vscode_3.png](./assets/w01_yapf_on_vscode_3.png "w01_yapf_on_vscode_3.png")

# Week 02 - Machine learning with scikit-learn

## What is machine learning?

<details>

<summary>What is machine learning?</summary>

A process whereby computers learn to make decisions from data without being explicitly programmed.

</details>

<details>

<summary>What are simple machine learning use-cases you've heard of?</summary>

- email: spam vs not spam;
- clustering books into different categories/genres based on their content;
- assigning any new book to one of the existing clusters.

</details>

<details>

<summary>What types of machine learning do you know?</summary>

- Unsupervised learning;
- Supervised learning;
- Reinforcement learning;
- Semi-supervised learning.

</details>

<details>

<summary>What is unsupervised learning?</summary>

Uncovering patterns in unlabeled data.

</details>

</details>

<details>

<summary>Knowing the definition of unsupervised learning, can you give some examples?</summary>

grouping customers into categories based on their purchasing behavior without knowing in advance what those categories are (clustering, one branch of unsupervised learning)

![w01_clustering01.png](./assets/w01_clustering01.png "w01_clustering01.png")

</details>

<details>

<summary>What is supervised learning?</summary>

Uncovering patterns in labeled data. Here all possible values to be predicted are already known, and a model is built with the aim of accurately predicting those values on new data.

</details>

</details>

<details>

<summary>Do you know any types of supervised learning?</summary>

- Regression.
- Classification.

</details>

<details>

<summary>What are features?</summary>

Properties of the examples that our model uses to predict the value of the target variable.

</details>

<details>

<summary>Do you know any synonyms of the "feature" term?</summary>

feature = characteristic = predictor variable = independent variable

</details>

<details>

<summary>Do you know any synonyms of the "target variable" term?</summary>

target variable = dependent variable = label = response variable

</details>

<details>

<summary>What features could be used to predict the position of a football player?</summary>

points_per_game, assists_per_game, steals_per_game, number_of_passes

Here how the same example task looks like for basketball:

![w01_basketball_example.png](./assets/w01_basketball_example.png "w01_basketball_example.png")

</details>

<details>

<summary>What is classification?</summary>

Classification is used to predict the label, or category, of an observation.

</details>

<details>

<summary>What are some examples of classification?</summary>

Predict whether a bank transaction is fraudulent or not. As there are two outcomes here - a fraudulent transaction, or non-fraudulent transaction, this is known as binary classification.

</details>

<details>

<summary>What is regression?</summary>

Regression is used to predict continuous values.

</details>

<details>

<summary>What are some examples of regression?</summary>

A model can use features such as the number of bedrooms, and the size of a property, to predict the target variable - the price of that property.

</details>

<details>

<summary>Let's say you want to create a model using supervised learning (for ex. to predict the price of a house). What requirements should the data, you want to use to train the model with, conform to?</summary>

It must not have missing values, must be in numeric format, and stored as CSV files, `pandas DataFrames` or `NumPy arrays`.

</details>

<details>

<summary>How can we make sure that our data conforms to those requirements?</summary>

We must look at our data, explore it. In other words, we need to **perform exploratory data analysis (EDA) first**. Various `pandas` methods for descriptive statistics, along with appropriate data visualizations, are useful in this step.

</details>

<details>

<summary>Have you heard of any supervised machine learning models?</summary>

- k-Nearest Neighbors (KNN);
- linear regression;
- logistic regression;
- support vector machines (SVM);
- decision tree;
- random forest;
- XGBoost;
- CatBoost.

</details>

<details>

<summary>Do you know how the k-Nearest Neighbors model works?</summary>

It uses distance between observations, spread in an `n`-dimensional plane, to predict labels or values, by using the labels or values of the closest `k` observations to them.

</details>

<details>

<summary>Do you know what `scikit-learn` is?</summary>

It is a Python package for using already implemented machine learning models and helpful functions centered around the process of creating and evaluating such models. You can find it's documentation [here](https://scikit-learn.org/).

Install using `pip install scikit-learn` and import using `import sklearn`.

</details>

## The `scikit-learn` syntax

`scikit-learn` follows the same syntax for all supervised learning models, which makes the workflow repeatable:

```python
# import a Model, which is a type of algorithm for our supervised learning problem, from an sklearn module
from sklearn.module import Model

# create a variable named "model", and instantiate the Model class
model = Model()

# fit to the data (X, an array of our features, and y, an array of our target variable values)
# notice the casing - capital letters represent matrices, lowercase - vectors
# during this step the model learns patterns about the features and the target variable
model.fit(X, y)

# use the model's "predict" method, passing X_new - new features of observations to get predictions
predictions = model.predict(X_new)

# for example, if feeding features from six emails to a spam classification model, an array of six values is returned.
# "1" indicates the model predicts that email is spam
# "0" indicates a prediction of not spam
print(predictions)
```

```console
array([0, 0, 0, 0, 1, 0])
```

## The classification challenge

- Classifying labels of unseen data:

    1. Build a model / Instantiate an object from the predictor class.
    2. Model learns from the labeled data we pass to it.
    3. Pass unlabeled data to the model as input.
    4. Model predicts the labels of the unseen data.

- As the classifier learns from the **labeled data**, we call this the **training data**.

- The first model we'll build is the `k-Nearest Neighbors` model. It predicts the label of a data point by:
  - looking at the `k` closest labeled data points;
  - taking a majority vote.

- What class would the black point by assigned to if `k = 3`?

![w01_knn_example.png](./assets/w01_knn_example.png "w01_knn_example.png")

<details>

<summary>Reveal answer</summary>

The red one, since from the closest three points, two of them are from the red class.

![w01_knn_example2.png](./assets/w01_knn_example2.png "w01_knn_example2.png")

</details>

- `k-Nearest Neighbors` is a **non-linear classification and regression model**:
  - it creates a decision boundary between classes (labels)/values. Here's what it looks like on a dataset of customers who churned vs those who did not.

    ![w01_knn_example3.png](./assets/w01_knn_example3.png "w01_knn_example3.png")

- Using `scikit-learn` to fit a classifier follows the standard syntax:

    ```python
    # import the KNeighborsClassifier from the sklearn.neighbors module
    from sklearn.neighbors import KNeighborsClassifier
    
    # split our data into X, a 2D NumPy array of our features, and y, a 1D NumPy array of the target values
    # scikit-learn requires that the features are in an array where each column is a feature and each row a different observation
    X = df_churn[['total_day_charge', 'total_eve_charge']].values
    y = df_churn['churn'].values

    # the target is expected to be a single column with the same number of observations as the feature data
    print(X.shape, y.shape)
    ```

    ```console
    (3333, 2), (3333,)
    ```

    We then instantiate the `KNeighborsClassifier`, setting `n_neighbors=15`, and fit it to the labeled data.

    ```python
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X, y)
    ```

- Predicting unlabeled data also follows the standard syntax:

    Let's say we have a set of new observations, `X_new`. Checking the shape of `X_new`, we see it has three rows and two columns, that is, three observations and two features.

    ```python
    X_new = np.array([[56.8, 17.5],
                      [24.4, 24.1],
                      [50.1, 10.9]])
    print(X_new.shape)
    ```

    ```console
    (3, 2)
    ```

    We use the classifier's `predict` method and pass it the unseen data, again, as a 2D NumPy array of features and observations.

    Printing the predictions returns a binary value for each observation or row in `X_new`. It predicts `1`, which corresponds to `'churn'`, for the first observation, and `0`, which corresponds to `'no churn'`, for the second and third observations.

    ```python
    predictions = knn.predict(X_new)
    print(f'{predictions=}') # notice this syntax! It's valid and cool!
    ```

    ```console
    predictions=[1 0 0]
    ```

## Measuring model performance

<details>

<summary>How do we know if the model is making correct predictions?</summary>

We can evaluate its performance on seen and unseen data.

</details>

<details>

<summary>What is a metric?</summary>

A number which characterizes the quality of the model - the higher the metric value is, the better.

</details>

<details>

<summary>What metrics could be useful for the task of classification?</summary>

A commonly-used metric is accuracy. Accuracy is the number of correct predictions divided by the total number of observations:

![w01_accuracy_formula.png](./assets/w01_accuracy_formula.png "w01_accuracy_formula.png")

There are other metrics which we'll explore further.

</details>

<details>

<summary>On which data should accuracy be measured?</summary>

<details>

<summary>What would be the training accuracy of a `KNN` model when `k=1`?</summary>

Always 100% because the model has seen the data. For every point we're asking the model to return the class of the closest labelled point, but that closest labelled point is the starting point itself (reflection).

</details>

We could compute accuracy on the data used to fit the classifier, however, as this data was used to train the model, performance will not be indicative of how well it can generalize to unseen data, which is what we are interested in!

We can still measure the training accuracy, but only for book-keeping purposes.

We should split the data into a part that is used to train the model and a part that's used to evaluate it.

![w01_train_test.png](./assets/w01_train_test.png "w01_train_test.png")

We fit the classifier using the training set, then we calculate the model's accuracy against the test set's labels.

![w01_training.png](./assets/w01_training.png "w01_training.png")

Here's how we can do this in Python:

```python
# we import the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split

# We call train_test_split, passing our features and targets.
# 
# parameter test_size: We commonly use 20-30% of our data as the test set. By setting the test_size argument to 0.3 we use 30% here.
# 
# parameter random_state: The random_state argument sets a seed for a random number generator that splits the data. Using the same number when repeating this step allows us to reproduce the exact split and our downstream results.
# 
# parameter stratify: It is best practice to ensure our split reflects the proportion of labels in our data. So if churn occurs in 10% of observations, we want 10% of labels in our training and test sets to represent churn. We achieve this by setting stratify equal to y.
# 
# return value: train_test_split returns four arrays: the training data, the test data, the training labels, and the test labels. We unpack these into X_train, X_test, y_train, and y_test, respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# We then instantiate a KNN model and fit it to the training data.
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# To check the accuracy, we use the "score" method, passing X_test and y_test.
print(knn.score(X_test, y_test))
```

```console
0.8800599700149925
```

</details>

<details>

<summary>If our labels have a 9 to 1 ratio, what would be your conclusion about a model that achieves an accuracy of 88%?</summary>

It is low, since even the greedy strategy of always assigning the most common class, would be more accurate than our model (90%).

</details>

## Model complexity (overfitting and underfitting)

Let's discuss how to interpret `k`.

We saw that `KNN` creates decision boundaries, which are thresholds for determining what label a model assigns to an observation.

In the image shown below, as **`k` increases**, the decision boundary is less affected by individual observations, reflecting a **simpler model**:

![w01_k_interpretation.png](./assets/w01_k_interpretation.png "w01_k_interpretation.png")

**Simpler models are less able to detect relationships in the dataset, which is known as *underfitting***. In contrast, complex models can be sensitive to noise in the training data, rather than reflecting general trends. This is known as ***overfitting***.

So, for any `KNN` classifier:

- Larger `k` = Less complex model = Can cause underfitting
- Smaller `k` = More complex model = Can cause overfitting

## Hyperparameter optimization (tuning) / Model complexity curve

We can also interpret `k` using a model complexity curve.

With a KNN model, we can calculate accuracy on the training and test sets using incremental `k` values, and plot the results.

```text
We create empty dictionaries to store our train and test accuracies, and an array containing a range of "k" values.

We can use a for-loop to repeat our previous workflow, building several models using a different number of neighbors.

We loop through our neighbors array and, inside the loop, we instantiate a KNN model with "n_neighbors" equal to the current iterator, and fit to the training data.

We then calculate training and test set accuracy, storing the results in their respective dictionaries. 
```

After our for loop, we can plot the training and test values, including a legend and labels:

```python
plt.figure(figsize=(8, 6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

We see that as `k` increases beyond `15` we see underfitting where performance plateaus on both test and training sets. The peak test accuracy actually occurs at around `13` neighbors.

![w01_knn_results.png](./assets/w01_knn_results.png "w01_knn_results.png")

- Which of the following situations looks like an example of overfitting?

```text
A. Training accuracy 50%, testing accuracy 50%.
B. Training accuracy 95%, testing accuracy 95%.
C. Training accuracy 95%, testing accuracy 50%.
D. Training accuracy 50%, testing accuracy 95%.
```

<details>

<summary>Reveal answer</summary>

Answer: C.

</details>

# Week 03 - Regression

---

## Which are the two main jobs connected to machine learning?

1. `Data Scientist`: Responsible for creating, training, evaluating and improving machine learning models.
2. `Machine Learning Engineer`: Responsible for taking the model produced by the data scientist and deploying it (integrating it) in existing business applications.

### Example

Let's say you are a doctor working in an Excel file. Your task is to map measurements (i.e. `features`) of patients (`age`, `gender`, `cholesterol_level`, `blood_pressure`, `is_smoking`, etc) to the `amount of risk` (a whole number from `0` to `10`) they have for developing heart problems, so that you can call them to come for a visit. For example, for a patient with the following features:

| age | gender | cholesterol_level | blood_pressure | is_smoking | risk_level |
|--------------|-----------|------------|------------|------------|------------|
| 40 | female | 5 | 6 | yes |

you might assign `risk_level=8`, thus the end result would be:

| age | gender | cholesterol_level | blood_pressure | is_smoking | risk_level |
|--------------|-----------|------------|------------|------------|------------|
| 40 | female | 5 | 6 | yes | 8

Throughout the years of working in the field, you have gathered 500,000 rows of data. Now, you've heard about the hype around AI and you want to use a model instead of you manually going through the data. For every patient you want the model to `predict` the amount the risk, so that you can only focus on the ones that have `risk_level > 5`.

- You hire a `data scientist` to create the model using your `training` data.
- You hire a `machine learning engineer` to integrate the created model with your Excel documents.

---

In regression tasks, the target variable typically has **continuous values**, such as a country's GDP, or the price of a house.

## Loading and exploring the data

As an example dataset, we're going to use one containing women's health data and we're going to create models that predict blood glucose levels. Here are the first five rows:

|  idx | pregnancies | glucose | diastolic | triceps | insulin | bmi  | dpf   | age | diabetes |
| ---: | ----------- | ------- | --------- | ------- | ------- | ---- | ----- | --- | -------- |
|    0 | 6           | 148     | 72        | 35      | 0       | 33.6 | 0.627 | 50  | 1        |
|    1 | 1           | 85      | 66        | 29      | 0       | 26.6 | 0.351 | 31  | 0        |
|    2 | 8           | 183     | 64        | 0       | 0       | 23.3 | 0.672 | 32  | 1        |
|    3 | 1           | 89      | 66        | 23      | 94      | 28.1 | 0.167 | 21  | 0        |
|    4 | 0           | 137     | 40        | 35      | 168     | 43.1 | 2.288 | 33  | 1        |

Our goal is to predict blood glucose levels from a single feature. This is known as **simple linear regression**. When we're using two or more features to predict a target variable using linear regression, the process is known as **multiple linear regression**.

We need to decide which feature to use. We talk with internal consultants (our domain experts) and they advise us to check whether there's any relationship between between blood glucose levels and body mass index. We plot them using a [`scatterplot`](https://en.wikipedia.org/wiki/Scatter_plot):

![w02_bmi_bg_plot.png](./assets/w02_bmi_bg_plot.png "w02_bmi_bg_plot.png")

We can see that, generally, as body mass index increases, blood glucose levels also tend to increase. This is great - we can use this feature to create our model.

## Data Preparation

To do simple linear regression we slice out the column `bmi` of `X` and use the `[[]]` syntax so that the result is a dataframe (i.e. two-dimensional array) which `scikit-learn` requires when fitting models.

```python
X_bmi = X[['bmi']]
print(X_bmi.shape, y.shape)
```

```console
(752, 1) (752,)
```

## Modelling

Now we're going to fit a regression model to our data.

We're going to use the model `LinearRegression`. It fits a straight line through our data.

```python
# import
from sklearn.linear_model import LinearRegression

# train
reg = LinearRegression()
reg.fit(X_bmi, y)

# predict
predictions = reg.predict(X_bmi)

# evaluate (visually) - see what the model created (what the model is)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('Body Mass Index')
plt.show()
```

![w02_linear_reg_predictions_plot.png](./assets/w02_linear_reg_predictions_plot.png "w02_linear_reg_predictions_plot.png")

The black line represents the linear regression model's fit of blood glucose values against body mass index, which appears to have a weak-to-moderate positive correlation.

## Diving Deep (Regression mechanics)

Linear regression is the process of fitting a line through our data. In two dimensions this takes the form:

$$y = ax + b$$

When doing simple linear regression:

- $y$ is the target;
- $x$ is the single feature;
- $a, b$ are the parameters (coefficients) of the model - the slope and the intercept.

How do we choose $a$ and $b$?

- define an **error function** that can evaluate any given line;
- choose the line that minimizes the error function.

    > Note: error function = loss function = cost function.

Let's visualize a loss function using this scatter plot.

![w02_no_model.png](./assets/w02_no_model.png "w02_no_model.png")

<details>

<summary>Where do we want the line to be?</summary>

As close to the observations as possible.

![w02_goal_model.png](./assets/w02_goal_model.png "w02_goal_model.png")

</details>

<details>

<summary>How do we obtain such a line (as an idea, not mathematically)?</summary>

We minimize the vertical distance between the fit and the data.

The distance between a single point and the line is called a ***residual***.

![w02_min_vert_dist.png](./assets/w02_min_vert_dist.png "w02_min_vert_dist.png")

</details>

<details>

<summary>Why is minimizing the sum of the residuals not a good idea?</summary>

Because then each positive residual would cancel out each negative residual.

![w02_residuals_cancel_out.png](./assets/w02_residuals_cancel_out.png "w02_residuals_cancel_out.png")

</details>

<details>

<summary>How could we avoid this?</summary>

We square the residuals.

By adding all the squared residuals, we calculate the **residual sum of squares**, or `RSS`. When we're doing linear regression by **minimizing the `RSS`** we're performing what's also called **Ordinary Least Squares Linear Regression**.

![w02_ols_lr.png](./assets/w02_ols_lr.png "w02_ols_lr.png")

In `scikit-learn` linear regression is implemented as `OLS`.

</details>

<details>

<summary>How is linear regression called when we're using more than 1 feature to predict the target?</summary>

Multiple linear regression.

Fitting a multiple linear regression model means specifying a coefficient, $a_n$, for $n$ number of features, and a single $b$.

$$y = a_1x_1 + a_2x_2 + a_3x_3 + \dots + a_nx_n + b$$

</details>

## Model evaluation

### Using a metric

The default metric for linear regression is $R^2$. It quantifies **the amount of variance in the target variable that is explained by the features**.

Values range from `0` to `1` with `1` meaning that the features completely explain the target's variance.

Here are two plots visualizing high and low R-squared respectively:

![w02_r_sq.png](./assets/w02_r_sq.png "w02_r_sq.png")

To compute $R^2$ in `scikit-learn`, we can call the `.score` method of a linear regression class passing test features and targets.

```python
reg_all.score(X_test, y_test)
```

```console
0.356302876407827
```

<details>

<summary>Is this a good result?</summary>

No. Here the features only explain about 35% of blood glucose level variance.

</details>

#### Adjusted $R^2$ ($R_{adj}^2$)

Using $R^2$ could have downsides in some situations. In `Task 03` you'll investigate what they are and how the extension $R_{adj}^2$ can help.

### Using a loss function

Another way to assess a regression model's performance is to take the mean of the residual sum of squares. This is known as the **mean squared error**, or (`MSE`).

$$MSE = \frac{1}{n} \sum (y_i - \hat{y_i})^2$$

> **Note**: Every time you see a hat above a letter (for example $\hat{y_i}$), think of it as if that variable holds the model predictions.

`MSE` is measured in units of our target variable, squared. For example, if a model is predicting a dollar value, `MSE` will be in **dollars squared**.

This is not very easy to interpret. To convert to dollars, we can take the **square root** of `MSE`, known as the **root mean squared error**, or `RMSE`.

$$RMSE = \sqrt{MSE}$$

`RMSE` has the benefit of being in the same unit as the target variable.

To calculate the `RMSE` in `scikit-learn`, we can use the `root_mean_squared_error` function in the `sklearn.metrics` module.

# Week 04 - Cross Validation, Regularized Regression, Classification Metrics

## Cross Validation

Currently, we're using train-test split to compute model performance.

<details>

<summary>What are the potential downsides of using train-test split?</summary>

1. Model performance is dependent on the way we split up the data: we may get different results if we do another split.
2. The data points in the test set may have some peculiarities: the R-squared computed on it is not representative of the model's ability to generalize to unseen data.
3. The points in the test set will never be used for training the model: we're missing out on potential benefits.

</details>

<details>

<summary>Have you heard of the technique called cross-validation?</summary>

It is a vital approach to evaluating a model. It maximizes the amount of data that is available to the model, as the model is not only trained but also tested on all of the available data.

Here's a visual example of what cross-validation comprises of:

![w03_cv_example1.png](./assets/w03_cv_example1.png "w03_cv_example1.png")

We begin by splitting the dataset into `k` groups or folds - ex. `5`. Then we set aside the first fold as a test set, fit our model on the remaining four folds, predict on our test set, and compute the metric of interest, such as R-squared.

Next, we set aside the second fold as our test set, fit on the remaining data, predict on the test set, and compute the metric of interest.

![w03_cv_example2.png](./assets/w03_cv_example2.png "w03_cv_example2.png")

Then similarly with the third fold, the fourth fold, and the fifth fold. As a result we get five values of R-squared from which we can compute statistics of interest, such as the mean, median, and 95% confidence intervals.

![w03_cv_example3.png](./assets/w03_cv_example3.png "w03_cv_example3.png")

Usually the value for `k` is either `5` or `10`.

</details>

<details>

<summary>What is the trade-off of using cross-validation compared to train-test split?</summary>

Using more folds is more computationally expensive. This is because we're fitting and predicting multiple times, instead of just `1`.

</details>

To perform k-fold cross-validation in `scikit-learn`, we can use the function [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#cross-val-score) and the class [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#kfold) that are part of `sklearn.model_selection`.

- the `KFold` class allows us to set a seed and shuffle our data, making our results repeatable downstream. The `n_splits` argument has a default of `5`, but in this case we assign `2`, allowing us to use `2` folds from our dataset for cross-validation. We also set `shuffle=True`, which shuffles our dataset **before** splitting into folds. We assign a seed to the `random_state` keyword argument, ensuring our data would be split in the same way if we repeat the process making the results repeatable downstream. We save this as the variable `kf`.

```python
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2, shuffle=True, random_state=42)

print(list(kf.split(X)))
print(list(kf.split(y)))
```

```console
[(array([0, 2]), array([1, 3])), (array([1, 3]), array([0, 2]))]
[(array([0, 2]), array([1, 3])), (array([1, 3]), array([0, 2]))]
```

The result is a list of tuples of arrays with training and testing indices. In this case, we would use elements at indices `0` and `2` to train a model and evaluate it on elements at indices `1` and `3`.

- in practice, you wouldn't call `kf.split` directly. Instead, you would pass the `kf` object to `cross_val_score`. It accepts a model, feature data and target data as the first three positional arguments. We also specify the number of folds by setting the keyword argument `cv` equal to our `kf` variable.

```python
cv_results = cross_val_score(linear_reg, X, y, cv=kv)
print(cv_results)

# we can calculate the 95% confidence interval passing our results followed by a list containing the upper and lower limits of our interval as decimals 
print(np.quantile(cv_results, [0.025, 0.975]))
```

```console
[0.70262578, 0.7659624, 0.75188205, 0.76914482, 0.72551151, 0.736]
array([0.7054865, 0.76874702])
```

This returns an array of cross-validation scores, which we assign to `cv_results`. The length of the array is the number of folds utilized.

> **Note:** the reported score is the result of calling `linear_reg.score`. Thus, when the model is linear regression, the score reported is $R^2$.

## Regularized Regression

### Regularization

<details>

<summary>Have you heard of regularization?</summary>

Regularization is a technique used to avoid overfitting. It can be applied in any task - classification or regression.

![example](https://www.mathworks.com/discovery/overfitting/_jcr_content/mainParsys/image.adapt.full.medium.svg/1718273106637.svg)

Its main idea is to reduce the size / values of model parameters / coefficients as large coefficients lead to overfitting.

Linear regression models minimize a loss function to choose a coefficient - $a$, for each feature, and an intercept - $b$. When we apply regularization to them, we "extend" the loss function, adding one more variable to the sum, that grows in value as coefficients grow.

</details>

### Ridge Regression

The first type of regularized regression that we'll look at is called `Ridge`. With `Ridge`, we use the `Ordinary Least Squares` loss function plus the squared value of each coefficient, multiplied by a constant - `alpha`.

$$J = \sum_{i=1}^n(y_i - \hat{y_i})^2 + \alpha \sum_{i=1}^na_i^2$$

So, when minimizing the loss function, models are penalized both *for creating a line that's far from the ideal one* **and** *for coefficients with large positive or negative values*.

When using `Ridge`, we need to choose the `alpha` value in order to fit and predict.

- we can select the `alpha` for which our model performs best;
- picking alpha for `Ridge` is similar to picking `k` in `KNN`;
- multiple experiments with different values required - choose local minimum; hope it is the global one.

`Alpha` controls model complexity. When alpha equals `0`, we are performing `OLS`, where large coefficients are not penalized and overfitting *may* occur. A high alpha means that large coefficients are significantly penalized, which *can* lead to underfitting (we're making our model dumber).

`Scikit-learn` comes with a ready-to-use class for Ridge regression - check it out [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#ridge).

### Hyperparameters

<details>

<summary>What are hyperparameters?</summary>

A hyperparameter is a variable used for selecting a model's parameters.

</details>

<details>

<summary>What are some examples?</summary>

- $a$ in `Ridge`;
- $k$ in `KNN`.

</details>

### Lasso Regression

There is another type of regularized regression called Lasso, where our loss function is the `OLS` loss function plus the absolute value of each coefficient multiplied by some constant - `alpha`:

$$J = \sum_{i=1}^n(y_i - \hat{y_i})^2 + \alpha \sum_{i=1}^n|a_i|$$

`Scikit-learn` also comes with a ready-to-use class for Lasso regression - check it out [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#lasso).

### Feature Importance

Feature importance is the amount of added value that a feature provides to a model when that model is trying to predict the target variable. The more important a feature is, the better it is to be part of a model.

Assessing the feature importance of all features can be used to perform **feature selection** - choosing which features will be part of the final model.

### Lasso Regression and Feature Importance

Lasso regression can actually be used to assess feature importance. This is because **it shrinks the coefficients of less important features to `0`**. The features whose coefficients are not shrunk to `0` are, essentially, selected by the lasso algorithm - when summing them up, **the coefficients act as weights**.

Here's how this can be done in practice:

```python
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```

![w03_lasso_coefs_class.png](./assets/w03_lasso_coefs_class.png "w03_lasso_coefs_class.png")

We can see that the most important predictor for our target variable, `blood glucose levels`, is the binary value for whether an individual has `diabetes` or not! This is not surprising, but is a great sanity check.

Benefits:

- allows us to communicate results to non-technical audiences (stakeholders, clients, management);
- helps us eliminate non-important features when we have too many;
- identifies which factors are important predictors for various physical phenomena.

## Classification Metrics

### A problem with using `accuracy` always

**Situation:**

A bank contacts our company and asks for a model that can predict whether a bank transaction is fraudulent or not.

Keep in mind that in practice, 99% of transactions are legitimate and only 1% are fraudulent.

> **Definition:** The situation where classes are not equally represented in the data is called ***class imbalance***.

**Problem:**

<details>

<summary>Do you see any problems with using accuracy as the primary metric here?</summary>

The accuracy of a model that predicts every transactions as legitimate is `99%`.

</details>

**Solution:**

<details>

<summary>How do we solve this?</summary>

We have to use other metrics that put focus on the **per-class** performance.

</details>

<details>

<summary>What can we measure then?</summary>

We have to count how the model treats every observation and define the performance of the model based on the number of times that an observation:

- is positive and the model predicts it to be negative;
- or is negative and the model predicts it to be positive;
- or the model predicts its class correctly.

We can store those counts in a table:

![w03_conf_matrix.png](./assets/w03_conf_matrix.png "w03_conf_matrix.png")

> **Definition:** A **confusion matrix** is a table that is used to define the performance of a classification algorithm.

- Across the top are the predicted labels, and down the side are the actual labels.
- Usually, the class of interest is called the **positive class**. As we aim to detect fraud, **the positive class is an *illegitimate* transaction**.
  - The **true positives** are the number of fraudulent transactions correctly labeled;
  - The **true negatives** are the number of legitimate transactions correctly labeled;
  - The **false negatives** are the number of legitimate transactions incorrectly labeled;
  - And the **false positives** are the number of transactions incorrectly labeled as fraudulent.

</details>

**Benefit:**

<details>

<summary>We can retrieve the accuracy. How?</summary>

It's the sum of true predictions divided by the total sum of the matrix.

![w03_cm_acc.png](./assets/w03_cm_acc.png "w03_cm_acc.png")

</details>

<details>

<summary>Do you know what precision is?</summary>

`precision` is the number of true positives divided by the sum of all positive predictions.

- also called the `positive predictive value`;
- in our case, this is the number of correctly labeled fraudulent transactions divided by the total number of transactions classified as fraudulent:

![w03_cm_precision.png](./assets/w03_cm_precision.png "w03_cm_precision.png")

- **high precision** means having a **lower false positive rate**. For our classifier, it means predicting most fraudulent transactions correctly.

$$FPR = \frac{FP}{FP + TN}$$

</details>

<details>

<summary>Do you know what recall is?</summary>

`recall` is the number of true positives divided by the sum of true positives and false negatives

- also called `sensitivity`;

![w03_cm_recall.png](./assets/w03_cm_recall.png "w03_cm_recall.png")

- **high recall** reflects a **lower false positive rate**. For our classifier, this translates to fewer legitimate transactions being classified as fraudulent.

$$FNR = \frac{FN}{TP + FN}$$

Here is a helpful table that can serve as another example:

![w03_example_metrics_2.png](./assets/w03_example_metrics_2.png "w03_example_metrics_2.png")

</details>

<details>

<summary>Do you know what the f1-score is?</summary>

The `F1-score` is the harmonic mean of precision and recall.

- gives equal weight to precision and recall -> it factors in both the number of errors made by the model and the type of errors;
- favors models with similar precision and recall;
- useful when we are seeking a model which performs reasonably well across both metrics.

![w03_cm_f1.png](./assets/w03_cm_f1.png "w03_cm_f1.png")

Another interpretation of the link between precision and recall:

![w03_prec_rec.png](./assets/w03_prec_rec.png "w03_prec_rec.png")

Why is the harmonic mean used? Since both precision and recall are rates (ratios) between `0` and `1`, the harmonic mean helps balance these two metrics by considering their reciprocals. This ensures that a low value in either one has a significant impact on the overall `F1` score, thus incentivizing a balance between the two.

How does gradient descend happen when we have regularization in linear regression?

- <https://towardsdatascience.com/create-a-gradient-descent-algorithm-with-regularization-from-scratch-in-python-571cb1b46642>;
- <https://math.stackexchange.com/questions/1652661/gradient-descent-l2-norm-regularization>.

</details>

### Confusion matrix in scikit-learn

We can use the [`confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#confusion-matrix) function in `sklearn.metrics`:

```python
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
```

```console
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
```

We can also use the `from_predictions` static function of the [`ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#confusionmatrixdisplay) class, also in `sklearn.metrics` to plot the matrix:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.tight_layout()
plt.show()
```

![w03_cm_plot.png](./assets/w03_cm_plot.png "w03_cm_plot.png")

We can get the discussed metrics from the confusion matrix, by calling the [`classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#classification-report) function in `sklearn.metrics`:

```python
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```

```console
              precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
```

```python
y_pred = [1, 1, 0]
y_true = [1, 1, 1]
print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
```

```console
              precision    recall  f1-score   support

           1       1.00      0.67      0.80         3
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0

   micro avg       1.00      0.67      0.80         3
   macro avg       0.33      0.22      0.27         3
weighted avg       1.00      0.67      0.80         3
```

`Support` represents the number of instances for each class within the true labels. If the column with `support` has different numbers, then we have class imbalance.

- `macro average` = $\frac{F1_{class1} + F1_{class2} + F1_{class3}}{3}$
- `weighted average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{3}$
- `micro average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{SUPPORT_{class1} + SUPPORT_{class2} + SUPPORT_{class3}}$

# Week 05 - Logistic Regression, The ROC curve, Hyperparameter optimization/tuning

## Logistic Regression

### Introduction

It's time to introduce another model: `Logistic Regression`.

<details>

<summary>Is logistic regression used for regression or classification problem?</summary>

Despite its name, **logistic regression is used for solving classification problems**.

This model:

- calculates the probability - `p`, that an observation belongs to a class;
- is usually used in binary classification, though it's also popular in multi-class classification;
- produces a linear decision boundary.

![w04_log_reg_boundary.png](./assets/w04_log_reg_boundary.png "w04_log_reg_boundary.png")

Using our diabetes dataset as an example:

- if `p >= 0.5`, we label the data as `1`, representing a prediction that an individual is more likely to have diabetes;
- if `p < 0.5`, we label it `0` to represent that they are more likely to not have diabetes.

</details>

<details>

<summary>Have you heard about the iris dataset?</summary>

- The [`iris` dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is a collection of measurements of many [iris plants](https://en.wikipedia.org/wiki/Iris_flower_data_set).
- Three species of iris:
  - *setosa*;
  - *versicolor*;
  - *virginica*.
- Features: petal length, petal width, sepal length, sepal width.

Iris setosa:

![w05_setosa.png](./assets/w05_setosa.png "w05_setosa.png")

Iris versicolor:

![w05_versicolor.png](./assets/w05_versicolor.png "w05_versicolor.png")

Iris virginica:

![w05_virginica.png](./assets/w05_virginica.png "w05_virginica.png")

</details>

### In `scikit-learn`

In scikit-learn logistic regression is implemented in the [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression) module:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(f'Possible classes: {set(y)}')
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
```

```console
Possible classes: {0, 1, 2}
array([0, 0])
```

> **Note:** The hyperparameter `C` is the inverse of the regularization strength - larger `C` means less regularization and smaller `C` means more regularization.

```python
clf.predict_proba(X[:2]) # probabilities of each instance belonging to each of the three classes
```

```console
array([[9.81780846e-01, 1.82191393e-02, 1.44184120e-08],
       [9.71698953e-01, 2.83010167e-02, 3.01417036e-08]])
```

```python
clf.predict_proba(X[:2])[:, 2] # probabilities of each instance belonging to the third class only
```

```console
array([1.44184120e-08, 3.01417036e-08])
```

```python
clf.score(X, y) # returns the accuracy
```

```console
0.97
```

## The receiver operating characteristic curve (`ROC` curve)

<details>

<summary>Do you know what the ROC curve shows?</summary>

> **Note:** Use the `ROC` curve only when doing ***binary* classification**.

The default probability threshold for logistic regression in scikit-learn is `0.5`. What happens as we vary this threshold?

We can use a receiver operating characteristic, or ROC curve, to visualize how different thresholds affect `true positive` and `false positive` rates.

![w04_roc_example.png](./assets/w04_roc_example.png "w04_roc_example.png")

The dotted line represents a random model - one that randomly guesses labels.

When the threshold:

- equals `0` (`p=0`), the model predicts `1` for all observations, meaning it will correctly predict all positive values, and incorrectly predict all negative values;
- equals `1` (`p=1`), the model predicts `0` for all data, which means that both true and false positive rates are `0` (nothing gets predicted as positive).

![w04_roc_edges.png](./assets/w04_roc_edges.png "w04_roc_edges.png")

If we vary the threshold, we get a series of different false positive and true positive rates.

![w04_roc_vary.png](./assets/w04_roc_vary.png "w04_roc_vary.png")

A line plot of the thresholds helps to visualize the trend.

![w04_roc_line.png](./assets/w04_roc_line.png "w04_roc_line.png")

<details>

<summary>What plot would be produced by the perfect model?</summary>

One in which the line goes straight up and then right.

![w04_perfect_roc.png](./assets/w04_perfect_roc.png "w04_perfect_roc.png")

</details>

</details>

### In `scikit-learn`

In scikit-learn the `roc_curve` is implemented in the `sklearn.metrics` module.

```python
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

fpr
```

```console
array([0. , 0. , 0.5, 0.5, 1. ])
```

```python
tpr
```

```console
array([0. , 0.5, 0.5, 1. , 1. ])
```

```python
thresholds
```

```console
array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])
```

To plot the curve can create a `matplotlib` plot or use a built-in function.

Using `matplotlib` would look like this:

```python
plt.plot([0, 1], [0, 1], 'k--') # to draw the dashed line
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

![w04_example_roc_matlotlib.png](./assets/w04_example_roc_matlotlib.png "w04_example_roc_matlotlib.png")

We could also use the `from_predictions` function in the `RocCurveDisplay` class to create plots.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y, scores)
plt.tight_layout()
plt.show()
```

![w04_example_roc_display.png](./assets/w04_example_roc_display.png "w04_example_roc_display.png")

<details>

<summary>The above figures look great, but how do we quantify the model's performance based on them?</summary>

### The Area Under the Curve (`AUC`)

If we have a model with `true_positive_rate=1` and `false_positive_rate=0`, this would be the perfect model.

Therefore, we calculate the area under the ROC curve, a **metric** known as `AUC`. Scores range from `0` to `1`, with `1` being ideal.

In the below figure, the model scores `0.67`, which is `34%` better than a model making random guesses.

![w04_example_roc_improvement.png](./assets/w04_example_roc_improvement.png "w04_example_roc_improvement.png")

### In `scikit-learn`

In scikit-learn the area under the curve can be calculated in two ways.

Either by using the `RocCurveDisplay.from_predictions` function:

![w04_example_roc_display_note.png](./assets/w04_example_roc_display_note.png "w04_example_roc_display_note.png")

or by using the `roc_auc_score` function in the `sklearn.metrics` module:

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

```console
0.6700964152663693
```

</details>

## Hyperparameter optimization/tuning

### Introduction

Recall that we had to choose a value for `alpha` in ridge and lasso regression before fitting it.

Likewise, before fitting and predicting `KNN`, we choose `n_neighbors`.

> **Definition:** Parameters that we specify before fitting a model, like `alpha` and `n_neighbors`, are called **hyperparameters**.

<details>

<summary>So, a fundamental step for building a successful model is choosing the correct hyperparameters. What's the best way to go about achieving this?</summary>

We can try lots of different values, fit all of them separately, see how well they perform, and choose the best values!

> **Definition:** The process of trying out different hyperparameters until a satisfactory performance threshold is reached is called **hyperparameter tuning**.

</details>

When fitting different hyperparameter values, we use cross-validation to avoid overfitting the hyperparameters to the test set:

- we split the data into train and test;
- and perform cross-validation on the training set.

We withhold the test set and use it only for evaluating the final, tuned model.

> **Definition:** The set on which the model is evaluated on during cross-validation is called the **validation set**.

Notice how:

- the training set is used to train the model;
- the validation set is used to tune the model until a satisfactory performance threshold is used;
- the test set is used as the final set on which model performance is reported. It is data that the model hasn't seen, but because it is labeled, we can use to evaluate the performance eliminating bias.

### Grid search cross-validation

<details>

<summary>Do you know what grid search cross-validation comprises of?</summary>

One approach for hyperparameter tuning is called **grid search**, where we choose a grid of possible hyperparameter values to try. In Python this translates to having a dictionary that maps strings to lists/arrays of possible values to choose from.

For example, we can search across two hyperparameters for a `KNN` model - the type of metric and a different number of neighbors. We perform `k`-fold cross-validation for each combination of hyperparameters. The mean scores (in this case, accuracies) for each combination are shown here:

![w04_grid_search_cv.png](./assets/w04_grid_search_cv.png "w04_grid_search_cv.png")

We then choose hyperparameters that performed best:

![w04_grid_search_cv_choose_best.png](./assets/w04_grid_search_cv_choose_best.png "w04_grid_search_cv_choose_best.png")

To get a list of supported values for the model we're building, we can use the scikit-learn documentation. For example, in the documentation for [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression), the possible values for the `solver` parameter can be seen as one scrolls down the page.

### In `scikit-learn`

`scikit-learn` has an implementation of grid search using cross-validation in the `sklearn.model_selection` module:

```python
from sklearn.model_selection import GridSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

ridge_cv = GridSearchCV(Ridge(), param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

</details>

<details>

<summary>What is the main problem of grid search?</summary>

Grid search is great - it allows us to scan a predefined parameter space fully. However, it does not scale well:

<details>

<summary>How many fits will be done while performing 3-fold cross-validation for 1 hyperparameter with 10 values?</summary>

Answer: 30.

</details>

<details>

<summary>How many fits will be done while performing 10-fold cross-validation for 3 hyperparameters with 10 values each?</summary>

Answer: 10,000!

We can verify this:

```python
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


def main():
    iris = datasets.load_iris()

    parameters = {
        'degree': np.arange(10),
        'C': np.linspace(0, 10, 10),
        'tol': np.linspace(0.0001, 0.01, 10),
    }

    print(len(parameters['degree']))
    print(len(parameters['C']))
    print(len(parameters['tol']))

    svc = svm.SVC() # This is a support vector machine. We'll talk about it soon.
    clf = GridSearchCV(svc, parameters, cv=10, verbose=1)
    clf.fit(iris.data, iris.target)
    print(sorted(clf.cv_results_))


if __name__ == '__main__':
    main()
```

This is because:

1. The total number of parameter combinations is `10^3 = 1000` (we have one for-loop with two nested ones inside).

  ```python
  # pseudocode
  for d in degrees:
    for c in cs:
      for tol in tols:
        # this is one combination
  ```

2. For every single one combination we do a `10`-fold cross-validation to get the mean metric. This means that every single one of the paramter combinations is the same while we shift the trainig and testing sets `10` times.

</details>

<details>

<summary>What is the formula in general then?</summary>

```text
number of fits = number of folds * number of total hyperparameter values
```

</details>

<details>

<summary>How can we go about solving this?</summary>

### Randomized search cross-validation

We can perform a random search, which picks random hyperparameter values rather than exhaustively searching through all options.

```python
from sklearn.model_selection import RandomizedSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

# optionally set the "n_iter" argument, which determines the number of hyperparameter values tested (default is 10)
# 5-fold cross-validation with "n_iter" set to 2 performs 10 fits
ridge_cv = RandomizedSearchCV(Ridge(), param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

In this case it is able to find the best hyperparameters from our previous grid search!

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

#### Benefits

This allows us to search from large parameter spaces efficiently.

### Evaluating on the test set

We can evaluate model performance on the test set by passing it to a call of the grid/random search object's `.score` method.

```python
test_score = ridge_cv.score(X_test, y_test)
test_score
```

```console
0.7564731534089224
```

</details>

</details>

# Week 06 - Preprocessing and Pipelines

## Dealing with categorical features

`scikit-learn` requires data that:

- is in **numeric** format;
- has **no missing** values.

All the data that we have used so far has been in this format. However, with real-world data:

- this will rarely be the case;
- typically we'll spend around 80% of our time solely focusing on preprocessing it before we can build models (may come as a shoker).

Say we have a dataset containing categorical features, such as `color` and `genre`. Those features are not numeric and `scikit-learn` will not accept them.

<details>

<summary>How can we solve this problem?</summary>

We can substitute the strings with numbers.

<details>

<summary>What approach can we use to do this?</summary>

We need to convert them into numeric features. We can achieve this by **splitting the features into multiple *binary* features**:

- `0`: observation was not that category;
- `1`: observation was that category.

![w05_dummies.png](./assets/w05_dummies.png "w05_dummies.png")

> **Definition:** Such binary features are called **dummy variables**.

We create dummy features for each possible `genre`. As each song has one `genre`, each row will have a `1` in only one of the ten columns and `0` in the rest.

**Benefit:** We can now pass categorical features to models as well.

</details>

</details>

<details>

<summary>What is one problem of this approach?</summary>

### Dropping one of the categories per feature

If a song is not any of the first `9` genres, then implicitly, it is a `Rock` song. That means we only need nine features, so we can delete the `Rock` column.

If we do not do this, we are duplicating information, which might be an issue for some models (we're essentially introducing linear dependence - if I know the values for the first `9` columns, I for sure know the value of the `10`-th one as well).

![w05_dummies.png](./assets/w05_dummies_drop_first.png "w05_dummies.png")

</details>

### In `scikit-learn` and `pandas`

To create dummy variables we can use:

- the `OneHotEncoder` class if we're working with `scikit-learn`;
- or `pandas`' `get_dummies` function.

We will use `get_dummies`, passing the categorical column.

```python
df_music.head()
```

```console
   popularity  acousticness  danceability  duration_ms  energy  instrumentalness  liveness  loudness  speechiness       tempo  valence       genre
0          41        0.6440         0.823       236533   0.814          0.687000    0.1170    -5.611       0.1770  102.619000    0.649        Jazz
1          62        0.0855         0.686       154373   0.670          0.000000    0.1200    -7.626       0.2250  173.915000    0.636         Rap
2          42        0.2390         0.669       217778   0.736          0.000169    0.5980    -3.223       0.0602  145.061000    0.494  Electronic
3          64        0.0125         0.522       245960   0.923          0.017000    0.0854    -4.560       0.0539  120.406497    0.595        Rock
4          60        0.1210         0.780       229400   0.467          0.000134    0.3140    -6.645       0.2530   96.056000    0.312         Rap
```

```python
# As we only need to keep nine out of our ten binary features, we can set the "drop_first" argument to "True".
music_dummies = pd.get_dummies(df_music['genre'], drop_first=True)
music_dummies.head()
```

```console
   Anime  Blues  Classical  Country  Electronic  Hip-Hop   Jazz    Rap   Rock
0  False  False      False    False       False    False   True  False  False
1  False  False      False    False       False    False  False   True  False
2  False  False      False    False        True    False  False  False  False
3  False  False      False    False       False    False  False  False   True
4  False  False      False    False       False    False  False   True  False
```

Printing the first five rows, we see pandas creates `9` new binary features. The first song is `Jazz`, and the second is `Rap`, indicated by a `True`/`1` in the respective columns.

```python
music_dummies = pd.get_dummies(df_music['genre'], drop_first=True, dtype=int)
music_dummies.head()
```

```console
   Anime  Blues  Classical  Country  Electronic  Hip-Hop  Jazz  Rap  Rock
0      0      0          0        0           0        0     1    0     0
1      0      0          0        0           0        0     0    1     0
2      0      0          0        0           1        0     0    0     0
3      0      0          0        0           0        0     0    0     1
4      0      0          0        0           0        0     0    1     0
```

To bring these binary features back into our original DataFrame we can use `pd.concat`, passing a list containing the music DataFrame and our dummies DataFrame, and setting `axis=1`. Lastly, we can remove the original genre column using `df.drop`, passing the `columns=['genre']`.

```python
music_dummies = pd.concat([df_music, music_dummies], axis=1)
music_dummies = music_dummies.drop(columns=['genre'])
```

If the DataFrame only has one categorical feature, we can pass the entire DataFrame, thus skipping the step of combining variables.

If we don't specify a column, the new DataFrame's binary columns will have the original feature name prefixed, so they will start with `genre_`.

```python
music_dummies = pd.get_dummies(df_music, drop_first=True)
music_dummies.columns
```

```console
Index(['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',   
       'valence', 'genre_Anime', 'genre_Blues', 'genre_Classical',
       'genre_Country', 'genre_Electronic', 'genre_Hip-Hop', 'genre_Jazz',   
       'genre_Rap', 'genre_Rock'],
      dtype='object')
```

Notice the original genre column is automatically dropped. Once we have dummy variables, we can fit models as before.

## EDA with categorical feature

We will be working with the above music dataset this week, for both classification and regression problems.

Initially, we will build a regression model using all features in the dataset to predict song `popularity`. There is one categorical feature, `genre`, with ten possible values.

We can use a `boxplot` to visualize the relationship between categorical and numeric features:

![w05_eda.png](./assets/w05_eda.png "w05_eda.png")

## Handling missing data

<details>

<summary>How can we define missing data?</summary>

When there is no value for a feature in a particular row, we call it missing data.

</details>

<details>

<summary>Why might this happen?</summary>

- there was no observation;
- the data might be corrupt;
- the value is invalid;
- etc, etc.

</details>

<details>

<summary>What pandas functions/methods can we use to check how much of our data is missing?</summary>

We can use the `isna()` pandas method:

```python
# get the number of missing values per column
df_music.isna().sum().sort_values(ascending=False)
```

```console
acousticness        200
energy              200
valence             143
danceability        143
instrumentalness     91
duration_ms          91
speechiness          59
tempo                46
liveness             46
loudness             44
popularity           31
genre                 8
dtype: int64
```

We see that each feature is missing between `8` and `200` values!

Sometimes it's more appropriate to see the percentage of missing values:

```python
# get the number of missing values per column
df_music.isna().mean().sort_values(ascending=False)
```

```console
acousticness        0.200
energy              0.200
valence             0.143
danceability        0.143
instrumentalness    0.091
duration_ms         0.091
speechiness         0.059
tempo               0.046
liveness            0.046
loudness            0.044
popularity          0.031
genre               0.008
dtype: float64
```

</details>

<details>

<summary>How could we handle missing data in your opinion?</summary>

1. Remove it.
2. Substitute it with a plausible value.

</details>

<details>

<summary>What similar analysis could we do to find columns that are not useful?</summary>

We can check the number of unique values in categorical columns. If every row has a unique value, then this feature is useless - there is no pattern.

</details>

### Removing missing values

A common approach is to remove missing observations accounting for less than `5%` of all data. To do this, we use pandas `dropna` method, passing a list of columns to the `subset` argument.

If there are missing values in our subset column, **the entire row** is removed.

```python
df_music = df_music.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
df_music.isna().mean().sort_values(ascending=False)
```

```console
acousticness        0.199552
energy              0.199552
valence             0.142377
danceability        0.142377
speechiness         0.059417
duration_ms         0.032511
instrumentalness    0.032511
popularity          0.000000
loudness            0.000000
liveness            0.000000
tempo               0.000000
genre               0.000000
dtype: float64
```

Other rules of thumb include:

- removing every missing value from the target feature;
- removing columns whose missing values are above `65%`;
- etc, etc.

### Imputing missing values

> **Definition:** Making an educated guess as to what the missing values could be.

Which value to use?

- for numeric features, it's best to use the `median` of the column;
- for categorical values, we typically use the `mode` - the most frequent value.

<details>

<summary>What should we do to our data before imputing missing values?</summary>

We must split our data before imputing to avoid leaking test set information to our model, a concept known as **data leakage**.

</details>

Here is a workflow for imputation:

```python
from sklearn.impute import SimpleImputer
imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)
```

For our numeric data, we instantiate and use another imputer.

```python
imp_num = SimpleImputer(strategy='median') # note that default is 'mean'
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
```

> **Definition:** Due to their ability to transform our data, imputers are known as **transformers**.

### Using pipelines

> **Definition:** A pipeline is an object used to run a series of transformers and build a model in a single workflow.

```python
from sklearn.pipeline import Pipeline

df_music = df_music.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
df_music['genre'] = np.where(df_music['genre'] == 'Rock', 1, 0)
X = df_music.drop(columns=['genre'])
y = df_music['genre']
```

To build a pipeline we construct a list of steps containing tuples with the step names specified as strings, and instantiate the transformer or model.

> **Note:** In a pipeline, each step but the last must be a transformer.

```python
steps = [('imputation', SimpleImputer()),
('logistic_regression', LogisticRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

## Centering and scaling

Let's use the `.describe().T` function composition to check out the ranges of some of our feature variables in the music dataset.

![w05_scaling_problem.png](./assets/w05_scaling_problem.png "w05_scaling_problem.png")

We see that the ranges vary widely:

- `duration_ms` ranges from `-1` to `1.6` million;
- `speechiness` contains only decimal values;
- `loudness` only has negative values.

<details>

<summary>What is the problem here?</summary>

Some machine learning models use some form of distance to inform them, so if we have features on far larger scales, they can disproportionately influence our model.

For example, `KNN` uses distance explicitly when making predictions.

</details>

<details>

<summary>What are the possible solutions?</summary>

We actually want features to be on a similar scale. To achieve this, we can `normalize` or `standardize` our data, often also referred to as scaling and centering.

As benefits we get:

1. Model agnostic data, meaning that any model would be able to work with it.
2. All features have equal meaning/contribution/weight.

</details>

## Definitions

Given any column, we can subtract the mean and divide by the variance:

![w05_standardization_formula.png](./assets/w05_standardization_formula.png "w05_standardization_formula.png")

- Result: All features are centered around `0` and have a variance of `1`.
- Terminology: This is called **standardization**.

We can also subtract the minimum and divide by the range of the data:

![w05_normalization_formula.png](./assets/w05_normalization_formula.png "w05_normalization_formula.png")

- Result: The normalized dataset has minimum of `0` and maximum of `1`.
- This is called **normalization**.

Or, we can center our data so that it ranges from `-1` to `1` instead. In general to get a value in a new interval `[a, b]` we can use the formula:

$$x''' = (b-a)\frac{x - \min{x}}{\max{x} - \min{x}} + a$$

## Scaling in `scikit-learn`

To scale our features, we can use the `StandardScaler` class from `sklearn.preprocessing`:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```

```console
19801.42536120538, 71343.52910125865
2.260817795600319e-17, 1.0
```

Looking at the mean and standard deviation of the columns of both the original and scaled data verifies the change has taken place.

We can also put a scaler in a pipeline!

```python
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps).fit(X_train, y_train)
```

and we can use that pipeline in cross validation. When we specify the hyperparameter space the dictionary has keys that are formed by the pipeline step name followed by a double underscore, followed by the hyperparameter name. The corresponding value is a list or an array of the values to try for that particular hyperparameter.

In this case, we are tuning `n_neighbors` in the `KNN` model:

```python
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
```

## How do we decide which model to try out in the first place?

### The size of our dataset

- Fewer features = a simpler model and can reduce training time.
- Some models, such as Artificial Neural Networks, require a lot of data to perform well.

### Interpretability

- Some models are easier to explain which can be important for stakeholders.
- Linear regression has high interpretability as we can understand the coefficients.

### Flexibility

- More flexibility = higher accuracy, because fewer assumptions are made about the data.
- A KNN model does not assume a linear relationship between the features and the target.

### Train several models and evaluate performance out of the box (i.e. without hyperparameter tuning)

- Regression model performance
  - RMSE
  - $R^2$

- Classification model performance
  - Accuracy
  - Confusion matrix
  - Precision, Recall, F1-score
  - ROC AUC

### Scale the data

Recall that the performance of some models is affected by scaling our data. Therefore, it is generally best to scale our data before evaluating models out of the box.

Models affected by scaling:

- KNN;
- Linear Regression (+ Ridge, Lasso);
- Logistic Regression;
- etc, etc, in general, every model that uses distance when predicting or has internal logic that works with intervals (activation functions in NN)

Models not affected by scaling:

- Decision Trees;
- Random Forest;
- XGBoost;
- Catboost;
- etc, etc, in general, models that are based on trees.
