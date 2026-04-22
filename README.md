# AI/ML Engineering Tasks

## Task 1: Exploring and Visualizing a Simple Dataset

### Objective

Perform foundational data inspection and statistical visualization to map feature distributions and understand the dataset.

### Dataset

* Iris Dataset (Loaded via Seaborn)

* Contains botanical attributes for classifying three distinct Iris species. Features include:

    * Sepal Length (cm)
    * Sepal Width (cm)
    * Petal Length (cm)
    * Petal Width (cm)

### Execution

* **Data Profiling:** Validated dimensionality and zero null values using Pandas.
* **Visualization:** Mapped correlations via Pair plots and detected IQR outliers via Box plots.

### Key Insights
1.  Strong linear correlation exists between Petal Length and Petal Width.
2.  **Iris setosa** is linearly separable; **versicolor** and **virginica** share overlapping feature boundaries.

### Tech Stack
* Python
* Pandas
* MatplotLib
* Seaborn

## Task 2: Predict Future Stock Prices (Short-Term)

### Objective

Forecast the next-day closing price of a selected equity (AAPL) using historical market data, moving averages, and machine learning regression techniques.

### Dataset

* Source: Live market data retrieved dynamically via the yfinance API.
* Core Features: Open, High, Low, Close, Volume (OHLCV).
* Engineered Features: 10-day and 20-day Moving Averages (MA_10, MA_20).
* Target Variable: Next-day percentage return (shifted chronologically to prevent data leakage).
