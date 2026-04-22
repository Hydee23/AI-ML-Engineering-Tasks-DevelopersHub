# AI/ML Engineering Tasks

## 1️⃣ Task 1: Exploring and Visualizing a Simple Dataset

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

## 2️⃣ Task 2: Predict Future Stock Prices (Short-Term)

### Objective

Forecast the next-day closing price of a selected equity (AAPL) using historical market data, moving averages, and machine learning regression techniques.

### Dataset

* **Source:** Live market data via the **yfinance API** (OHLCV features).
* **Core Features:** Open, High, Low, Close, Volume (OHLCV).
* **Engineered Features:** 10-day & 20-day Moving Averages.
* **Target Variable:** Next-day percentage return (shifted to prevent data leakage).

### Execution

* **Data Engineering:** Ingested live API data and engineered moving average features.
* **Modeling:** Trained a Random Forest Regressor using a strict chronological 80/20 split.
* **Evaluation:** Assessed model accuracy via RMSE and plotted 1-step forward predictions.

### Result

**RMSE:** $2.74 (model's average prediction error per day)

### Key Insights

1. Framing the target as percentage returns rather than absolute prices provides a mathematically sound, stationary target.
2. Chronological data splitting is mandatory; standard randomized splits cause catastrophic look-ahead bias in time-series data.

### Tech Stack

* Python
* Pandas & NumPy
* Scikit-Learn (Random Forest Regressor)
* Matplotlib
* yfinance (Data API)

## 4️⃣ Task 4: General Health Query Chatbot (Prompt Engineering Based)

### Objective

Build a conversational AI agent using Prompt Engineering to handle general health inquiries while strictly enforcing medical safety guardrails and API token constraints.

### Architecture

* **LLM Engine:** OpenAI API (gpt-3.5-turbo).
* **Core Logic:** Python CLI with stateful context management.

###Execution

* **Prompt Design:** Built a medical persona with strict negative constraints (no diagnosing or prescribing).
* **Memory Management:** Engineered a 10-message **"Sliding Window"** to preserve multi-turn context while mathematically preventing API token overflow.
* **Safety Guardrails:** Embedded routing logic to detect crisis triggers and redirect to human professionals.

###Key Insights

* **Negative Constraints:** Explicitly programming what the model cannot do is the primary defense for medical AI liability.
* **Token Stability:** Infinite memory loops eventually crash APIs; a sliding window is mandatory for production-safe conversational state.

### Tech Stack

* Python
* OpenAI API
* Prompt Engineering & Context Management
