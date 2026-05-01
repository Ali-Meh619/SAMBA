# 📊 SAMBA Dataset & Feature Construction

Welcome to the SAMBA dataset directory. This folder contains the historical stock market datasets and the automated feature builder (`samba_feature_builder.py`) used to reconstruct the 82-variable input space required by the SAMBA and CNNpred models.

## 📖 Overview

The datasets capture the complex, interconnected nature of global financial markets. To predict the directional movement of a target index (e.g., S&P 500, NASDAQ, DJI), the model looks far beyond the target's own price history. 

Each sample (representing a single trading day) is composed of exactly **82 features**:
*   **14-15 Local Features**: Primitive data and technical indicators derived directly from the target market.
*   **67-68 External Features**: Macroeconomic variables, global indices, commodities, exchange rates, and leading stocks.

> **Note on Feature Count:** To maintain a strict 82-node graph structure across different target markets, the target's *own* external return column is dynamically dropped and replaced by the `Name` column. This ensures exactly 82 numeric features are fed into the network after preprocessing.

---

## 🧬 Feature Dictionary (The 82 Variables)

The variables are grouped into distinct categories based on their financial origin and purpose, mirroring the original CNNpred paper specifications.

### 1. Primitive Variables & Technical Indicators (Local)
Derived solely from the target market's historical OHLCV data.
*   **Primitives**: `Price` (Raw Close), `Vol.` (Relative change in volume), `weekday` (Day of the week, 0-4).
*   **Momentum (MOM)**: `mom` (1-day return), `mom1`, `mom2`, `mom3` (lagged returns representing past momentum).
*   **Rate of Change (ROC)**: `ROC_5`, `ROC_10`, `ROC_15`, `ROC_20` (Percentage change over specific window lengths).
*   **Exponential Moving Average (EMA)**: `EMA_10`, `EMA_20`, `EMA_50`, `EMA_200` (Trend-following smoothing filters).

### 2. World Stock Indices
Tracks the globalization of economies and timezone-lagged impacts from foreign markets.
*   `IXIC` (NASDAQ), `GSPC` (S&P 500), `DJI` (Dow Jones), `NYSE`, `RUT` (Russell 2000)
*   `FTSE` (UK), `GDAXI` (Germany DAX), `FCHI` (France CAC 40)
*   `HSI` (Hong Kong), `SSEC` (Shanghai), `Nikkei-F` (Japan)

### 3. Exchange Rates (USD vs. World)
Fluctuations impact the profitability of multinational corporations which make up major U.S. indices.
*   `JPY`, `GBP`, `CAD`, `CNY`, `AUD`, `NZD`, `CHF`, `EUR`
*   `Dollar Index`, `Dollar Index-F` (Futures)

### 4. Commodities
Reflects the global economic health, inflation, and raw material costs.
*   **Energy**: `WTI-oil`, `Brent`, `oil`, `GAS-F` (Natural Gas)
*   **Metals**: `Gold-F` (Gold Futures), `XAU` (Gold Spot), `silver-F`, `XAG` (Silver Spot), `copper-F`
*   **Agriculture**: `wheat-F`

### 5. Major U.S. Companies
Bellwether stocks that carry significant weight in index calculations or dictate sector trends.
*   `AAPL` (Apple), `MSFT` (Microsoft), `AMZN` (Amazon)
*   `XOM` (Exxon Mobil), `JPM` (JPMorgan), `WFC` (Wells Fargo)
*   `GE` (General Electric), `JNJ` (Johnson & Johnson)

### 6. Futures Contracts
Provides the market's expected future value of indices.
*   `NASDAQ-F`, `S&P-F`, `DJI-F`, `RUSSELL-F`
*   `FTSE-F`, `HSI-F`, `CAC-F` (FCHI-F), `DAX-F`, `KOSPI-F`

### 7. Treasury Rates & Spreads
Macroeconomic debt indicators critical for assessing market risk and credit health.
*   **Bill/Bond Yields**: `DTB4WK`, `DTB3`, `DTB6`, `DGS5`, `DGS10`
*   **Corporate Yields**: `DAAA` (Aaa), `DBAA` (Baa)
*   **Yield Changes**: `CTB3M`, `CTB6M`, `CTB1Y`
*   **Term Spreads (TE)**: `TE1` to `TE6` (Differences between long-term bonds and short-term bills, e.g., DGS10 - DTB4WK).
*   **Default Spreads (DE)**: `DE1` to `DE6` (Differences between Baa corporate yields and safer assets. *Note: DE1 is corrected to DBAA - DAAA, fixing a typo in the original CNNpred paper*).

---

## ⚙️ Feature Builder (`samba_feature_builder.py`)

The python script located in this folder is responsible for converting raw data downloads into the perfectly aligned, 82-feature CSVs used for training.

### Legacy vs. Causal Modes
The script supports two distinct alignment modes:

*   **Legacy Mode (Original CNNpred/SAMBA Replica)**
    ```bash
    python samba_feature_builder.py --target-csv target.csv --mode legacy --external-alignment reverse-position
    ```
    Replicates the exact mathematical quirks (e.g., reversed EMA smoothing, row-position-based alignment) of the original published CNNpred datasets. Use this to achieve bit-for-bit parity with the legacy benchmarks.

*   **Causal Mode (Chronologically Pure)**
    ```bash
    python samba_feature_builder.py --target-csv target.csv --mode causal --external-alignment date
    ```
    Enforces strict chronological sorting and standard forward-looking TA-Lib style calculations (e.g., preventing data leakage from future dates). **Use this mode when applying the model to new, real-world data or extending the dataset to present day.**

### Example Output
The builder produces files like `combined_dataframe_IXIC.csv`, matching the structure required by `utils/data_utils.py` for PyTorch sequence loading.