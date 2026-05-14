# Recession Lookback: Using "Hard" Macroeconomic Data and "Soft" Market Sentiment for 6-Month Forward Forecasting

This project stems from a curiosity of whether 'Hard' economic data (e.g., Yield Curve spreads, Payrolls, CPI) or 'Soft' market sentiment (FinBERT analysis of S&P 500 headlines) lag or lead each other in predicting economic downturn. I also wanted to get a gauge of how good the two types of data is at predicting economic downturns - given the recent events such as COVID-19, rate hike cycles, etc. The results highlight the limitations of historical machine learning in the face of exogenous shocks (COVID-19) and unprecedented monetary tightening cycles (2022-2024)."

## Ingestion and Feature Engineering
### Hard Macroeconomic Data Pipeline
* Point-in-Time (PiT) Ingestion: Leveraged the fredapi to extract historical macroeconomic indicators (e.g., CPI, Payrolls, Unemployment, Yield Curves). Crucially, the pipeline queried "first-release" data using the realtime_start parameter to replicate the exact knowledge state of the market on the publication day, entirely eliminating look-ahead bias.
* Target Variable Formulation: Engineered a 6-month forward-looking binary classification target by applying a 6-month rolling maximum window to the NBER USREC indicator and shifting the results backward. This trained the model to predict if a recession will begin at any point within the next half-year rather than just the current month.
* Stationarity & Unit Root Treatment: Absolute macroeconomic metrics intrinsically drift over time (non-stationarity). Augmented Dickey-Fuller (ADF) tests were deployed to identify unit roots across the dataset.
* Momentum Transformations: To resolve non-stationarity, cumulative data (e.g., CPI, Industrial Production) were transformed into Month-over-Month (MoM) and Year-over-Year (YoY) percentage changes, while rate-based data (e.g., Treasury Yields) were converted into absolute differences. For deeply trending features like YoY CPI, the 1st-order difference of the rate was calculated to capture inflation acceleration rather than absolute levels.

### Soft Market Sentiment Pipeline
* NLP Sentiment Extraction: Processed over 19,000 historical S&P 500 financial headlines (2008–2024) using the Hugging Face ProsusAI/finbert transformer pipeline to compute Positive, Negative, and Neutral sentiment probabilities for each headline.
* Directional Signal Filtration: Because financial journalism often states objective facts resulting in high "Neutral" FinBERT scores, a "Neutrality Ratio" (Positive / (Positive + Negative)) was engineered. This effectively stripped out journalistic noise and isolated the market's true directional bias.
* Exponential Decay Aggregation: Daily sentiment scores were aggregated into a rolling 30-day Exponential Moving Average (EMA). This transformation mirrors real-world market microstructure, weighting recent news shocks significantly more heavily than stale information.
* Unified Dataset Alignment: Transformed hard and soft datasets were merged on a standardized, 1st-of-the-month aligned_date (and later aligned_week) index to simulate a live, production-ready inference environment.

## Methodology
* Hard Data (XGBoost): Extracted policy pillars (CPI, BAA10Y) and expectation pillars (T10Y3M, T10Y2Y). Handled the severe class imbalance of recessions using scale_pos_weight.
* Soft Data (LightGBM & FinBERT): Extracted financial headlines from 2008-2024, utilized FinBERT for sentiment scoring, and applied a 30-day Exponential Moving Average (EMA) and a "Neutrality Ratio" to cut through the noise of objective financial journalism.
* Calibration: Used Platt Scaling (Sigmoid Calibration) via CalibratedClassifierCV to convert raw model outputs into reliable, ensemble-ready probabilities for unseen data (2018-2024).
## Conclusion
* Finding 1: The Crowd Anticipates the Data (Sentiment Leads)
    * Through Time Lagged Cross Correlation (TLCC) between the Neutrality Ratio (Soft Data) and Unemployment Rate MoM Difference (Hard Data), the project mathematically proves a peak correlation at a lag of +4.
    * Macro Takeaway: Market sentiment and news headlines act as a 4-month leading indicator for physical employment output. The market "prices in" the economic reality before it hits the lagging hard data.
* Finding 2: The Limits of Historical ML on Exogenous Shocks (The 2020 Miss)
    * The model failed to forecast the March 2020 COVID-19 crash.
    * Macro Takeaway: A model trained on endogenous credit cycles, yield curves, and inflation cannot predict an exogenous global pandemic. However, as the Fed injected trillions in stimulus, the model correctly read the "Hard Data" intervention and predicted a rapid recovery 6 months forward.
* Finding 3: The Unprecedented Tightening Cycle (The 2022-2024 False Alarm)
    * The model persistently hallucinated a recession throughout 2023 and 2024.
    * Macro Takeaway: In 2022-2023, the 10Y-3M Yield curve experienced its deepest inversion since the 1980s alongside 9% inflation. The model learned from 2001 and 2008 that this always triggers a recession. It lacked the historical context that pandemic-era cash buffers and structural labor shortages would mask the traditional liquidity squeeze, resulting in a false alarm.

## Future Work 
* The COVID Recession: How do we address "Exogenous Shocks"?
    * Shift toward "Nowcasting" by incorporating high-frequency (daily/weekly) alternative data sets: real-time mobility data, maritime shipping volumes, weekly retail transaction APIs, to measure the velocity of sudden economic halts and recoveries in real-time
    * The goal is to bridge the gap between monthly government data releases.
* The 2023-2024 "False Alarm": Correcting the Macro Premise
    * To accurately predict "Soft Landings," the feature space must be expanded to include Balance Sheet and Fiscal indicators, such as M2 Money Supply velocity, Excess Household Savings, and Fiscal Deficit ratios, allowing the model to weigh interest rate pressure against available systemic liquidity.
* Transitioning to Regime-Switching Models (HMM)
    * Instead of a binary output, the engine will classify the economy into dynamic states (e.g., Reflation, Stagflation, Soft Landing, Contraction), allowing portfolio managers and corporate planners to adapt their asset allocation and inventory strategies dynamically based on the prevailing macro regime.