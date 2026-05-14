# Recession Lookback: Using "Hard" Macroeconomic Data and "Soft" Market Sentiment for 6-Month Forward Forecasting

This project stems from a curiosity of whether 'Hard' economic data (e.g., Yield Curve spreads, Payrolls, CPI) or 'Soft' market sentiment (FinBERT analysis of S&P 500 headlines) lag or lead each other in predicting economic downturn. I also wanted to get a gauge of how good the two types of data is at predicting economic downturns - given the recent events such as COVID-19, rate hike cycles, etc. The results highlight the limitations of historical machine learning in the face of exogenous shocks (COVID-19) and unprecedented monetary tightening cycles (2022-2024)."

## Ingestion and Feature Engineering

## Methodology

## Conclusion

## Future Work 
* The COVID Recession: How do we address "Exogenous Shocks"?
    * Shift toward "Nowcasting" by incorporating high-frequency (daily/weekly) alternative data sets: real-time mobility data, maritime shipping volumes, weekly retail transaction APIs, to measure the velocity of sudden economic halts and recoveries in real-time
    * The goal is to bridge the gap between monthly government data releases.
* The 2023-2024 "False Alarm": Correcting the Macro Premise
    * To accurately predict "Soft Landings," the feature space must be expanded to include Balance Sheet and Fiscal indicators, such as M2 Money Supply velocity, Excess Household Savings, and Fiscal Deficit ratios, allowing the model to weigh interest rate pressure against available systemic liquidity.
* Transitioning to Regime-Switching Models (HMM)
    * Instead of a binary output, the engine will classify the economy into dynamic states (e.g., Reflation, Stagflation, Soft Landing, Contraction), allowing portfolio managers and corporate planners to adapt their asset allocation and inventory strategies dynamically based on the prevailing macro regime.