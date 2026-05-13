# recession_lookback

**Core Objective:** To build, evaluate, and compare two distinct machine learning models—one driven by macroeconomic indicators and one driven by real-time text sentiment—to predict a US economic recession 6 months into the future, culminating in an interactive simulation tool. The project is a one time "point-in-time" snapshot, there will be no need to continually update the data in the back end - it is merely a proof of concept and knowledge. So all the API calls should just be one time. 

---
### Phase 1: The "Hard Data" Macro Model
This track focuses on the traditional, quantitative approach to economic forecasting.

**Required APIs:** 
- `fredapi` (Federal Reserve Economic Data) for features; NBER (National Bureau of Economic Research) for the target variable (0 or 1 for recession).
**Data Ingestion:** 
- The American Federal Reserve Operates on a dual mandate: maximum employment and stable prices. Ideally, our macro model should encompass the factors that represent this dual mandate. [[Data Points Behind Fed Macro Decisions]]. Note: ***Macro data is not published on the 1st of the month. February's employment data, for example, is released in early March. When doing your historical train/test split, you must align your rows based on when the data was known to the public, not the month the data describes. Otherwise, you are leaking future information into your model.***
- 10y-3m Yield Curve
	- Spread between the interest rate on 10-year Treasury bonds and 3-month Treasury bills. Normally, long-term bonds yield more than short-term bonds.
		- Normal/Positive (1% or higher): Indicates a healthy economy. Investors demand higher yields for locking money away for 10 years compared to 3 months.
		- Flattening (but decreasing): Suggests economic growth is slowing or the Fed is raising short-term rates.
		- Inverted/Negative: A strong recession warning. Investors believe the future economy will be weak, leading to lower rates later, making the short-term risky and unattractive compared to the long-term. 
	- The spread is often affected by the tailends of the bond curve e.g. more people buying 10 y bonds now to lock in rates = lower/negative spread / more people selling 10 y bonds to buy other things = positive spread because of the rising yield due to cheaper 10 y bond prices
- Consumer Price Index (CPI)
	- If CPI runs hot (like in 2022), the Fed is forced to raise interest rates to cool the economy down. Raising rates aggressively often triggers a recession.
- Non-Farm Payrolls
	- NFP measures the absolute number of jobs added, showing raw economic momentum.
- Unemployment Rate
	- Unemployment Rate is a percentage that highlights labor market slack. A sudden uptick in the unemployment rate (even a small one, known as the "Sahm Rule") is a highly reliable recession trigger.
- Industrial Production Index (IPI) 
	- This measures the real output of manufacturing, mining, and utilities. Manufacturing is highly cyclical; it usually contracts _before_ the service sector does, making it an excellent leading indicator.
- Credit Spreads (e.g., Baa Corporate vs. 10y Treasury)
	- This measures how much extra yield investors demand to hold risky corporate debt over safe government debt. When spreads "widen" (go up), it means banks and investors are scared corporations will default.
**Feature Engineering:** 
- Calculate month-over-month (MoM) and year-over-year (YoY) momentum for each non-stationary indicator to capture velocity rather than just absolute values.
	- Macro data like Total Payrolls goes up indefinitely over time. Machine learning models struggle with this non-stationary data. You must convert these into Month-over-Month (MoM) or Year-over-Year (YoY) percentage changes. Sentiment scores, however, are usually bounded (e.g., -1 to 1) and are naturally stationary, so they don't need a MoM conversion.
	- Add the **Augmented Dickey-Fuller (ADF) Test**: a formal statistical test where the null hypothesis is that the time series is non-stationary. You will run this test on your features to quantitatively prove to your evaluators that your transformations successfully stabilized the data before feeding it to your models.
- To account for imbalance you should avoid using SMOTE in time-series forecasting, to avoid look ahead bias 
	- Instead, use scale_pos_weight parameter build into XGBoost to penalise the model more for a false negative (missing a recession) than incorrectly guessing a recession (false positive) - such that the minority class is more balanced in the voting.
**Model Selection:** 
- Logistic Regression Model
	- Standard Logistic Regression struggles heavily with multicollinearity by assigning massive weights to both to cancel them out (when features are highly correlated with each other e.g. Non-Farm Payrolls and the Unemployment Rate are highly negatively correlated.)
	- To avoid this you can add a penalty to the loss functions for having large weights.
		- L1 Regularisation (Lasso): This applies a penalty equal to the absolute value of the magnitude of coefficients. $$Penalty = \lambda \sum_{i=1}^{n} |w_i|$$
			-  It is a ruthless feature selector. If two features are highly correlated, Lasso will often shrink the weight of one feature entirely to $0$ and keep the other. It essentially says, "I only need one of these to make my point."
		- **L2 Regularization (Ridge):** This applies a penalty equal to the square of the magnitude of coefficients. $$Penalty = \lambda \sum_{i=1}^{n} w_i^2$$
			- It shrinks all coefficients down evenly but rarely forces them to exactly zero. If it sees NFP and Unemployment, it will share the weight between them. Ridge is generally the go-to solution for dealing with multicollinearity in standard datasets.
	- If you use LogisticRegression from scikit-learn, it actually applies L2 (Ridge) regularization by default
- XGBoost model 
	- Given I want to predict a 6 month or a 3 month forecast, I need to change the target variable when forecasting
	- As $y_t$ for month $t$ is 1 if there is a recession in any of the subsequent 6 months where $R_i \in \{0, 1\}$ is the NBER recession indicator for month $i$ $$y_t = \max(R_{t+1}, R_{t+2}, \dots, R_{t+6})$$
	- Note: while the NBER label is the historical gold standard for training, a real-time production model cannot rely on it for the most recent 12 months of training data because the label is technically unknown (a "ghost" label). In a live trading environment, you would substitute the most recent labels with a mechanical proxy, such as two consecutive quarters of negative GDP growth or the triggering of the Sahm Rule.
**Testing Protocol:** 
- Use walk-forward time-series cross-validation. Evaluate using Precision, Recall, and the ROC-AUC score, as recessions are highly imbalanced minority classes.
**Key Considerations:** 
- Macroeconomic data is subject to "look-ahead bias" due to retrospective revisions. You must acknowledge that training on revised historical data gives the model an unnatural advantage compared to real-time forecasting.

---
### Phase 2: The "Soft Data" Sentiment Model
This track captures the psychological and reactionary "vibes" of the market. Note: to avoid this project from turning into a bot-evading scraping project I evaded google trends, twitter, reddit, etc.

Dataset: [S&P 500 with Financial News Headlines (2008–2024)](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024)

Processing
- Load and Sanitise: Ingest the historical headline data. Crucially, immediately drop the CP (S&P 500 Closing Price) column. Keeping this column would introduce massive data leakage, as the model would learn to track stock market momentum rather than pure text sentiment, ruining the Phase 3 head-to-Head evaluation.
- Timestamp Alignment: Map the daily headline dates to ensure they align cleanly with the Point-in-Time (PiT) monthly NBER recession labels, preserving the strict 6-month forward forecasting rule.

Feature Engineering
- Granular FinBERT Processing: Pass every individual headline through FinBERT (via Hugging Face) to generate baseline probability scores (Positive, Negative, Neutral). Do not concatenate multiple daily headlines into one string, as FinBERT relies on short-context sentences. Instead, process them individually and calculate the daily mean of the probabilities.
- The "Fear/Greed" Index (SMA vs. EMA): Aggregate the daily scores into a rolling 30-day sentiment index. Use an Exponential Moving Average (EMA) rather than a Simple Moving Average (SMA), ensuring that recent market panic heavily outweighs month-old news.
- The Neutrality Ratio Filter: Because short financial headlines lack deep context, FinBERT will default to "Neutral" frequently, causing the baseline index to flatline. Engineer a feature that calculates the ratio of Positive / (Positive + Negative) probabilities. This mathematically strips out the objective noise and isolates the true directional bias of the media.
- Accounting for the "Bad News is Good News" Paradox: Financial journalism often exhibits a quirk where bad macro data (e.g., job losses) generates "Positive" S&P 500 headlines because the market expects Fed rate cuts. This psychological divergence will be a key talking point when evaluating the model's feature importance compared to the Phase 1 "Hard Data" model.

Model Selection: LightGBM: Train a Gradient Boosting Machine (LightGBM) using purely the historical rolling text sentiment features (EMA, Positive/Negative ratios, momentum) to predict the 6-month forward NBER recession label. LightGBM is chosen for its high efficiency and strong handling of dense, continuous engineered features.

---
### Phase 3: Conclusion Synthesis
This is where you bridge the two tracks to answer a fundamental financial research question: _Does the crowd anticipate the data, or react to it?_

One of the main focuses for the data engineering is matching the release dates. In quantitative finance, this is called using Point-in-Time (PiT) data. For example, the Unemployment Rate for January is usually released on the first Friday of February. If your model is trying to predict a recession using data available on January 31st, it cannot use January's unemployment rate because the public didn't know it yet. It must use December's rate (which was released in early January). If you align your "soft data" (daily tweets/news) with the release date of the "hard data," you will completely avoid look-ahead bias.

Implementing Polymarket and Kalshi
- **Required APIs:** Polymarket CLOB (Central Limit Order Book) API or Kalshi API.
- **Target Contracts:** Look for contracts like "US Recession in 2023", "US Recession in 2024", or "Fed Interest Rate by [Month]".
- **Implied Probability:** The raw price of a "Yes" share (e.g., 20¢ = 20% probability of a recession).
	- **Market Volatility:** Calculate the 7-day and 14-day rolling standard deviation of the contract's price. This isolates the specific macroeconomic uncertainty.

Data focused synthesis: The Lead/Lag Hypothesis
- **Time-Lagged Cross-Correlation (TLCC):** Calculate the correlation coefficient between your "Soft Data" Sentiment Index and your "Hard Data" indicators (like Unemployment) across different time shifts. If sentiment shifted forward by two months correlates perfectly with a spike in unemployment, sentiment _leads_. 
	- We run this on the underlying cleaned data/features
	- For this you must ensure you are correlating the MoM or YoY changes of your macro data, not the absolute numbers. Correlating a bounded sentiment score with a perpetually rising GDP number will result in mathematical garbage
		1. Take your Monthly Sentiment Score array and your MoM Unemployment Change array. 
		2. Calculate the correlation when Lag = `0 (same month). 
		3. Shift the Sentiment array back by 1 month (Lag = -1) and calculate. 
		4. Shift the Sentiment array forward by 1 month (Lag = +1) and calculate. 
		5. Plot these correlation values on a bar chart. If the highest correlation occurs at Lag = +2 (meaning sentiment from two months ago highly correlates with the unemployment change today), you have quantitative proof that sentiment _leads_ the hard data.
- **Granger Causality Test:** Run a formal statistical test to determine if past values of your Sentiment Index provide statistically significant predictive power for future macroeconomic data, beyond what the macroeconomic data's own history provides.

Predictive Synthesis
- **The Head-to-Head Model Evaluation:**
	- Your models are competing to see which domain (hard economics vs. soft psychology) is actually better at predicting a 6-month forward NBER recession.
	- **The Execution:** You will plot the ROC (Receiver Operating Characteristic) curves of your Macro XGBoost model and your Sentiment LightGBM model on the same graph.
	- **The Insight:** If the Sentiment model has a higher AUC (Area Under the Curve) than the Macro model, you have a massive talking point for your portfolio: you proved that tracking Reddit/Twitter panic is mathematically superior to tracking the Federal Reserve's data for this specific forecasting window.
-  **The Ensemble:**
	- In the real world, hedge funds rarely rely on one type of data. In Phase 3, you should combine the two models to create an "Ensemble Model."
	- **The Execution:** Before you combine these models in an ensemble, you must pass them through a **Probability Calibration** step. Why?
		- Imagine a single tree in your XGBoost model trying to classify historical months. It splits the data based on the Yield Curve, then splits it again based on Unemployment. Finally, the data lands in an end node (a "leaf").
			- The leaf may contain 10 months, but actually only 8 were recessions and 2 were not. The model shows that this leaf has a raw score of 0.8. The XGBoost model then combines all these trees and aggregates all the leaf scores and pushes it through a math function to give you a final number between 0 and 1
			- If XGBoost finds a few strong features (like an inverted yield curve), it will rapidly push its output score toward the extremes (like 0.95 or 0.05), even if the real-world probability isn't actually that certain.
		- So to calibrate take a chunk of data the model has never seen before (your validation set). You run it through your trained XGBoost model and get the raw output scores.
		- You look at the model's outputs and group them into buckets (e.g., 0%–10%, 10%–20%, ..., 70%–80%).
		- Let's focus on the 70% to 80% bucket. Let's say there are 100 different months in your validation set where XGBoost output a score between 0.70 and 0.80. The average score in this bucket is 0.75. The model is essentially claiming, "I am 75% confident a recession is coming for these months."
		- Now, you look at the actual NBER historical data for those exact same 100 months. You ask: Out of these 100 months where the model was 75% confident, how many actually resulted in a recession? Because XGBoost is overconfident, you might find that only 50 of those months actually resulted in a recession.
			- The Model's Score: 0.75 
			- The Real-World Probability: 0.50
		- Calibration is the process of building a mathematical "translator" (often using a technique called Isotonic Regression or Platt Scaling) to fix this.
			- You apply this translator to the end of your model. From now on, whenever your XGBoost model looks at new data and confidently spits out a raw score of 0.75, your calibration layer intercepts it, says "I know you're overconfident," and adjusts the final output to 0.50.
		- You need to apply this for all classification models that output probability.
			- Tree-Based Ensembles (XGBoost, Random Forest, LightGBM) / Support Vector Machines (SVMs) / Naive Bayes / Modern Deep Neural Networks
	- **The Insight:** Often, combining uncorrelated models yields a final model that is vastly superior to either individual model. 
- **Explaining the Model Behaviour (SHAP):**
	- While the Granger Causality test tells you if sentiment leads unemployment in the _data_, your models tell you what actually triggers a _recession prediction_.
	- **The Execution:** You run SHAP values on your final models.
	- **The Insight:** You might find that the Sentiment Index is the most important feature for predicting a recession _6 months out_, but the Yield Curve is the most important feature for predicting a recession _1 month out_.
	- **Output:** A clear, data-backed conclusion for your portfolio detailing whether news and retail sentiment acts as an early warning system or merely a lagging echo chamber.
---
### Phase 4: The Interactive Simulator UI
A model sitting in a Jupyter Notebook is invisible. Building a front-end application proves your engineering and deployment capabilities.
**Framework & Tools**
- **Library:** Streamlit or Dash (Python-based, perfect for data science web apps).
	- use Streamlit's **`@st.cache_resource`** and **`@st.cache_data`** to load models and datasets in the background
- **Deployment:** Streamlit Community Cloud
**UI Components**
- Two tabs: one for the data-focused synthesis, and one for the predictive synthesis
	- Data-focused Synthesis Tab
		- Should be a study of the correlation of the two sets of data based on the two tests
	- Predictive Synthesis Tab
		- Model Section
			- Displaying the most recent data I include, and the subsequent 6 month probability of a recession based on both models and perhaps the ensemble model
			- **The "What-If" Sliders:** Interactive sliders for adjusting each and every metric I have incorporated 
			- **Feature Importance Chart:** A dynamic SHAP (SHapley Additive exPlanations) waterfall chart that updates alongside the sliders to show _why_ the model is outputting that specific probability.
				- Look into optimizing the `TreeExplainer` or pre-calculating the background datasets to keep the UI snappy.
		- Explanation Section
			- The head to head results, and how the ensemble was created. 
			- The difference in accuracy and precision of each model individually vs logistic regression. 
			- The difference in accuracy and precision of ensemble vs the individual models

---