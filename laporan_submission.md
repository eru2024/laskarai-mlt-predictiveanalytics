# Machine Learning Project Report - Booking Cancellation Prediction Model - Fatih El Haq

## Executive Summary

Booking cancellations pose a significant challenge for hotel operations, impacting revenue forecasts, resource allocation, and overall business efficiency. This project addresses two key objectives: identifying the most influential factors contributing to booking cancellations and developing a predictive model capable of accurately forecasting such events.

To begin, a structured feature selection process was implemented using statistical methods. Categorical variables were evaluated through Chi-Square tests, while numerical variables were assessed using correlation analysis. This ensured that only statistically relevant features were considered during model development.

Multiple classification algorithms were tested, including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Adaptive Boosting and Gradient Boosting. After evaluation using Precision, Recall, and F1-score, the Random Forest model emerged as the best performer with an F1-score of 0.91. This indicates a strong balance between identifying actual cancellations and avoiding false positives, which is crucial for operational decision-making.

To interpret the model and determine feature influence, permutation importance was applied using the F1-score as the performance baseline. The analysis revealed that bookings made from Portugal (PRT), lead time, and the total number of special requests were the top three predictors of cancellation. Bookings with longer lead times and those made through travel agencies showed a higher likelihood of cancellation, while reservations with more special requests were less likely to be canceled.

The combination of statistical analysis, machine learning modeling, and feature interpretation provides valuable insights for hotel management. By understanding cancellation patterns and deploying a reliable predictive model, hotels can improve resource planning, minimize revenue loss from late cancellations, and enhance overall booking strategy.

## Project Domain
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/background.jpg" alt="Problem Analysis Diagram" width="50%">
</div>
<br>

In hospitality industry, one of core issues that is usually faced by hoteliers and lodging owners are booking cancellations. Hotel and lodging owners are facing booking cancellation with more than 20% rate on average (D-EDGE Hospitality Solutions, 2024). Think about that – on average, more than one out of every five bookings made ends up getting cancelled. That's a significant amount of uncertainty for any business, and for hotels, it creates some real difficulties.


This high rate of cancellations leads to several **negative impacts on hospitality business**, which background diagram categorizes as **“Impact”**.
- **Significant Revenue Loss**: When a room is booked but then cancelled, that potential income is lost. Since a hotel room can't be saved and sold later like a physical product, that revenue is often gone for good (Antonio et al., 2019; Sánchez-Medina & C-Sánchez, 2020). Studies consistently show this directly impacts the bottom line.
- **Hindered Demand Forecasting and Planning**: Trying to accurately predict how many guests will actually arrive becomes very tough. This uncertainty makes it harder for hotels to plan everything from staffing levels and resources to setting the right prices, impacting overall efficiency and revenue management (Satu et al., 2020).
- **Increased Operational Inefficiencies and Costs**: Dealing with a high volume of cancellations, especially close to the arrival date, means more work and can disrupt daily operations. It can lead to situations where rooms sit empty that could have been used, making the hotel less efficient and potentially increasing costs (Chen et al., 2023; Satu et al., 2020).
- **Negative Impact on Customer Relationships** and Experience: Sometimes, the strategies hotels use to cope with cancellations, like very strict rules or the risk of overbooking, can unfortunately lead to unhappy guests, which isn't good for the hotel's reputation or future business (Satu et al., 2020; Sánchez-Medina and C-Sánchez, 2020).


So, what's behind these high cancellation rates? Background diagram points to several Root Causes of Booking Cancellations:
- **Flexible Cancellation Policies and Ease of Online Cancellation**: It's often very easy and sometimes free to cancel a booking, especially with online travel agencies. This makes it simple for travelers to book rooms even if their plans aren't set in stone (Sánchez-Medina and C-Sánchez, 2020).
- **Guest Behavior (Booking Multiple Options & Price Shopping)**: Many guests book several different places or dates, or they keep looking for better deals even after booking. If they find a more attractive option, they'll cancel their original reservation (Sánchez-Medina and C-Sánchez, 2020; Antonio et al., 2019).
- **External Factors & Changes in Travel Plans**: Unforeseen circumstances like sudden changes in personal plans, business trips getting cancelled, or even external events like bad weather or wider travel disruptions can force people to cancel their bookings (Antonio et al., 2019).
- **Lack of Effective Cancellation Prediction and Management**: This cause highlights that the problem's severity persists because hotels lack the ability to accurately predict which bookings will be cancelled and when. The inability to predict leads to uncertainty in demand forecasting, suboptimal revenue management, and inefficient operational planning, allowing the high cancellation rate to continue causing significant financial and operational harm (Sánchez-Medina and C-Sánchez, 2020; Antonio et al., 2019).


**Focusing on Cancellation Prediction**:

While external factors and guest motivations for canceling are difficult for hotels to directly control, the inability to anticipate cancellations renders hotels reactive rather than proactive in mitigating the ensuing negative impacts. The literature underscores that accurate cancellation forecasting is an underdeveloped area, and this lack of predictive capability directly contributes to the persistence of revenue loss, operational inefficiencies, and forecasting challenges (Sánchez-Medina and C-Sánchez, 2020; Antonio et al., 2019). By focusing on this root cause through predictive analytics, the aim is to equip hotels with the foresight needed to implement targeted interventions, thereby reducing uncertainty and mitigating the adverse effects of high cancellation rates.

## Business Understanding
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/business_understanding.jpg" alt="Problem Analysis Diagram" width="75%">
</div>
<br>

This project is focused on **Lack of Effective Cancellation Prediction** as one of root causes of high booking cancellation rate. The diagram above shows this project approach to address the root cause. This diagram illustrates a logical flow: recognizing that the Lack of Effective Cancellation Prediction is a core Root Cause of problems for hotels, identifying the key Problem Statements that need to be answered to enable prediction, setting clear Goals to address these problems, and outlining concrete Solution Statements (steps) using data analysis and machine learning techniques to achieve those goals. Explanation is below.

### Problem Statements

Stemming from this root cause are two key Problem Statements. These are the specific questions that arise because of the lack of effective prediction:

1. **What factors or features have the greatest influence on booking cancellations?** To predict cancellations, we first need to understand what makes a booking likely to be cancelled. This problem statement focuses on identifying the key pieces of information (features) that are most predictive.
2. **What is the best model to predict booking cancellations?** Once we understand the influencing factors, the next problem is determining the most effective method or algorithm to use those factors to make accurate predictions.

### Goals

To address these problem statements, the project defines specific Goals:

1. **Identify the factors or features that have the greatest influence on booking cancellations.** This goal directly corresponds to the first problem statement, aiming to pinpoint the most important predictors from the available data.
2. **Develop a machine learning model that can predict booking cancellations as best as possible.** This goal aligns with the second problem statement, focusing on building a high-performing predictive model.

### Solution Statements
Finally, the diagram outlines the Solution Statements, which are the practical steps planned to achieve the stated goals:
1. **Filter features based on appropriate statistical measures.** This involves using statistical techniques to evaluate the relevance and predictive power of different pieces of data (features) to help identify the most influential ones.
2. **Evaluate feature importance after selecting best model.** By using permutation importance, features are ranked based on importance mean (performance changes on each feature).
3. **Use at least five machine learning algorithms with hyperparameter tuning if needed.** To find the "best model," the approach involves experimenting with multiple machine learning techniques. Hyperparameter tuning is a process to optimize these algorithms for the specific dataset to achieve the best possible performance.
4. **Evaluate different machine learning algorithms by Precision, Recall, and F1-Score.** After training the models, their performance will be assessed using key evaluation metrics. Precision and Recall are particularly important for this problem to ensure the model correctly identifies actual cancellations (Recall) and that its predictions of cancellation are reliable (Precision), while Accuracy and F1-Score provide overall performance insights.

## Data Understanding
The [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset (119 390 records, 32 variables) captures detailed reservation information for two Portuguese properties—a city hotel and a resort hotel—over a 26-month span (July 1, 2015 through August 31, 2017). It was originally published alongside the “Hotel Booking Demand Datasets” article by Nuno António, Ana Almeida, and Luís Nunes (Data in Brief, Feb 2019) and later cleaned for the #TidyTuesday initiative by Thomas Mock and Antoine Bichat. All personally identifying fields have been removed, making it ideal for modeling cancellations, forecasting demand, pricing analysis, and understanding guest behavior patterns.

### Variable Description

Based on the journal article ["Hotel booking demand datasets"](https://doi.org/10.1016/j.dib.2018.11.126) as the dataset original source, there are 32 variables or features with discription below:

1. `hotel`. Categorical, Type of hotel (City Hotel or Resort Hotel)
2. `is_canceled`. Integer, Value indicating if the booking was canceled (1) or not (0)
3. `lead_time`. Integer, Number of days between booking date and arrival date
4. `arrival_date_year`. Integer, Year of arrival date
5. `arrival_date_month`. Categorical, Month of arrival date (January to December)
6. `arrival_date_week_number`. Integer, Week number of arrival date
7. `arrival_date_day_of_month`. Integer, Day of arrival date
8. `stays_in_weekend_nights`. Integer, Number of weekend nights (Saturday or Sunday) the customer stayed or booked to stay at the hotel
9. `stays_in_week_nights`. Integer, Number of week nights (Monday to Friday) the customer stayed or booked to stay at the hotel
10. `adults`. Integer, Number of adults
11. `children`. Integer, Number of children
12. `babies`. Integer, Number of babies
13. `meal`. Categorical, Type of meal booked. Categories are Undefined, BB (Bed & Breakfast), HB (Half Board), FB (Full Board)
14. `country`. Categorical, Country of origin. Categories are represented in the ISO 3166 – 1 alpha-3 format
15. `market_segment`. Categorical, Market segment designation. In categories such as Online TA, Offline TA/TO, Groups, Corporate, Complementary, Aviation
16. `distribution_channel`. Categorical, Booking distribution channel. Categories are Corporate, GDS, TA/TO, Direct, Undefined
17. `is_repeated_guest`. Integer, Value indicating if the booking person is a repeated guest (1) or not (0)
18. `previous_cancellations`. Integer, Number of previous bookings that were canceled by the customer prior to the current booking
19. `previous_bookings_not_canceled`. Integer, Number of previous bookings that were not canceled by the customer prior to the current booking
20. `reserved_room_type`. Categorical, Code of room type reserved. Codes are for illustration purposes only.
21. `assigned_room_type`. Categorical, Code of room type assigned. Codes are for illustration purposes only.
22. `booking_changes`. Integer, Number of changes/amendments made to the booking from the moment it was entered on the PMS until the moment of check-in or cancellation
23. `deposit_type`. Categorical, Indication on if the customer made a deposit to guarantee the booking. Categories are No Deposit, Non Refund, Refundable
24. `agent`. Categorical, ID of the travel agency that made the booking
25. `company`. Categorical, ID of the company/entity that made the booking or responsible for paying the booking
26. `days_in_waiting_list`. Integer, Number of days the booking was in the waiting list before it was confirmed to the customer
27. `customer_type`. Categorical, Type of customer, assuming one of four categories: Contract, Group, Transient, Transient-Party
28. `adr`. Numeric, Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights
29. `required_car_parking_spaces`. Integer, Number of required car parking spaces
30. `total_of_special_requests`. Integer, Number of special requests made by the customer (e.g. twin bed or high floor)
31. `reservation_status`. Categorical, Reservation status (Canceled, Check-Out, No-Show)
32. `reservation_status_date`. Date, Date at which the last status was set. This variable can be used in conjunction with the `is_canceled` variable to understand when a booking was really canceled.

### Missing Value and Outliers
| Feature  | Missing Values | Missing Handling | Notes                         |
|----------|----------------|------------------|-------------------------------|
| children | 4              | row removal      | small % of missing values     |
| country  | 488            | row removal      | small % of missing values     |
| agent    | 16,340         | column removal   | high cardinality              |
| company  | 112,593        | column removal   | high cardinality              |

Missing data was handled with two strategies based on the size and complexity of each gap. For **children** and **country**, the missing rates were very small (4 and 488 missing values out of 119 390 records). Those rows were dropped, preserving almost all information and avoiding unnecessary imputation.

In contrast, **agent** and **company** had high missing rates (≈13.7 % and ≈94.3 % of records) and thousands of unique values. Imputation or encoding would add complexity and risk overfitting. Those columns were removed entirely, focusing the model on features with more complete and informative data.

| Feature | Outlier                             | Handling      | Notes                                   |
|---------|-------------------------------------|---------------|-----------------------------------------|
| adr     | negative value                      | row removal   | most likely a data error                |
| adr     | 5400                                | winsorizing   | replace with second highest (510)       |
| adults  | Adults=0, Children=0, Babies=0      | row removal   | unrealistic combinations                |
| adults  | Adults=0, Children>0, Babies>0      | keep          | edge cases (school/child‐only bookings) |
| adults  | > 10                                | row removal   | rare occurrences                        |

Outliers in **adr** were handled by removing rows with negative values (likely data errors) and winsorizing extreme highs (5400 replaced with the second-highest value, 510) to limit the influence of improbable rates on the model.

For **adults**, rows with unrealistic combinations (0 adults, 0 children, 0 babies) and rare extreme cases (>10 adults) were removed because they either reflect data entry errors or occur too infrequently to support reliable learning. Cases with 0 adults but children and/or babies were retained to capture valid child-only or unaccompanied-minor bookings.

### Univariate Analysis
#### Categorical Features
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/univariate_categorical.jpg" alt="Problem Analysis Diagram" width="100%">
</div>
<br>

Categorical features exhibited diverse distributions: **hotel** bookings split between City (66.7 %) and Resort (33.3 %); **arrival_date_month** peaked in August (11.7 %) and July (10.6 %), reflecting seasonality; **meal** plans were dominated by BB (77.3 %) with smaller HB (12.2 %), SC (8.9 %) and rare Undefined/FB (<2 %); **market_segment** was led by Online TA (47.5 %) and Offline TA/TO (20.3 %), with Complementary/Aviation below 1 %; **distribution_channel** was overwhelmingly TA/TO (82.2 %); top two **reserved_room_type** and **assigned_room_type** (A and D) covered over 88 % of bookings; **deposit_type** was mostly No Deposit (87.6 %); **customer_type** was mainly Transient (75.0 %) and Transient-Party (21.1 %); **reservation_status** was chiefly Check-Out (62.8 %) and Canceled (36.2 %).

#### Numerical Features
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/univariate_numerical.jpg" alt="Problem Analysis Diagram" width="100%">
</div>
<br>

Numerical features display a variety of distribution shapes and scales. The cancellation flag (**is_canceled**) is bimodal, with roughly 63 000 non-cancellations and 43 000 cancellations, underscoring class imbalance considerations. **Lead_time** is heavily right-skewed: most bookings occur within 50 days of arrival, but a long tail extends past 500 days. Arrival dates by year cluster around 2016 and 2017, while week-of-year and day-of-month are nearly uniform, capturing seasonal patterns without obvious peaks (aside from a modest spike in week 27). Stay durations show that most guests book 0–2 weekend nights and 2–4 week nights, with extreme stays being rare. Party size features (**adults**, **children**, **babies**, **total_guests**) concentrate at 2 adults and zero children/babies, with very few large groups. Low-variance counts (e.g. **is_repeated_guest**, **previous_cancellations**, **booking_changes**, **days_in_waiting_list**, **required_car_parking_spaces**) are zero for the vast majority of records, indicating potential for binary encoding or capping.

Continuous variables like **lead_time**, **adr**, and **days_in_waiting_list** may benefit from log transformation or binning to reduce skew and stabilize variance. Rare extreme values in **stays_in_weekend_nights** or **stays_in_week_nights** can be capped or grouped into an “Other” category. Year and week-number fields can be treated as categorical to capture seasonal effects. Low-frequency numeric indicators (previous cancellations, special requests) could be recast as binary flags or binned into a few categories to simplify the feature set and prevent overfitting.  

### Multivariate Analysis
#### Categorical Features
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/multivariate_categorical.jpg" alt="Problem Analysis Diagram" width="100%">
</div>
<br>

Bar graphs for each categorical features show that:
- Only **deposit_type** that shows visually significant difference between canceled bookings and non-canceled bookings.
- bookings with is_canceled equal 1 has two **reservation_status** (Canceled or No Show). This shows redudancy between is_canceled and reservation_status

The signifance of correlation between categorical features and is_canceled can be evaluated on the difference within categorical features with Chi-Squared test.

| Variable                  | Chi² Statistic | P-Value | Degrees of Freedom | Conclusion                                                                                           |
|---------------------------|---------------:|--------:|-------------------:|------------------------------------------------------------------------------------------------------|
| hotel                     |      2,154.441 |   0.000 |                  1 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| arrival_date_month        |        554.097 |   0.000 |                 11 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| meal                      |        305.210 |   0.000 |                  4 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| market_segment            |      8,393.078 |   0.000 |                  6 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| distribution_channel      |      3,646.293 |   0.000 |                  4 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| reserved_room_type        |        638.159 |   0.000 |                  8 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| assigned_room_type        |      4,783.708 |   0.000 |                 10 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| deposit_type              |     27,515.935 |   0.000 |                  2 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| customer_type             |      2,281.962 |   0.000 |                  3 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |
| reservation_status        |    118,715.000 |   0.000 |                  2 | Reject H₀ – significant difference between is_canceled = 0 and is_canceled = 1.                       |

#### Numerical Features
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/multivariate_numerical.jpg" alt="Problem Analysis Diagram" width="100%">
</div>
<br>

The heatmap visualizes pairwise correlation between numerical features. Absolute value from the correlation value shows correlation strength between the pair of numerical features. For prediction model, it is more better to select features with high correlation with target variabel (for this project, is_canceled) and not select features with high correlation with other than target variabel due to potential correlation issue. For example, lead_time and total_of_special_requests are great features candidate for prediction model because of high correlation with is_canceled and low correlation with other features.

The signifance of correlation between numerical features and is_canceled can also be evaluated by statistical test on pearson correlation.

| Feature                          | Correlation | P-value        | Significance     |
|----------------------------------|-----------:|---------------:|:-----------------|
| lead_time                        |    0.291417 | 0.000000e+00   | Significant      |
| arrival_date_year                |    0.016635 | 9.933873e-09   | Significant      |
| arrival_date_week_number         |    0.007545 | 9.330803e-03   | Significant      |
| arrival_date_day_of_month        |   -0.005952 | 4.029552e-02   | Significant      |
| stays_in_weekend_nights          |   -0.002290 | 4.301691e-01   | Not Significant  |
| stays_in_week_nights             |    0.024727 | 1.584766e-17   | Significant      |
| adults                           |    0.058846 | 1.491891e-91   | Significant      |
| children                         |    0.004568 | 1.155424e-01   | Not Significant  |
| babies                           |   -0.032598 | 2.760390e-29   | Significant      |
| is_repeated_guest                |   -0.084100 | 2.940225e-185  | Significant      |
| previous_cancellations           |    0.109932 | 7.871143e-316  | Significant      |
| previous_bookings_not_canceled   |   -0.055488 | 1.346099e-81   | Significant      |
| booking_changes                  |   -0.145109 | 0.000000e+00   | Significant      |
| days_in_waiting_list             |    0.054134 | 9.477144e-78   | Significant      |
| adr                              |    0.046283 | 2.618103e-57   | Significant      |
| required_car_parking_spaces      |   -0.194998 | 0.000000e+00   | Significant      |
| total_of_special_requests        |   -0.235860 | 0.000000e+00   | Significant      |
| total_guests                     |    0.041802 | 4.529052e-47   | Significant      |

#### Geolocation Feature
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/bookings_country.jpg" alt="Problem Analysis Diagram" width="75%">
</div>
<br>

The dataset includes a geolocation feature represented by the `country` variable, indicating the country of origin for each booking. Upon analysis, it was found that bookings from Portugal (PRT) account for more than 60% of the total, significantly dominating the distribution. This high imbalance reduces the overall usefulness of the `country` variable in its original form for predictive modeling.

To address this, the `country` feature was simplified into a binary variable indicating whether the booking originated from Portugal (PRT) or not. This transformation retains the influence of the dominant country while reducing dimensionality and preventing overfitting during model training.

### Summary of EDA
<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/summary_eda.jpg" alt="Problem Analysis Diagram" width="100%">
</div>
<br>

Exploratory analysis identified strong associations between several categorical variables (hotel, meal, market_segment, room types, deposit_type, customer_type, reservation_status) and the target (`is_canceled`), confirmed by chi-square tests. Encoding strategies include binary or one-hot transformation for these features to preserve their predictive power while avoiding unnecessary dimensionality.

Numeric features with significant correlations (lead_time, stays, previous_cancellations, booking_changes, waiting_list days, ADR, parking, special requests, total_guests) were scaled to standardize distributions. Redundant or weak predictors (arrival_date_year, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, is_repeated_guest) were removed or replaced by derived features to reduce multicollinearity.

New binary and aggregated features were suggested to capture key behaviors:  
- `PRT` flag for Portugal bookings (over 60% of total)  
- `match_room_type` to unify reserved vs. assigned room categories  
- `has_weekend_stay`, `total_stay_duration` for combined weekend/weekday nights  
- `has_children`, `has_babies` instead of raw counts  
- `changes_made`, `was_on_waiting_list`, `parking_requested`, `has_special_requests` to reflect customer actions  

This summary shows balancing of dimensionality reduction, meaningful feature representation, and model robustness.

## Data Preparation

Based on Summary of EDA, data preparation for this project consists of add new features, scaling for numerical features, and encoding for categorical features. The data preparation detail can bee seen in tabel below.

| Feature                          | Preparation                                              | Reason                                                                          |
|----------------------------------|----------------------------------------------------------|---------------------------------------------------------------------------------|
| is_canceled                      | target variable                                          | defines prediction objective                                                    |
| hotel                            | binary encoding                                          | converts categories to numeric format                                           |
| arrival_date_month               | convert to numeric                                       | enables chronological ordering and model interpretation                         |
| meal                             | one-hot encoding (BB, HB, SC, Other)                     | preserves category distinctions without ordinal bias                            |
| market_segment                   | one-hot encoding (Online TA, Offline TA/TO, Direct, Corporate, Other) | captures booking channel effects                                    |
| reserved_room_type               | one-hot encoding (A, D, Other)                           | differentiates room preferences                                                 |
| deposit_type                     | one-hot encoding (No Deposit, Refundable, Non Refund)    | reflects payment commitment levels                                              |
| customer_type                    | one-hot encoding (Transient, Transient-Party, Other)     | models different customer behaviors                                             |
| lead_time                        | scaling                                                  | standardizes range for improved model convergence                               |
| arrival_date_week_number         | retain as-is                                            | captures seasonal booking patterns                                              |
| arrival_date_day_of_month        | retain as-is                                            | captures intra-month booking trends                                             |
| previous_cancellations           | retain as-is                                            | indicates cancellation history impact                                           |
| previous_bookings_not_canceled   | retain as-is                                            | balances cancellation history signal                                           |
| booking_changes                  | retain as-is                                            | measures booking flexibility                                                   |
| days_in_waiting_list             | retain as-is                                            | reflects booking demand vs. availability                                        |
| adr                              | scaling                                                  | normalizes price distribution                                                   |
| required_car_parking_spaces      | retain as-is                                            | indicates extra service requests                                                |
| total_of_special_requests        | retain as-is                                            | summarizes guest preferences                                                    |
| match_room_type                  | binary encoding (0: reserved ≠ assigned, 1: reserved = assigned) | detects room allocation discrepancies                                   |
| has_weekend_stay                 | binary encoding (0: no weekend nights, 1: any weekend nights) | highlights weekend travel behavior                                    |
| total_stay_duration              | stays_in_weekend_nights + stays_in_week_nights           | aggregates total nights stayed                                                  |
| has_babies                       | binary encoding (0: no babies, 1: babies > 0)            | simplifies family composition signal                                            |
| total_guests                     | retain as-is                                            | represents group size directly                                                  |
| PRT                              | binary encoding (0: country ≠ 'PRT', 1: country = 'PRT')  | captures dominant market segment (Portugal > 60% bookings)                       |

Before model training, the dataset was balanced using the Synthetic Minority Oversampling Technique (SMOTE) to address the class imbalance between canceled and non-canceled bookings. The processed dataset was then split into training and testing subsets using an 80/20 ratio to ensure a fair evaluation of model performance on unseen data.

## Modeling
### Model 1: Logistic Regression

Logistic Regression is a linear classification algorithm used to model the probability of a binary outcome. It estimates the relationship between input features and the log-odds of the target variable using a logistic (sigmoid) function. The algorithm assigns weights to input features and applies the sigmoid function to map the result into a probability between 0 and 1, which is then used for classification.

In this model, a pipeline is used to standardize the input features with `StandardScaler()` before applying `LogisticRegression()`. Standardization ensures that all features contribute equally to the model, which improves convergence and performance for algorithms that are sensitive to feature scale. All other parameters of the Logistic Regression classifier are left at their default settings, which include using the L2 regularization (`penalty='l2'`), the 'lbfgs' solver, and no class weighting.

The main advantages of Logistic Regression include its simplicity, efficiency, and interpretability. It performs well when the relationship between features and the target variable is approximately linear and provides probabilistic outputs. However, it can struggle with non-linear relationships, high-dimensional data with multicollinearity, and imbalanced class distributions. In cases of class imbalance, such as booking cancellation data, the model's performance may be biased toward the majority class. To address this, SMOTE (Synthetic Minority Oversampling Technique) is used to balance the classes during training by synthetically generating samples for the minority class, improving the model’s ability to generalize to underrepresented cases.

### Model 2: K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm that makes predictions based on the closest training examples in the feature space. For classification tasks, the algorithm determines the class of a new data point by identifying the majority class among its *k* nearest neighbors. Distance between data points is typically calculated using Euclidean distance, although other metrics can be used.

In this implementation, `KNeighborsRegressor` with `n_neighbors=3` is used. This setting instructs the model to consider the three nearest data points in the training set when predicting the outcome for a new instance. All other parameters are left at their default values, including `weights='uniform'`, which means each of the three neighbors contributes equally to the prediction.

The main advantage of KNN is its simplicity and effectiveness for datasets with clear patterns and well-separated classes. It makes no assumptions about the underlying data distribution and adapts naturally to complex decision boundaries. However, KNN is sensitive to the choice of *k*, the presence of irrelevant or highly correlated features, and variations in feature scale. It is also computationally expensive for large datasets, as it requires storing the entire training set and computing distances at prediction time. Moreover, KNN does not handle imbalanced datasets well by default, so prior use of SMOTE to balance the class distribution is important to improve predictive fairness and performance.

### Model 3: Decision Tree Classifier

Decision Tree Classifier is a supervised learning algorithm that splits the dataset into subsets based on the value of input features, forming a tree-like structure. At each node, the algorithm selects a feature and a threshold that best separate the data into classes by optimizing a metric such as Gini impurity or entropy. The process continues recursively until the stopping criteria are met, such as a maximum depth or minimum samples per leaf.

In this model, `DecisionTreeClassifier` is initialized with `random_state=42` to ensure reproducibility. All other parameters use default values, including the splitting criterion (`criterion='gini'`), which measures the impurity of a node. The model automatically determines the best splits and the tree's depth based on the training data unless explicitly constrained.

Decision trees are easy to interpret, handle both numerical and categorical data, and require little data preprocessing. They can capture non-linear relationships and interactions between features. However, they are prone to overfitting, especially if not pruned or limited in depth. Overfitting can reduce the model’s ability to generalize to unseen data. Decision trees also tend to be unstable to slight changes in data, which can lead to different tree structures. Although class imbalance can negatively affect the split quality, using SMOTE before training helps mitigate this issue by balancing the class distribution.

### Model 4: Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees during training and combines their predictions to improve accuracy and stability. Each tree is trained on a random subset of the data using the bagging method (bootstrap aggregation), and a random subset of features is considered at each split to ensure diversity among trees. The final prediction is made by majority vote in classification problems.

In this implementation, `RandomForestClassifier` is initialized with `random_state=42` to ensure reproducibility. Other parameters are set to their default values, including the number of trees (`n_estimators=100`) and the splitting criterion (`criterion='gini'`). These defaults provide a strong baseline performance without manual tuning.

Random Forest offers several advantages: it reduces overfitting compared to a single decision tree, handles large datasets and high-dimensional feature spaces well, and provides feature importance scores. It also manages missing values and maintains good performance even when data contains outliers or noise. However, it can be computationally intensive with large datasets, and the resulting model is less interpretable than a single decision tree. While class imbalance may affect the performance, using SMOTE before training helps address this limitation by balancing the dataset.

### Model 5: Adaptive Boosting (AdaBoost) Classifier

Adaptive Boosting, or AdaBoost, is an ensemble method that combines multiple weak learners—typically shallow decision trees—to create a strong classifier. The algorithm works by training weak learners sequentially, with each learner focusing more on instances misclassified by the previous ones. It assigns weights to training instances and updates these weights after each round based on the classification errors, allowing the model to concentrate on difficult cases.

In the implementation, `AdaBoostClassifier` is used with the `random_state=42` parameter to ensure reproducibility. All other parameters are kept at their default values. This includes `n_estimators=50`, which defines the number of weak learners (typically decision stumps), and `learning_rate=1.0`, which controls the contribution of each learner.

AdaBoost is advantageous in improving the accuracy of weak learners, reducing bias, and being less prone to overfitting on clean datasets. It can handle both binary and multiclass classification tasks effectively. However, it is sensitive to noisy data and outliers, as it puts increasing focus on difficult samples. Additionally, its sequential nature may lead to longer training times compared to parallel ensemble methods like Random Forest. Class imbalance can also affect performance, but pre-processing with techniques like SMOTE helps mitigate this issue.

### Model 6: Gradient Boosting Classifier

Gradient Boosting is an ensemble machine learning algorithm that builds models sequentially, where each new model attempts to correct the errors made by the previous one. It works by optimizing a loss function through gradient descent, using shallow decision trees as weak learners. The algorithm combines these learners to minimize prediction errors iteratively.

The implementation uses `GradientBoostingClassifier` with the default parameters, except for `random_state=42` to ensure reproducibility. The default configuration includes `n_estimators=100`, which determines the number of boosting stages, `learning_rate=0.1`, which controls the contribution of each tree, and `max_depth=3`, which restricts the complexity of individual decision trees.

Gradient Boosting offers strong predictive performance, particularly for structured tabular data. It is effective at reducing both bias and variance and can model complex non-linear relationships. However, it is more computationally intensive than simpler models and may require careful tuning to prevent overfitting. Gradient Boosting is also sensitive to noisy data and outliers, and performance can be impacted in imbalanced datasets without pre-processing techniques like SMOTE.

## Evaluation

### Choosing the Best Model
Each algorithm brings a balance between performance, interpretability, and computational cost. Multiple models were tested to evaluate which approach yields the most effective prediction of booking cancellations. The performance for each algorithm can be seen in table below.

| Model                | Accuracy | Precision | Recall   | F1-Score |
|----------------------|----------|-----------|----------|----------|
| Logistic Regression  | 0.822899 | 0.834688  | 0.802965 | 0.818519 |
| KNN                  | 0.845824 | 0.818566  | 0.886523 | 0.851190 |
| Decision Tree        | 0.867945 | 0.868243  | 0.865903 | 0.867072 |
| Random Forest        | 0.906757 | 0.913002  | 0.898113 | 0.905496 |
| AdaBoost             | 0.816597 | 0.842598  | 0.776280 | 0.808081 |
| Gradient Boosting    | 0.845958 | 0.865127  | 0.817790 | 0.840793 |

Model evaluation was based on four key metrics: accuracy, precision, recall, and F1-score. These metrics offer a balanced view of model performance, especially in the context of imbalanced classification such as predicting booking cancellations.

1. **Logistic Regression** achieved an accuracy of 82.29% with a precision of 83.47% and recall of 80.30%, resulting in an F1-score of 81.85%. This performance is solid for a baseline model, but it struggles to capture more complex patterns in the data.

2. **K-Nearest Neighbors (KNN)** performed better, with a recall of 88.65%, indicating it effectively identified most cancellations. However, its precision (81.86%) was slightly lower, leading to more false positives. Its overall F1-score was 85.12%, suggesting a stronger performance than logistic regression, but it remains sensitive to data scaling and outliers.

3. **Decision Tree Classifier** achieved balanced results across all metrics, with an accuracy of 86.79% and F1-score of 86.71%. It demonstrates the ability to capture non-linear relationships, though it may risk overfitting without ensemble support.

4. **Random Forest Classifier** outperformed all other models, with the highest accuracy (90.68%), precision (91.30%), recall (89.81%), and F1-score (90.55%). This model benefits from its ensemble approach, which reduces overfitting and improves generalization, making it the most reliable choice for this classification task. An ensemble approach combines the outputs of multiple weaker models—in this case, decision trees—to produce a stronger and more accurate final prediction. Random Forest builds many decision trees using different random subsets of the data and features, then aggregates their results, which reduces variance and minimizes overfitting that often affects individual trees.

5. **Adaptive Boosting (AdaBoost)** produced the lowest recall (77.63%) despite a decent precision (84.26%), which suggests it struggled to identify all cancellation cases. Its overall F1-score (80.81%) was the lowest among the tested ensemble models.

6. **Gradient Boosting** showed competitive results with an F1-score of 84.08%, better than logistic regression and AdaBoost, but still trailing behind Random Forest in overall performance.

Considering all evaluation metrics, **Random Forest Classifier** is selected as the best model. It consistently achieved the highest scores across all metrics, offering strong predictive capability, balanced precision and recall, and robustness against overfitting through its randomized ensemble design.

### Best Model Performance
To evaluate the hotel booking cancellation prediction model, three key classification metrics were used: **precision**, **recall**, and **F1-score**. These metrics were selected based on the nature of the business problem, where incorrect predictions can lead to significant operational and financial consequences for hotel management.

- **Precision** measures the proportion of predicted cancellations that were actually canceled. High precision is important to prevent overestimating cancellations, which could lead to underutilized rooms and lost revenue.  
  *Formula: True Positives / (True Positives + False Positives)*

- **Recall** measures the proportion of actual cancellations that were correctly identified by the model. High recall ensures that most cancellations are detected, which is critical for reducing overbooking and improving planning.  
  *Formula: True Positives / (True Positives + False Negatives)*

- **F1-score** is the harmonic mean of precision and recall. It provides a single score that balances both concerns, especially valuable when the costs of false positives and false negatives are high.  
  *Formula: 2 * (Precision * Recall) / (Precision + Recall)*

Among all the models tested, the **Random Forest Classifier** delivered the best performance with a **precision of 0.9130**, **recall of 0.8981**, and an **F1-score of 0.9055**. These results indicate that the model not only accurately identifies most actual cancellations but also minimizes the number of bookings incorrectly predicted as cancellations.

In a hotel booking context, such predictive accuracy has direct business implications. **Incorrectly predicting a cancellation (false positive)** can cause the hotel to unnecessarily release or reallocate a room, resulting in **lost revenue opportunities**. Conversely, **failing to predict a cancellation (false negative)** may lead to **overbooking**, strained customer service, or even reputational damage if accommodations are unavailable upon guest arrival. 

By achieving high precision and recall, the Random Forest model supports **efficient resource planning**, such as staff scheduling, room inventory management, and revenue forecasting. Its ensemble approach aggregates multiple decision trees, improving generalization and reducing overfitting, which contributes to its superior and robust performance in operational settings.

Overall, the Random Forest model provides a reliable solution for anticipating cancellations, enabling hotels to optimize capacity, improve guest experience, and reduce unnecessary operational costs.

### Feature Importance

To interpret the results of the hotel booking cancellation prediction model, feature importance was evaluated using **permutation importance** with **F1-score** as the baseline metric. Permutation importance works by randomly shuffling the values of each feature and measuring the change in model performance. A significant drop in F1-score indicates that the shuffled feature had a strong impact on the model, making it important. This method is model-agnostic and provides an intuitive measure of feature influence based on how much performance deteriorates when feature information is disrupted.

F1-score was chosen as the baseline for permutation importance due to its balanced consideration of both **precision** and **recall**. In the context of predicting booking cancellations, the cost of false positives (unnecessarily releasing rooms) and false negatives (failing to anticipate a cancellation) are both high. Therefore, a balanced metric like F1-score is more appropriate than accuracy, which can be misleading when dealing with class imbalance or unequal error costs.

| Feature                               | Importance Mean | Importance Std |
|---------------------------------------|------------------|-----------------|
| PRT                                   | 0.073583         | 0.001823        |
| lead_time_norm                        | 0.071465         | 0.002175        |
| total_of_special_requests             | 0.068611         | 0.002435        |
| adr_norm                              | 0.056767         | 0.001475        |
| market_segment_Offline TA/TO          | 0.033684         | 0.001872        |
| match_room_type                       | 0.028342         | 0.001146        |
| market_segment_Online TA              | 0.026975         | 0.001291        |
| required_car_parking_spaces           | 0.018803         | 0.000728        |
| customer_type_Transient-Party         | 0.017788         | 0.000621        |
| booking_changes                       | 0.015642         | 0.001194        |

The permutation importance results as show above highlighted the three most important features:

<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/mean_prob_prt.jpg" alt="Problem Analysis Diagram" width="50%">
</div>
<br>

1. **PRT (Portugal)** – This binary feature (1: country is Portugal, 0: other) had the highest importance mean at **0.0736**. Analysis of booking behavior shows a clear pattern: bookings from Portugal have a significantly higher cancellation probability (**57.5%**) compared to those from other countries (**25.4%**). This strong contrast in behavior makes the PRT feature a key predictor in the model.

<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/mean_prob_lead_time.jpg" alt="Problem Analysis Diagram" width="50%">
</div>
<br>

2. **lead_time** – With an importance mean of **0.0715**, the normalized lead time between the booking date and check-in date proves to be a strong indicator of cancellation likelihood. The longer the lead time, the higher the chance of cancellation. Cancellations increase steadily from **37.3%** in the 4–104 day interval to **81.8%** in the 504–604 day range, eventually reaching **100%** in the longest lead time range (604–704 days). This trend highlights the risk of long-term bookings and emphasizes the need for closer monitoring or flexible policies for such cases.

<div align="center">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-predictiveanalytics/master/img/mean_prob_special_requests.jpg" alt="Problem Analysis Diagram" width="50%">
</div>
<br>

3. **total_of_special_requests** – This feature, with an importance mean of **0.0686**, reflects the level of customer engagement through requests made during booking. The data shows that bookings with **no special requests** have a cancellation rate of **49.1%**, while those with **five special requests** drop sharply to only **7.6%**. This pattern suggests that guests who invest effort in customizing their stay are more likely to follow through with their bookings.

These insights provide valuable inputs for hotel management to improve **operational forecasting**, **tailored customer retention strategies**, and **revenue optimization**, ultimately reducing the impact of cancellations on business performance.

## Conclusion

This project was designed to address two key challenges in hotel booking management: understanding the primary factors driving booking cancellations and building a predictive model capable of accurately forecasting such cancellations. To meet these goals, the project followed a structured approach that began with feature selection, continued with model development, and concluded with model interpretation through feature importance analysis.

To identify the most influential features, statistical methods were applied before model training. For **categorical features**, a **Chi-Square test of independence** was conducted to assess whether there was a significant association between each feature and the target variable (cancellation status). Features with p-values below the significance threshold (typically 0.05) were considered statistically relevant. For **numerical features**, **Pearson correlation tests** were used to determine the strength and direction of the relationship between each numerical variable and the cancellation label. Features that showed moderate to strong correlation were retained for modeling.

After model development, **Permutation Importance** was applied to the best-performing model to evaluate each feature’s impact on prediction performance. This technique measures how much the model's performance (based on the F1-score) decreases when a feature’s values are randomly shuffled. Features causing greater performance drops are interpreted as more important. The **F1-score** was selected as the baseline metric due to its ability to balance **Precision** and **Recall**. This is especially important in cancellation prediction, where both false positives (e.g., incorrectly flagging a booking as likely to cancel) and false negatives (e.g., missing an actual cancellation) have financial implications such as unnecessary overbooking strategies or missed opportunities for reallocating rooms—ultimately affecting revenue and operational efficiency.

Based on the permutation importance results, the **top three most influential features** were:
- **PRT (bookings from Portugal or not):** Bookings made from Portugal were significantly more likely to be canceled (57.5%) compared to those without such history (25.4%).
- **Lead time:** A clear upward trend was observed where cancellation probability increased with longer booking lead times. For example, bookings made between 504–604 days in advance had a cancellation probability of 81.8%, and bookings beyond 604 days were always canceled.
- **Total of special requests:** Guests with more special requests were less likely to cancel. Cancellation probability dropped from 49.1% for those with no requests to just 7.6% for those with five requests.

To determine the best model for cancellation prediction, five classification algorithms were tested and evaluated: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosting. Each model underwent hyperparameter tuning where necessary. Among them, the **Random Forest** model achieved the best results with a **Precision of 0.91**, **Recall of 0.90**, and an **F1-score of 0.91**. This indicates strong predictive capability and reliability in identifying both actual cancellations and non-cancellations.

In conclusion, the project successfully identified the most predictive features using a combination of statistical testing and model-based feature importance, and developed a highly accurate machine learning model. These insights empower hotel managers to better anticipate cancellations, optimize room allocation, and improve decision-making around pricing and promotions—ultimately contributing to more effective resource planning and cost control.

## References
Antonio, N., De Almeida, A., & Nunes, L. (2019). Big Data in Hotel Revenue Management: Exploring Cancellation Drivers to Gain Insights Into Booking Cancellation Behavior. Cornell Hospitality Quarterly, 60(4), 298–319. https://doi.org/10.1177/1938965519851466

Chen, S., Ngai, E. W. T., Ku, Y., Xu, Z., Gou, X., & Zhang, C. (2023). Prediction of hotel booking cancellations: Integration of machine learning and probability model based on interpretable feature     interaction. Decision Support Systems, 170, 113959. https://doi.org/10.1016/j.dss.2023.113959

D-EDGE Hospitality Solutions. (2024, October 31). How to prevent hotel no-show and last-minute cancellations? Hospitality Financial and Technology Professionals. https://www.hftp.org/news/4124422/how-to-prevent-hotel-no-show-and-last-minute-cancellations

Sánchez-Medina, A. J., & C-Sánchez, E. (2020). Using machine learning and big data for efficient forecasting of hotel booking cancellations. International Journal of Hospitality Management, 89, 102546. https://doi.org/10.1016/j.ijhm.2020.102546

Satu, Md. S., Ahammed, K., & Abedin, M. Z. (2020). Performance Analysis of Machine Learning Techniques to Predict Hotel booking Cancellations in Hospitality Industry. 2020 23rd International Conference on Computer and Information Technology (ICCIT), 1–6. https://doi.org/10.1109/ICCIT51783.2020.9392648



