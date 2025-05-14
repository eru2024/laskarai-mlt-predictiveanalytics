# Machine Learning Project Report - Booking Cancellation Prediction Model - Fatih El Haq

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
### Feature Engineering Summary

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## References
Antonio, N., De Almeida, A., & Nunes, L. (2019). Big Data in Hotel Revenue Management: Exploring Cancellation Drivers to Gain Insights Into Booking Cancellation Behavior. Cornell Hospitality Quarterly, 60(4), 298–319. https://doi.org/10.1177/1938965519851466

Chen, S., Ngai, E. W. T., Ku, Y., Xu, Z., Gou, X., & Zhang, C. (2023). Prediction of hotel booking cancellations: Integration of machine learning and probability model based on interpretable feature     interaction. Decision Support Systems, 170, 113959. https://doi.org/10.1016/j.dss.2023.113959

D-EDGE Hospitality Solutions. (2024, October 31). How to prevent hotel no-show and last-minute cancellations? Hospitality Financial and Technology Professionals. https://www.hftp.org/news/4124422/how-to-prevent-hotel-no-show-and-last-minute-cancellations

Sánchez-Medina, A. J., & C-Sánchez, E. (2020). Using machine learning and big data for efficient forecasting of hotel booking cancellations. International Journal of Hospitality Management, 89, 102546. https://doi.org/10.1016/j.ijhm.2020.102546

Satu, Md. S., Ahammed, K., & Abedin, M. Z. (2020). Performance Analysis of Machine Learning Techniques to Predict Hotel booking Cancellations in Hospitality Industry. 2020 23rd International Conference on Computer and Information Technology (ICCIT), 1–6. https://doi.org/10.1109/ICCIT51783.2020.9392648



