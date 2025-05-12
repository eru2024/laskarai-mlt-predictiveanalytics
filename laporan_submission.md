# Machine Learning Project Report - Booking Cancellation Predictive Model - Fatih El Haq

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
2. **Search for well-performing subsets of features by RFE.** Recursive Feature Elimination (RFE) is a technique used to find the best combination of features that contribute most effectively to the model's performance, further refining the set of influential factors.
3. **Use at least five machine learning algorithms with hyperparameter tuning if needed.** To find the "best model," the approach involves experimenting with multiple machine learning techniques. Hyperparameter tuning is a process to optimize these algorithms for the specific dataset to achieve the best possible performance.
4. **Evaluate different machine learning algorithms by Accuracy, Precision, Recall, and F1-Score.** After training the models, their performance will be assessed using key evaluation metrics. As discussed, Precision and Recall are particularly important for this problem to ensure the model correctly identifies actual cancellations (Recall) and that its predictions of cancellation are reliable (Precision), while Accuracy and F1-Score provide overall performance insights.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
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



