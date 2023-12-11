# Bristol Air Quality Analysis

Bristol Air Quality Analysis

## Project Overview

This project aims to investigate the air quality in Bristol, identify crucial influencing factors, and propose improvement strategies. The analysis is based on machine learning and statistical tests, utilizing urban environmental, traffic, and other data from 2020 to 2022. Air quality is classified into 10 AQI categories according to COMEAP criteria. The XGBOOST classifier is employed to identify features significantly impacting AQI. To enhance model interpretability, both global and local causal explanations are provided. The study explores the effects of various factors on Bristol's air quality, including the impact of the COVID-19 outbreak on air quality using LSTM self-prediction and the effectiveness of Bristol's Clean Air Zone policy using paired t-tests.

## Background

Air pollution is linked to numerous negative health consequences, including heart disease and cancer. The impact of air pollution is more severe on vulnerable members of society, such as children, the elderly, and individuals with pre-existing heart or lung conditions. Bristol, as the 8th largest city and the 11th largest local authority, faces significant challenges in improving air quality, breaching annual targets of nitrogen dioxide. The increase in traffic volume in Bristol has had a negative impact on air quality. To address this issue, a clean air zone and Local Cycling and Walking Infrastructure Plan (LCWIP) have been implemented. The COVID-19 pandemic has also changed people's interactions with the urban environment, leading to reduced traffic volumes and significant improvements in air quality.

## Data Sources

To evaluate the impact of different factors on air quality, relevant datasets were collected from the Open Data Bristol website, containing raw data from Bristol. Based on the Air Quality Index, concentrations of ozone, nitrogen dioxide, PM2.5, and PM10 were extracted. Covid lockdowns resulted in a dramatic decrease in the number of cars on the road, providing an opportunity to understand the exaggerated effect of how traffic affects air quality. Meteorology plays a crucial role, and certain weather conditions can trap pollutants close to the ground. Trees absorb carbon dioxide and other pollutants from the air through photosynthesis.

## Data Processing Methods

- **Outlier Treatment:** Tukey's test was used to ensure data conforms to normal distribution characteristics.
- **Filling in Nulls:** Mean of data points most similar in space or time for datasets with multiple temporal and spatial dimensions. Aggregation for datasets with too many missing values.
- **Label Encoder:** Qualitative features designed for this report, such as area postcodes and tree species, were encoded using a label encoder.
- **Classifying Air Quality:** The Daily Air Quality Index recommended by COMEAP was calculated using concentrations of ozone, nitrogen dioxide, PM2.5, and PM10. Geographical location data were classified using the geopy library.

## Research Methods

### Qualitative Analysis

1. **Initial Data Exploration:** Two-dimensional plots or category-specific plots to identify prevalent patterns and potential correlations among features.
2. **Correlation Coefficient Matrix:** Computed to uncover correlations across all data, visualized using a heatmap.
3. **XGBoost Classifier:** Employed to classify AQI indices into categories 0-10 using a decision tree. Feature importance analysis to identify variables significantly impacting AQI.

### Quantitative Analysis

1. **LSTM (Long Short-Term Memory):** Utilized for univariate time series forecasting to explore the impact of COVID on air quality.
2. **Paired T-Test:** Conducted to determine the effectiveness of the Clean Air Zone (CAZ) policy in improving air quality.

## Data Observation

### Correlation

Pearson product moment correlation coefficients were used to explore the nature of the correlation between processed data. Preliminary conclusions include strong negative correlation between AQI and visibility, strong positive correlation between temperature and dewpoint temperature, positive correlation between AQI and COVID data, and positive correlation between AQI and day, hour, and negative correlation with other variables.

### XGBoost Output

The XGBoost model outputted accuracy, F1 Score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). The model demonstrated good performance with an accuracy of 0.73, F1 Score of 0.72, MSE of 1.03, and RMSE of 1.01.

### Interpretability of XGBoost

#### Global Interpretability

SHAP (SHapley Additive exPlanations) feature attribution method was used for consistent feature importance across all target individuals. The analysis revealed the impact of month, confirmed COVID-19 cases, wind speed, and hour on higher AQI levels. The SHAP analysis suggested interventions for managing confirmed cases and wind speed to improve air quality.

#### Local Interpretability

Local Interpretable Model-agnostic Explanations (LIME) was employed to interpret model predictions for individual samples. The analysis revealed factors contributing to shifts in AQI levels, such as the number of confirmed COVID-19 cases.

## LSTM

Daily traffic flow was used for LSTM analysis, aggregating the

 data at the daily level. LSTM was trained on the training set and tested on the test set to predict AQI levels. The model showed good predictive performance with low Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Paired T-Test

To assess the impact of the Clean Air Zone (CAZ) policy on air quality, a paired t-test was conducted comparing AQI values before and after the policy implementation. The test resulted in a p-value of 0.002, indicating a statistically significant improvement in air quality after the implementation of the CAZ policy.

## Conclusion

The analysis provides insights into the factors influencing air quality in Bristol. The XGBoost model demonstrated good predictive performance, and interpretability analyses highlighted the importance of variables such as month, confirmed COVID-19 cases, and wind speed. The LSTM analysis further supported the impact of reduced traffic flow on improved air quality. The paired t-test confirmed the effectiveness of the Clean Air Zone policy in enhancing air quality. The findings contribute valuable information for policymakers and urban planners to formulate strategies for sustainable air quality management in Bristol.
