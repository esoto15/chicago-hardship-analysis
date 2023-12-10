# Unveiling Chicago's Socioeconomic Challenges: A Hardship Index-Based Analysis of Chicago (2008-2012)

## Dependencies
[![Pandas](https://img.shields.io/badge/pandas-1.3.3-blue)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.21.4-blue)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-blue)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.7.3-blue)](https://www.scipy.org/)

## How to Run
1. Clone the repository.
2. Ensure dependencies are installed (`pip install -r requirements.txt`).
3. Run `python main.py`.

## Abstract
In the United States, enduring challenges of poverty and racial disparities have marked the past three decades, disproportionately impacting Black and Hispanic communities. Notably, their poverty rates stand at 25% and 22%, respectively—twice the rate observed in white communities. Chicago, like numerous other cities, vividly illustrates significant socioeconomic disparities among its various neighborhoods and communities.

To comprehensively grasp the impact of socioeconomic challenges and the adversities faced by these communities, we delved into a census dataset sourced from the city of Chicago for the years 2008 to 2012. Our main goal was to determine the impact of socioeconomic indicators on its hardship index. For this reason, we implemented a Linear Regression, Decision Tree, and Agglomerative Hierarchical Clustering algorithms. The linear regression model outperformed the decision tree regressor model with 96% accuracy. The results pointed to three socioeconomic factors linked to hardship: unemployment, lack of a high school diploma, and living below the poverty line. These factors consistently showed a strong correlation with the hardship index in both linear regression and decision tree regressor models. 

## Models Performance
We assessed the performance of both the linear regression and decision tree regressor models using the R-Squared metric, Mean Absolute Error (MAE), and Root Mean Square Deviation (RMSE). These metrics were applied to both the actual and predicted results from each model. The linear regression model demonstrated a strong performance with an R-Squared value of 0.96, a Mean Absolute Error of 3.69, and a RMSE of 4.966. In contrast, the Decision Tree Regressor yielded an R-Squared value of 0.89, a Mean Absolute Error of 7.04, and a RMSE of 9.187. In comparing these metrics, the higher R-Squared value and lower values for MAE and RMSE in the linear regression model suggested a better overall performance and predictive accuracy compared to the decision tree regressor.

### Visualizations
- Scatter plots for linear regression.
- Decision tree plot.
- Dendrogram for agglomerative clustering.

### Data
- Initial dataset: [Census Data - Chicago 2008-2012](link_to_dataset.csv)
- Clustered data: [Clustered Data](clustereddata.csv)

**Note:** Close generated graphs to see complete results in the terminal.
