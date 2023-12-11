# Elizabeth Soto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# models
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestRegressor
# metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#
import shap
shap.initjs()


def main():
    # ----preprocessing----
    df = pd.read_csv('Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')
    df.drop('Community Area Number', axis=1, inplace=True)
    df.drop(77, inplace=True)
    print('Dataset Name: Census Data Selected socioeconomic indicators in Chicago 2008-2012')
    print('Initial dataset shape: ', df.shape)
    print('Initial column names: ', df.columns.tolist())
    ignored_columns = ['COMMUNITY AREA NAME', 'PERCENT OF HOUSING CROWDED', 'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME ']
    # dependent(X) and target variables(y)
    X = df[['PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
            'PERCENT HOUSEHOLDS BELOW POVERTY']]  # .95
    y = df['HARDSHIP INDEX']
    # splitting the training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ----1.training linear regression model----

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    result_df = pd.DataFrame({
        'Actual': y_test.values.flatten(),
        'Predicted': predictions
    })
    # test set
    print('Training set (shape):', X_train.shape, ' Test set (shape): ', X_test.shape)
    print('-----------------')
    print('Linear Regression')
    print(result_df.head(6).to_string(index=False))
    print('Performance Measurements')
    mse1 = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse1}")
    # performance measurement
    r2 = r2_score(y_test, predictions)
    print(f'R-squared: {r2}')
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

    actual = np.array(y_test)
    pred = np.array(predictions)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    print('RMSE: ', rmse)

    # Visualizations

    # 3 Features Selected
    x_1 = X_test.iloc[:, 0]
    x_2 = X_test.iloc[:, 1]
    x_3 = X_test.iloc[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.set_facecolor('#fff')

    # PERCENT AGED 16+ UNEMPLOYED
    axes[0].scatter(x_1, y_test, color='orange', label='Actual')
    axes[0].scatter(x_1, predictions, color='blue', label='Predicted')
    axes[0].set_xlabel('% 16+ Unemployed', fontsize=14)
    axes[0].set_ylabel('HARDSHIP INDEX', fontsize=14)
    axes[0].set_title('Impact of Percent Aged 16+ Unemployed on Hardship Index', wrap=True, fontsize=18, pad=25)
    axes[0].legend()
    axes[0].text(0.01, -0.20, 'Figure 1.1', transform=axes[0].transAxes, fontsize=12)

    # PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA
    axes[1].scatter(x_2, y_test, color='orange', label='Actual')
    axes[1].scatter(x_2, predictions, color='blue', label='Predicted')
    axes[1].set_xlabel('% 25+ Without High School Diploma', fontsize=14)
    axes[1].set_ylabel('HARDSHIP INDEX', fontsize=14)
    axes[1].set_title('Impact of Percent aged 25+ with no High \nSchool Diploma on Hardship Index', wrap=True,
                      fontsize=18, pad=25)
    axes[1].legend()
    axes[1].text(0.01, -0.20, 'Figure 1.2', transform=axes[1].transAxes, fontsize=12)

    # PERCENT HOUSEHOLDS BELOW POVERTY
    axes[2].scatter(x_3, y_test, color='orange', label='Actual')
    axes[2].scatter(x_3, predictions, color='blue', label='Predicted')
    axes[2].set_xlabel('% Households Below Poverty', fontsize=14)
    axes[2].set_ylabel('HARDSHIP INDEX', fontsize=14)
    axes[2].set_title('Impact of Living Below Poverty on Hardship Index', wrap=True, fontsize=18, pad=25)
    axes[2].legend()
    axes[2].text(0.01, -0.20, 'Figure 1.3', transform=axes[2].transAxes, fontsize=12)
    plt.tight_layout()
    plt.savefig('images/linear_reg_scatter_plot.png')
    plt.show()
    plt.close()
    # ----2.training Decision Tree Regressor model----

    dt_regressor = DecisionTreeRegressor(max_depth=3)
    dt_regressor.fit(X_train, y_train)
    predictions2 = dt_regressor.predict(X_test)

    result_df_2 = pd.DataFrame({
        'Actual': y_test.values.flatten(),
        'Predicted': predictions2
    })

    print('-----------------')
    print('Decision Tree Regressor')
    print(result_df_2.head(6).to_string(index=False))
    print('Performance Measurements')
    # performance measurement
    mse = mean_squared_error(y_test, predictions2)
    print(f"Mean Squared Error: {mse}")
    mae = mean_absolute_error(y_test, predictions2)
    print(f'Mean Absolute Error: {mae}')
    r2 = r2_score(y_test, predictions2)
    print(f"R-squared: {r2}")
    pred2 = np.array(predictions2)
    rmse_2 = np.sqrt(np.mean((actual - pred2) ** 2))
    print('RMSE: ', rmse_2)

    # Visualizations
    decision_tree = plt.figure(figsize=(23, 10))
    plot_tree(dt_regressor, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=12)
    decision_tree.set_facecolor('#fff')
    plt.savefig('images/decision_tree_regressor.png')
    plt.show()
    plt.close()

    # ---3. Clustering: Agglomerative Clustering

    linkage_matrix1 = linkage(X_test, method='ward', metric='euclidean')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix1, leaf_rotation=90., leaf_font_size=8.)
    plt.title(
        'Hierarchical Clustering Dendrogram: Community Area Similarity Based on Employment, Education, and Poverty'
        , loc='center', fontsize=18, wrap=True, pad=20)
    plt.xlabel('Community Areas(Data points)', fontsize=14)
    plt.ylabel('Euclidean Distance', fontsize=14)
    cutting_height = 60
    plt.axhline(y=cutting_height, color='gray', linestyle='--')
    plt.savefig('images/dendrogram.png')
    plt.show()
    plt.close()

    model = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
    clusters = model.fit_predict(X_test)
    print("-----------------------")
    print("Agglomerative Clustering")
    final = X_test.copy()
    final['Hardship Index'] = y_test
    final['Cluster'] = np.array(clusters)
    print(final.head(3).to_string(index=False))
    # final.to_csv('cluster_data.csv', index=False)
    # Save original X_test and y_test to a DataFrame
    test_data = pd.DataFrame(X_test, columns=X_test.columns)
    test_data['HARDSHIP INDEX'] = y_test.values.flatten()

    # Add cluster assignments to the DataFrame
    test_data['Cluster'] = clusters

    # Add back the ignored columns with their corresponding data
    for col in ignored_columns:
        test_data[col] = df[col]

    # Save the DataFrame to a CSV file
    test_data.to_csv('test_data_with_clusters_and_ignored_columns.csv', index=False)

    # random forest regressor
    rand_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rand_model.fit(X_train, y_train)
    predictions = rand_model.predict(X_test)
    print("Random forest Regressor")

    # performance measurements
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')
    r22 = r2_score(y_test, predictions)
    print(f"R-squared: {r22}")
    actual3 = np.array(y_test)
    pred3 = np.array(predictions)
    rmse3 = np.sqrt(np.mean((actual3 - pred3) ** 2))
    print('RMSE: ', rmse3)

    # SHAP Values
    explainer = shap.Explainer(rand_model)
    shap_values = explainer(X_test)
    print(shap_values.shape)
    # waterfall plot
    shap.plots.waterfall(shap_values[23])
    plt.show()

    # tree plot summary
    shap.summary_plot(shap_values, X_test)
    plt.show()

    # bar plot
    shap.plots.bar(shap_values)
    plt.show()


if __name__ == "__main__":
    main()