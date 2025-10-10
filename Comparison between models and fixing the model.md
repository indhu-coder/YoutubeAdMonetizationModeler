Five algorithms has been used for this project and summarised as follows:

Differences between Linear, Lasso, and Ridge are very small —  About stability/regularization I chose Ridge (since I expect multicollinearity) or Lasso (if sparse coefficients needed).
For a single automatic ranking using average test-metric ranks LinearRegression is the best.
Avoid: ElasticNet and SVR here — they show substantially worse errors (ElasticNet especially in this run).

Training and Testing scores are given in tabular format in .JPG file.

    data = {
        "Model": ["LinearRegression", "Lasso", "Ridge", "SVR", "ElasticNet"],
        "RMSE_Train": [2.049220, 2.050380, 2.051054, 2.212353, 4.328588],
        "RMSE_Test": [2.017386, 2.018675, 2.019581, 2.183719, 4.325544],
        "MAE_Test": [0.492748, 0.519704, 0.538546, 1.079240, 3.553403],
        "R2_Test": [0.998943, 0.998941, 0.998940, 0.998761, 0.995139]
    }
    
    df = pd.DataFrame(data)
    st.title("📊 Model Performance Comparison")

    st.write("### Metrics Summary")
    st.dataframe(df.style.format({col: "{:.6f}" for col in df.select_dtypes(include=["float", "int"]).columns}))

Model Selection Rationale

Although multiple regression models demonstrated strong predictive performance, the Ridge Regression model was ultimately selected.

Ridge Regression was chosen primarily to address potential multicollinearity among the predictor variables — a condition where independent features are highly correlated, potentially destabilizing coefficient estimates in ordinary least squares regression.

By applying an L2 regularization penalty, Ridge Regression constrains the magnitude of coefficients without driving them to zero. 
This not only reduces variance but also enhances model generalization when new or correlated features are introduced in future datasets.
Moreover, Ridge Regression maintains model interpretability and provides consistent performance across training and testing data, making it a robust choice for evolving datasets where additional predictors may be incorporated over time.
