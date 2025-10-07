import streamlit as st
from Main import *
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import date
import plotly.express as px


# -------------------------
# Function to prepare user input
# -------------------------
def prepare_user_input(user_input_dict):

  # Load feature names saved during training
    with open("feature_names.pkl", "rb") as f:
        feature_cols = pickle.load(f)
        # Encode categorical columns
    le = LabelEncoder()
    user_input_dict['device'] = le.fit_transform([user_input_dict['device']])[0]
    
    # Map country and category to numeric
    country_map = {'US':0, 'UK':1, 'CA':2, 'AU':3, 'DE':4, 'IN':5, 'Others':6}
    category_map = {'Entertainment':0, 'Education':1, 'Music':2, 'Gaming':3, 'Others':4, 'Tech':5, 'Lifestyle':6}
    
    user_input_dict['country'] = country_map.get(user_input_dict['country'], 6)
    user_input_dict['category'] = category_map.get(user_input_dict['category'], 6)
    
   
    
    user_df = pd.DataFrame([user_input_dict], columns=feature_cols)
    print(user_df.columns)
    # Ensure 'date' column exists
    if 'date' not in user_df.columns:
        user_df['date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
    else:
        user_df['date'] = pd.to_datetime(user_df['date']).dt.strftime('%Y-%m-%d')

    
    
    user_df['engagements'] = user_df['likes'] + user_df['comments']
    user_df['engagement_rate (%)'] = round((user_df['engagements'] / user_df['views']) * 100, 2)

    
    
    # Date features
   
    user_df['date'] = pd.to_datetime(user_df['date'],format='mixed')

    # Drop unused columns
    user_df = user_df.drop(columns=['video_id','year','month','day','engagements','engagement_rate (%)','date'], errors='ignore')
    
    return user_df



options = st.sidebar.selectbox("Select Page", options=["Home","EDA & Visualizations","Modeling Prediction"])
if options == "Home":
    st.title("YouTube Ad Revenue Prediction App")
    st.write("This app predicts YouTube ad revenue based on user inputs.")
    st.write("The YouTube content Monetization Modeler aims to predict YouTube revenue based on different factors affecting monetization, such as views, engagement rate, video length, content type, and ad types." \
    " Using linear regression, the model provides a way to forecast potential earnings for content creators or advertisers." \
    " This system can help YouTubers or advertisers make data-driven decisions by predicting revenue from various parameters, optimizing the performance of their content or ad campaigns.")


if options == "Modeling Prediction":
    # -------------------------
    # Streamlit UI
    # -------------------------
    st.title("🎬 YouTube Ad Revenue Prediction")
    st.write("Estimate your channel's ad revenue (USD) based on performance metrics and date.")

    # User input
    user_input_dict = {
        "video_id": st.text_input("video_id", value="abc123"),
        "views": st.number_input("views", min_value=0, step=1000),
        "likes": st.number_input("likes", min_value=0, step=100),
        "comments": st.number_input("comments", min_value=0, step=100),
        "subscribers": st.number_input("subscribers", min_value=0, step=100),
        "watch_time_minutes": st.number_input("watch_time_minutes", min_value=0.0, step=10.0, format="%.2f"),
        "date": st.date_input("date", value=date.today()),
        "video_length_minutes": st.number_input("video_length_minutes", min_value=1, step=1),
        "country": st.selectbox("country", options=['US', 'IN', 'UK', 'CA', 'AU', 'DE', 'Others']),
        "category": st.selectbox("category", options=['Education', 'Entertainment', 'Tech', 'Lifestyle', 'Music','Gaming','Others']),
        "device": st.selectbox("device", options=['Mobile', 'Desktop', 'Tablet', 'TV', 'Others'])
    }

    # -------------------------
    # Predict button
    # -------------------------
    if st.button("Predict Revenue"):
        # Load model and scaler
        with open('ridge_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare user input
        X_new = prepare_user_input(user_input_dict)
        
        # Apply scaler
        X_new_scaled = scaler.transform(X_new)
        
        # Predict
        y_prediction = loaded_model.predict(X_new_scaled)
        
       
        st.subheader("📊 Prediction Insights")

        # 1️⃣ Display the prediction clearly
        st.metric("💰 Predicted Ad Revenue (USD)", f"${y_prediction[0]:,.2f}")
        user_input_dict['engagement_rate (%)'] = round(((user_input_dict['likes'] + user_input_dict['comments']) / user_input_dict['views']) * 100, 2)
        st.metric(label="🎯 Engagement Rate",value=f"{user_input_dict['engagement_rate (%)']:.2f} %")
        
        st.subheader("📅 Date Features")

        date_obj = pd.to_datetime(user_input_dict['date'])

        metrics = {
            "📅 Date": date_obj.strftime("%Y-%m-%d"),
            "🗓️ Weekday": date_obj.strftime("%A"),
            "📊 Quarter": f"Q{((date_obj.month-1)//3)+1}",
            "🕒 Week of Year": date_obj.isocalendar().week,
            "🏖️ Weekend": "Yes" if date_obj.weekday() >= 5 else "No"
        }

        for k, v in metrics.items():
            st.write(f"**{k}:** {v}")

        if date_obj.month in [10, 11, 12]:
            season = "🎄 Holiday Season (Q4 boost likely)"
        elif date_obj.month in [6, 7, 8]:
            season = "☀️ Mid-Year Engagement Season"
        else:
            season = "📈 Regular Period"

        st.info(f"**Time Context Insight:** {season}")
        st.subheader("📊 Comparative Analysis")

        values = [y_prediction[0], y_pred3.mean()]
        labels = ['Predicted', 'Average Channel Revenue']

        plt.bar(labels, values)
        plt.title('Predicted vs Average Revenue')
        plt.ylabel('USD')
        st.pyplot(plt)
        st.subheader("📉 Feature Importance & Comparison")
        importance = pd.Series(loaded_model.coef_, index=scaler.feature_names_in_)
        importance.sort_values().plot(kind='barh', figsize=(10,6))
        plt.title("Feature Importance in Ad Revenue Prediction")
        plt.xlabel("Coefficient Value")
        st.pyplot(plt)
        


        # --------------------------
        # Mapping: encoded value → full country name → ISO-3 code
        # --------------------------
        country_map = {'US':0, 'UK':1, 'CA':2, 'AU':3, 'DE':4, 'IN':5, 'Others':6}
        iso_map = {'US':'USA','UK':'GBR','CA':'CAN','AU':'AUS','DE':'DEU','IN':'IND','Others':'XXX'}

        # Reverse mapping to get country name from encoded value
        country_name_list = [k for k,v in country_map.items() if v == user_input_dict['country']]
        country_name = country_name_list[0] if country_name_list else 'Others'

        # Map to ISO code for plotting
        country_iso = iso_map.get(country_name, 'XXX')

        # --------------------------
        # Prepare DataFrame for Plotly
        # --------------------------
        all_iso_codes = list(iso_map.values())
        df = pd.DataFrame({
            'country_iso': all_iso_codes,
            'predicted_revenue': [0]*len(all_iso_codes)  # all zero initially
        })

        # Highlight only the selected country
        df.loc[df['country_iso'] == country_iso, 'predicted_revenue'] = y_prediction[0]

        # --------------------------
        # Plot choropleth map
        # --------------------------
        fig = px.choropleth(
            df,
            locations='country_iso',
            color='predicted_revenue',
            color_continuous_scale='Reds',
            range_color=[0,y_prediction[0]+50],
            labels={'predicted_revenue':'Predicted Revenue (USD)'},
            title=f"🌍 Highlighted Country: {country_name}"
        )

        # Make other countries light gray
        fig.update_traces(marker_line_width=0.5, marker_line_color='white')
        fig.update_layout(coloraxis_showscale=False)

        # --------------------------
        # Display in Streamlit
        # --------------------------
        st.plotly_chart(fig)



        
if options == "EDA & Visualizations":
        st.title("📊 EDA & Visualizations")
        st.write("Explore various visualizations related to YouTube ad revenue and performance metrics.")
        plt.figure(figsize=(14,6))
        sns.heatmap(model_data.corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlations with Ad Revenue")
        st.pyplot(plt)
        
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=model_data, x='views', y='ad_revenue_usd', hue='category')
        plt.title("Views vs Ad Revenue by Category")
        plt.xlabel("Views")
        plt.ylabel("Ad Revenue (USD)")
        st.pyplot(plt)

          # # #boxplot for outlier detection
        plt.figure(figsize=(25,20))
        sns.boxplot(data=data)
        plt.title('Boxplot for Outlier Detection')
        st.pyplot(plt)

           
        # # #distribution of target variable
        plt.figure(figsize=(8,6))
        sns.histplot(data['ad_revenue_usd'], bins=30, kde=True)
        plt.title('Distribution of Ad Revenue (USD)')
        st.pyplot(plt)



        