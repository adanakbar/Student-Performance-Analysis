import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats


# Set up the page configuration
st.set_page_config(page_title="Student Performance Data Analysis", layout="wide")

df = pd.read_csv(r"D:\IDS_Project\StudentPerformanceFactors.csv")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Introduction",
        "Exploratory Data Analysis",
        "Data Cleaning",
        "Statistical Analysis",
        "Predictive Modeling",
        "Download Cleaned Data",
    ]
)
categorical_cols = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

numeric_cols = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score",
]
# Introduction Page
final_df = pd.DataFrame()

with tab1:
    st.markdown(
        """
        # 🌟 Welcome to the Student Performance Analysis App! 🌟

        ## **Project Overview:**

        This interactive app is designed to provide a comprehensive analysis of the factors that impact student performance. By analyzing various elements such as study hours, attendance, sleep patterns, parental involvement, and more, we aim to uncover the key drivers behind students' academic outcomes. Our goal is to identify these influential factors and visually explore their relationship with exam scores.

        ## **What You'll Discover:**

        - **Exploratory Data Analysis (EDA):** We'll begin by diving into the dataset—understanding its structure, columns, and key statistics.

        - **Data Cleaning:** Learn how we handle missing values, perform necessary transformations, and ensure the dataset is ready for analysis.

        - **Data Visualization:** Explore insightful visualizations that highlight key trends in the data.

        - **Statistical Analysis & Modeling:** We perform regression analysis and predictive modeling to understand the factors influencing performance.

        - **Insights & Recommendations:** Based on the data, we provide actionable insights and suggest potential improvements for student outcomes.

        ## **What You'll Be Exploring:**
        - **Data Preprocessing:** How we clean the data—filling missing values, detecting outliers, and transforming the data to ensure quality.
        - **Exploratory Analysis:** Interactive charts and graphs to help you understand the relationships between key variables.
        - **Predictive Modeling:** See how machine learning models can forecast student performance based on the various factors.

        The app is designed to offer both an in-depth analysis of the data and a user-friendly interface for anyone interested in understanding student performance. 

         **Ready to start your exploration? Let's uncover the hidden patterns together!** 
        """
    )

    st.subheader("👨‍💻 How to Use This App")

    st.markdown(
        """
        Here's a simple guide to navigate through the app:

        1. **EDA (Exploratory Data Analysis)**: Begin by exploring the dataset. Understand its structure, features, and see some basic statistics. 
        2. **Data Cleaning**: In this section, we take care of missing values, perform transformations, and ensure the data is clean for further analysis.
        3. **Statistical Analysis**: This section covers hypothesis testing and regression analysis to gain deeper insights.
        4. **Predictive Modeling**: See how we use machine learning models to predict student performance based on key factors.
        5. **Download Cleaned Data**: If you'd like, you can download the cleaned and preprocessed data for your own analysis.

        Use the **sidebar** to easily switch between sections, explore the visuals, and uncover valuable insights.
        """
    )

    st.markdown(
        """
        --- 
        ## **Project Creator:**
        - Adan Akbar

        ## **Data Source:**
        - Dataset: [Student Performance Factors Dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

        """
    )

    st.markdown(
        """
        ### 🚀 Let's get started! 
        """
    )

# EDA Page
with tab2:
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Preview")
    st.write(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(
            f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
        )

    with col2:
        st.subheader("Column Names")
        st.write(df.columns)

    st.subheader("Data Types and Non-Null Values")
    dtype_info = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isna().sum(),
        }
    )
    st.write(dtype_info)

    # Descriptive Statistics for Numeric Columns
    st.subheader("Descriptive Statistics for Numeric Columns")
    st.write(df.describe())

    duplicates_count = df.duplicated().sum()
    st.subheader("Duplicate Rows Check")
    st.write(f"The dataset contains **{duplicates_count} duplicate rows**.")

# Data Cleaning Page
with tab3:
    st.title("Data Cleaning And Transformation")

    st.subheader("Inconsistent or Faulty Data in Categorical Columns")
    categorical_summary = pd.DataFrame(
        {
            "Column Name": categorical_cols,
            "Unique Value Count": [df[col].nunique() for col in categorical_cols],
            "Sample Values": [
                ", ".join(map(str, df[col].unique()[:5]))
                + ("..." if df[col].nunique() > 5 else "")
                for col in categorical_cols
            ],
        }
    )
    st.write("Below is a summary of the categorical columns:")
    st.dataframe(categorical_summary)

    with st.expander("View Detailed Unique Values"):
        selected_col = st.selectbox(
            "Select a column to view unique values", categorical_cols
        )
        st.write(
            f"**{selected_col}**: {', '.join(map(str, df[selected_col].unique()))}"
        )

    st.markdown(
        """
        - All values are correctly spelled and meaningful.
        - Binary columns have consistent "Yes" or "No" values.
        - No unexpected characters, typos, or case inconsistencies were found.
        """
    )

    st.subheader("Inconsistent or Faulty Data in Numerical Columns")

    numeric_summary = pd.DataFrame(
        {
            "Column Name": numeric_cols,
            "Min Value": [df[col].min() for col in numeric_cols],
            "Max Value": [df[col].max() for col in numeric_cols],
            "Mean Value": [df[col].mean() for col in numeric_cols],
            "Unique Value Count": [df[col].nunique() for col in numeric_cols],
        }
    )
    st.write("Below is a summary of the numeric columns:")
    st.dataframe(numeric_summary)

    df["Exam_Score"] = df["Exam_Score"].apply(lambda x: min(x, 100))

    st.markdown(
        """
        - The numeric data is largely consistent with plausible ranges across most columns.
        - **Potential issue:** The value 101 in "Exam_Score" exceeds the typical range (0–100) and has been corrected to fit within this range.
        """
    )

    st.subheader("Data After Filling Missing Values")
    st.write(df.head())

    st.subheader("Handle Missing Values")
    columns = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
        "Exam_Score",
    ]
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
        df[col] = df[col].fillna(df[col].mean())
    df[categorical_cols] = df[categorical_cols].astype("category")

    df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
        df["Distance_from_Home"].mode()[0]
    )
    df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
        df["Parental_Education_Level"].mode()[0]
    )
    df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
        df["Teacher_Quality"].mode()[0]
    )

    with st.expander("Approach For Handling Missing Values"):
        st.markdown(
            """
            **The missing values in the categorical columns Distance_from_Home, Parental_Education_Level, and Teacher_Quality
            were filled using the **mode** of each respective column. This approach helps avoid bias while maintaining the
            original data distribution.**
            """
        )

    st.subheader("Visualizing Distribution and Detecting Outliers for Numeric Features")
    # Loop through numeric columns for histograms and box plots
    for col in numeric_cols:
        with st.expander(f"View Distribution And Box Plot of {col}"):
            st.subheader(f"Distribution of {col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)
            st.subheader(f"Box Plot for {col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"{col} Box Plot")
            st.pyplot(fig)

    def cap_outliers(df, numeric_cols):
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = np.where(
                df[col] > upper_bound,
                upper_bound,
                np.where(df[col] < lower_bound, lower_bound, df[col]),
            )

        return df

    # Cap the outliers
    df = cap_outliers(df, numeric_cols)
    st.subheader("Data After Handling Outliers")
    st.write(df)
    with st.expander("Outliers Handling Approach"):
        st.markdown(
            """**I capped the outliers to maintain data integrity, preserve the extreme data points so that
            they do not disrupt the distribution, and reduce the influence of outliers on analysis and modeling.**"""
        )

    st.title("Categorical Encoding with Streamlit")

    label_encode_cols = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality",
        "Parental_Education_Level",
        "Distance_from_Home",
    ]

    one_hot_encode_cols = [
        "Extracurricular_Activities",
        "Internet_Access",
        "School_Type",
        "Peer_Influence",
        "Learning_Disabilities",
        "Gender",
    ]

    label_encoded_df = df.copy()
    label_encoder = LabelEncoder()

    for col in label_encode_cols:
        label_encoded_df[col] = label_encoder.fit_transform(label_encoded_df[col])

    one_hot_encoded_df = pd.get_dummies(
        df[one_hot_encode_cols], drop_first=True
    ).astype(int)

    final_df = pd.concat(
        [label_encoded_df[label_encode_cols], one_hot_encoded_df, df[numeric_cols]],
        axis=1,
    )
    st.session_state.final_df = final_df
    st.subheader("Label Encoded Columns")
    st.dataframe(label_encoded_df[label_encode_cols])

    st.subheader("One-Hot Encoded Columns")
    st.dataframe(one_hot_encoded_df)

    st.subheader("Final Dataset")
    st.dataframe(final_df)

    st.download_button(
        label="Download Final Dataset as CSV",
        data=final_df.to_csv(index=False),
        file_name="encoded_dataset.csv",
        mime="text/csv",
    )

# Statistical Analysis Page
with tab4:
    st.title("Statistical Analysis")
    st.subheader("Descriptive Statistics")

    numeric_data = df[numeric_cols]
    st.write("**Summary Statistics for Numeric Data:**")
    st.dataframe(numeric_data.describe())

    st.write("**Distribution of Categorical Data**")
    for col in categorical_cols:
        with st.expander(f"View Distribution of {col}"):
            st.write(f"Distribution of {col}:")
            st.dataframe(df[col].value_counts())

    with st.expander("**Importance of Descriptive Statistics in Analysis**"):
        st.markdown(
            """**I used descriptive statistics to get a clear understanding of the data’s trends, variability, and overall distribution, which helped lay the groundwork for further analysis.**"""
        )

    st.subheader("Correlation Analysis")
    correlation = df[numeric_cols].corr()
    st.dataframe(correlation)

    st.write("Correlation Matrix (Heatmap)")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Matrix", fontsize=16)
    st.pyplot(fig)

    with st.expander("Correlation Analysis Insights"):
        st.title("Correlation Analysis Insights")

        st.subheader("• Attendance is Key")
        st.write(
            "• Attendance shows the strongest positive correlation with exam scores (0.581), highlighting that consistent class participation significantly impacts performance."
        )

        st.subheader("• Study Hours Matter")
        st.write(
            "• Hours spent studying have a moderate positive correlation with exam scores (0.445), indicating that dedicating more time to studying generally leads to better results."
        )

        st.subheader("• Past Performance is a Factor")
        st.write(
            "• Previous scores have a weak positive correlation with current exam scores (0.175), suggesting that while past performance helps, it is not the sole determinant of success."
        )

        st.subheader("• Tutoring Sessions Help Slightly")
        st.write(
            "• Tutoring sessions have a weak positive correlation with exam scores (0.157), indicating a small but noticeable benefit."
        )

        st.subheader("• Physical Activity and Exam Scores")
        st.write(
            "• Physical activity has a negligible correlation with exam scores (0.028), implying no significant direct impact."
        )

        st.subheader("• Sleep Hours and Exam Scores")
        st.write(
            "• Sleep hours show a near-zero negative correlation (-0.017), indicating no meaningful relationship with exam performance in this dataset."
        )

    with st.expander("Problem to Solve for Students"):
        st.title("Problem to Solve for Students")
        st.write(
            "Based on this analysis, the key issue students face is prioritizing effective habits that directly impact academic performance."
        )

    with st.expander("Suggested Solutions to Improve Student Performance"):
        st.title("Solutions You Can Suggest")

        st.subheader("• Improving Class Attendance")
        st.write(
            "• Encourage students to attend classes regularly by highlighting the strong correlation between attendance and better grades."
        )
        st.write(
            "• Develop incentives or support programs to help students stay consistent in attendance."
        )

        st.subheader("• Encouraging Structured Study Habits")
        st.write(
            "• Promote time management techniques to help students dedicate sufficient and effective study hours."
        )
        st.write(
            "• Offer workshops on efficient study strategies, such as active recall and spaced repetition."
        )

        st.subheader("• Leveraging Tutoring Sessions")
        st.write(
            "• Although tutoring has a smaller impact, providing personalized tutoring or mentorship programs could help bridge learning gaps for struggling students."
        )

        st.subheader("• Holistic Well-Being Awareness")
        st.write(
            "• While physical activity and sleep have negligible correlations in this dataset, ensure students understand their broader benefits on mental health and energy levels."
        )

    categorical_cols = [
        ("Parental_Involvement", "Motivation_Level"),
        ("Internet_Access", "Motivation_Level"),
        ("Peer_Influence", "Extracurricular_Activities"),
    ]

    def perform_chi_square_test(data, col1, col2):
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value, contingency_table, expected

    st.title("Chi-Square Test of Independence")

    for col1, col2 in categorical_cols:
        st.subheader(f"Chi-Square Test: {col1} vs {col2}")

        chi2, p_value, contingency_table, expected = perform_chi_square_test(
            df, col1, col2
        )

        st.write("### Contingency Table")
        st.dataframe(contingency_table)

        st.write("### Expected Counts")
        st.dataframe(
            pd.DataFrame(
                expected,
                columns=contingency_table.columns,
                index=contingency_table.index,
            )
        )

        st.write(f"Chi-Square Statistic: {chi2:.2f}")
        st.write(f"P-Value: {p_value:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap="Blues", fmt="d", ax=ax)
        ax.set_title(f"Contingency Table: {col1} vs {col2}")
        st.pyplot(fig)

        # Interpretation based on p-value
        if p_value < 0.05:
            st.markdown(
                "**Conclusion:** There is a significant association between the two variables."
            )
        else:
            st.markdown(
                "**Conclusion:** There is no significant association between the two variables."
            )

    st.title("Chi-Square Test Insights")

    with st.expander("1. Parental Involvement vs Motivation Level"):
        st.write(
            "The Chi-Square test reveals that there is no significant relationship between parental involvement and motivation level. "
            "While parental involvement is often assumed to strongly affect motivation, this analysis suggests that other factors may play a larger role."
        )

    with st.expander("2. Internet Access vs Motivation Level"):
        st.write(
            "There is no significant relationship between internet access and motivation level. "
            "This finding indicates that simply having internet access does not directly influence a student's motivation to succeed academically."
        )

    with st.expander("3. Peer Influence vs Extracurricular Activities"):
        st.write(
            "The Chi-Square test shows a significant relationship between peer influence and participation in extracurricular activities. "
            "This suggests that students are more likely to engage in extracurricular activities if they are influenced or encouraged by their peers, emphasizing the importance of social factors in student engagement."
        )

    st.title("Student Engagement Solutions Based on Insights")

    with st.expander("1. Lack of Motivation Despite Parental Involvement"):
        st.write(
            "Parental involvement alone may not significantly influence motivation. Other factors could be more impactful."
        )
        st.write("**Solutions:**")
        st.write(
            "- **Peer Mentorship Programs:** Pair students with mentors or create study groups for mutual motivation."
        )
        st.write(
            "- **Goal Setting & Self-Reflection:** Develop programs to help students set personal and academic goals."
        )
        st.write(
            "- **Personalized Academic Support:** Offer tutoring, academic guidance, and one-on-one meetings with counselors."
        )

    with st.expander("2. Internet Access Not Having a Direct Impact on Motivation"):
        st.write(
            "Internet access does not have a direct relationship with student motivation in this dataset."
        )
        st.write("**Solutions:**")
        st.write(
            "- **Promote Purposeful Internet Use:** Teach students to use the internet for academic purposes."
        )
        st.write(
            "- **Digital Literacy Workshops:** Educate students on using the internet effectively for learning."
        )
        st.write(
            "- **Access to Online Study Groups:** Encourage joining study groups or accessing academic resources online."
        )

    with st.expander("3. Low Participation in Extracurricular Activities"):
        st.write(
            "Peer influence significantly impacts participation in extracurricular activities."
        )
        st.write("**Solutions:**")
        st.write(
            "- **Peer-Led Programs:** Create peer-led clubs or activities for leadership and community-building."
        )
        st.write(
            "- **Peer-to-Peer Encouragement:** Develop systems where students motivate each other to participate."
        )
        st.write(
            "- **Group-Based Activities:** Focus on extracurriculars that involve group participation, like sports or team projects."
        )

# # Predictive Modeling Page
with tab5:
    st.title("Random Forest Regression - Predicting Exam Scores")

    st.subheader("Model Overview")
    st.write(
        "In this section, we will apply the Random Forest Regression model to predict exam scores based on various features such as parental involvement, internet access, and more."
    )

    X = final_df.drop("Exam_Score", axis=1)
    y = final_df["Exam_Score"]

    # Splitting the data into features and target variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R-squared (R²): {r2:.4f}")

    # Plot Actual vs Predicted Exam Scores
    st.subheader("Actual vs Predicted Exam Scores")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    ax.set_title("Actual vs Predicted Exam Scores")
    ax.set_xlabel("Actual Exam Scores")
    ax.set_ylabel("Predicted Exam Scores")
    st.pyplot(fig)

    feature_importance = rf_regressor.feature_importances_
    feature_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    st.subheader("Feature Importance")
    st.write(
        "The following table shows the importance of each feature in predicting the exam score."
    )
    st.dataframe(feature_df)

    with st.expander("Insights from Actual vs Predicted Exam Scores Scatter Plot"):
        st.write(
            "The scatter plot of actual vs. predicted exam scores shows a strong positive relationship, "
            "meaning the Random Forest model predicts exam scores well. Most predictions are very close to the actual scores, "
            "as seen by the points near the diagonal line. However, the model has more variation in predictions for very low "
            "(below 62) and very high (above 72) scores. Fixing these differences and improving the features used in the model could make it even better."
        )

with tab6:
    st.title("Download Cleaned Data")

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
