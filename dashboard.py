import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Employee Attrition Prediction & HR Analytics")
st.write(
    "Interactive dashboard for analysing employee attrition "
    "and predicting employee attrition risk."
)

st.divider()

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


df = load_data()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")

department_list = ["All"] + sorted(
    df["Department"].unique().tolist()
)

job_role_list = ["All"] + sorted(
    df["JobRole"].unique().tolist()
)

gender_list = ["All"] + sorted(
    df["Gender"].unique().tolist()
)

overtime_list = ["All"] + sorted(
    df["OverTime"].unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "Department",
    department_list
)

selected_job_role = st.sidebar.selectbox(
    "Job Role",
    job_role_list
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    gender_list
)

selected_overtime = st.sidebar.selectbox(
    "OverTime",
    overtime_list
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]

if selected_job_role != "All":
    filtered_df = filtered_df[
        filtered_df["JobRole"] == selected_job_role
    ]

if selected_gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == selected_gender
    ]

if selected_overtime != "All":
    filtered_df = filtered_df[
        filtered_df["OverTime"] == selected_overtime
    ]

# ============================================================
# KPI CALCULATIONS
# ============================================================

total_employees = len(filtered_df)

employees_left = (
    filtered_df["Attrition"] == "Yes"
).sum()

employees_stayed = (
    filtered_df["Attrition"] == "No"
).sum()

if total_employees > 0:
    attrition_rate = (
        employees_left / total_employees
    ) * 100

    average_age = filtered_df["Age"].mean()

    average_income = filtered_df["MonthlyIncome"].mean()
else:
    attrition_rate = 0
    average_age = 0
    average_income = 0

# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📌 HR Summary")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Total Employees",
        total_employees
    )

with c2:
    st.metric(
        "Employees Left",
        employees_left
    )

with c3:
    st.metric(
        "Employees Stayed",
        employees_stayed
    )

with c4:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

with c5:
    st.metric(
        "Average Age",
        f"{average_age:.1f}"
    )

st.divider()

# ============================================================
# TABS
# ============================================================

overview_tab, workforce_tab, drivers_tab, ml_tab, data_tab = st.tabs(
    [
        "📊 Overview",
        "👥 Workforce Analysis",
        "🔥 Attrition Drivers",
        "🤖 ML Prediction",
        "📋 Employee Data"
    ]
)

# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with overview_tab:

    st.subheader("📊 Employee Attrition Overview")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # ATTRITION DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        attrition_counts = (
            filtered_df["Attrition"]
            .value_counts()
        )

        fig1, ax1 = plt.subplots()

        attrition_counts.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_title(
            "Employee Attrition Distribution"
        )

        ax1.set_xlabel("Attrition")
        ax1.set_ylabel("Number of Employees")

        plt.xticks(rotation=0)

        st.pyplot(fig1)

        plt.close(fig1)

    # --------------------------------------------------------
    # DEPARTMENT ATTRITION
    # --------------------------------------------------------

    with col2:

        department_attrition = pd.crosstab(
            filtered_df["Department"],
            filtered_df["Attrition"]
        )

        fig2, ax2 = plt.subplots()

        department_attrition.plot(
            kind="bar",
            ax=ax2
        )

        ax2.set_title(
            "Department-wise Attrition"
        )

        ax2.set_xlabel("Department")
        ax2.set_ylabel("Employees")

        plt.xticks(rotation=0)

        st.pyplot(fig2)

        plt.close(fig2)

    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🎂 Age Distribution")

    fig3, ax3 = plt.subplots()

    ax3.hist(
        filtered_df["Age"],
        bins=15
    )

    ax3.set_title(
        "Employee Age Distribution"
    )

    ax3.set_xlabel("Age")
    ax3.set_ylabel("Number of Employees")

    st.pyplot(fig3)

    plt.close(fig3)

    # --------------------------------------------------------
    # MONTHLY INCOME
    # --------------------------------------------------------

    st.subheader("💰 Monthly Income Analysis")

    fig4, ax4 = plt.subplots()

    filtered_df.boxplot(
        column="MonthlyIncome",
        by="Attrition",
        ax=ax4
    )

    plt.suptitle("")

    ax4.set_title(
        "Monthly Income by Attrition"
    )

    ax4.set_xlabel("Attrition")
    ax4.set_ylabel("Monthly Income")

    st.pyplot(fig4)

    plt.close(fig4)


# ============================================================
# TAB 2 - WORKFORCE ANALYSIS
# ============================================================

with workforce_tab:

    st.subheader("👥 Workforce Analysis")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # JOB ROLE
    # --------------------------------------------------------

    with col1:

        role_attrition = pd.crosstab(
            filtered_df["JobRole"],
            filtered_df["Attrition"]
        )

        fig5, ax5 = plt.subplots(
            figsize=(7, 5)
        )

        role_attrition.plot(
            kind="bar",
            ax=ax5
        )

        ax5.set_title(
            "Job Role vs Attrition"
        )

        ax5.set_xlabel("Job Role")
        ax5.set_ylabel("Employees")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        st.pyplot(fig5)

        plt.close(fig5)

    # --------------------------------------------------------
    # OVERTIME
    # --------------------------------------------------------

    with col2:

        overtime_attrition = pd.crosstab(
            filtered_df["OverTime"],
            filtered_df["Attrition"]
        )

        fig6, ax6 = plt.subplots()

        overtime_attrition.plot(
            kind="bar",
            ax=ax6
        )

        ax6.set_title(
            "OverTime vs Attrition"
        )

        ax6.set_xlabel("OverTime")
        ax6.set_ylabel("Employees")

        plt.xticks(rotation=0)

        st.pyplot(fig6)

        plt.close(fig6)

    # --------------------------------------------------------
    # BUSINESS TRAVEL
    # --------------------------------------------------------

    st.subheader("✈️ Business Travel Analysis")

    travel_attrition = pd.crosstab(
        filtered_df["BusinessTravel"],
        filtered_df["Attrition"]
    )

    st.dataframe(
        travel_attrition,
        use_container_width=True
    )

    # --------------------------------------------------------
    # JOB LEVEL
    # --------------------------------------------------------

    st.subheader("📈 Job Level Analysis")

    level_attrition = pd.crosstab(
        filtered_df["JobLevel"],
        filtered_df["Attrition"]
    )

    st.dataframe(
        level_attrition,
        use_container_width=True
    )


# ============================================================
# TAB 3 - ATTRITION DRIVERS
# ============================================================

with drivers_tab:

    st.subheader("🔥 Attrition Drivers")

    st.write(
        "This section identifies employee groups where "
        "attrition rates can be examined in more detail."
    )

    # --------------------------------------------------------
    # OVERTIME ATTRITION RATE
    # --------------------------------------------------------

    st.write("### ⏰ Attrition Rate by OverTime")

    overtime_rate = (
        filtered_df.groupby("OverTime")["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Attrition Rate (%)"
        )
    )

    st.dataframe(
        overtime_rate.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # JOB SATISFACTION
    # --------------------------------------------------------

    st.write(
        "### 😊 Attrition Rate by Job Satisfaction"
    )

    satisfaction_rate = (
        filtered_df.groupby(
            "JobSatisfaction"
        )["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Attrition Rate (%)"
        )
    )

    fig7, ax7 = plt.subplots()

    ax7.bar(
        satisfaction_rate[
            "JobSatisfaction"
        ].astype(str),
        satisfaction_rate[
            "Attrition Rate (%)"
        ]
    )

    ax7.set_title(
        "Job Satisfaction vs Attrition Rate"
    )

    ax7.set_xlabel(
        "Job Satisfaction Level"
    )

    ax7.set_ylabel(
        "Attrition Rate (%)"
    )

    st.pyplot(fig7)

    plt.close(fig7)

    # --------------------------------------------------------
    # JOB ROLE ATTRITION RATE
    # --------------------------------------------------------

    st.write(
        "### 💼 Attrition Rate by Job Role"
    )

    role_rate = (
        filtered_df.groupby(
            "JobRole"
        )["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .sort_values(
            ascending=False
        )
        .reset_index(
            name="Attrition Rate (%)"
        )
    )

    st.dataframe(
        role_rate.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # YEARS AT COMPANY
    # --------------------------------------------------------

    st.write(
        "### 🏢 Years at Company vs Attrition"
    )

    tenure_rate = (
        filtered_df.groupby(
            "YearsAtCompany"
        )["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Attrition Rate (%)"
        )
    )

    st.dataframe(
        tenure_rate.round(2),
        use_container_width=True
    )


# ============================================================
# TAB 4 - MACHINE LEARNING
# ============================================================

with ml_tab:

    st.subheader(
        "🤖 Employee Attrition Prediction"
    )

    st.write(
        "A Random Forest classification model is used "
        "to estimate employee attrition probability."
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [
        "Age",
        "MonthlyIncome",
        "JobLevel",
        "JobSatisfaction",
        "OverTime",
        "YearsAtCompany"
    ]

    ml_df = df[
        features + ["Attrition"]
    ].copy()

    # Convert categorical values
    ml_df["OverTime"] = (
        ml_df["OverTime"]
        .map(
            {
                "Yes": 1,
                "No": 0
            }
        )
    )

    ml_df["Attrition"] = (
        ml_df["Attrition"]
        .map(
            {
                "Yes": 1,
                "No": 0
            }
        )
    )

    X = ml_df[features]
    y = ml_df["Attrition"]

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with m2:

        st.metric(
            "Training Records",
            len(X_train)
        )

    with m3:

        st.metric(
            "Testing Records",
            len(X_test)
        )

    st.divider()

    # --------------------------------------------------------
    # EMPLOYEE PREDICTION
    # --------------------------------------------------------

    st.write(
        "### 🔮 Predict Employee Attrition Risk"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        input_age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30
        )

        input_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=50000,
            value=5000
        )

    with col2:

        input_job_level = st.slider(
            "Job Level",
            min_value=1,
            max_value=5,
            value=2
        )

        input_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

    with col3:

        input_overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

        input_years = st.number_input(
            "Years at Company",
            min_value=0,
            max_value=40,
            value=5
        )

    predict = st.button(
        "🔮 Predict Attrition Risk",
        use_container_width=True
    )

    if predict:

        overtime_value = (
            1
            if input_overtime == "Yes"
            else 0
        )

        input_data = pd.DataFrame(
            [[
                input_age,
                input_income,
                input_job_level,
                input_satisfaction,
                overtime_value,
                input_years
            ]],
            columns=features
        )

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        probability_percentage = (
            probability * 100
        )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability_percentage >= 70:

            st.error(
                f"🔴 HIGH RISK — "
                f"Attrition Probability: "
                f"{probability_percentage:.2f}%"
            )

        elif probability_percentage >= 40:

            st.warning(
                f"🟡 MEDIUM RISK — "
                f"Attrition Probability: "
                f"{probability_percentage:.2f}%"
            )

        else:

            st.success(
                f"🟢 LOW RISK — "
                f"Attrition Probability: "
                f"{probability_percentage:.2f}%"
            )

        st.progress(
            min(
                int(probability_percentage),
                100
            )
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.divider()

    st.write(
        "### 🔍 Feature Importance"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True
    )

    fig8, ax8 = plt.subplots()

    ax8.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax8.set_title(
        "Random Forest Feature Importance"
    )

    ax8.set_xlabel(
        "Importance"
    )

    ax8.invert_yaxis()

    st.pyplot(fig8)

    plt.close(fig8)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.write(
        "### 📈 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.write(
        "### 🧩 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig9, ax9 = plt.subplots()

    ax9.imshow(cm)

    ax9.set_title(
        "Confusion Matrix"
    )

    ax9.set_xlabel(
        "Predicted"
    )

    ax9.set_ylabel(
        "Actual"
    )

    ax9.set_xticks(
        [0, 1]
    )

    ax9.set_yticks(
        [0, 1]
    )

    ax9.set_xticklabels(
        ["Stayed", "Left"]
    )

    ax9.set_yticklabels(
        ["Stayed", "Left"]
    )

    for i in range(2):

        for j in range(2):

            ax9.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig9)

    plt.close(fig9)


# ============================================================
# TAB 5 - EMPLOYEE DATA
# ============================================================

with data_tab:

    st.subheader(
        "📋 Employee Data"
    )

    search = st.text_input(
        "🔎 Search by Employee Number, Job Role or Department"
    )

    display_df = filtered_df.copy()

    if search:

        search = search.lower()

        employee_match = (
            display_df["EmployeeNumber"]
            .astype(str)
            .str.lower()
            .str.contains(
                search,
                na=False
            )
        )

        role_match = (
            display_df["JobRole"]
            .astype(str)
            .str.lower()
            .str.contains(
                search,
                na=False
            )
        )

        department_match = (
            display_df["Department"]
            .astype(str)
            .str.lower()
            .str.contains(
                search,
                na=False
            )
        )

        display_df = display_df[
            employee_match
            | role_match
            | department_match
        ]

    st.write(
        f"Showing **{len(display_df)}** employees"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Employee Data",
        data=csv_data,
        file_name="employee_attrition_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Prediction & HR Analytics | "
    "Python • Pandas • Scikit-learn • Matplotlib • Streamlit"
)
