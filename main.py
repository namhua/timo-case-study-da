import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Timo Digital Bank Case Study",
        page_icon="",
    )

    st.title("Timo Digital Bank Case Study - DA")
    st.write("Author: Hua Dai Nam")
    st.write("Published date: April 17, 2024")

    st.markdown("# Required packages")
    with st.echo():
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sys
        from datetime import datetime
        from io import StringIO

    st.markdown("# Dataset understanding")
    st.markdown("## Read the dataset")
    with st.echo():
        file_path = r"./data/txn_history_dummysample.csv"
        df = pd.read_csv(file_path)

        output = df.head(5)
        print(output)
    st.dataframe(output)

    st.markdown("## Inspect the dataset")

    with st.echo():
        def get_n_row_col(df: pd.DataFrame):
            df_shape = df.shape
            n_rows = f"{df_shape[0]:,}"
            n_columns = f"{df_shape[1]:,}"
            return n_rows, n_columns

        n_rows, n_columns = get_n_row_col(df)

        output = f"This dataset has {n_rows} rows and {n_columns} columns."
        print(output)
    st.write(output)

    st.write("\nBasic information of the dataset:")
    with st.echo():
        string_buffer = StringIO()
        sys.stdout = string_buffer

        df.info()

        sys.stdout = sys.__stdout__

        output = string_buffer.getvalue()
        print(output)
    st.text(output)

    st.write(
        """
Dataset description:
- `account_id`: a code consisting of 8 characters, starting with "ID" followed by unique numbers assigned to each customer.
- `date_of_birth`: the date of birth of each customer.
- `txn_ts`: the timestamp of each transaction.
- `txn_amount`: the amount of each transaction.
- `txn_type_code`: a code representing the type of each transaction.
"""
    )

    st.write(
        """
Initially, ensure the `date_of_birth` column is converted to the *date* data type, the `txn_ts` column to *datetime*, and the `txn_type_code` column to *string*, aligning them accurately with the nature of the data they contain:
"""
    )

    with st.echo():
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"]).dt.date
        df["txn_ts"] = pd.to_datetime(df["txn_ts"])
        df["txn_type_code"] = df["txn_type_code"].astype(str)

        string_buffer = StringIO()
        sys.stdout = string_buffer

        df.info()
        
        sys.stdout = sys.__stdout__

        output = string_buffer.getvalue()
        print(output)
    st.text(output)

    st.markdown("## Explain \"txn_type_code\"")
    st.write(
        """
The business believes that the transaction types contained are:
- **Internal Transfers** (sender & recipient are both SuperBank customers): This type may involve negative values for senders and positive values for recipients. The `txn_ts` values and the range of `txn_amount` values vary.
- **Interbank Transfer** (recipient is not a SuperBank customer): This type typically involves negative values for senders. The `txn_ts` values and the range of `txn_amount` values vary.
- **Savings Account Withdrawal** (money pulled from savings account to the customer’s current account): This type involves only positive values. The `txn_ts` values and the range of `txn_amount` values vary.
- **Auto Recurring Savings Account Contribution** (money automatically pulled from savings account to the customer’s current account based on e.g. monthly rule): This type involves only positive values. The `txn_ts` values correspond to the customer's chosen interest accrual schedule (e.g., daily, monthly, or annually), and the `txn_amount` values vary.
- **Manual Savings Account Contribution** (one-time contribution from a current to a savings account): This type involves only negative values. The `txn_ts` values and the range of `txn_amount` values vary.
- **Phone Top-Up** (airtime purchase): This type involves only negative values. The `txn_ts` values vary, and the range of `txn_amount` values is relatively small.
- **Gift Payments** (special transaction type – recipient can but does not have to be a SuperBank customer): This type involves only positive values. The `txn_ts` values vary, and the range of `txn_amount` values is relatively small.
"""
    )

    with st.echo():
        total_unique_code = df["txn_type_code"].nunique()
        output = f"There are {total_unique_code} transaction types in the dataset, each corresponding to one of the seven transaction types described by the business."
        print(output)
    st.write(output)

    st.write("Function to inspect each type:")
    with st.echo():
        def inspect_txn_type_code(df: pd.DataFrame, type_code: str) -> pd.DataFrame:
            result = df[df["txn_type_code"]==type_code]
            total_trans = len(result)
            min_amount = result["txn_amount"].min()
            max_amount = result["txn_amount"].max()
            mean_amount = result["txn_amount"].mean()
            mode_amount = result["txn_amount"].mode()

            # mode can have multiple values, so it's returned as a Series
            for i, mode_value in enumerate(mode_amount):
                mode_frequency = result["txn_amount"].value_counts()[mode_value]

            # redirect stdout to capture the output of print statements
            string_buffer = StringIO()
            sys.stdout = string_buffer
            
            print(f"- total_transactions: {total_trans:,.0f}")
            print(f"- min_amount: {min_amount:,.2f}")
            print(f"- max_amount: {max_amount:,.2f}")
            print(f"- mean_amount: {mean_amount:,.2f}")
            print(f"- mode {i+1}: {mode_value:,.2f}, frequency: {mode_frequency:,.0f}")
            print(f"- result:")

            # reset stdout
            sys.stdout = sys.__stdout__

            # get the captured output
            printed_output = string_buffer.getvalue()

            return result, printed_output

    st.write("type_1:")
    with st.echo():
        type_1_result, type_1_output = inspect_txn_type_code(df, "1")
        print(type_1_output)
    st.text(type_1_output)
    st.dataframe(type_1_result.head(5))
    st.write(
        """
The txn_type_code `1` corresponds to the `Internal Transfers` type. This is because the `txn_amount` contains both negative and positive values, indicating transactions between SuperBank customers, and both the `txn_ts` values and `txn_amount` values exhibit variability.
"""
    )

    st.write("type_2:")
    with st.echo():
        type_2_result, type_2_output = inspect_txn_type_code(df, "2")
        print(type_2_output)
    st.text(type_2_output)
    st.dataframe(type_2_result.head(5))
    st.write(
        """
The txn_type_code `2` corresponds to the `Interbank Transfer` type. This is because the `txn_amount` contains only negative values, indicating transactions involving senders are SuperBank customers. Additionally, both the `txn_ts` values and `txn_amount` values exhibit variability.
"""
    )

    st.write("type_3:")
    with st.echo():
        type_3_result, type_3_output = inspect_txn_type_code(df, "3")
        print(type_3_output)
    st.text(type_3_output)
    st.dataframe(type_3_result.head(5))

    with st.echo():
        output = type_3_result[type_3_result["account_id"]=="ID000007"]
        print(output)
    st.dataframe(output)

    st.write("type_4:")
    with st.echo():
        type_4_result, type_4_output = inspect_txn_type_code(df, "4")
        print(type_4_output)
    st.text(type_4_output)
    st.dataframe(type_4_result.head(5))

    with st.echo():
        output = type_4_result.sort_values(["account_id", "txn_ts"]).head(10)
        print(output)
    st.dataframe(output)
    st.write(
        """
The txn_type_code `4` corresponds to the `Auto Recurring Savings Account Contribution` type. This is because the `txn_ts` values vary according to daily, monthly, and yearly rules, which correspond to the customer's chosen interest accrual schedule. Additionally, the `txn_amount` values exhibit variability.
"""
    )

    st.write("type_5:")
    with st.echo():
        type_5_result, type_5_output = inspect_txn_type_code(df, "5")
        print(type_5_output)
    st.text(type_5_output)
    st.dataframe(type_5_result.head(5))
    st.write(
        """
The txn_type_code `5` corresponds to the `Phone Top-Up` type. This is because the `txn_amount` contains only negative values, indicating deductions for airtime purchases. Additionally, the `txn_ts` values vary, and the range of `txn_amount` values is relatively small.
"""
    )

    st.write("type_6:")
    with st.echo():
        type_6_result, type_6_output = inspect_txn_type_code(df, "5")
        print(type_6_output)
    st.text(type_6_output)
    st.dataframe(type_6_result.head(5))
    st.write(
        """
The txn_type_code `6` corresponds to the `Gift Payments` type. This is because the `txn_amount` contains only positive values, indicating transactions involving gift payment receiving. Additionally, the `txn_ts` values vary, and the range of `txn_amount` values is relatively small.
"""
    )

    st.write("type_7:")
    with st.echo():
        type_7_result, type_7_output = inspect_txn_type_code(df, "7")
        print(type_7_output)
    st.text(type_7_output)
    st.dataframe(type_7_result.head(5))

    with st.echo():
        output = type_7_result[type_7_result["account_id"]=="ID000038"]
        print(output)
    st.dataframe(output)

    st.write(
        """
Two transaction types, namely `Saving Account Withdrawal` and `Manual Savings Account Contribution`, have not been assigned a `txn_type_code`, corresponding to unclassified codes `3` and `7`.
For clarity, `Saving Account Withdrawal` transactions can only contain positive values, while `Manual Savings Account Contribution` transactions can only contain negative values. However, both `3` and `7` codes currently only contain positive values. This discrepancy may arise from:
- Mislabeling: Data originally belonging to type `7` or type `3` may have been mistakenly recorded as positive values instead of negative values.
- Omission: The business may have failed to include the appropriate type in the initial classification.
"""
    )
    st.write("> To address this issue, assuming that data associated with type `7` comprises only negative values. Consequently, type `7` should be relabeled as `Manual Savings Account Contribution`, while type `3` should be identified as `Saving Account Withdrawal`.")

    st.markdown("# Dataset cleansing")

    st.write(
        """
Based on the assumption, convert the values of amount associated with type 7 transactions into negative values:
"""
    )
    with st.echo():
        cleaned_df = df.copy()
        cleaned_df.loc[cleaned_df["txn_type_code"]=="7", "txn_amount"] *= -1

        output = cleaned_df[cleaned_df["txn_type_code"]=="7"].head(5)
        print(output)
    st.dataframe(output)

    st.write(
        """
Display the total number of null values and their respective percentages compared to the total number of records per column:
"""
    )
    with st.echo():
        def null_summary(df: pd.DataFrame) -> pd.DataFrame:
            total_null = df.isnull().sum().map("{:,}".format)
            percentage_null = ((df.isnull().sum() / len(df)) * 100).map("{:,.2f}%".format)
            result = pd.concat([total_null, percentage_null], axis=1, keys=["Total null", "Percentage null"])
            return result

        output = null_summary(cleaned_df)
        print(output)
    st.dataframe(output)
    st.write("> There are no null values to handle.")
    st.write(
        """
In case of null values needs to be handled:
Set the **threshold** to handle with null values:
- `the percentage of null values > threshold`: remove null value records to avoid excessive distortion of the data.
- `the percentage of null values < threshold`: fill null values with the dummy data based on the context (ffill, bfill, linear, polynomial, etc.).
In this scenario, employ the *mean* method to handle missing values in numerical data, and *mode* method to handle missing values in the categorical data:
"""
    )
    with st.echo():
        def handle_null_values(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
            df_clean = df.copy()
            for col in df_clean.columns:
                percentage_null = (df_clean[col].isnull().sum() / len(df_clean[col])) * 100
                if percentage_null == 0:
                    continue
                else:
                    if percentage_null > threshold:
                        df_clean.dropna(subset=col, inplace=True)
                    else:
                        # numeric column
                        if df_clean[col].dtype in ["int64", "float64"]:
                            mean_val = df_clean[col].mean()
                            df_clean[col].fillna(value=mean_val, inplace=True)
                        # categorical (object) column
                        else:
                            mode_val = df_clean[col].mode()[0]
                            df_clean[col].fillna(value=mode_val, inplace=True)
            return df_clean

        threshold = 0.05 # 5%
        cleaned_df = handle_null_values(cleaned_df, threshold)

        # check null values after handeling
        output = null_summary(cleaned_df)
        print(output)
    st.dataframe(output)

    with st.echo():
        n_rows, n_columns = get_n_row_col(df)
        output = f"The cleaned dataset now consists of {n_rows} rows and {n_columns} columns."
        print(output)
    st.write(output)

    st.write("Create a \"txn_type\" column to map \"txn_type_code\":")
    with st.echo():
        type_explain = {
            "1": "Internal Transfers",
            "2": "Interbank Transfer",
            "3": "Saving Account Withdrawal",
            "4": "Auto Recurring Savings Account Contribution",
            "5": "Phone Top-Up",
            "6": "Gift  Payments",
            "7": "Manual Savings Account Contribution",
        }

        # create a "txn_type" column based on the mapping
        cleaned_df["txn_type"] = cleaned_df["txn_type_code"].astype(str).map(type_explain)

        output = cleaned_df.head(5)
        print(output)
    st.dataframe(output)

    st.markdown("# Exploratory data analysis (EDA)")

    st.markdown("## Demographic information")

    st.write("Calculate the \"age\" column:")
    with st.echo():
        today = datetime.today()

        # conversion to make sure the data type of "date_of_birth" column is datetime
        cleaned_df["date_of_birth"] = pd.to_datetime(cleaned_df["date_of_birth"])

        # calculate "age" column
        cleaned_df["year_of_birth"] = cleaned_df["date_of_birth"].dt.year
        cleaned_df["current_year"] = today.year
        cleaned_df["age"] = cleaned_df["current_year"] - cleaned_df["year_of_birth"]

        output = cleaned_df.head(5)
        print(output)
    st.dataframe(output)

    st.write("Get demographic information:")
    with st.echo():
        demographic_info = cleaned_df[["account_id", "age"]]
        demographic_info = demographic_info.drop_duplicates(subset=["account_id", "age"], ignore_index=True)

        output = demographic_info.head(5)
        print(output)
    st.dataframe(output)

    st.write("Describe \"age\" column:")
    with st.echo():
        output = demographic_info["age"].describe()
        print(output)
    st.dataframe(output)

    st.write("Age distribution chart:")
    with st.echo():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(demographic_info["age"], bins=30, color="skyblue", edgecolor="black")
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        st.pyplot(fig)

    st.write(
        """
This age distribution chart illustrates the frequency of age groups within the dataset. Observing this chart, several insights can be gleaned:
- **Age Distribution:** The histogram depicts the distribution of user ages in the dataset. Notably, the most prevalent age group appears to be customers aged between 20 and 30 years old.
- **Data Dispersion:** The width and shape of the age distribution reveal the diversity of ages among users. This graph exhibits an uneven distribution, indicating a significant presence of users within the 20-30 age group.
- **Identification of Outliers:** Anomalies in the age distribution, particularly age values beyond 60, may suggest outliers. These outliers could indicate potentially inaccurate data or prompt further investigation into user groups aged over 60.
- **Relevance:** Histograms offer insights into the relationship between age and other dataset variables such as transaction count, transaction value, and transaction type. This analysis aids in understanding the behavior and preferences of distinct age demographics.
"""
    )

    st.markdown("## Average  balances and total transactions per user")

    st.write("Assuming that the customer's opening balance is 0:")
    with st.echo():
        mapping_groupby = {
            "txn_ts": "count",
            "txn_amount": "sum",
        }

        # get info per user
        info_per_user = cleaned_df.groupby("account_id").agg(mapping_groupby).reset_index()
        info_per_user.columns = ["account_id", "total_transactions", "trans_delta"]
        info_per_user["opening_balane"] = 0
        info_per_user["ending_balance"] = info_per_user["opening_balane"] + info_per_user["trans_delta"]

        output = info_per_user.head(10)
        print(output)
    st.dataframe(output)

    st.write("Detect outliers:")
    with st.echo():
        outliers = info_per_user.sort_values(by=["total_transactions", "ending_balance"], ascending=[True, False])

        # top 10 accounts with the fewest number of transactions and the highest amount of transactions
        output = outliers.head(10)
        print(output)
    st.dataframe(output)

    st.write(
        """
This scenario could arise due to two potential factors:
- Assuming that the customer's opening balance is zero.
- The dataset covers only a specific period of time, without including historical balances.        
"""
    )

    st.write(
        """
These 10 exception customers will be stored separately for further actions and removed from the dataset to avoid affecting the calculation of the average value:
"""
    )
    with st.echo():
        top_10_outliers = outliers.head(10)
        outlier_account_ids = top_10_outliers["account_id"].tolist()

        # remove these customers from info_per_user
        filtered_info_per_user = info_per_user[~info_per_user["account_id"].isin(outlier_account_ids)]

    st.write(
        """
In addition, customers over the age of 60 are also stored separately for further actions and removed from the dataset when calculating average values:
"""
    )
    with st.echo():
        above_60 = cleaned_df[cleaned_df["age"] > 60]
        above_60_account_ids = above_60["account_id"].tolist()

        # remove these customers from info_per_user
        filtered_info_per_user = filtered_info_per_user[~filtered_info_per_user["account_id"].isin(above_60_account_ids)]
    
    st.write("Get average balances (based on ending balances):")
    with st.echo():
        avg_balances = filtered_info_per_user["ending_balance"].mean()

        output = f"The average balances of this dataset is {avg_balances:,.2f}"
        print(output)
    st.write(output)

    st.write("Get total transactions per user:")
    with st.echo():
        trans_per_user = info_per_user[["account_id", "total_transactions"]]

        output = trans_per_user.head(10)
        print(output)
    st.dataframe(output)

    st.markdown("## Average transaction sizes")
    
    st.write("Quick preprocessing before calculating:")
    with st.echo():
        cleaned_df = cleaned_df.copy() # to avoid SettingWithCopyWarning
        cleaned_df["abs_txn_amount"] = cleaned_df["txn_amount"].abs()
        
        # remove outliers from "cleaned_df"
        filtered_cleaned_df = cleaned_df[~cleaned_df["account_id"].isin(outlier_account_ids)]
        filtered_cleaned_df = filtered_cleaned_df[~filtered_cleaned_df["account_id"].isin(above_60_account_ids)]

        output = filtered_cleaned_df.head(5)
        print(output)
    st.dataframe(output)

    st.write("Average transaction size per account:")
    with st.echo():
        avg_trans_per_user = cleaned_df.groupby("account_id")["abs_txn_amount"].mean().reset_index()
        avg_trans_per_user.columns = ["account_id", "avg_transaction_size"]
        avg_trans_per_user["avg_transaction_size"] = avg_trans_per_user["avg_transaction_size"].round(2)

        output = avg_trans_per_user.head(5)
        print(output)
    st.dataframe(output)

    st.write("Average transaction sizes of the dataset:")
    with st.echo():
        avg_trans_sizes = filtered_cleaned_df["abs_txn_amount"].mean()

        output = f"The average transaction sizes of the dataset is {avg_trans_sizes:,.2f}"
        print(output)
    st.write(output)

    st.markdown("## Information per \"txn_type\"")

    st.write("Get information per \"txn_type\":")
    with st.echo():
        mapping_groupby = {
            "account_id": "nunique",
            "age": "mean",
            "abs_txn_amount": "mean",
        }

        info_per_type = filtered_cleaned_df.groupby("txn_type").agg(mapping_groupby).reset_index()
        info_per_type.columns = ["txn_type", "total_accounts", "avg_age", "avg_amount"]
        info_per_type["avg_age"] = info_per_type["avg_age"].round(0)
        info_per_type["avg_amount"] = info_per_type["avg_amount"].round(2)

        output = info_per_type
        print(output)
    st.dataframe(output)

    st.write("Visualization:")
    # plotting total_accounts per txn_type
    with st.echo():
        fig, ax = plt.subplots(figsize=(5, 5))
        x = info_per_type["txn_type"]
        y = info_per_type["total_accounts"]
        ax.bar(x, y, color="skyblue")
        ax.set_title("Total Accounts per Transaction Type")
        ax.set_xlabel("Transaction Type")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=90)
        ax.set_ylabel("Total Accounts")
        st.pyplot(fig)

    # plotting avg_age per txn_type
    with st.echo():
        fig, ax = plt.subplots(figsize=(5, 5))
        x = info_per_type["txn_type"]
        y = info_per_type["avg_age"]
        ax.bar(x, y, color="skyblue")
        ax.set_title("Average ages per Transaction Type")
        ax.set_xlabel("Transaction Type")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=90)
        ax.set_ylabel("Average ages")
        st.pyplot(fig)

    # plotting avg_amount per txn_type
    with st.echo():
        fig, ax = plt.subplots(figsize=(5, 5))
        x = info_per_type["txn_type"]
        y = info_per_type["avg_amount"]
        ax.bar(x, y, color="skyblue")
        ax.set_title("Average amounts per Transaction Type")
        ax.set_xlabel("Transaction Type")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=90)
        ax.set_ylabel("Average amounts")
        st.pyplot(fig)

if __name__ == "__main__":
    run()
