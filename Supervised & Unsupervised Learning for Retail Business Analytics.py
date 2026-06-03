#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IOM103 Coursework Source Code
Sales Prediction and Customer Segmentation Analysis

Converted from the original Jupyter notebook into a single runnable Python file.

How to run:
    python IOM103_Coursework.py

Required CSV files should be placed in the same folder as this script, or in /mnt/data:
    1. Sales Dataset_Task A1.csv
    2. Customer Segmentation_Task B 2.csv

Optional:
    python IOM103_Coursework.py --data-dir "path/to/csv/folder" --output-dir "outputs"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

SALES_FILENAME = "Sales Dataset_Task A1.csv"
CUSTOMER_FILENAME = "Customer Segmentation_Task B 2.csv"


def configure_plot_style() -> None:
    """Configure a consistent plotting style for all figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    )
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.unicode_minus"] = False


def print_section(title: str) -> None:
    """Print a clear section heading in the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_df(df: pd.DataFrame, title: Optional[str] = None, max_rows: int = 12) -> None:
    """Print a DataFrame in a readable console-friendly format."""
    if title:
        print_section(title)

    if len(df) > max_rows:
        print(df.head(max_rows).to_string())
        print(f"... ({len(df) - max_rows} more rows)")
    else:
        print(df.to_string())


def locate_file(filename: str, data_dir: Optional[Path] = None) -> Path:
    """
    Locate an input CSV file.

    Search order:
    1. The user-specified data directory, if provided.
    2. The current working directory.
    3. The folder where this script is stored.
    4. /mnt/data, useful when running in the original notebook environment.
    """
    candidates = []

    if data_dir is not None:
        candidates.append(data_dir / filename)

    candidates.extend(
        [
            Path.cwd() / filename,
            Path(__file__).resolve().parent / filename,
            Path("/mnt/data") / filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched_locations = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot find required file: {filename}\n"
        f"Please place it in the same folder as this script or use --data-dir.\n"
        f"Searched locations:\n{searched_locations}"
    )


def parse_dates_safely(date_series: pd.Series) -> pd.Series:
    """
    Convert dates robustly.

    The original notebook used format='mixed', which requires newer pandas versions.
    This helper first tries that method and then falls back to a more compatible parser.
    """
    try:
        return pd.to_datetime(date_series, format="mixed", dayfirst=True, errors="coerce")
    except TypeError:
        return pd.to_datetime(date_series, dayfirst=True, errors="coerce")


def save_or_show(output_dir: Path, filename: str, show_plots: bool) -> None:
    """Save the current figure and optionally display it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Task A1: Sales Prediction
# ---------------------------------------------------------------------

def load_and_preprocess_sales_data(sales_file: Path) -> pd.DataFrame:
    """Load the sales dataset and create date/business-related variables."""
    sales_df = pd.read_csv(sales_file)

    print_section("Task A1 - Sales Dataset Loading")
    print(f"Dataset shape: {sales_df.shape[0]} rows × {sales_df.shape[1]} columns")
    print(f"Original columns: {list(sales_df.columns)}")
    print("\nMissing value check:")
    print(sales_df.isnull().sum().to_string())
    print_df(sales_df.head(), "\nData preview")

    sales_df["Date"] = parse_dates_safely(sales_df["Date"])

    # Time features for exploring seasonal or calendar-related effects.
    sales_df["Year"] = sales_df["Date"].dt.year
    sales_df["Month"] = sales_df["Date"].dt.month
    sales_df["Quarter"] = sales_df["Date"].dt.quarter
    sales_df["Weekday"] = sales_df["Date"].dt.day_name()

    # Business indicators for descriptive analysis.
    sales_df["Revenue"] = sales_df["Price"] * sales_df["Sales"]
    sales_df["NetPrice"] = sales_df["Price"] * (1 - sales_df["Discount"] / 100)
    sales_df["DiscountValue"] = sales_df["Price"] * sales_df["Discount"] / 100

    print_section("Task A1 - Preprocessing Summary")
    valid_dates = sales_df["Date"].dropna()
    if not valid_dates.empty:
        print(f"Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
    else:
        print("Date range: no valid date values were parsed")

    print(f"Product categories: {sales_df['Product'].nunique()} -> {list(sales_df['Product'].unique())}")
    print(
        f"Customer types: {sales_df['CustomerType'].nunique()} "
        f"-> {list(sales_df['CustomerType'].unique())}"
    )
    print(f"Missing values after preprocessing: {sales_df.isnull().sum().sum()}")
    print_df(sales_df.head(), "\nProcessed sales data preview")

    return sales_df


def analyse_sales_descriptively(
    sales_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform descriptive, correlation, and marketing-effectiveness analysis."""
    print_section("Core Business Metrics")

    core_metrics = {
        "Total Transactions": f"{len(sales_df):,}",
        "Total Units Sold": f"{sales_df['Sales'].sum():,}",
        "Total Gross Revenue": f"JPY {sales_df['Revenue'].sum():,.2f}",
        "Average Price": f"JPY {sales_df['Price'].mean():.2f}",
        "Average Discount Rate": f"{sales_df['Discount'].mean():.2f}%",
        "Average Marketing Effort": f"JPY {sales_df['MarketingEffort'].mean():,.2f}",
        "Average Sales per Transaction": f"{sales_df['Sales'].mean():.2f} units",
    }

    for key, value in core_metrics.items():
        print(f"{key:<32}: {value}")

    numeric_summary = (
        sales_df[["Price", "Discount", "MarketingEffort", "Sales", "Revenue", "NetPrice"]]
        .describe()
        .round(2)
    )
    print_df(numeric_summary, "\nNumeric variable summary")

    product_summary = (
        sales_df.groupby("Product")
        .agg(
            Transactions=("Sales", "count"),
            TotalUnits=("Sales", "sum"),
            AvgSales=("Sales", "mean"),
            AvgPrice=("Price", "mean"),
            AvgDiscount=("Discount", "mean"),
            TotalRevenue=("Revenue", "sum"),
        )
        .round(2)
        .sort_values("TotalRevenue", ascending=False)
    )

    customer_type_summary = (
        sales_df.groupby("CustomerType")
        .agg(
            Transactions=("Sales", "count"),
            TotalUnits=("Sales", "sum"),
            AvgSales=("Sales", "mean"),
            AvgMarketing=("MarketingEffort", "mean"),
            TotalRevenue=("Revenue", "sum"),
        )
        .round(2)
        .sort_values("TotalRevenue", ascending=False)
    )

    print_df(product_summary, "Product-Level Summary")
    print_df(customer_type_summary, "Customer-Type Summary")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    sns.barplot(
        data=product_summary.reset_index(),
        x="TotalRevenue",
        y="Product",
        ax=axes[0, 0],
        color="#4C72B0",
    )
    axes[0, 0].set_title("Revenue by Product Category")
    axes[0, 0].set_xlabel("Total Revenue")
    axes[0, 0].set_ylabel("Product")

    sns.barplot(
        data=customer_type_summary.reset_index(),
        x="CustomerType",
        y="TotalRevenue",
        ax=axes[0, 1],
        color="#4C72B0",
    )
    axes[0, 1].set_title("Revenue by Customer Type")
    axes[0, 1].set_xlabel("Customer Type")
    axes[0, 1].set_ylabel("Total Revenue")

    sns.histplot(sales_df["Sales"], bins=25, kde=True, ax=axes[1, 0], color="#4C72B0")
    axes[1, 0].set_title("Distribution of Sales Units")
    axes[1, 0].set_xlabel("Sales")

    sns.histplot(sales_df["Price"], bins=25, kde=True, ax=axes[1, 1], color="#4C72B0")
    axes[1, 1].set_title("Distribution of Product Price")
    axes[1, 1].set_xlabel("Price")
    save_or_show(output_dir, "01_sales_overview.png", show_plots)

    corr_columns = [
        "Price",
        "Discount",
        "MarketingEffort",
        "Sales",
        "Revenue",
        "NetPrice",
        "DiscountValue",
    ]
    corr_matrix = sales_df[corr_columns].corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Matrix of Sales Variables")
    save_or_show(output_dir, "02_sales_correlation_matrix.png", show_plots)

    key_corr = pd.DataFrame(
        {
            "Relationship": [
                "MarketingEffort vs Sales",
                "Price vs Sales",
                "Discount vs Sales",
                "Price vs Revenue",
                "NetPrice vs Revenue",
            ],
            "Correlation": [
                sales_df["MarketingEffort"].corr(sales_df["Sales"]),
                sales_df["Price"].corr(sales_df["Sales"]),
                sales_df["Discount"].corr(sales_df["Sales"]),
                sales_df["Price"].corr(sales_df["Revenue"]),
                sales_df["NetPrice"].corr(sales_df["Revenue"]),
            ],
            "Interpretation": [
                "Negligible",
                "Negligible",
                "Very weak negative",
                "Very strong positive",
                "Strong positive",
            ],
        }
    )

    print_df(key_corr.round(3), "Key Correlation Coefficients")

    sales_df["MarketingLevel"] = pd.qcut(
        sales_df["MarketingEffort"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

    marketing_summary = (
        sales_df.groupby("MarketingLevel", observed=True)
        .agg(
            Transactions=("Sales", "count"),
            AvgSales=("Sales", "mean"),
            AvgRevenue=("Revenue", "mean"),
            AvgMarketing=("MarketingEffort", "mean"),
            AvgDiscount=("Discount", "mean"),
        )
        .round(2)
    )

    print_df(marketing_summary, "Marketing Effort Binning Analysis")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(
        data=marketing_summary.reset_index(),
        x="MarketingLevel",
        y="AvgSales",
        ax=axes[0],
        color="#4C72B0",
    )
    axes[0].set_title("Average Sales by Marketing Level")
    axes[0].set_xlabel("Marketing Effort Level")
    axes[0].set_ylabel("Average Sales")
    axes[0].tick_params(axis="x", rotation=25)

    sns.scatterplot(
        data=sales_df,
        x="MarketingEffort",
        y="Sales",
        hue="CustomerType",
        alpha=0.65,
        ax=axes[1],
    )
    axes[1].set_title("Marketing Effort and Sales by Customer Type")
    axes[1].set_xlabel("Marketing Effort")
    axes[1].set_ylabel("Sales")
    save_or_show(output_dir, "03_marketing_effectiveness.png", show_plots)

    return product_summary, customer_type_summary, marketing_summary


def train_sales_prediction_models(
    sales_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train Linear Regression, Regression Tree, and Random Forest models."""
    model_df = sales_df[
        [
            "Price",
            "Discount",
            "MarketingEffort",
            "Product",
            "CustomerType",
            "Month",
            "Quarter",
            "Sales",
        ]
    ].copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["Product", "CustomerType"],
        drop_first=True,
    )

    X = model_df.drop("Sales", axis=1)
    y = model_df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print_section("Model Dataset Preparation")
    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    print(f"Number of model features after encoding: {X_train.shape[1]}")
    print("\nFeature list:")
    print(list(X.columns))

    models = {
        "Mean Baseline": None,
        "Linear Regression": LinearRegression(),
        "Regression Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    model_results = []
    trained_models = {}

    for model_name, model in models.items():
        if model is None:
            train_pred = np.full(y_train.shape, y_train.mean())
            test_pred = np.full(y_test.shape, y_train.mean())
        else:
            model.fit(X_train_scaled, y_train)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            trained_models[model_name] = model

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        model_results.append(
            {
                "Model": model_name,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "RMSE Gap": test_rmse - train_rmse,
                "Test MAE": mean_absolute_error(y_test, test_pred),
                "Test R2": r2_score(y_test, test_pred),
            }
        )

    model_results_df = pd.DataFrame(model_results).round(4)

    print_df(model_results_df, "Model Performance Comparison")

    required_models = model_results_df[
        model_results_df["Model"].isin(
            ["Linear Regression", "Regression Tree", "Random Forest"]
        )
    ]
    best_required_model = required_models.sort_values("Test RMSE").iloc[0]
    print(
        f"\nBest required model based on Test RMSE: "
        f"{best_required_model['Model']} ({best_required_model['Test RMSE']:.2f})"
    )

    rmse_plot_df = required_models.melt(
        id_vars="Model",
        value_vars=["Train RMSE", "Test RMSE"],
        var_name="Dataset",
        value_name="RMSE",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(data=rmse_plot_df, x="Model", y="RMSE", hue="Dataset", ax=axes[0])
    axes[0].set_title("Train-Test RMSE Comparison")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=required_models, x="Model", y="RMSE Gap", ax=axes[1], color="#4C72B0")
    axes[1].set_title("Generalisation Gap")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Test RMSE - Train RMSE")
    axes[1].tick_params(axis="x", rotation=15)
    save_or_show(output_dir, "04_model_performance.png", show_plots)

    rf_model = trained_models["Random Forest"]
    feature_importance = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": rf_model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print_df(feature_importance.round(4), "Random Forest Feature Importance")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(10),
        x="Importance",
        y="Feature",
        color="#4C72B0",
    )
    plt.title("Top Random Forest Feature Importance Scores")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_or_show(output_dir, "05_random_forest_feature_importance.png", show_plots)

    return model_results_df, required_models, feature_importance


# ---------------------------------------------------------------------
# Task B: Customer Segmentation
# ---------------------------------------------------------------------

def load_customer_data(customer_file: Path) -> pd.DataFrame:
    """Load and inspect the customer segmentation dataset."""
    customer_df = pd.read_csv(customer_file)

    print_section("Task B - Customer Dataset Loading")
    print(f"Dataset shape: {customer_df.shape[0]} rows × {customer_df.shape[1]} columns")
    print(f"Columns: {list(customer_df.columns)}")

    print("\nMissing value check:")
    print(customer_df.isnull().sum().to_string())

    print("\nGender distribution:")
    print(customer_df["Gender"].value_counts().to_string())

    print_df(customer_df.describe().round(2), "\nDescriptive statistics")
    print_df(customer_df.head(), "\nCustomer data preview")

    return customer_df


def analyse_customer_data(
    customer_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
) -> None:
    """Create exploratory visualisations for customer variables."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(customer_df["Age"], bins=20, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")

    sns.histplot(
        customer_df["Annual Income (k$)"],
        bins=20,
        kde=True,
        ax=axes[1],
        color="#4C72B0",
    )
    axes[1].set_title("Annual Income Distribution")
    axes[1].set_xlabel("Annual Income (k$)")

    sns.histplot(
        customer_df["Spending Score (1-100)"],
        bins=20,
        kde=True,
        ax=axes[2],
        color="#4C72B0",
    )
    axes[2].set_title("Spending Score Distribution")
    axes[2].set_xlabel("Spending Score")
    save_or_show(output_dir, "06_customer_variable_distributions.png", show_plots)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=customer_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Gender",
        alpha=0.75,
    )
    plt.title("Income and Spending Score by Gender")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    save_or_show(output_dir, "07_income_spending_by_gender.png", show_plots)


def build_customer_segments(
    customer_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate cluster numbers and build the final K-Means segmentation model."""
    cluster_features = customer_df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()

    inertia_values = []
    silhouette_values = []

    for k in range(2, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(cluster_features)
        inertia_values.append(kmeans_temp.inertia_)
        silhouette_values.append(silhouette_score(cluster_features, labels_temp))

    cluster_eval_df = pd.DataFrame(
        {
            "K": range(2, 11),
            "Inertia": inertia_values,
            "Silhouette Score": silhouette_values,
        }
    ).round(4)

    print_df(cluster_eval_df, "Cluster Number Evaluation")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=cluster_eval_df, x="K", y="Inertia", marker="o", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")

    sns.lineplot(
        data=cluster_eval_df,
        x="K",
        y="Silhouette Score",
        marker="o",
        ax=axes[1],
        color="#55A868",
    )
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    save_or_show(output_dir, "08_cluster_number_evaluation.png", show_plots)

    final_k = 5
    kmeans_model = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    customer_df["Cluster"] = kmeans_model.fit_predict(cluster_features)

    segment_names = {
        0: "Central Mainstream Customers",
        1: "Priority High-Spenders",
        2: "Young Promotion Responders",
        3: "Affluent Low-Engagement Customers",
        4: "Price-Sensitive Minimal Spenders",
    }
    customer_df["Segment Name"] = customer_df["Cluster"].map(segment_names)

    cluster_centers = pd.DataFrame(
        kmeans_model.cluster_centers_,
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    ).round(2)

    cluster_profile = (
        customer_df.groupby(["Cluster", "Segment Name"])
        .agg(
            Count=("CustomerID", "count"),
            AvgAge=("Age", "mean"),
            AvgIncome=("Annual Income (k$)", "mean"),
            AvgSpendingScore=("Spending Score (1-100)", "mean"),
            Female=("Gender", lambda x: (x == "Female").sum()),
            Male=("Gender", lambda x: (x == "Male").sum()),
        )
        .round(2)
    )

    print_df(cluster_centers, "Final Cluster Centers")
    print_df(cluster_profile, "Customer Segment Profiles")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.scatterplot(
        data=customer_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment Name",
        s=75,
        alpha=0.8,
        ax=axes[0, 0],
    )
    axes[0, 0].scatter(
        cluster_centers["Annual Income (k$)"],
        cluster_centers["Spending Score (1-100)"],
        marker="X",
        s=250,
        c="black",
        label="Centroids",
    )
    axes[0, 0].set_title("Customer Segments in Income-Spending Space")
    axes[0, 0].legend(fontsize=7, loc="best")

    cluster_size_df = customer_df["Segment Name"].value_counts().reset_index()
    cluster_size_df.columns = ["Segment Name", "Count"]
    sns.barplot(data=cluster_size_df, x="Count", y="Segment Name", ax=axes[0, 1], color="#4C72B0")
    axes[0, 1].set_title("Segment Size Distribution")

    profile_plot_df = cluster_profile.reset_index().melt(
        id_vars=["Cluster", "Segment Name"],
        value_vars=["AvgAge", "AvgIncome", "AvgSpendingScore"],
        var_name="Metric",
        value_name="Average",
    )
    sns.barplot(data=profile_plot_df, x="Average", y="Segment Name", hue="Metric", ax=axes[1, 0])
    axes[1, 0].set_title("Average Segment Characteristics")

    segment_gender_df = (
        customer_df.groupby(["Segment Name", "Gender"])
        .size()
        .reset_index(name="Count")
    )
    sns.barplot(data=segment_gender_df, x="Count", y="Segment Name", hue="Gender", ax=axes[1, 1])
    axes[1, 1].set_title("Gender Composition by Segment")
    save_or_show(output_dir, "09_customer_segmentation_profiles.png", show_plots)

    return cluster_eval_df, cluster_centers, cluster_profile


# ---------------------------------------------------------------------
# Final summary and main workflow
# ---------------------------------------------------------------------

def print_final_summary(
    sales_df: pd.DataFrame,
    model_results_df: pd.DataFrame,
) -> None:
    """Print a concise final business summary."""
    required_models = model_results_df[
        model_results_df["Model"].isin(
            ["Linear Regression", "Regression Tree", "Random Forest"]
        )
    ]
    best_required_model = required_models.sort_values("Test RMSE").iloc[0]

    print_section("Final Business Summary")

    valid_dates = sales_df["Date"].dropna()
    if not valid_dates.empty:
        date_text = f"{valid_dates.min().date()} to {valid_dates.max().date()}"
    else:
        date_text = "the available period"

    summary_text = f"""
Task A1 - Sales Prediction
1. The sales dataset contains {len(sales_df):,} transactions from {date_text}.
2. Total gross revenue is JPY {sales_df['Revenue'].sum():,.2f}, with {sales_df['Sales'].sum():,} units sold.
3. MarketingEffort has a very weak direct correlation with Sales: {sales_df['MarketingEffort'].corr(sales_df['Sales']):.3f}.
4. Among Linear Regression, Regression Tree and Random Forest, {best_required_model['Model']} has the lowest test RMSE.
5. The Regression Tree overfits heavily, while Random Forest still shows a sizeable train-test gap.

Task B - Customer Segmentation
1. Annual Income and Spending Score were used as the clustering variables.
2. K=5 is selected because it provides a clear and interpretable five-segment structure.
3. The five segments are Central Mainstream Customers, Priority High-Spenders,
   Young Promotion Responders, Affluent Low-Engagement Customers, and Price-Sensitive Minimal Spenders.
4. Priority High-Spenders and Young Promotion Responders should be prioritised for promotion events.
5. Affluent Low-Engagement Customers should be activated with personalised premium recommendations.
"""
    print(summary_text)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the IOM103 sales prediction and customer segmentation analysis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing the two required CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iom103_outputs"),
        help="Folder used to save generated figures.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full coursework workflow from data loading to final summary."""
    args = parse_args()
    configure_plot_style()

    sales_file = locate_file(SALES_FILENAME, args.data_dir)
    customer_file = locate_file(CUSTOMER_FILENAME, args.data_dir)

    print_section("Environment Setup Completed")
    print(f"Sales dataset path: {sales_file}")
    print(f"Customer dataset path: {customer_file}")
    print(f"Output directory: {args.output_dir.resolve()}")

    sales_df = load_and_preprocess_sales_data(sales_file)
    analyse_sales_descriptively(sales_df, args.output_dir, args.show_plots)
    model_results_df, _, _ = train_sales_prediction_models(
        sales_df,
        args.output_dir,
        args.show_plots,
    )

    customer_df = load_customer_data(customer_file)
    analyse_customer_data(customer_df, args.output_dir, args.show_plots)
    build_customer_segments(customer_df, args.output_dir, args.show_plots)

    print_final_summary(sales_df, model_results_df)


if __name__ == "__main__":
    main()
