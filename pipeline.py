"""
Project AI Native: AI-Assisted Coding Assignment

Complete, robust pipeline for a small Customer 360 analytics use case.
Handles data anomalies, tracks dropped records, and exports clean 360/KPI tables.

Expected usage:
    python pipeline.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_sources():
    """
    Dynamically load Excel files with safety checks and error handling.
    """
    file_mapping = {
        "orders": DATA_DIR / "orders_source.xlsx",
        "customers": DATA_DIR / "customers_source.xlsx",
        "tickets": DATA_DIR / "support_tickets_source.xlsx"
    }
    
    dataframes = {}
    
    for name, file_path in file_mapping.items():
        if not file_path.exists():
            error_msg = f"Critical Error: Required file not found at {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logging.info(f"Loading {name} from {file_path.name}...")
            dataframes[name] = pd.read_excel(file_path)
            logging.info(f"Successfully loaded {len(dataframes[name])} rows for {name}.")
            
        except pd.errors.EmptyDataError:
            logging.error(f"The file {file_path.name} is empty.")
            raise
        except ValueError as ve:
            logging.error(f"Value error while parsing {file_path.name}. Is it a valid Excel file? Details: {ve}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading {name}: {e}")
            sys.exit(1)
            
    return dataframes["orders"], dataframes["customers"], dataframes["tickets"]


def clean_customers(customers: pd.DataFrame) -> tuple:
    """Clean customers and track data quality metrics."""
    dq = {}
    initial_count = len(customers)
    
    customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
    customers["customer_id"] = customers["customer_id"].replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})
    
    missing_ids = customers["customer_id"].isna()
    dq["customers_missing_id"] = missing_ids.sum()
    customers = customers[~missing_ids]
    
    duplicates = customers.duplicated(subset=["customer_id"], keep="first")
    dq["customers_duplicate_ids"] = duplicates.sum()
    customers = customers[~duplicates]
    
    customers["customer_name"] = customers["customer_name"].astype(str).str.strip().str.title()
    
    valid_dates = customers["signup_date"].notna().sum()
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    dq["customers_invalid_signup_dates"] = valid_dates - customers["signup_date"].notna().sum()
    
    dq["customers_total_dropped"] = initial_count - len(customers)
    return customers.copy(), dq


def clean_orders(orders: pd.DataFrame) -> tuple:
    """Clean orders, calculate net revenue, and track data quality metrics."""
    dq = {}
    initial_count = len(orders)
    
    orders["order_id"] = orders["order_id"].astype(str).str.strip()
    orders["customer_id"] = orders["customer_id"].astype(str).str.strip()
    orders["customer_id"] = orders["customer_id"].replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})
    
    missing_cust = orders["customer_id"].isna()
    dq["orders_missing_customer_id"] = missing_cust.sum()
    orders = orders[~missing_cust]
    
    duplicates = orders.duplicated(subset=["order_id"], keep="first")
    dq["orders_duplicate_ids"] = duplicates.sum()
    orders = orders[~duplicates]
    
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    invalid_dates = orders["order_date"].isna()
    dq["orders_invalid_dates"] = invalid_dates.sum()
    orders = orders[~invalid_dates]
    
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    invalid_amount = orders["order_amount"].isna() | (orders["order_amount"] <= 0)
    dq["orders_invalid_or_negative_amounts"] = invalid_amount.sum()
    orders = orders[~invalid_amount]
    
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    invalid_discount = orders["discount_pct"].isna() | (orders["discount_pct"] < 0) | (orders["discount_pct"] > 100)
    dq["orders_invalid_discounts_imputed_to_0"] = invalid_discount.sum()
    orders.loc[invalid_discount, "discount_pct"] = 0
    
    orders["net_revenue"] = orders["order_amount"] * (1 - (orders["discount_pct"] / 100))
    
    dq["orders_total_dropped"] = initial_count - len(orders)
    return orders.copy(), dq


def clean_tickets(tickets: pd.DataFrame) -> tuple:
    """Clean tickets, calculate resolution in hours, and track data quality."""
    dq = {}
    initial_count = len(tickets)
    
    tickets["customer_id"] = tickets["customer_id"].astype(str).str.strip()
    tickets["customer_id"] = tickets["customer_id"].replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})
    missing_cust = tickets["customer_id"].isna()
    dq["tickets_missing_customer_id"] = missing_cust.sum()
    tickets = tickets[~missing_cust]
    
    tickets["ticket_id"] = tickets["ticket_id"].astype(str).str.strip()
    duplicates = tickets.duplicated(subset=["ticket_id"], keep="first")
    dq["tickets_duplicate_ids"] = duplicates.sum()
    tickets = tickets[~duplicates]
    
    tickets["created_date"] = pd.to_datetime(tickets["created_date"], errors="coerce")
    invalid_created = tickets["created_date"].isna()
    dq["tickets_invalid_created_dates"] = invalid_created.sum()
    tickets = tickets[~invalid_created] 
    
    tickets["resolved_date"] = pd.to_datetime(tickets["resolved_date"], errors="coerce")
    
    # Calculate resolution in HOURS
    tickets["resolution_hours"] = (tickets["resolved_date"] - tickets["created_date"]).dt.total_seconds() / 3600.0
    
    tickets["satisfaction_score"] = pd.to_numeric(tickets["satisfaction_score"], errors="coerce")
    out_of_bounds = (tickets["satisfaction_score"] < 1) | (tickets["satisfaction_score"] > 5)
    dq["tickets_out_of_bounds_scores_nullified"] = out_of_bounds.sum()
    tickets.loc[out_of_bounds, "satisfaction_score"] = np.nan
    
    dq["tickets_total_dropped"] = initial_count - len(tickets)
    return tickets.copy(), dq


def build_customer_360(orders: pd.DataFrame, customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """
    Build the master Customer 360 dataset.
    Uses strict LEFT JOIN from Customer master.
    """
    # Summarize Orders
    order_summary = orders.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        total_net_revenue=("net_revenue", "sum")
    ).reset_index()

    # Summarize Tickets
    ticket_summary = tickets.groupby("customer_id").agg(
        ticket_count=("ticket_id", "count"),
        average_resolution_hours=("resolution_hours", "mean"),
        average_satisfaction_score=("satisfaction_score", "mean")
    ).reset_index()

    # Apply strict LEFT JOIN starting from Customers
    c360 = customers.merge(order_summary, on="customer_id", how="left")
    c360 = c360.merge(ticket_summary, on="customer_id", how="left")

    # Fill NaNs for operational counts
    c360["order_count"] = c360["order_count"].fillna(0)
    c360["total_net_revenue"] = c360["total_net_revenue"].fillna(0)
    c360["ticket_count"] = c360["ticket_count"].fillna(0)

    # Calculate Value Tier
    tier_conditions = [
        c360["total_net_revenue"] >= 5000,
        c360["total_net_revenue"] >= 1500
    ]
    tier_choices = ["Tier 1 (VIP)", "Tier 2 (Core)"]
    c360["value_tier"] = np.select(tier_conditions, tier_choices, default="Tier 3 (Standard)")

    # Calculate Risk Flag
    risk_condition = (
        (c360["average_satisfaction_score"] < 2.5) | 
        ((c360["ticket_count"] > 4) & (c360["average_resolution_hours"] > 48))
    )
    c360["risk_flag"] = np.where(risk_condition, 1, 0)

    return c360


def build_dashboard_outputs(customer_360: pd.DataFrame, orders: pd.DataFrame) -> tuple:
    """Generate summary dataframes for BI dashboard usage."""
    
    kpi_summary = pd.DataFrame([
        {"metric": "total_customers", "value": customer_360["customer_id"].nunique()},
        {"metric": "total_net_revenue", "value": customer_360["total_net_revenue"].sum()},
        {"metric": "total_orders", "value": customer_360["order_count"].sum()},
        {"metric": "average_satisfaction_score", "value": customer_360["average_satisfaction_score"].mean()}
    ])

    region_revenue = (
        customer_360.groupby("region")
        .agg(
            total_net_revenue=("total_net_revenue", "sum"), 
            customers=("customer_id", "count")
        )
        .reset_index()
    )

    category_revenue = (
        orders.groupby("product_category")
        .agg(
            total_net_revenue=("net_revenue", "sum"), 
            order_count=("order_id", "count")
        )
        .reset_index()
    )

    return kpi_summary, region_revenue, category_revenue


def save_outputs(c360, kpis, regions, categories, dq_report):
    """Write all 5 pipeline outputs to the outputs folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    c360.to_csv(OUTPUT_DIR / "customer_360.csv", index=False)
    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)
    regions.to_csv(OUTPUT_DIR / "region_revenue.csv", index=False)
    categories.to_csv(OUTPUT_DIR / "category_revenue.csv", index=False)
    dq_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
    
    logging.info("Success! All 5 outputs written to the outputs/ directory.")


def main():
    logging.info("Starting Customer 360 Pipeline...")
    
    # 1. Load Data
    orders_raw, customers_raw, tickets_raw = load_sources()
    
    # 2. Clean Data & Capture DQ Metrics
    orders, ord_dq = clean_orders(orders_raw)
    customers, cust_dq = clean_customers(customers_raw)
    tickets, tck_dq = clean_tickets(tickets_raw)
    
    # Consolidate Data Quality Metrics
    combined_dq = {**cust_dq, **ord_dq, **tck_dq}
    dq_report = pd.DataFrame(list(combined_dq.items()), columns=["Data_Quality_Metric", "Record_Count"])
    
    # 3. Build Features
    customer_360 = build_customer_360(orders, customers, tickets)
    kpi_summary, region_revenue, category_revenue = build_dashboard_outputs(customer_360, orders)
    
    # 4. Save Outputs
    save_outputs(customer_360, kpi_summary, region_revenue, category_revenue, dq_report)
    
    logging.info("Pipeline Execution Complete.")


if __name__ == "__main__":
    main()