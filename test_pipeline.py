import pytest
import pandas as pd
import numpy as np

# Assuming the pipeline script is saved as pipeline.py in the project root
# If it's in a src/ folder, change this to: from src.pipeline import ...
from pipeline import (
    clean_customers,
    clean_orders,
    clean_tickets,
    build_customer_360
)


# ==========================================
# FIXTURES (Mocked DataFrames)
# ==========================================

@pytest.fixture
def sample_customers():
    """Mock raw customers DataFrame with string inconsistencies."""
    return pd.DataFrame([
        {"customer_id": " C-001 ", "customer_name": "  customer 1 analytics  ", "signup_date": "2024-01-12", "region": "North"},
        {"customer_id": "C-002", "customer_name": "CUSTOMER 2", "signup_date": "bad-date", "region": "South"},
    ])

@pytest.fixture
def sample_orders():
    """Mock raw orders DataFrame with duplicates, negative amounts, and mixed discounts."""
    return pd.DataFrame([
        # Valid order
        {"order_id": "O-01", "customer_id": "C-001", "order_date": "2026-01-01", "order_amount": 1000.0, "discount_pct": 20},
        # Duplicate of O-01 (should be dropped)
        {"order_id": "O-01", "customer_id": "C-001", "order_date": "2026-01-01", "order_amount": 1000.0, "discount_pct": 20},
        # Negative order amount (should be dropped)
        {"order_id": "O-02", "customer_id": "C-002", "order_date": "2026-01-02", "order_amount": -50.0, "discount_pct": 0},
        # Invalid discount (should be imputed to 0, net_revenue = 500)
        {"order_id": "O-03", "customer_id": "C-001", "order_date": "2026-01-03", "order_amount": 500.0, "discount_pct": 150}, 
    ])

@pytest.fixture
def sample_tickets():
    """Mock raw tickets DataFrame with out-of-bound satisfaction scores."""
    return pd.DataFrame([
        {"ticket_id": "T-01", "customer_id": "C-001", "created_date": "2026-01-01 10:00:00", "resolved_date": "2026-01-02 10:00:00", "satisfaction_score": 5},
        {"ticket_id": "T-02", "customer_id": "C-001", "created_date": "2026-01-03 10:00:00", "resolved_date": "2026-01-10 10:00:00", "satisfaction_score": 0}, # Out of bounds (<1)
        {"ticket_id": "T-03", "customer_id": "C-002", "created_date": "2026-01-04 10:00:00", "resolved_date": "2026-01-05 10:00:00", "satisfaction_score": 6}, # Out of bounds (>5)
    ])


# ==========================================
# TEST CASES
# ==========================================

def test_string_cleaning(sample_customers):
    """Test whitespace trimming and Title Case conversion in customers data."""
    cleaned_df, dq_metrics = clean_customers(sample_customers)
    
    # Assert whitespace is trimmed from customer_id
    assert cleaned_df.loc[0, "customer_id"] == "C-001"
    
    # Assert customer_name is title-cased and stripped
    assert cleaned_df.loc[0, "customer_name"] == "Customer 1 Analytics"
    assert cleaned_df.loc[1, "customer_name"] == "Customer 2"


def test_duplicate_handling(sample_orders):
    """Test purging of duplicate order_id records while keeping the first."""
    cleaned_df, dq_metrics = clean_orders(sample_orders)
    
    # Assert duplicate O-01 was dropped (1 occurrence remains)
    assert len(cleaned_df[cleaned_df["order_id"] == "O-01"]) == 1
    
    # Assert the DQ metric correctly tracked the 1 dropped duplicate
    assert dq_metrics["orders_duplicate_ids"] == 1


def test_net_revenue_computation(sample_orders):
    """Test mathematical computation of net revenue: amount * (1 - discount/100)."""
    cleaned_df, _ = clean_orders(sample_orders)
    
    # Order O-01: 1000 * (1 - 20/100) = 800
    net_rev_o1 = cleaned_df.loc[cleaned_df["order_id"] == "O-01", "net_revenue"].iloc[0]
    assert net_rev_o1 == 800.0

    # Order O-03: 500 with invalid discount (150). Discount imputed to 0. Net = 500
    net_rev_o3 = cleaned_df.loc[cleaned_df["order_id"] == "O-03", "net_revenue"].iloc[0]
    assert net_rev_o3 == 500.0


def test_invalid_record_filtering(sample_orders, sample_tickets):
    """Test dropping negative orders and nullifying out-of-bound satisfaction scores."""
    cleaned_orders, ord_dq = clean_orders(sample_orders)
    cleaned_tickets, tck_dq = clean_tickets(sample_tickets)
    
    # Assert order with -50 amount (O-02) is completely dropped
    assert "O-02" not in cleaned_orders["order_id"].values
    assert ord_dq["orders_invalid_or_negative_amounts"] == 1
    
    # Assert out-of-bound ticket scores (0 and 6) are converted to NaN but rows remain
    t2_score = cleaned_tickets.loc[cleaned_tickets["ticket_id"] == "T-02", "satisfaction_score"].iloc[0]
    t3_score = cleaned_tickets.loc[cleaned_tickets["ticket_id"] == "T-03", "satisfaction_score"].iloc[0]
    
    assert pd.isna(t2_score)
    assert pd.isna(t3_score)
    assert tck_dq["tickets_out_of_bounds_scores_nullified"] == 2
    assert len(cleaned_tickets) == 3 # Rows shouldn't be dropped, just nullified


def test_output_schema_and_business_logic(sample_orders, sample_customers, sample_tickets):
    """Verify final customer_360 schema match, value_tier generation, and risk_flag calculation."""
    # Pre-clean the mocked data using our tested functions
    clean_ord, _ = clean_orders(sample_orders)
    clean_cust, _ = clean_customers(sample_customers)
    clean_tck, _ = clean_tickets(sample_tickets)
    
    # Build 360 table
    customer_360 = build_customer_360(clean_ord, clean_cust, clean_tck)
    
    # 1. Structural Checks
    expected_columns = [
        "customer_id", 
        "customer_name", 
        "signup_date", 
        "region", 
        "order_count", 
        "total_net_revenue", 
        "ticket_count", 
        "average_resolution_hours", 
        "average_satisfaction_score", 
        "value_tier", 
        "risk_flag"
    ]
    # Check that both sets of columns match exactly (order doesn't matter)
    assert set(customer_360.columns) == set(expected_columns)
    
    # 2. Assert Strict LEFT JOIN behavior (C-002 should exist even with 0 valid orders left)
    assert len(customer_360) == 2
    c2_row = customer_360[customer_360["customer_id"] == "C-002"].iloc[0]
    assert c2_row["order_count"] == 0
    assert c2_row["total_net_revenue"] == 0
    
    # 3. Assert value_tier logic
    c1_row = customer_360[customer_360["customer_id"] == "C-001"].iloc[0]
    # C-001 revenue = 800 + 500 = 1300. (< 1500, so Tier 3)
    assert c1_row["value_tier"] == "Tier 3 (Standard)"
    # C-002 revenue = 0
    assert c2_row["value_tier"] == "Tier 3 (Standard)"
    
    # 4. Assert risk_flag logic
    # C-001 ticket average score: Ticket T-01 is 5, T-02 is NaN -> Average is 5.0 (No risk)
    assert c1_row["risk_flag"] == 0
    
    # C-002 ticket average score: Ticket T-03 is NaN -> Average is NaN.
    # Risk flag triggers if score < 2.5. NaN < 2.5 evaluates to False in Pandas -> Risk flag 0.
    assert c2_row["risk_flag"] == 0


def test_high_value_and_risk_flag_logic():
    """Specific targeted test to ensure VIP calculation and Risk Flag triggers properly."""
    # Create specific data for just this test to hit the thresholds
    customers = pd.DataFrame([{"customer_id": "C-99", "customer_name": "Risk VIP"}])
    orders = pd.DataFrame([{"order_id": "O-99", "customer_id": "C-99", "net_revenue": 6000.0}])
    tickets = pd.DataFrame([{"ticket_id": "T-99", "customer_id": "C-99", "resolution_hours": 10.0, "satisfaction_score": 1.5}])
    
    # Inject directly into build_customer_360 (assuming upstream cleaning is already done)
    c360 = build_customer_360(orders, customers, tickets)
    
    assert c360.loc[0, "value_tier"] == "Tier 1 (VIP)"  # > 5000 revenue
    assert c360.loc[0, "risk_flag"] == 1                # score < 2.5