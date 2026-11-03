# Feature Availability and Data Leakage Assessment

## Prediction Objective

Predict whether an Olist order will arrive later than its estimated delivery date, using information assumed to be available at the time of purchase.

**Target:** `is_delayed`

- `1`: Actual delivery date is later than estimated delivery date.
- `0`: Actual delivery date is on or before estimated delivery date.

The actual delivery date is used only for target construction and evaluation, never as a model input.

## Feature Assessment

| Feature | Source | Checkout availability | Decision |
|---|---|---|---|
| purchase_month | Purchase timestamp | Available | Include |
| purchase_day_of_week | Purchase timestamp | Available | Include |
| purchase_hour | Purchase timestamp | Available | Include |
| estimated_lead_days | Purchase and estimated delivery timestamps | Assumed available | Include with documented assumption |
| item_count | Order items | Assumed available | Include with documented assumption |
| total_price | Order items | Assumed available | Include with documented assumption |
| total_freight_value | Order items | Assumed available | Include with documented assumption |
| customer_state | Customer information | Assumed available | Include with documented assumption |
| unique_sellers | Final order-item aggregation | Not verified | Exclude |
| seller_states | Final order-item aggregation | Not verified | Exclude |
| order_delivered_customer_date | Delivery event | Not available | Target construction only |
| order_purchase_timestamp | Prediction timestamp | Available | Exclude as raw model input |

## Key Assumptions

1. The Olist dataset documentation describes `order_estimated_delivery_date` as the estimated date communicated to the customer at purchase.
2. The dataset does not provide estimate creation or update timestamps.
3. Item and freight information is assumed to represent information available when the order was placed.
4. Seller assignment timing has not been verified, so seller-derived features are excluded.

## Data Leakage Prevention

- Actual delivery timestamps must never be model inputs.
- Validation and test data must not be used to fit preprocessing transformations.
- Data splitting must preserve chronological order.
- The test set must remain untouched until final model evaluation.

## Experimental Evidence

The Logistic Regression feature-ablation experiment produced:

| Metric | With estimated_lead_days | Without estimated_lead_days |
|---|---:|---:|
| PR-AUC | 0.137704 | 0.079881 |
| ROC-AUC | 0.750263 | 0.616885 |

The feature improves validation performance, but predictive improvement alone does not establish feature availability.

## Current Limitation

The checkout-time availability of certain fields is based on dataset documentation and assumptions rather than timestamp-level event logs. The model should therefore be treated as an experimental checkout-time prediction system until these assumptions are independently verified.