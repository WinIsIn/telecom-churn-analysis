-- KiwiTel Churn Analysis — SQLite Queries
-- Load data first: CREATE TABLE customers AS SELECT * FROM read_csv_auto('data/raw/customers.csv');
-- Or use Python sqlite3 to load the CSV before running these queries.

-- 1. Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 2. Churn rate by contract type (sorted by churn %)
SELECT
    contract_type,
    COUNT(*) AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY contract_type
ORDER BY churn_rate_pct DESC;

-- 3. Churn rate by region
SELECT
    region,
    COUNT(*) AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY region
ORDER BY churn_rate_pct DESC;

-- 4. Average monthly charges: churned vs retained
SELECT
    churn AS customer_status,
    COUNT(*) AS total,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges_nzd,
    ROUND(MIN(monthly_charges), 2) AS min_monthly_charges_nzd,
    ROUND(MAX(monthly_charges), 2) AS max_monthly_charges_nzd
FROM customers
GROUP BY churn;

-- 5. Churn rate by support call bucket (0, 1-2, 3-4, 5+)
SELECT
    CASE
        WHEN num_support_calls = 0 THEN '0 calls'
        WHEN num_support_calls BETWEEN 1 AND 2 THEN '1-2 calls'
        WHEN num_support_calls BETWEEN 3 AND 4 THEN '3-4 calls'
        ELSE '5+ calls'
    END AS support_bucket,
    COUNT(*) AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY support_bucket
ORDER BY
    CASE support_bucket
        WHEN '0 calls'   THEN 1
        WHEN '1-2 calls' THEN 2
        WHEN '3-4 calls' THEN 3
        ELSE 4
    END;

-- 6. Top churn risk segment: month-to-month + fiber + tenure < 12 months
SELECT
    COUNT(*) AS segment_size,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges_nzd,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue_nzd
FROM customers
WHERE contract_type = 'Month-to-month'
  AND internet_service = 'Fiber optic'
  AND tenure_months < 12;

-- 7. Revenue at risk: total monthly charges from churned customers
SELECT
    churn,
    COUNT(*) AS customers,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue_nzd,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_revenue_nzd,
    ROUND(SUM(monthly_charges) * 12, 2) AS annualised_revenue_nzd
FROM customers
GROUP BY churn
ORDER BY churn DESC;
