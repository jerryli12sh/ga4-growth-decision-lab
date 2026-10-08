-- 92-day source event inventory: independent reconciliation anchor.

SELECT event_date,
       event_name,
       device.category AS device_category,
       COUNT(*) AS event_count,
       COUNT(DISTINCT user_pseudo_id) AS observed_users,
       COUNTIF(user_pseudo_id IS NULL) AS missing_user_events,
       COUNTIF(
                 (SELECT value.int_value
                  FROM UNNEST(event_params)
                  WHERE KEY='ga_session_id') IS NULL) AS missing_session_events,
       COUNT(DISTINCT IF(user_pseudo_id IS NOT NULL, CONCAT(user_pseudo_id, '|', CAST(
                                                                                        (SELECT value.int_value
                                                                                         FROM UNNEST(event_params)
                                                                                         WHERE KEY='ga_session_id') AS STRING)), NULL)) AS event_sessions,
       COUNTIF(ARRAY_LENGTH(items)>0) AS events_with_items,
       SUM(IF(event_name='purchase', ecommerce.purchase_revenue_in_usd, 0)) AS purchase_revenue_usd,
       COUNT(DISTINCT IF(event_name='purchase', ecommerce.transaction_id, NULL)) AS distinct_transaction_ids,
       COUNTIF(event_name='purchase'
               AND (ecommerce.transaction_id IS NULL
                    OR ecommerce.transaction_id IN ('', '(not set)', '<Other>'))) AS purchase_bad_txid
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY 1,
         2,
         3
ORDER BY 1,
         2,
         3;
