-- Grain: distinct literal purchase/refund event record with identical copies counted.
-- Do not deduplicate by transaction_id alone: IDs are obfuscated and can collide.
-- All purchase/refund events retained, including invalid session/user identifiers.
WITH purchase_rows AS
  (SELECT e.*,
          TO_JSON_STRING(e) AS event_record,

     (SELECT COUNT(*)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid_param_n,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS ga_session_id
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` e
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND event_name IN ('purchase',
                        'refund')),
     marked AS
  (SELECT *,
          COUNT(*) OVER(PARTITION BY event_record) AS identical_raw_row_count,
          ROW_NUMBER() OVER(PARTITION BY event_record
                            ORDER BY event_timestamp) AS copy_rank
   FROM purchase_rows)
SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date,
       event_timestamp,
       event_name,
       user_pseudo_id,
       ga_session_id,
       sid_param_n,
       CAST(user_pseudo_id IS NOT NULL
            AND user_pseudo_id!=''
            AND ga_session_id IS NOT NULL
            AND sid_param_n=1 AS INT64) AS valid_session_key,
       device.category AS device_category,
       geo.country AS country,
       traffic_source.source AS first_user_source,
       traffic_source.medium AS first_user_medium,
       ecommerce.transaction_id AS transaction_id,
       ecommerce.purchase_revenue_in_usd AS purchase_revenue_usd,
       ecommerce.refund_value_in_usd AS refund_value_usd,
       ecommerce.shipping_value_in_usd AS shipping_value_usd,
       ecommerce.tax_value_in_usd AS tax_value_usd,
       ecommerce.total_item_quantity AS ecommerce_total_item_quantity,
       ecommerce.unique_items AS ecommerce_unique_items,
       ARRAY_LENGTH(items) AS item_row_count,

  (SELECT SUM(COALESCE(i.quantity, 0))
   FROM UNNEST(items) i) AS item_quantity_sum,

  (SELECT SUM(COALESCE(i.item_revenue_in_usd, 0))
   FROM UNNEST(items) i) AS item_revenue_sum_usd,

  (SELECT COUNTIF(i.item_id IS NULL
                  OR i.item_id IN ('', '(not set)', '<Other>'))
   FROM UNNEST(items) i) AS missing_item_id_rows,
       TO_JSON_STRING(items) AS items_json,
       FORMAT('purchase_%05d', ROW_NUMBER() OVER (
                                                  ORDER BY event_date, event_timestamp, user_pseudo_id, event_record)) AS event_id,
       identical_raw_row_count
FROM marked
WHERE copy_rank=1
ORDER BY event_date,
         event_timestamp,
         user_pseudo_id,
         event_id;
