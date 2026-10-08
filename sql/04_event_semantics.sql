-- Event semantics audit, not a funnel or a conversion model.
-- Grain: property date x event_name x normalized page path x item-array length.
-- Exact paths retained; normalization removes origin/query/fragment, lowercases, trims slash.
WITH extracted AS
  (SELECT PARSE_DATE('%Y%m%d', event_date) AS property_date,
          event_name,
          user_pseudo_id,

     (SELECT COUNT(*)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid_param_n,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='page_location') AS page_location,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='page_title') AS page_title,

     (SELECT MAX(COALESCE(p.value.double_value, CAST(p.value.int_value AS FLOAT64), p.value.float_value))
      FROM UNNEST(event_params) p
      WHERE p.key='value') AS event_value,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='currency') AS currency,
          COALESCE(ecommerce.transaction_id,
                     (SELECT MAX(p.value.string_value)
                      FROM UNNEST(event_params) p
                      WHERE p.key='transaction_id')) AS transaction_id,
          ARRAY_LENGTH(items) AS item_array_length,
          ARRAY
     (SELECT DISTINCT i.item_id
      FROM UNNEST(items) i
      WHERE i.item_id IS NOT NULL
        AND i.item_id NOT IN ('', '(not set)', '<Other>')) AS valid_sku_ids
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND event_name IN ('view_item',
                        'select_item',
                        'begin_checkout',
                        'add_payment_info',
                        'add_shipping_info')),
     paths AS
  (SELECT *,
          IF(page_location IS NULL, '__missing__', COALESCE(NULLIF(REGEXP_REPLACE(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(page_location, r'^https?://[^/]+', ''), r'[?#].*$', '')), r'/+$', ''), ''), '/')) AS normalized_page_path,
          IF(user_pseudo_id IS NOT NULL
             AND user_pseudo_id!=''
             AND sid_param_n=1
             AND sid IS NOT NULL, TO_JSON_STRING(STRUCT(user_pseudo_id, sid)), NULL) AS session_key
   FROM extracted),
     grouped AS
  (SELECT property_date,
          event_name,
          normalized_page_path,
          item_array_length,
          COUNT(*) AS event_count,
          COUNT(DISTINCT user_pseudo_id) AS observed_users,
          COUNT(DISTINCT session_key) AS valid_sessions,
          COUNTIF(ARRAY_LENGTH(valid_sku_ids)>0) AS events_with_valid_sku,
          SUM(ARRAY_LENGTH(valid_sku_ids)) AS valid_sku_event_occurrences,
          MIN(ARRAY_LENGTH(valid_sku_ids)) AS min_valid_skus_per_event,
          MAX(ARRAY_LENGTH(valid_sku_ids)) AS max_valid_skus_per_event,
          ARRAY_CONCAT_AGG(valid_sku_ids) AS all_valid_sku_ids,
          COUNTIF(event_value IS NOT NULL) AS events_with_value,
          COUNTIF(event_value>0) AS events_with_positive_value,
          COUNTIF(currency IS NOT NULL
                  AND currency!=''
                  AND currency!='(not set)') AS events_with_currency,
          COUNTIF(transaction_id IS NOT NULL
                  AND transaction_id NOT IN ('', '(not set)', '<Other>')) AS events_with_valid_transaction_id,
          STRING_AGG(DISTINCT page_title, ' | '
                     LIMIT 3) AS observed_page_title_examples
   FROM paths
   GROUP BY property_date,
            event_name,
            normalized_page_path,
            item_array_length)
SELECT * EXCEPT(all_valid_sku_ids),

  (SELECT COUNT(DISTINCT sku)
   FROM UNNEST(all_valid_sku_ids) sku) AS unique_valid_skus
FROM grouped
ORDER BY property_date,
         event_name,
         normalized_page_path,
         item_array_length;
