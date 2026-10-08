-- Pre-outcome basket/features at the first begin_checkout timestamp, one valid session each.
-- Multiple payloads at an identical first timestamp are flagged, not assigned an invented order.
-- Serialized payloads provide a stable choice; variant counts expose timestamp ties.
WITH base AS
  (SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date,
          event_timestamp,
          user_pseudo_id,
          device.category AS device_category,
          geo.country AS country,

     (SELECT COUNT(*)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid_param_n,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS ga_session_id,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='page_location') AS page_location,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='currency') AS currency,

     (SELECT MAX(COALESCE(p.value.double_value, CAST(p.value.int_value AS FLOAT64), p.value.float_value))
      FROM UNNEST(event_params) p
      WHERE p.key='value') AS event_value,
          items
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND event_name='begin_checkout'),
     VALID AS
  (SELECT *,
          MIN(event_timestamp) OVER(PARTITION BY user_pseudo_id, ga_session_id) AS checkout_index_ts
   FROM base
   WHERE user_pseudo_id IS NOT NULL
     AND user_pseudo_id!=''
     AND sid_param_n=1
     AND ga_session_id IS NOT NULL), at_index AS
  (SELECT *,
          TO_JSON_STRING(STRUCT(event_date, device_category, country, page_location, currency, event_value, items)) AS index_payload
   FROM VALID
   WHERE event_timestamp=checkout_index_ts),
                                     summarized AS
  (SELECT user_pseudo_id,
          ga_session_id,
          checkout_index_ts,
          event_date,
          device_category,
          country,
          IF(page_location IS NULL, NULL, COALESCE(NULLIF(REGEXP_REPLACE(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(page_location, r'^https?://[^/]+', ''), r'[?#].*$', '')), r'/+$', ''), ''), '/')) AS normalized_page_path,
          currency,
          event_value,
          index_payload,
          ARRAY_LENGTH(items) AS item_array_length,

     (SELECT COUNT(DISTINCT i.item_id)
      FROM UNNEST(items) i
      WHERE i.item_id IS NOT NULL
        AND i.item_id NOT IN ('',
                              '(not set)',
                              '<Other>')) AS valid_sku_count,

     (SELECT COUNTIF(i.item_id IS NULL
                     OR i.item_id IN ('', '(not set)', '<Other>'))
      FROM UNNEST(items) i) AS placeholder_item_rows,

     (SELECT SUM(i.quantity)
      FROM UNNEST(items) i) AS observed_quantity_sum,

     (SELECT SUM(i.price_in_usd*i.quantity)
      FROM UNNEST(items) i
      WHERE i.price_in_usd IS NOT NULL
        AND i.quantity IS NOT NULL) AS observed_price_quantity_usd,

     (SELECT COUNTIF(i.price_in_usd IS NOT NULL
                     AND i.quantity IS NOT NULL)
      FROM UNNEST(items) i) AS price_quantity_complete_rows,

     (SELECT MIN(i.price_in_usd)
      FROM UNNEST(items) i) AS min_item_price_usd,

     (SELECT MAX(i.price_in_usd)
      FROM UNNEST(items) i) AS max_item_price_usd,

     (SELECT COUNT(DISTINCT i.item_category)
      FROM UNNEST(items) i
      WHERE i.item_category IS NOT NULL
        AND i.item_category NOT IN ('',
                                    '(not set)',
                                    '<Other>')) AS observed_category_count,

     (SELECT COUNTIF(i.item_category IS NULL
                     OR i.item_category IN ('', '(not set)', '<Other>'))
      FROM UNNEST(items) i) AS missing_category_item_rows,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'apparel')) AS INT64) AS contains_apparel,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'drinkware')) AS INT64) AS contains_drinkware,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'accessor')) AS INT64) AS contains_accessories,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'bag')) AS INT64) AS contains_bags,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'stationery|office|notebook|writing')) AS INT64) AS contains_stationery_office,
          CAST(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE REGEXP_CONTAINS(LOWER(i.item_category), r'lifestyle')) AS INT64) AS contains_lifestyle,
          TO_JSON_STRING(ARRAY
                           (SELECT DISTINCT i.item_id
                            FROM UNNEST(items) i
                            WHERE i.item_id IS NOT NULL
                              AND i.item_id NOT IN ('', '(not set)', '<Other>')
                            ORDER BY i.item_id)) AS valid_sku_ids_json,
          TO_JSON_STRING(ARRAY
                           (SELECT DISTINCT i.item_category
                            FROM UNNEST(items) i
                            WHERE i.item_category IS NOT NULL
                              AND i.item_category NOT IN ('', '(not set)', '<Other>')
                            ORDER BY i.item_category)) AS observed_categories_json
   FROM at_index),
                                     choices AS
  (SELECT user_pseudo_id,
          ga_session_id,
          checkout_index_ts,
          COUNT(*) AS index_rows_same_timestamp,
          COUNT(DISTINCT index_payload) AS index_payload_variants,
          ARRAY_AGG(STRUCT(event_date, device_category, country, normalized_page_path, currency, event_value, item_array_length, valid_sku_count, placeholder_item_rows, observed_quantity_sum, observed_price_quantity_usd, price_quantity_complete_rows, min_item_price_usd, max_item_price_usd, observed_category_count, missing_category_item_rows, contains_apparel, contains_drinkware, contains_accessories, contains_bags, contains_stationery_office, contains_lifestyle, valid_sku_ids_json, observed_categories_json)
                    ORDER BY index_payload
                    LIMIT 1)[OFFSET(0)] AS chosen
   FROM summarized
   GROUP BY user_pseudo_id,
            ga_session_id,
            checkout_index_ts)
SELECT user_pseudo_id,
       ga_session_id,
       checkout_index_ts,
       index_rows_same_timestamp,
       index_payload_variants,
       chosen.*
FROM choices
ORDER BY chosen.event_date,
         user_pseudo_id,
         ga_session_id;
