-- Grain: property date x device x exact item ID/name/category tuple.
-- Item-grain amounts ONLY: do not multiply event-level ecommerce revenue by item count.
-- All events included regardless of session validity; valid sessions reported separately.
WITH EVENTS AS
  (SELECT event_date,
          event_timestamp,
          event_name,
          user_pseudo_id,
          device.category AS device_category,
          items,

     (SELECT COUNT(*)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid_param_n,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND event_name IN ('view_item_list',
                        'select_item',
                        'view_item',
                        'add_to_cart',
                        'remove_from_cart',
                        'begin_checkout',
                        'purchase',
                        'refund')), item_rows_cte AS
  (SELECT event_date,
          event_name,
          user_pseudo_id,
          device_category,
          IF(user_pseudo_id IS NOT NULL
             AND user_pseudo_id!=''
             AND sid_param_n=1
             AND sid IS NOT NULL, TO_JSON_STRING(STRUCT(user_pseudo_id, sid)), NULL) AS session_key,
          i.item_id,
          i.item_name,
          i.item_category,
          i.item_brand,
          i.price_in_usd,
          i.quantity,
          i.item_revenue_in_usd,
          i.item_refund_in_usd
   FROM EVENTS
   CROSS JOIN UNNEST(items) i)
SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date,
       device_category,
       item_id,
       item_name,
       item_category,
       item_brand,
       COUNT(*) AS item_rows,
       COUNTIF(event_name='view_item_list') AS list_item_rows,
       COUNTIF(event_name='select_item') AS selected_item_rows,
       COUNTIF(event_name='view_item') AS view_item_rows,
       COUNTIF(event_name='add_to_cart') AS cart_item_rows,
       COUNTIF(event_name='remove_from_cart') AS remove_cart_item_rows,
       COUNTIF(event_name='begin_checkout') AS checkout_item_rows,
       COUNTIF(event_name='purchase') AS purchase_item_rows,
       COUNT(DISTINCT IF(event_name='view_item', user_pseudo_id, NULL)) AS view_users,
       COUNT(DISTINCT IF(event_name='view_item', session_key, NULL)) AS view_sessions,
       COUNT(DISTINCT IF(event_name='add_to_cart', session_key, NULL)) AS cart_sessions,
       COUNT(DISTINCT IF(event_name='begin_checkout', session_key, NULL)) AS checkout_sessions,
       COUNT(DISTINCT IF(event_name='purchase', session_key, NULL)) AS purchase_sessions,
       SUM(IF(event_name='purchase', COALESCE(quantity, 0), 0)) AS purchase_quantity,
       SUM(IF(event_name='purchase', COALESCE(item_revenue_in_usd, 0), 0)) AS item_purchase_revenue_usd,
       SUM(IF(event_name='refund', COALESCE(item_refund_in_usd, 0), 0)) AS item_refund_usd,
       MIN(IF(event_name='view_item', price_in_usd, NULL)) AS min_view_price_usd,
       MAX(IF(event_name='view_item', price_in_usd, NULL)) AS max_view_price_usd,
       COUNTIF(event_name='purchase'
               AND item_revenue_in_usd IS NULL) AS missing_item_revenue_purchase_rows
FROM item_rows_cte
GROUP BY item_rows_cte.event_date,
         device_category,
         item_id,
         item_name,
         item_category,
         item_brand
ORDER BY event_date,
         device_category,
         item_id,
         item_name,
         item_category,
         item_brand;
