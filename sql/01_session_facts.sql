-- GA4 rebuild: one row per valid (user_pseudo_id, ga_session_id).
-- Prefer full-window query + outer session_date export filter for exact weekly exports.
WITH extracted AS
  (SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date,
          event_timestamp,
          event_name,
          user_pseudo_id,
          user_first_touch_timestamp,
          device.category AS device_category,
          device.operating_system AS operating_system,
          device.web_info.browser AS browser,
          geo.country AS country,
          traffic_source.source AS first_user_source,
          traffic_source.medium AS first_user_medium,

     (SELECT COUNT(*)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS sid_param_n,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS ga_session_id,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_number') AS ga_session_number,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='source') AS event_source,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='medium') AS event_medium,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='page_location') AS page_location,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='page_referrer') AS page_referrer,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='engagement_time_msec') AS engagement_time_msec,
          ecommerce.transaction_id AS transaction_id,
          ecommerce.purchase_revenue_in_usd AS purchase_revenue_usd,
          ARRAY_LENGTH(items) AS item_rows
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'),
     VALID AS
  (SELECT *
   FROM extracted
   WHERE user_pseudo_id IS NOT NULL
     AND user_pseudo_id!=''
     AND sid_param_n=1
     AND ga_session_id IS NOT NULL), sessions AS
  (SELECT user_pseudo_id,
          ga_session_id,
          MIN(event_date) AS session_date,
          MAX(event_date) AS session_last_date,
          MIN(event_timestamp) AS session_start_ts,
          MAX(event_timestamp) AS session_end_ts,
          MIN(user_first_touch_timestamp) AS user_first_touch_ts,
          MIN(ga_session_number) AS ga_session_number,
          ARRAY_AGG(STRUCT(device_category, operating_system, browser, country, first_user_source, first_user_medium, event_source, event_medium, REGEXP_REPLACE(page_location, r'[?#].*$', '') AS landing_page, REGEXP_REPLACE(page_referrer, r'[?#].*$', '') AS entry_referrer)
                    ORDER BY event_timestamp, IF(event_name='session_start', 0, 1), COALESCE(page_location, ''), COALESCE(device_category, '')
                    LIMIT 1)[OFFSET(0)] AS attrs,
          COUNT(DISTINCT device_category) AS device_value_count,
          COUNT(*) AS event_count,
          COUNT(*)-COUNT(DISTINCT event_timestamp) AS timestamp_tie_excess,
          COUNTIF(event_name='session_start') AS session_start_events,
          COUNTIF(event_name='first_visit') AS first_visit_events,
          COUNTIF(event_name='page_view') AS page_view_events,
          COUNTIF(event_name IN ('view_search_results', 'search')) AS search_events,
          COUNTIF(event_name='view_item_list') AS item_list_events,
          COUNTIF(event_name='select_item') AS select_item_events,
          COUNTIF(event_name='view_item') AS view_events,
          COUNTIF(event_name='add_to_cart') AS cart_events,
          COUNTIF(event_name='remove_from_cart') AS remove_cart_events,
          COUNTIF(event_name='view_cart') AS view_cart_events,
          COUNTIF(event_name='begin_checkout') AS checkout_events,
          COUNTIF(event_name='add_shipping_info') AS shipping_events,
          COUNTIF(event_name='add_payment_info') AS payment_events,
          COUNTIF(event_name='purchase') AS purchase_events,
          COUNTIF(event_name='refund') AS refund_events,
          COUNTIF(event_name='purchase'
                  AND purchase_revenue_usd IS NULL) AS missing_revenue_purchase_events,
          COUNTIF(event_name='purchase'
                  AND (transaction_id IS NULL
                       OR transaction_id IN ('', '(not set)', '<Other>'))) AS invalid_transaction_purchase_events,
          COUNT(DISTINCT IF(event_name='purchase'
                            AND transaction_id NOT IN ('', '(not set)', '<Other>'),transaction_id, NULL)) AS distinct_transaction_ids,
          SUM(IF(event_name='purchase', COALESCE(purchase_revenue_usd, 0), 0)) AS raw_purchase_revenue_usd,
          SUM(COALESCE(engagement_time_msec, 0)) AS engagement_time_msec,
          SUM(IF(event_name='view_item', item_rows, 0)) AS viewed_item_rows,
          MIN(IF(event_name='view_item', event_timestamp, NULL)) AS first_view_ts,
          MIN(IF(event_name='add_to_cart', event_timestamp, NULL)) AS first_cart_ts,
          MIN(IF(event_name='begin_checkout', event_timestamp, NULL)) AS first_checkout_ts,
          MIN(IF(event_name='add_shipping_info', event_timestamp, NULL)) AS first_shipping_ts,
          MIN(IF(event_name='add_payment_info', event_timestamp, NULL)) AS first_payment_ts,
          MIN(IF(event_name='purchase', event_timestamp, NULL)) AS first_purchase_ts,
          MIN(IF(event_name IN ('view_search_results', 'search'),event_timestamp, NULL)) AS first_search_ts,
          ARRAY_AGG(STRUCT(event_name, event_timestamp, transaction_id, purchase_revenue_usd)) AS EVENTS
   FROM VALID
   GROUP BY user_pseudo_id,
            ga_session_id),
                                     stages1 AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='begin_checkout'
        AND e.event_timestamp>first_view_ts) AS checkout_after_view_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='add_to_cart'
        AND e.event_timestamp>first_view_ts) AS cart_after_view_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='purchase'
        AND e.event_timestamp>first_checkout_ts) AS purchase_after_checkout_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='purchase'
        AND e.event_timestamp>=first_checkout_ts) AS purchase_after_checkout_weak_ts,

     (SELECT SUM(COALESCE(d.purchase_revenue_usd, 0))
      FROM
        (SELECT DISTINCT e.event_timestamp,
                         e.transaction_id,
                         e.purchase_revenue_usd
         FROM UNNEST(EVENTS) e
         WHERE e.event_name='purchase') d) AS timestamp_signature_purchase_revenue_usd,

     (SELECT COUNT(*)
      FROM
        (SELECT DISTINCT e.event_timestamp,
                         e.transaction_id,
                         e.purchase_revenue_usd
         FROM UNNEST(EVENTS) e
         WHERE e.event_name='purchase')) AS purchase_timestamp_signatures
   FROM sessions),
                                     stages2 AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='purchase'
        AND e.event_timestamp>checkout_after_view_ts) AS purchase_after_view_checkout_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='begin_checkout'
        AND e.event_timestamp>cart_after_view_ts) AS checkout_after_view_cart_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='add_shipping_info'
        AND e.event_timestamp>first_checkout_ts) AS shipping_after_checkout_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='add_payment_info'
        AND e.event_timestamp>first_checkout_ts) AS payment_after_checkout_ts,
          EXISTS
     (SELECT 1
      FROM UNNEST(EVENTS) e
      WHERE e.event_name='add_to_cart'
        AND e.event_timestamp>first_view_ts
        AND e.event_timestamp<checkout_after_view_ts) AS cart_between_view_checkout
   FROM stages1)
SELECT user_pseudo_id,
       ga_session_id,
       session_date,
       session_last_date,
       session_start_ts,
       session_end_ts,
       user_first_touch_ts,
       ga_session_number,
       attrs.device_category,
       attrs.operating_system,
       attrs.browser,
       attrs.country,
       attrs.first_user_source,
       attrs.first_user_medium,
       attrs.event_source AS entry_event_source,
       attrs.event_medium AS entry_event_medium,
       attrs.landing_page,
       attrs.entry_referrer,
       device_value_count,
       event_count,
       timestamp_tie_excess,
       session_start_events,
       first_visit_events,
       page_view_events,
       search_events,
       item_list_events,
       select_item_events,
       view_events,
       cart_events,
       remove_cart_events,
       view_cart_events,
       checkout_events,
       shipping_events,
       payment_events,
       purchase_events,
       refund_events,
       missing_revenue_purchase_events,
       invalid_transaction_purchase_events,
       distinct_transaction_ids,
       raw_purchase_revenue_usd,
       COALESCE(timestamp_signature_purchase_revenue_usd, 0) AS timestamp_signature_purchase_revenue_usd,
       purchase_timestamp_signatures,
       engagement_time_msec,
       viewed_item_rows,
       first_view_ts,
       first_cart_ts,
       first_checkout_ts,
       first_shipping_ts,
       first_payment_ts,
       first_purchase_ts,
       first_search_ts,
       checkout_after_view_ts,
       cart_after_view_ts,
       purchase_after_checkout_ts,
       purchase_after_checkout_weak_ts,
       purchase_after_view_checkout_ts,
       checkout_after_view_cart_ts,
       shipping_after_checkout_ts,
       payment_after_checkout_ts,
       CAST(cart_between_view_checkout AS INT64) AS cart_between_view_checkout,

  (SELECT MIN(e.event_timestamp)
   FROM UNNEST(EVENTS) e
   WHERE e.event_name='purchase'
     AND e.event_timestamp>checkout_after_view_cart_ts) AS purchase_after_view_cart_checkout_ts,

  (SELECT MIN(e.event_timestamp)
   FROM UNNEST(EVENTS) e
   WHERE e.event_name='purchase'
     AND e.event_timestamp>shipping_after_checkout_ts
     AND e.event_timestamp>payment_after_checkout_ts) AS purchase_after_checkout_shipping_payment_ts
FROM stages2
WHERE TRUE
ORDER BY session_date,
         user_pseudo_id,
         ga_session_id;
