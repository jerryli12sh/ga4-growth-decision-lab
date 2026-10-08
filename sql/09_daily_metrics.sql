-- Daily session counts, purchase events, revenue and duration quantiles.
-- Grain: session_date. Source window: 2020-11-01 through 2021-01-31.
-- Primary Key: session_date + device_category + source_group
-- 字段：sessions, view_sessions, checkout_sessions, purchase_sessions, revenue_usd, observed_purchase_sessions, all_purchase_revenue_usd
WITH events_log AS
  (SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date,
          event_name,
          event_timestamp,
          ecommerce.purchase_revenue_in_usd AS purchase_revenue_in_usd,
          user_pseudo_id,
          CONCAT(user_pseudo_id, '-', CAST(params.value.int_value AS STRING)) AS session_key,
          params.value.int_value AS ga_session_id,
          device.category AS device_category,
          device.mobile_brand_name AS device_brand,
          device.operating_system AS device_os,
          geo.continent AS continent,
          geo.country AS country,
          traffic_source.source AS user_source,
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
        UNNEST(event_params) AS params
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND params.key = 'ga_session_id'
     AND user_pseudo_id IS NOT NULL
     AND params.value.int_value IS NOT NULL),
     session_base AS
  (SELECT session_key,
          MIN(user_pseudo_id) AS user_pseudo_id,
          MIN(event_date) AS session_date,
          ARRAY_AGG(STRUCT(event_name AS event_name, event_timestamp AS event_timestamp)
                    ORDER BY event_timestamp) AS EVENTS,
          ARRAY_AGG(device_category
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS device_category,
          ARRAY_AGG(device_brand
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS device_brand,
          ARRAY_AGG(device_os
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS device_os,
          ARRAY_AGG(continent
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS continent,
          ARRAY_AGG(country
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS country,
          ARRAY_AGG(user_source
                    ORDER BY event_timestamp
                    LIMIT 1)[OFFSET(0)] AS user_source,
          SUM(CASE
                  WHEN purchase_revenue_in_usd IS NULL THEN 0
                  ELSE purchase_revenue_in_usd
              END) AS session_revenue_usd,
          COUNTIF(event_name = 'view_item') AS view_item,
          COUNTIF(event_name = 'add_to_cart') AS add_to_cart,
          COUNTIF(event_name = 'begin_checkout') AS begin_checkout,
          COUNTIF(event_name = 'add_shipping_info') AS add_shipping_info,
          COUNTIF(event_name = 'add_payment_info') AS add_payment_info,
          COUNTIF(event_name = 'purchase') AS purchase,
   FROM events_log
   GROUP BY session_key),
     view_stage AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e
      WHERE e.event_name = 'view_item') AS view_item_ts
   FROM session_base),
     checkout_stage AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e
      WHERE e.event_name = 'begin_checkout'
        AND e.event_timestamp > view_item_ts) AS begin_checkout_ts
   FROM view_stage),
     shipping_stage AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e
      WHERE e.event_name = 'add_shipping_info'
        AND e.event_timestamp > begin_checkout_ts) AS add_shipping_info_ts
   FROM checkout_stage),
     payment_stage AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e
      WHERE e.event_name = 'add_payment_info'
        AND e.event_timestamp > begin_checkout_ts) AS add_payment_info_ts
   FROM shipping_stage),
     purchase_stage AS
  (SELECT *,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e
      WHERE e.event_name = 'purchase'
        AND e.event_timestamp > add_payment_info_ts
        AND e.event_timestamp > add_shipping_info_ts) AS purchase_ts,

     (SELECT MIN(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e) AS session_begin_ts,

     (SELECT MAX(e.event_timestamp)
      FROM UNNEST(EVENTS) AS e) AS session_close_ts,
   FROM payment_stage),
     session_metrics AS
  (SELECT *, -- event_timestamp 的单位是微秒，因此统一转换为秒。
 -- 同时排除缺失或顺序异常的时间戳。
  IF(session_begin_ts IS NOT NULL
     AND session_close_ts IS NOT NULL
     AND session_close_ts >= session_begin_ts, SAFE_DIVIDE(session_close_ts - session_begin_ts, 1000000), NULL) AS session_duration_sec,
  IF(view_item_ts IS NOT NULL
     AND begin_checkout_ts IS NOT NULL
     AND begin_checkout_ts >= view_item_ts, SAFE_DIVIDE(begin_checkout_ts - view_item_ts, 1000000), NULL) AS view_to_checkout_duration_sec,
  IF(begin_checkout_ts IS NOT NULL
     AND add_shipping_info_ts IS NOT NULL
     AND add_shipping_info_ts >= begin_checkout_ts, SAFE_DIVIDE(add_shipping_info_ts - begin_checkout_ts, 1000000), NULL) AS checkout_to_shipping_duration_sec,
  IF(add_shipping_info_ts IS NOT NULL
     AND add_payment_info_ts IS NOT NULL
     AND add_payment_info_ts >= add_shipping_info_ts, SAFE_DIVIDE(add_payment_info_ts - add_shipping_info_ts, 1000000), NULL) AS shipping_to_payment_duration_sec,
  IF(add_payment_info_ts IS NOT NULL
     AND purchase_ts IS NOT NULL
     AND purchase_ts >= add_payment_info_ts, SAFE_DIVIDE(purchase_ts - add_payment_info_ts, 1000000), NULL) AS payment_to_purchase_duration_sec,
  IF(begin_checkout_ts IS NOT NULL
     AND purchase_ts IS NOT NULL
     AND purchase_ts >= begin_checkout_ts, SAFE_DIVIDE(purchase_ts - begin_checkout_ts, 1000000), NULL) AS checkout_to_purchase_duration_sec
   FROM purchase_stage), -- session_metrics 是完整的 session-level fact table
 -- 聚合为 daily 数据
-- 聚合为轻量的 daily 基础指标
daily_base AS
  (SELECT session_date, /* 规模与主要漏斗节点 */  COUNT(*) AS sessions,
                                         COUNTIF(view_item > 0) AS view_sessions,
                                         COUNTIF(begin_checkout > 0) AS checkout_sessions,
                                         COUNTIF(purchase > 0) AS purchase_sessions, /* 交易与收入 */  SUM(COALESCE(purchase, 0)) AS transactions,
                                                                                                  SUM(COALESCE(session_revenue_usd, 0)) AS revenue_usd, /* Session 时长：只保留中位数和慢尾 */  APPROX_QUANTILES(session_duration_sec, 100
                                                                                                                                                                                                     IGNORE NULLS) AS session_duration_quantiles, /* Checkout → Purchase：保留有效样本量及关键分位数 */  COUNT(checkout_to_purchase_duration_sec) AS checkout_to_purchase_duration_n,
                                                                                                                                                                                                                                                                                           APPROX_QUANTILES(checkout_to_purchase_duration_sec, 100
                                                                                                                                                                                                                                                                                                            IGNORE NULLS) AS checkout_to_purchase_duration_quantiles
   FROM session_metrics
   GROUP BY session_date),
                    daily_metrics AS
  (SELECT session_date,   sessions,
                                                                     view_sessions,
                                                                     checkout_sessions,
                                                                     purchase_sessions,
                                                                     transactions,
                                                                     revenue_usd,   SAFE_DIVIDE(view_sessions, sessions) AS session_to_view_rate,
                                                                     SAFE_DIVIDE(checkout_sessions, view_sessions) AS view_to_checkout_rate,
                                                                     SAFE_DIVIDE(purchase_sessions, checkout_sessions) AS checkout_to_purchase_rate, /* 端到端核心转化率 */  SAFE_DIVIDE(purchase_sessions, view_sessions) AS view_to_purchase_rate,   SAFE_DIVIDE(revenue_usd, sessions) AS revenue_per_session,
                                                                     SAFE_DIVIDE(revenue_usd, transactions) AS revenue_per_transaction,   STRUCT(session_duration_quantiles[SAFE_OFFSET(50)] AS p50, session_duration_quantiles[SAFE_OFFSET(90)] AS p90) AS session_duration_sec,
                                                                     STRUCT(checkout_to_purchase_duration_n AS n, checkout_to_purchase_duration_quantiles[SAFE_OFFSET(50)] AS p50, checkout_to_purchase_duration_quantiles[SAFE_OFFSET(90)] AS p90) AS checkout_to_purchase_duration_sec
   FROM daily_base)
SELECT *
FROM daily_metrics
ORDER BY session_date;
