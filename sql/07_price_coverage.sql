SELECT event_name,
       COUNT(*) AS EVENTS,
       COUNTIF(ARRAY_LENGTH(items)>0) AS events_with_items,
       COUNTIF(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE i.price IS NOT NULL)) AS events_with_local_price,
       COUNTIF(EXISTS
                 (SELECT 1
                  FROM UNNEST(items) i
                  WHERE i.price_in_usd IS NOT NULL)) AS events_with_usd_price,
       COUNTIF(EXISTS
                 (SELECT 1
                  FROM UNNEST(event_params) p
                  WHERE p.key='currency'
                    AND p.value.string_value IS NOT NULL)) AS events_with_currency,
       STRING_AGG(DISTINCT
                    (SELECT MAX(p.value.string_value)
                     FROM UNNEST(event_params) p
                     WHERE p.key='currency'),' | '
                  LIMIT 10) AS currencies
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  AND event_name IN ('view_item',
                     'begin_checkout',
                     'purchase')
GROUP BY event_name
ORDER BY event_name;
