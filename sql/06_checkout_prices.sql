WITH x AS
  (SELECT user_pseudo_id,

     (SELECT MAX(p.value.int_value)
      FROM UNNEST(event_params) p
      WHERE p.key='ga_session_id') AS ga_session_id,
          event_timestamp,

     (SELECT MAX(p.value.string_value)
      FROM UNNEST(event_params) p
      WHERE p.key='currency') AS currency,
          items
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
     AND event_name='begin_checkout'),
     y AS
  (SELECT *,
          ROW_NUMBER() OVER(PARTITION BY user_pseudo_id, ga_session_id
                            ORDER BY event_timestamp, TO_JSON_STRING(items)) AS rn
   FROM x
   WHERE user_pseudo_id IS NOT NULL
     AND ga_session_id IS NOT NULL)
SELECT user_pseudo_id,
       ga_session_id,
       event_timestamp AS checkout_index_ts,
       currency,
       ARRAY_LENGTH(items) AS item_rows,

  (SELECT COUNTIF(i.price IS NOT NULL)
   FROM UNNEST(items)i) AS local_price_rows,

  (SELECT COUNTIF(i.price=0)
   FROM UNNEST(items)i) AS zero_local_price_rows,

  (SELECT COUNTIF(i.price<0)
   FROM UNNEST(items)i) AS negative_local_price_rows,

  (SELECT MIN(i.price)
   FROM UNNEST(items)i) AS min_local_price,

  (SELECT MAX(i.price)
   FROM UNNEST(items)i) AS max_local_price,

  (SELECT SUM(i.price*i.quantity)
   FROM UNNEST(items)i
   WHERE i.price IS NOT NULL
     AND i.quantity IS NOT NULL) AS local_price_quantity_sum
FROM y
WHERE rn=1
ORDER BY user_pseudo_id,
         ga_session_id;
