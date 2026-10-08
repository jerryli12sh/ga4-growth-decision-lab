-- 用户留存
 -- Purpose: calculate exact-day D0-D14 retention for November first-visit cohorts.
-- Input grain: one GA4 event.
-- Output grain: one cohort_date + day_n.
-- Cohort window: 2020-11-01 to 2020-11-30.
-- Activity window: 2020-11-01 to 2020-12-14.
-- Denominator: distinct users in each first_visit cohort.
-- Numerator: cohort users active on the exact nth calendar day.
-- QA: D0 retained_users = cohort_size; retained_users <= cohort_size;
--     0 <= retention_rate <= 1; every cohort has 15 rows.
 WITH first_visit_cohort AS
  (SELECT user_pseudo_id,
          MIN(PARSE_DATE('%Y%m%d', event_date)) AS first_date
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201130'
     AND event_name = 'first_visit'
     AND user_pseudo_id IS NOT NULL
   GROUP BY user_pseudo_id),
      activity AS
  (SELECT DISTINCT user_pseudo_id,
                   PARSE_DATE('%Y%m%d', event_date) AS activity_date
   FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
   WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201214'),
      nov_date AS
  (SELECT date
   FROM UNNEST (GENERATE_DATE_ARRAY(DATE '2020-11-01', DATE '2020-11-30', INTERVAL 1 DAY)) AS date),
      delta_days AS
  (SELECT num AS delta_day
   FROM UNNEST(GENERATE_ARRAY(0, 14)) AS num),
      retained AS
  (SELECT f.first_date AS cohort_date,
          d.delta_day,
          DATE_ADD(f.first_date, INTERVAL d.delta_day DAY) AS activity_date,
          COUNT(DISTINCT a.user_pseudo_id) AS active_size,
          COUNT(DISTINCT f.user_pseudo_id) AS cohort_size,
          SAFE_DIVIDE(COUNT(DISTINCT a.user_pseudo_id), COUNT(DISTINCT f.user_pseudo_id)) AS active_perc
   FROM first_visit_cohort f
   CROSS JOIN delta_days d
   LEFT JOIN activity a ON DATE_ADD(f.first_date, INTERVAL d.delta_day DAY) = a.activity_date
   AND a.user_pseudo_id = f.user_pseudo_id
   GROUP BY f.first_date,
            d.delta_day)
SELECT *
FROM retained
ORDER BY cohort_date,
         delta_day;
