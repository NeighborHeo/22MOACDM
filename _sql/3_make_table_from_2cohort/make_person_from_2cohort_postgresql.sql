---- 1) load cohort1 (total) ----
DROP TABLE IF EXISTS temp_cohort_total;

SELECT person_id AS subject_id, cohort_start_date AS cohort_start_date
INTO TEMP temp_cohort_total
FROM @target_database_schema.@target_person_table_total;

---- 2) load cohort2 (case) ---
DROP TABLE IF EXISTS temp_cohort_case;

SELECT person_id AS subject_id, cohort_start_date AS first_abnormal_date
INTO TEMP temp_cohort_case
FROM @target_database_schema.@target_person_table_case;

---- 3) total + case cohort ---
DROP TABLE IF EXISTS temp_cohort;

SELECT t.subject_id
, t.cohort_start_date
, c.first_abnormal_date
, (CAST(first_abnormal_date AS date) - CAST(cohort_start_date AS date)) AS n_diff
INTO TEMP temp_cohort
FROM (SELECT * FROM temp_cohort_total) t
LEFT JOIN (SELECT * FROM temp_cohort_case) c
ON t.subject_id = c.subject_id;

---- 4) Filtering of patients with adverse events occurring within 60 days ---
DROP TABLE IF EXISTS temp_cohort_filter;

SELECT *, ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY n_diff ASC, cohort_start_date) AS row_count
INTO TEMP temp_cohort_filter
FROM temp_cohort 
WHERE (n_diff <= 60 and n_diff > 0) or n_diff is null;

---- 5) select first row in each patient using not exists ---
DROP TABLE IF EXISTS temp_cohort_final;

SELECT * 
INTO TEMP temp_cohort_final
FROM temp_cohort_filter AS co
WHERE co.row_count=1
ORDER by subject_id;

---- 6) join table (cohort & person) ----
DROP TABLE IF EXISTS temp_person;

SELECT *
INTO TEMP temp_person
FROM temp_cohort_final co
LEFT JOIN @cdm_database_schema.person ps
ON co.subject_id = ps.person_id;

--select * from temp_person;

---- 6) save table (person table) ----
DROP TABLE IF EXISTS @target_database_schema.@target_person_table;

SELECT person_id
, cohort_start_date
, first_abnormal_date
, (date_part('year', cohort_start_date)-year_of_birth) as age
, gender_source_value
, gender_source_concept_id
, n_diff
INTO @target_database_schema.@target_person_table
FROM temp_person;

-- SELECT * FROM @target_database_schema.@target_person_table
DROP TABLE IF EXISTS temp_cohort_total;
DROP TABLE IF EXISTS temp_cohort_case;
DROP TABLE IF EXISTS temp_cohort;
DROP TABLE IF EXISTS temp_cohort_filter;
DROP TABLE IF EXISTS temp_cohort_final;
DROP TABLE IF EXISTS temp_person;