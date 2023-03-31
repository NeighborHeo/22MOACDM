SET ANSI_WARNINGS OFF;

---- 1) load cohort1 (total) ----
IF OBJECT_ID('tempdb..#temp_cohort_total') IS NOT NULL
	DROP TABLE #temp_cohort_total;

SELECT person_id AS subject_id, cohort_start_date AS cohort_start_date
INTO #temp_cohort_total
FROM @target_database_schema.@target_person_table_total;

---- 2) load cohort2 (case) ---

IF OBJECT_ID('tempdb..#temp_cohort_case') IS NOT NULL
	DROP TABLE #temp_cohort_case;

SELECT person_id AS subject_id, cohort_start_date AS first_abnormal_date
INTO #temp_cohort_case
FROM @target_database_schema.@target_person_table_case;

---- 3) total + case cohort ---

IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort;

SELECT t.subject_id
, t.cohort_start_date
, c.first_abnormal_date
, DATEDIFF(DAY, t.cohort_start_date, c.first_abnormal_date) AS n_diff 
INTO #temp_cohort
FROM (SELECT * FROM #temp_cohort_total) t
LEFT JOIN (SELECT * FROM #temp_cohort_case) c
ON t.subject_id = c.subject_id;

---- 4) Filtering of patients with adverse events occurring within 60 days ---

IF OBJECT_ID('tempdb..#temp_cohort_filter') IS NOT NULL
	DROP TABLE #temp_cohort_filter;

SELECT *, ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY n_diff ASC, cohort_start_date) AS row_count
INTO #temp_cohort_filter
FROM #temp_cohort 
WHERE (n_diff <= 60 and n_diff > 0) or n_diff is null;

---- 5) select first row in each patient using not exists ---

IF OBJECT_ID('tempdb..#temp_cohort_final') IS NOT NULL
	DROP TABLE #temp_cohort_final;

SELECT * 
INTO #temp_cohort_final
FROM #temp_cohort_filter AS co
WHERE co.row_count=1
ORDER by subject_id;

---- 6) join table (cohort & person) ----

IF OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	DROP TABLE #temp_person;

SELECT *
INTO #temp_person
FROM #temp_cohort_final co
LEFT JOIN @cdm_database_schema.person ps
ON co.subject_id = ps.person_id;

--select * from #temp_person;

---- 6) save table (person table) ----

IF OBJECT_ID('@target_database_schema.@target_person_table') IS NOT NULL
	DROP TABLE @target_database_schema.@target_person_table;

SELECT person_id
, cohort_start_date
, first_abnormal_date
, (YEAR(cohort_start_date)-year_of_birth) AS age
, gender_source_value
, gender_source_concept_id
, n_diff
INTO @target_database_schema.@target_person_table
FROM #temp_person;

-- SELECT * FROM @target_database_schema.@target_person_table

IF OBJECT_ID('tempdb..#temp_cohort_total') IS NOT NULL
	DROP TABLE #temp_cohort_total;
IF OBJECT_ID('tempdb..#temp_cohort_case') IS NOT NULL
	DROP TABLE #temp_cohort_case;
IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort;
IF OBJECT_ID('tempdb..#temp_cohort_filter') IS NOT NULL
	DROP TABLE #temp_cohort_filter;
IF OBJECT_ID('tempdb..#temp_cohort_final') IS NOT NULL
	DROP TABLE #temp_cohort_final;
IF OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	DROP TABLE #temp_person;
