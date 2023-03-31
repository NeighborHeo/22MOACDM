-- 1. cohort 
if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort

select subject_id, cohort_start_date 
into #temp_cohort
from @cohort_database_schema.@target_cohort_table where cohort_definition_id = @target_cohort_id

-- 2. join table (cohort + person)
if OBJECT_ID('tempdb..#temp_cohortperson') IS NOT NULL
	drop table #temp_cohortperson

select * 
into #temp_cohortperson
from #temp_cohort
left join @cdm_database_schema.person
on #temp_cohort.subject_id = @cdm_database_schema.person.person_id

-- 2-1. subject_id, cohort_start_date, age, gender_source_value
if OBJECT_ID('@target_database_schema.@target_person_table') IS NOT NULL
	drop table @target_database_schema.@target_person_table

select person_id, cohort_start_date, gender_source_value, (YEAR(cohort_start_date)-year_of_birth) as age
into @target_database_schema.@target_person_table
from #temp_cohortperson

-- del temp table
if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort
if OBJECT_ID('tempdb..#temp_cohortperson') IS NOT NULL
	drop table #temp_cohortperson