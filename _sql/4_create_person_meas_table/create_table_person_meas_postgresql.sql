-- select * from @cdm_database_schema.measurement 
-- where measurement_concept_id = @target_measurement_concept_id

-- -- 0. Check if use only one unit
-- select unit_concept_id, unit_source_value, COUNT(unit_source_value) from @cdm_database_schema.measurement 
-- where measurement_concept_id = @target_measurement_concept_id
-- group by unit_concept_id, unit_source_value

-- 1. 
DROP TABLE IF EXISTS temp_meas;

select distinct person_id, measurement_concept_id, measurement_date, value_as_number, unit_concept_id
into temp temp_meas
from @cdm_database_schema.measurement 
where measurement_concept_id = @target_measurement_concept_id;

---- 2-1. method to extract rows with max value (1)

--DROP TABLE IF EXISTS temp_meas_group;

--SELECT a.*
--into temp temp_meas_group
--FROM temp_meas a
--INNER JOIN (
--    SELECT person_id, MAX(value_as_number) as value_as_number
--    FROM temp_meas
--    GROUP BY person_id
--) b ON a.person_id = b.person_id AND a.value_as_number = b.value_as_number

-- 2-2. method to extract rows with max value (2)
DROP TABLE IF EXISTS temp_meas_group;

SELECT a.*
into temp temp_meas_group
FROM temp_meas a
LEFT JOIN temp_meas b
ON a.person_id = b.person_id AND a.value_as_number < b.value_as_number
WHERE b.person_id IS NULL;

-- 3. 
DROP TABLE IF EXISTS temp_person;

select m.person_id, m.measurement_concept_id, m.value_as_number, p.gender_source_value, (date_part('year', m.measurement_date)-p.year_of_birth) as age, p.year_of_birth, m.measurement_date
into temp temp_person
from (
	(select * from temp_meas_group) as m
	inner join
	(select person_id, year_of_birth, gender_source_value from @cdm_database_schema.person) as p
	on m.person_id = p.person_id
);

DROP TABLE IF EXISTS @target_database_schema.@target_person_table;

select person_id, measurement_concept_id, value_as_number, gender_source_value, age
into @target_database_schema.@target_person_table
from temp_person;

DROP TABLE IF EXISTS temp_meas;
DROP TABLE IF EXISTS temp_meas_group;
DROP TABLE IF EXISTS temp_person;
