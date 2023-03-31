-- SELECT *
-- FROM (
--     SELECT person_id, measurement_concept_id, measurement_date, value_as_number, range_low, range_high 
--     FROM {}
--     WHERE measurement_concept_id!=0 AND value_as_number IS NOT NULL
-- ) t1
-- INNER JOIN {} t2
-- ON t1.person_id = t2.person_id
-- INNER JOIN (SELECT concept_id, concept_name FROM {}) cc
-- ON t1.measurement_concept_id = cc.concept_id;

SELECT person_id, measurement_concept_id, measurement_date, value_as_number, range_low, range_high 
FROM {}
WHERE measurement_concept_id!=0 
AND value_as_number IS NOT NULL
AND person_id IN (SELECT person_id FROM {});