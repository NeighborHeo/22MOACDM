-- 1. drug concepts
DROP TABLE IF EXISTS temp_drug_concept;

SELECT DISTINCT concept_id, concept_name, standard_concept, vocabulary_id
INTO temp temp_drug_concept
FROM @cohort_database_schema.drug_exposure d
LEFT JOIN @cohort_database_schema.concept c
ON d.drug_concept_id = c.concept_id;

-- 2, search string
DROP TABLE IF EXISTS temp_concepts;

SELECT *
INTO temp temp_concepts
FROM temp_drug_concept
WHERE lower(concept_name) LIKE lower('%@drugname%') and lower(vocabulary_id) LIKE lower('%RxNorm%') and standard_concept='S';

-- 3, results 
-- SELECT * FROM temp_concepts

-- 4, make a list
SELECT DISTINCT string_agg(CAST(concept_id AS varchar), ',') AS concept_list
FROM temp_concepts;