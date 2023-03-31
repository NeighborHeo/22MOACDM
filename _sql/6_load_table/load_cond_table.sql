SELECT *
FROM (
    SELECT person_id, condition_concept_id, condition_start_date
    FROM {}
    where condition_concept_id!=0
) t1
INNER JOIN {} t2
ON t1.person_id = t2.person_id
INNER JOIN (SELECT concept_id, concept_name FROM {}) cc
ON t1.condition_concept_id = cc.concept_id;