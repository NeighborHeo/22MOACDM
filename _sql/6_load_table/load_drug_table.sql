SELECT *
FROM (
    SELECT person_id, drug_concept_id, drug_exposure_start_date, quantity
    FROM {}
    where drug_concept_id!=0 and quantity is not null
) t1
INNER JOIN {} t2
ON t1.person_id = t2.person_id
INNER JOIN (SELECT concept_id, concept_name FROM {}) cc
ON t1.drug_concept_id = cc.concept_id;