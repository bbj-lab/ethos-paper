ICD_QUERY = """ SELECT main.*, items.long_title
        FROM mimiciv_hosp.diagnoses_icd main 
        LEFT JOIN mimiciv_hosp.d_icd_diagnoses items 
        ON main.icd_code = items.icd_code AND main.icd_version = items.icd_version 
    """

ADMISSIONS_QUERY = f""" SELECT a.hadm_id, a.subject_id, a.admittime, 
        CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END AS death, p.gender, 
        DATE_PART('year', a.admittime) - p.anchor_year + p.anchor_age as admit_age
        FROM mimiciv_hosp.admissions a
        LEFT JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id 
    """

