# Medical Taxonomy & Justification
**Defense Document for Clinical Experts**

## Executive Summary
To improve model robustness without sacrificing clinical utility, we aggregated the 50+ granular SCP-ECG codes into **19 Major Clinical Categories**. This grouping strategy follows the **AHA/ACC/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram (2009)**, focusing on *Anatomical Localization* for Ischemia/Infarction and *Physiological Mechanism* for Arrhythmias/Blocks.

---

## 1. Myocardial Infarction & Ischemia (Anatomical Grouping)
*Logic: Clinically, the reperfusion strategy (opening the artery) depends on the **Territory** (Anterior vs. Inferior), not the specific age of the infarct (Acute vs Old) for a screening algorithm.*

| Our Category | Included SCP Codes | Medical Justification |
| :--- | :--- | :--- |
| **MI_Anterior** | `AMI` (Acute), `ALMI` (Anterolateral), `ASMI` (Anteroseptal), `INJAL`, `INJAS` | Corresponds to the **Left Anterior Descending (LAD)** artery territory. |
| **MI_Inferior** | `IMI`, `ILMI`, `IPLMI`, `IPMI`, `INJIL`, `INJIN` | Corresponds to the **Right Coronary Artery (RCA)** or **LCx** territory. |
| **MI_Lateral** | `LMI`, `PMI` (Posterior), `INJLA` | Corresponds to the **Left Circumflex (LCx)** territory. |
| **Ischemia** | `ISC_`, `ISCAL`, `ISCAN`, `ISCAS`, `ISCIL`, `ISCIN`, `ISCLA`, `NST_` | Represents **ST-T Wave Changes** indicative of supply-demand mismatch, distinct from infarction (tissue death). |

## 2. Conduction Abnormalities (Pathological Grouping)
*Logic: Grouped by the level of the block within the conduction system.*

| Our Category | Included SCP Codes | Medical Justification |
| :--- | :--- | :--- |
| **LBBB** | `CLBBB` (Complete), `ILBBB` (Incomplete) | **Left Bundle Branch Block**. Critical marker for structural heart disease and dyssynchrony. Kept distinct from RBBB. |
| **RBBB** | `CRBBB`, `IRBBB` | **Right Bundle Branch Block**. Often benign, but clinically distinct from LBBB. |
| **AV_Block** | `1AVB`, `2AVB`, `3AVB` | **Atrioventricular Node Dysfunction**. *Note: Experts may argue that 1st Degree and 3rd Degree have vastly different urgency, but mechanistically they are both AV Node failures.* |
| **Fascicular_Block**| `LAFB`, `LPFB` | Blocks in the sub-branches (fascicles) of the Left Bundle. |
| **IVCD** | `IVCD` | Non-specific delays not meeting LBBB/RBBB criteria. |

## 3. Hypertrophy (Structural Grouping)
| Our Category | Included SCP Codes | Medical Justification |
| :--- | :--- | :--- |
| **Left_Hypertrophy** | `LVH`, `LAO/LAE` | Indicates pressure overload (e.g., Hypertension). |
| **Right_Hypertrophy**| `RVH`, `RAO/RAE`, `SEHYP` | Indicates volume/pressure overload (e.g., Pulmonary Hypertension). |

## 4. Rhythm & Arrhythmia (Physiological Grouping)
| Our Category | Included SCP Codes | Medical Justification |
| :--- | :--- | :--- |
| **Sinus_Rhythm** | `SR`, `STACH` (Tachycardia), `SBRAD` (Bradycardia), `SARRH` (Arrhythmia) | All originate from the **SA Node**. The variation is only in *Rate*, not *Origin*. |
| **Atrial_Fibrillation**| `AFIB`, `AFLT` (Flutter) | Irregular rhythms originating from the atria. Critical stroke risk factors. |
| **SVT** | `SVT`, `PSVT` | Supraventricular Tachycardias (Re-entrant mechanisms). |
| **Paced** | `PACE` | Artificial Pacemaker rhythm. |

---

## Expert Defense Q&A
**Q: "Why group Acute MI with Old MI?"**
A: For a visual screening tool, identifying the **territory of damage** is the primary visual feature (Q-waves vs ST-elevation). The model detects "Damage in Anterior Wall". Clinical correlation is needed to determine age, but the localization is accurate.

**Q: "Why group 1st Degree and 3rd Degree AV Block?"**
A: This is a limitation of the 20-class taxonomy. However, the model successfully identifies "PR Interval Abnormalities". This flags the patient for manual review to determine severity.

**Q: "Why group Sinus Bradycardia with Normal Sinus?"**
A: Bradycardia (slow heart rate) is a vital sign measurement, not a morphological diagnosis. The *shape* of the P-QRS-T complex is identical. Calculating the rate is a simple mathematical step, not a classification task.
