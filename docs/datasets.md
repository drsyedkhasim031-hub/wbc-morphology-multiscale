# Dataset acquisition and verification

Checked on 2026-10-04. [Machine-readable audit](dataset_audit.json) records what was actually obtained.

| Dataset | Official source | Manuscript native task | Common-five task | Verification performed |
|---|---|---:|---:|---|
| Raabin-WBC | [Raabin](https://raabindata.com/) | 14,514, five classes, training + Test-A | 14,514 | Original portal and current archive reachable; image membership not downloaded/verified |
| PBC | [Mendeley v1](https://data.mendeley.com/datasets/snkd93bnjr/1) | 13,193, six WBC classes | 10,298 | Original source description checked; all 17,092 labels in official PBC-derived BloodMNIST verified and counts reconciled; original JPG membership not verified |
| AML-Cytomorphology_LMU | [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/) | 18,365, fifteen classes | 14,833 | Original abbreviations and annotation archive downloaded; all 18,365 original label rows counted and matched; TIFF pixels not downloaded |

The current Raabin archive advertised by its portal is [WBCData.rar](https://dl.raabindata.com/WBCData.rar), with a response content length of 55,781,603,692 bytes during the audit. This can contain broader material than the selected paper cohort. Do not mistake archive size or a reachable link for verified cohort membership. Use the original release's selected classification subsets and preserve Test-A/Test-B distinctions.

PBC original images are available through the Mendeley page (CC BY 4.0); its API returned an interactive challenge here, so no automatic original-archive download was claimed. The CPU demo uses the separately identified [BloodMNIST release](https://zenodo.org/records/10519652), not an alternative silently substituted into native experiments.

AML images use TCIA's linked download, approximately 11 GB with an Aspera access option. Its [abbreviations](https://www.cancerimagingarchive.net/wp-content/uploads/abbreviations.txt) confirm MYB=myelocyte and MYO=myeloblast. The [annotation archive](https://www.cancerimagingarchive.net/wp-content/uploads/AML-CYTOMORPHOLOGY_LMU-annotations-dat.zip) contains original labels and two re-annotation columns. The manuscript's native cohort uses original labels, so re-annotations are retained as audit metadata and do not silently replace targets. Class-number filenames alone do not identify patients.

## Labels

Raabin: basophil 301, eosinophil 1,066, lymphocyte 3,461, monocyte 795, neutrophil 8,891.

PBC WBC selection: basophil 1,218, eosinophil 3,117, lymphocyte 1,214, monocyte 1,420, neutrophil 3,329, immature granulocyte 2,895. Exclude erythroblasts 1,551 and platelets 2,348. Common-five additionally excludes immature granulocytes.

AML original labels: BAS79, EBO78, EOS424, KSC15, LYA11, LYT3937, MMZ15, MOB26, MON1789, MYB42, MYO3268, NGB109, NGS8484, PMB18, PMO70. Common-five retains BAS/EOS/MON, combines LYA+LYT and NGB+NGS, and excludes all remaining labels.

## Metadata schema

Supply UTF-8 CSV with relative POSIX paths from the chosen root:

```csv
path,original_label,collection,patient_id,slide_id,dataset_version,maturation_eligible
train/Neutrophil/example.jpg,neutrophil,train,,,original_release,true
Test-A/Basophil/example.jpg,basophil,test_a,,,original_release,true
```

These are schema examples, not real image memberships. Patient and slide columns remain blank until verified. Use dataset-specific IDs; confirmed cross-dataset relationships are expressed through the separate `id_a,id_b` link file. Set `maturation_eligible=false` for explicitly missing/ambiguous maturation targets while keeping valid classification supervision.

Generated manifests add stable cell IDs, pixel hash, pHash, image dimensions, native label, original label, group, and maturation target. The split manifest adds `split` and AML `outer_fold`. Hash groups connect decoded RGB duplicates, regardless of file compression. pHash distance ≤4 only produces review candidates. Do not pass the unreviewed candidate file as confirmed links.

Tests are image-level unless patient/slide provenance demonstrates otherwise. The public datasets remain at their original providers; observe their licenses and citation requirements in [DATA_LICENSES.md](../DATA_LICENSES.md).
