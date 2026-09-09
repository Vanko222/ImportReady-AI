# ImportReady AI — Data Conversion Report v1

Generated: `2026-09-09T11:13:35Z`  
Schema version: `1.0`

## Result

| Dataset | Final records |
|---|---:|
| Canonical product attributes | 86 |
| Compliance rules | 41 |
| Policy sources | 70 |
| Compliance-cost references | 32 |
| Persistent known gaps | 16 |
| Pending policy updates | 0 |

All 41 original Rule IDs and all 51 original Source IDs have direct destinations. The source union contains 70 IDs after adding S052–S070. No source IDs were consolidated because no cross-ID canonical-URL duplicate was identified in the approved files.

## Canonical attribute conversion

The 49 original attributes were retained. Thirty-seven distinct supplement fields received permanent IDs. `NEW cell_diameter_height_chemistry` was merged into existing `A-ELEC-017` because both describe button/coin-cell diameter, height, and chemistry; `A-ELEC-017` is now categorized as `common`, and every rule reference was updated. Repeated temporary names such as `asin`, `amazon_compliance_notification`, `fcc_authorization_path`, `fcc_id`, `us_responsible_party_identity`, and `recall_or_platform_prohibition_match` were assigned once and reused.

| Temporary supplement name | Canonical attribute_id | Note |
|---|---|---|
| `NEW amazon_compliance_notification` | `A-CMN-014` |  |
| `NEW amazon_store_country` | `A-CMN-012` |  |
| `NEW antenna_gain` | `A-ELEC-029` |  |
| `NEW applicable_children_product_rules` | `A-TOY-024` |  |
| `NEW asin` | `A-CMN-013` |  |
| `NEW business_employee_count` | `A-CMN-009` |  |
| `NEW cell_diameter_height_chemistry` | `A-ELEC-017` | Merged into existing canonical attribute. |
| `NEW changed_element` | `A-TOY-028` |  |
| `NEW children_product_classification_confirmed` | `A-TOY-027` |  |
| `NEW circuit_voltage_rms` | `A-TOY-026` |  |
| `NEW class_a_or_b` | `A-ELEC-026` |  |
| `NEW conducted_or_radiated_power` | `A-ELEC-028` |  |
| `NEW duty_cycle` | `A-ELEC-030` |  |
| `NEW e_mobility_exclusion` | `A-ELEC-037` |  |
| `NEW eps_exception_claim` | `A-ELEC-032` |  |
| `NEW eps_output_type_and_power` | `A-ELEC-031` |  |
| `NEW existing_report_date` | `A-CMN-015` |  |
| `NEW factory_or_component_source_change` | `A-TOY-029` |  |
| `NEW fcc_authorization_path` | `A-ELEC-023` |  |
| `NEW fcc_id` | `A-ELEC-025` |  |
| `NEW installed_or_separately_packaged` | `A-ELEC-040` |  |
| `NEW label_space_or_e_label` | `A-ELEC-027` |  |
| `NEW loose_18650_cell` | `A-ELEC-038` |  |
| `NEW marking_exception_claim` | `A-CMN-008` |  |
| `NEW marking_location` | `A-CMN-007` |  |
| `NEW medical_device_exclusion` | `A-ELEC-034` |  |
| `NEW no_authorization_basis` | `A-ELEC-036` |  |
| `NEW package_quantity` | `A-ELEC-035` |  |
| `NEW part1501_exclusion_claim` | `A-TOY-025` |  |
| `NEW prop65_listed_chemical_or_exposure_flag` | `A-CMN-010` |  |
| `NEW rated_battery_energy` | `A-ELEC-033` |  |
| `NEW recall_or_platform_prohibition_match` | `A-CMN-016` |  |
| `NEW residential_or_business_environment` | `A-ELEC-022` |  |
| `NEW safe_harbor_assessment` | `A-CMN-011` |  |
| `NEW substantial_transformation_basis` | `A-CMN-006` |  |
| `NEW toy_compliance_path` | `A-ELEC-041` |  |
| `NEW us_responsible_party_identity` | `A-ELEC-024` |  |
| `NEW zinc_air_cell` | `A-ELEC-039` |  |

`A-CMN-001.product_category` now uses the approved classification values `childrens_toys`, `small_consumer_electronics`, `dual`, `unsupported`, and `uncertain`. For `dual`, runtime collection uses the union of attributes required by the actually relevant rule paths, de-duplicated by canonical ID; it does not request both entire taxonomies.

## Schema and status corrections

- Original `verification_status` values were migrated to `evidence_status`.
- Independent `rule_status` was applied from the supplement patch.
- `R-ELEC-019` is `VERIFIED` + `PROPOSED`; it cannot generate an active obligation.
- `R-ELEC-018` is `VERIFIED` + `WATCHLIST`; it cannot generate an active obligation.
- The other 39 rules are `VERIFIED` + `EFFECTIVE`, but still require their runtime trigger to be satisfied.
- No fixed `applicability_status` is stored in any rule. Runtime order remains evidence check → lifecycle gate → trigger evaluation → applicability result.
- Only `VERIFIED` + `EFFECTIVE` rules may enter obligation evaluation. Unknown required trigger data returns the supplement-authored `NEEDS_INFO` or `REVIEW_REQUIRED` result for all 33 P0 rules.

## Gap consolidation and traceability

| Original/supplement record | Destination | Treatment |
|---|---|---|
| `GAP-001` | SUP-GAP-001 + SUP-GAP-002 | Targeted supplement replaces the broad toy-price gap. |
| `GAP-002` | SUP-GAP-003 | FCC lab ranges were added; public TCB fee remains the persistent gap. |
| `GAP-003` | SUP-GAP-004 | UN 38.3 attempt and remaining uncertainty retained. |
| `GAP-004` | SUP-GAP-005 | Amazon-approved provider price remains quote-required. |
| `GAP-005` | SUP-GAP-006 + SUP-GAP-007 + SUP-GAP-008 | Toy baseline improved; lithium detail and re-access remain unresolved. |
| `GAP-008` | SUP-GAP-010 | UL item retained as WATCHLIST, not a universal federal obligation. |
| `GAP-012` | SUP-GAP-009 | FCC item retained as VERIFIED + PROPOSED. |
| `GAP-011` | Resolved by A-CMN-001 dual routing | Both relevant rule paths are evaluated; required attributes are unioned and de-duplicated. |
| `GAP-013` | Resolved by status/activation schema | Missing evidence cannot produce a definitive green result. |
| `GAP-014` | Excluded from USA rule logic | EU GPSR material remains outside the target jurisdiction. |
| `SUP-GAP-011` | Resolved by 37 new canonical records plus one merge | Temporary P0 fields now have permanent IDs. |

Original `GAP-006`, `GAP-007`, `GAP-009`, and `GAP-010` remain directly in `known_gaps.json`. `CONV-GAP-001` records that the approved files do not provide P1 clarification questions or missing-information statuses; those rules remain review-only. Non-ID sheet labels in supplement gap linkage fields were not emitted as source IDs.

## Cost normalization

- The eight original cost records and 24 supplement records were retained: 32 total.
- Equal original low/high values were normalized to `exact_amount` without changing the amount.
- Public vendor prices/ranges are `DIRECT`; planning/regulatory estimates are `PLANNING_ONLY`; historical, expired, or specialty examples are `DISPLAY_ONLY`; absent prices are `QUOTE_REQUIRED`.
- No provider amounts were averaged and no currencies were converted.
- A required `QUOTE_REQUIRED` item has null amounts, never zero; the application may show a known-cost subtotal only with an explicit incomplete-total warning.

## Excluded from active MVP logic

- P1 rules: R-ELEC-008, R-ELEC-013, R-ELEC-018, R-ELEC-019, R-TOY-009, R-TOY-016, R-TOY-017, R-TOY-018. They remain reference/escalation records; source-authored P1 clarification mappings are not available.
- `R-ELEC-019`: excluded by `PROPOSED` lifecycle.
- `R-ELEC-018`: excluded by `WATCHLIST` lifecycle.
- Sources or cost records marked `UNVERIFIED`, `NOT_FOUND`, planning-only, quote-required, expired, or display-only remain contextual and cannot support a definitive compliance result.
- EU GPSR content is excluded by the USA jurisdiction scope. Tariffs, marketplace commissions, product cost, logistics cost, and carrier-specific fee logic were not added.

## Unresolved conflicts and uncertainties

No unresolved authoritative-source conflict was silently resolved. Remaining uncertainty is captured in `known_gaps.json`, including standalone lead/phthalates prices, TCB fees, UN 38.3 pricing quality, Amazon-approved-provider pricing, restricted Amazon lithium detail, source re-access, arbitrary-product ASTM/FCC scoping, unresearched state overlays, carrier-specific transport rules, and P1 clarification coverage. `NOT_FOUND` is never treated as `NOT_REQUIRED`.

## Validation

- PASS — Six JSON files created
- PASS — Unique rule_id values
- PASS — Unique source_id values
- PASS — Unique attribute_id values
- PASS — Every original rule preserved
- PASS — Every original source preserved
- PASS — All supplement source IDs preserved
- PASS — All rule source references resolve
- PASS — All required attributes resolve
- PASS — All attribute rule references resolve
- PASS — All P0 clarification mappings retained
- PASS — No static applicability_status stored
- PASS — Proposed/watchlist lifecycle preserved
- PASS — No temporary NEW attribute references in JSON data
- PASS — QUOTE_REQUIRED never converted to zero
- PASS — NOT_FOUND never converted to NOT_REQUIRED
- PASS — Source files unchanged during generation

All six JSON files are UTF-8, contain `schema_version`, `generated_at`, `source_files`, and `records`, and were parsed again after writing. The four approved research files matched their recorded SHA-256 hashes before and after conversion and were not edited, renamed, moved, or deleted.
