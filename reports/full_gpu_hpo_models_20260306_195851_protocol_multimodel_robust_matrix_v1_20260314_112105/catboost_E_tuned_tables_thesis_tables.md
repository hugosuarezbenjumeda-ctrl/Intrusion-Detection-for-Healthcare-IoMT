Table 5.1 CatBoost E Protocol Metrics (Tuned Thresholds)

| protocol_model | threshold | precision | recall | F1 | FPR | roc_auc | pr_auc | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wifi_catboost_E | 0.896667 | 0.999992 | 0.998153 | 0.999072 | 0.000288 | 0.999912 | 0.999998 | 1510066 | 41725 | 12 | 2794 |
| mqtt_catboost_E | 0.946667 | 1.000000 | 0.993738 | 0.996859 | n/a | n/a | 1.000000 | 63316 | 0 | 0 | 399 |
| bluetooth_catboost_E | 0.870000 | 1.000000 | 0.998649 | 0.999324 | 0.000000 | 1.000000 | 1.000000 | 25136 | 6671 | 0 | 34 |
| Global | Model Specific | 0.999992 | 0.997985 | 0.998988 | 0.000248 | 0.999886 | 0.999997 | 1598518 | 48396 | 12 | 3227 |


Note. MQTT FPR remains undefined because the test split contains no MQTT benign negatives.



Table 5.2 CatBoost E Robustness Metrics (Tuned Thresholds)

| Protocol | clean_f1 | clean_fpr | attacked_benign_fpr | adv_malicious_recall | robust_f1 | selected_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| WiFi | 0.999072 | 0.000288 | n/a | 0.996319 | 0.998152 | 0.896667 |
| Mqtt | 0.996859 | n/a | n/a | 0.051919 | 0.098712 | 0.946667 |
| Bluetooth | 0.999324 | 0.000000 | n/a | 0.999841 | 0.999921 | 0.870000 |
| Global | 0.998988 | 0.000248 | n/a | 0.958807 | 0.978967 | Model Specific |


Note. In this surrogate FGSM/PGD export, only malicious rows are perturbed, so attacked benign FPR is undefined. Global selected threshold is model specific, not an arithmetic average.


