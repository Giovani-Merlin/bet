# Examples

## Zeshel

First train a WBDSM model, then using its output path you can train with the zeshel dataset.

Considering that the output path was `model_en/lightning_logs/version_1`:

```bash

"""
python scripts/train.py \
    --data_data_path data/zeshel/wbdsm_format \
    --query_encoder_weights_path model/en/lightning_logs/version_1 \
    --candidate_encoder_weights_path model_en/lightning_logs/version_1 \
    --output_path models/zeshel
"""
