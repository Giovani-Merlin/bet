## Example to train BLINK on Zero-shot Entity Linking dataset

Follow BLINK [tutorial](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel) - download and convert the dataset to BLINK format.
After it, convert to BET format using the following command:

```bash
python examples/zeshel/convert_to_bet_format.py --input_path data/zeshel/blink_format --output_path data/zeshel/wbdsm_format
```

To eval the dataset, we need to split it by worlds (to keep BET format). For that, we can use the following command:

```bash
python examples/zeshel/split_by_worlds.py --input_path data/zeshel/wbdsm_format --output_path data/zeshel/wbdsm_format/split_by_worlds
```
