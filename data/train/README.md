# Training data

One JSON object per line in `*.jsonl` files:

```json
{"text": "Wat steiht morgen an?", "intent": "appointments_tomorrow", "timeframe": "tomorrow", "entity": null}
```

`timeframe` and `entity` are optional and default to `null`.

The holdout set in `data/holdout_benchmarks.json` is never used for training.
`tools/check_holdout_leak.py` rejects any training sentence that is identical
or close to a holdout sentence (character or word similarity of 0.85 or more);
the Data check workflow runs it on every change below `data/`.
