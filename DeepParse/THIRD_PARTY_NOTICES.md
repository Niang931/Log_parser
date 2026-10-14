# Third-party notices

This project includes or adapts code from the projects below.

## LogBERT

- Project: https://github.com/HelenGuohx/logbert (commit `a845e61`)
- Paper: Haixuan Guo, Shuhan Yuan, Xintao Wu. *LogBERT: Log Anomaly Detection via BERT.*
  IJCNN 2021. [arXiv:2103.04475](https://arxiv.org/abs/2103.04475)
- Used in: `logpipe/anomaly/logbert/` (`model.py`, `dataset.py`, `trainer.py`), and the
  scoring rule and threshold search in `logpipe/anomaly/scorer.py`.
- Changes: listed at the top of each adapted module. In summary:
  - Ported to torch 2.x and merged into one module per concern.
  - Segment and time embeddings removed.
  - `[UNK]` is handled as a trained token that is never masked.
  - Prediction-time masking is deterministic.
  - Models are saved as `state_dict` + JSON, not pickled modules.
  - Thresholds are tuned on a validation split.

```
MIT License

Copyright (c) 2023 Helennnnn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BERT-pytorch

- Project: https://github.com/codertimo/BERT-pytorch, by Junseong Kim and contributors
- LogBERT's `bert_pytorch` package derives from it, so the transformer, attention and
  embedding code in `logpipe/anomaly/logbert/model.py` does too.
- License: Apache License 2.0, http://www.apache.org/licenses/LICENSE-2.0

## DeepParse

- Project: https://github.com/NightBaRron1412/DeepParse, vendored in `DeepParse/`
- License: Apache License 2.0 (see `DeepParse/LICENSE` and `DeepParse/THIRD_PARTY_NOTICES.md`)
- `DeepParse/deepparse/drain/drain_engine.py` was extended with `match()`, `add_template()`
  and `clusters`.

## Datasets (downloaded at evaluation time, not redistributed)

- Loghub and Loghub-2.0: https://github.com/logpai/loghub, https://github.com/logpai/loghub-2.0.
  Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
  *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023.
- HDFS and BGL archives come from Zenodo record 3227177, the same archives LogBERT's
  download scripts use.
