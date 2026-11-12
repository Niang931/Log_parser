# Third-party notices

logpipe is MIT-licensed (see `LICENSE`). It includes or adapts code from the projects below,
and uses the models and datasets listed at the end.

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
- Paper: Amir Shetaia, Sean Kauffman. *DeepParse: Hybrid Log Parsing with LLM-Synthesized Regex
  Masks.* Proceedings of the 30th International Conference on Evaluation and Assessment in
  Software Engineering (EASE 2026), ACM, Glasgow. To appear.
- License: Apache License 2.0 (see `DeepParse/LICENSE`; DeepParse's own dependency notices are
  in `DeepParse/THIRD_PARTY_NOTICES.md`). The `DeepParse/` directory stays under Apache-2.0.
- Changes (NOTICE as required by section 4(b) of the Apache License):
  - `deepparse/drain/drain_engine.py` extended with `match()`, `add_template()` and `clusters`,
    plus changes to mask application, metrics, evaluation, the API, the CLI and sampling.
  - Added by this project: `deepparse/ai_scraper/` (driving signed-in Gemini, Claude and
    ChatGPT web apps), `deepparse/providers/` (Anthropic, OpenAI, Gemini, Groq and web chat
    providers), `deepparse/synth/chat_provider.py`, `deepparse/metrics/template_rules.py`,
    `deepparse/tools/retrain.py`, `deepparse/tools/sample_loghub_full.py`,
    `deepparse/tools/synth_iot_fleet.py`.
  - The full history of changes is in this repository's git log under `DeepParse/`.

## Models downloaded at runtime (not redistributed)

- `sentence-transformers/all-MiniLM-L6-v2` (Apache License 2.0), the default text model for
  Ask's template index (`LOGPIPE_ASK_MODEL`).
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (weights under MIT; derived from Llama 3.1 8B and
  so also subject to the Llama 3.1 Community License) and the LoRA adapter
  `MThien213/deepparse-iot-r1-8b`, only with `LLM_BACKEND=hf`.

## Datasets (downloaded at evaluation time, not redistributed)

- Loghub and Loghub-2.0: https://github.com/logpai/loghub, https://github.com/logpai/loghub-2.0.
  Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
  *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023.
- HDFS and BGL archives come from Zenodo record 3227177, the same archives LogBERT's
  download scripts use.
- RCAEval: https://github.com/phamquiluan/RCAEval (MIT). Run through `uvx` by
  `make eval-rcaeval`; its datasets are downloaded into `artifacts/` and not redistributed.
  Luan Pham, Hongyu Zhang, Huong Ha, Flora Salim, Xiuzhen Zhang. *RCAEval: A Benchmark for
  Root Cause Analysis of Microservice Systems with Telemetry Data.* WWW 2025 Companion.
  https://doi.org/10.1145/3701716.3715290
