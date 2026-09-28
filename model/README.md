# Model Evaluation Scripts

Scripts for the algorithm-side evaluation of the S2ST accelerator, built on [StreamSpeech](https://github.com/ictnlp/StreamSpeech) and evaluated on CVSS-C.

| Directory | Contents |
|---|---|
| `lp_decomposition/` | Block-wise linear prediction and residual sparsification of vocoder features, with a replay-based quality evaluation |
| `quantization/` | MXINT4 weight and activation quantization, logit drift, WER and BLEU scoring |
| `group_reordering/` | Scaling-factor gap statistics and quantization-group reordering on 8x8 tiles |
| `speculative_generation/` | Probability-triggered speculative speech generation agent, sweeps, latency analysis and FLOPs calibration |
| `profiling/` | Per-component FLOPs of StreamSpeech and the unit vocoder |
| `data_prep/` | CVSS-C / CoVoST 2 test-set and subset construction |
| `patches/` | Changes applied to the StreamSpeech tree |

## Setup

Python 3.10 and a CUDA GPU.

```bash
git clone https://github.com/ictnlp/StreamSpeech.git
cd StreamSpeech
git apply <this repo>/model/patches/streamspeech.patch

python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<1.24"
(cd fairseq && pip install --editable ./ --no-build-isolation)
(cd SimulEval && pip install --editable ./)
pip install sacrebleu==2.3.1 openai-whisper==20231117 soundfile pyarrow matplotlib pyyaml
```

`ffmpeg` must be on `PATH` for Whisper.

Download the simultaneous models (`streamspeech.simultaneous.{fr,es,de}-en.pt`) from [Hugging Face](https://huggingface.co/ICTNLP/StreamSpeech_Models/tree/main) into `pretrain_models/`, and the unit-based HiFi-GAN vocoder ([ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)) into `pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/`. Replace `/data/zhangshaolei/StreamSpeech` and `/data/zhangshaolei/pretrain_models` in `configs/*/config_gcmvn.yaml` and `configs/*/config_mtl_asr_st_ctcst.yaml` with the local paths.

For test data, place the CoVoST 2 test parquet files in `datasets/covost2/fr_en_parquet/` and the CVSS-C `test.tsv` in `datasets/cvss/cvss-c/fr-en/`, then run:

```bash
python <this repo>/model/data_prep/build_fulltest_data.py
python <this repo>/model/data_prep/make_subsets.py
```

Scripts run from the StreamSpeech root. Set `STREAMSPEECH_ROOT` to run from elsewhere.

Scripts are provided under `Apache-2.0`. `speculative_generation/speech_to_text.s2tt.confidence.agent.py` is derived from StreamSpeech and also carries its MIT notice (`LICENSE.StreamSpeech`).
