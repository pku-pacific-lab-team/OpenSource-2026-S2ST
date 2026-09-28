# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0
#
# Derived from agent/speech_to_text.s2tt.streamspeech.agent.py in StreamSpeech
# (https://github.com/ictnlp/StreamSpeech), Copyright (c) 2024 ICTNLP,
# licensed under the MIT License; see LICENSE.StreamSpeech.

##########################################
# Simultaneous Speech-to-Text Translation Agent for StreamSpeech
#
# StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning (ACL 2024)
##########################################

from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import extract_fbank_features
import ast
import math
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.file_io import PathManager
from fairseq import search
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform

from examples.speech_to_speech.asr_bleu.utils import retrieve_asr_config, ASRGenerator

SHIFT_SIZE = 10
WINDOW_SIZE = 25
ORG_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_EOS = 2


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args, cfg):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = "cuda" if args.device == "gpu" else "cpu"
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            {"feature_transforms": ["utterance_cmvn"]}
        )

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples, sr=ORG_SAMPLE_RATE):
        samples = new_samples
        # # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # # the number of frames used for feature extraction
        # # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )
        samples = samples[:effective_num_samples]
        waveform, sample_rate = convert_waveform(
            torch.tensor([samples]), sr, to_mono=True, to_sample_rate=16000
        )
        output = extract_fbank_features(waveform, 16000)
        output = self.transform(output)
        return torch.tensor(output, device=self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


@entrypoint
class StreamSpeechS2TTConfidenceAgent(SpeechToTextAgent):
    """
    Experimental policy:
      1. every time the streaming ASR (source CTC) emits >= `asr_stride` new words,
         run the MT decoder as a *beam search* continuing from the committed prefix;
      2. if the confidence of the top-1 beam's new tokens >= `conf_threshold`,
         commit them (this is where TTS would be triggered), else READ more audio.
    TTS itself is not run; this is a text-side (S2TT) agent so BLEU/AL are measured
    on the committed text. Every decision is traced to `--trace-file` (jsonl).
    """

    def __init__(self, args):
        super().__init__(args)
        self.eos = DEFAULT_EOS

        self.gpu = self.args.device == "gpu"
        self.device = "cuda" if args.device == "gpu" else "cpu"

        self.args = args

        self.load_model_vocab(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

        tgt_dict_mt = self.dict[f"{self.models[0].mt_task_name}"]
        tgt_dict = self.dict["tgt"]
        tgt_dict_asr = self.dict["source_unigram"]
        tgt_dict_st = self.dict["ctc_target_unigram"]
        args.user_dir=args.agent_dir
        utils.import_user_module(args)
        from agent.sequence_generator import SequenceGenerator
        from agent.ctc_generator import CTCSequenceGenerator
        from agent.ctc_decoder import CTCDecoder
        from agent.tts.vocoder import CodeHiFiGANVocoderWithDur

        self.ctc_generator = CTCSequenceGenerator(
            tgt_dict, self.models, use_incremental_states=True
        )

        self.asr_ctc_generator = CTCDecoder(tgt_dict_asr, self.models)
        self.st_ctc_generator = CTCDecoder(tgt_dict_st, self.models)

        self.generator = SequenceGenerator(
            self.models,
            tgt_dict,
            beam_size=1,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict),
            eos=tgt_dict.eos(),
            symbols_to_strip_from_output=None,
        )

        self.generator_mt = SequenceGenerator(
            self.models,
            tgt_dict_mt,
            beam_size=1,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict_mt),
            eos=tgt_dict_mt.eos(),
            symbols_to_strip_from_output=None,
            use_incremental_states=True,
        )
        self.generator_beam = SequenceGenerator(
            self.models,
            tgt_dict_mt,
            beam_size=args.beam_size,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict_mt),
            eos=tgt_dict_mt.eos(),
            symbols_to_strip_from_output=None,
            use_incremental_states=False,  # beams + persistent KV cache don't mix; recompute
        )
        self.beam_size = args.beam_size
        self.conf_threshold = args.conf_threshold
        self.conf_type = args.conf_type
        self.commit_mode = args.commit_mode
        self.asr_trigger = args.asr_trigger
        self.asr_stride = args.asr_stride
        self.whole_word = not args.no_whole_word
        self.max_commit_tokens = args.max_commit_tokens
        self.max_new = args.max_new
        self.early_thr = args.early_mass_thr
        self.early_mode = args.early_mode
        self.early_norm = args.early_norm
        self.min_steps = args.min_steps
        self.trace_file = args.trace_file
        self.tts_profile = args.tts_profile
        self.vocoder = None
        if self.tts_profile:
            assert args.vocoder and args.vocoder_cfg, "--tts-profile needs --vocoder/--vocoder-cfg"
            with open(args.vocoder_cfg) as f:
                vocoder_cfg = json.load(f)
            self.vocoder = CodeHiFiGANVocoderWithDur(args.vocoder, vocoder_cfg)
            if self.gpu:
                self.vocoder = self.vocoder.cuda()
            # full (non-incremental) unit CTC generator: recompute over the whole prefix
            self.unit_generator = CTCSequenceGenerator(
                tgt_dict, self.models, use_incremental_states=False
            )
        self._trace_fh = None
        if self.trace_file is not None:
            Path(self.trace_file).parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(self.trace_file, "w", encoding="utf-8")
        self.sent_idx = -1
        self.lagging_k1 = args.lagging_k1
        self.lagging_k2 = args.lagging_k2
        self.segment_size = args.segment_size
        self.stride_n = args.stride_n
        self.unit_per_subword = args.unit_per_subword
        self.stride_n2 = args.stride_n2
        if args.extra_output_dir is not None:
            self.asr_file = Path(args.extra_output_dir + "/asr.txt")
            self.st_file = Path(args.extra_output_dir + "/st.txt")
            self.unit_file = Path(args.extra_output_dir + "/unit.txt")
            self.quiet = False
        else:
            self.quiet = True

        self.reset()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="path to your pretrained model.",
        )
        parser.add_argument(
            "--data-bin", type=str, required=True, help="Path of data binary"
        )
        parser.add_argument(
            "--config-yaml", type=str, default=None, help="Path to config yaml file"
        )
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Path to config yaml file",
        )
        parser.add_argument(
            "--global-stats",
            type=str,
            default=None,
            help="Path to json file containing cmvn stats",
        )
        parser.add_argument(
            "--tgt-splitter-type",
            type=str,
            default="SentencePiece",
            help="Subword splitter type for target text",
        )
        parser.add_argument(
            "--tgt-splitter-path",
            type=str,
            default=None,
            help="Subword splitter model path for target text",
        )
        parser.add_argument(
            "--user-dir",
            type=str,
            default="researches/ctc_unity",
            help="User directory for model",
        )
        parser.add_argument(
            "--agent-dir",
            type=str,
            default="agent",
            help="User directory for agents",
        )
        parser.add_argument(
            "--max-len", type=int, default=200, help="Max length of translation"
        )
        parser.add_argument(
            "--force-finish",
            default=False,
            action="store_true",
            help="Force the model to finish the hypothsis if the source is not finished",
        )
        parser.add_argument(
            "--shift-size",
            type=int,
            default=SHIFT_SIZE,
            help="Shift size of feature extraction window.",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=WINDOW_SIZE,
            help="Window size of feature extraction window.",
        )
        parser.add_argument(
            "--sample-rate", type=int, default=ORG_SAMPLE_RATE, help="Sample rate"
        )
        parser.add_argument(
            "--feature-dim",
            type=int,
            default=FEATURE_DIM,
            help="Acoustic feature dimension.",
        )
        parser.add_argument("--lagging-k1", type=int, default=0, help="lagging number")
        parser.add_argument("--lagging-k2", type=int, default=0, help="lagging number")
        parser.add_argument(
            "--segment-size", type=int, default=320, help="segment-size"
        )
        parser.add_argument("--stride-n", type=int, default=1, help="lagging number")
        parser.add_argument("--stride-n2", type=int, default=1, help="lagging number")
        parser.add_argument(
            "--unit-per-subword", type=int, default=15, help="lagging number"
        )
        parser.add_argument(
            "--extra-output-dir", type=str, default=None, help="extra output dir"
        )

        parser.add_argument("--beam-size", type=int, default=5, help="MT beam size")
        parser.add_argument(
            "--conf-threshold", type=float, default=0.5,
            help="commit top-1 beam's new tokens only if confidence >= this",
        )
        parser.add_argument(
            "--conf-type", type=str, default="mean", choices=["mean", "min", "prod"],
            help="mean: geometric-mean prob of the new tokens; min: min token prob; "
                 "prod: joint prob (product) of the new tokens",
        )
        parser.add_argument(
            "--commit-mode", type=str, default="all", choices=["all", "prefix", "stepwise"],
            help="all: commit every new token of top-1 if conf>=thr (user's scheme); "
                 "prefix: commit the longest leading run of tokens whose own prob>=thr "
                 "(with --conf-type prod: longest prefix whose running product>=thr); "
                 "stepwise: beam search that checks the top-1 beam's product after every "
                 "step and stops at the first step it drops below thr, committing the "
                 "previous step's top-1",
        )
        parser.add_argument("--max-new", type=int, default=50,
                            help="stepwise mode: cap on new tokens decoded per trigger")
        parser.add_argument("--early-mode", type=str, default="pos", choices=["pos", "token"],
                            help="pos: mass summed over beams sharing the token at the same "
                                 "position, one entry per position; token: mass summed over "
                                 "hypotheses containing the token anywhere beyond the real "
                                 "commit (max over search steps), one entry per token, hits "
                                 "matched by token only")
        parser.add_argument("--early-norm", action="store_true",
                            help="token mode: at every step normalise the kept hypotheses' "
                                 "probabilities to sum to 1 before summing them into token masses")
        parser.add_argument("--min-steps", type=int, default=0,
                            help="stepwise mode: search at least this many steps (token mode)")
        parser.add_argument("--early-mass-thr", type=float, default=None,
                            help="stepwise mode: speculative pre-synthesis. At every search "
                                 "step, the token with the largest summed beam probability at "
                                 "that position is pre-synthesised (not output) if the mass >= "
                                 "this; positions need not be contiguous. Search continues "
                                 "until the mass drops below it.")
        parser.add_argument(
            "--asr-trigger", type=str, default="words", choices=["words", "tokens"],
            help="what counts as 'ASR emitted something new'",
        )
        parser.add_argument("--asr-stride", type=int, default=1,
                            help="need >= this many new ASR words/tokens to trigger MT")
        parser.add_argument("--no-whole-word", action="store_true",
                            help="allow committing a partial (subword-cut) word")
        parser.add_argument("--max-commit-tokens", type=int, default=-1,
                            help="cap on tokens committed per trigger (-1 = no cap)")
        parser.add_argument("--trace-file", type=str, default=None,
                            help="jsonl trace of every MT trigger / decision")
        parser.add_argument("--tts-profile", action="store_true",
                            help="stepwise mode: on every commit also run T2U + unit CTC + "
                                 "vocoder duration prediction and log the shapes needed "
                                 "for FLOPs/latency accounting (needs --vocoder/--vocoder-cfg)")
        parser.add_argument("--vocoder", type=str, default=None)
        parser.add_argument("--vocoder-cfg", type=str, default=None)

    def reset(self):
        self.sent_idx = getattr(self, "sent_idx", -1) + 1
        self.asr_count = 0
        self.committed = None
        self.n_triggers = 0
        self.units = []
        self.mt_state = {}  # persistent MT-decoder KV cache for the committed prefix (tts-profile)
        self.mt_hidden = None  # cached MT hidden states (T x 1 x C) of the committed prefix
        self.spec = {}  # speculative pre-synthesis: abs position -> {tok, units, expanded}
        self.spec_tok = {}  # token-mode speculation: tok id -> {units, expanded}
        self.src_seg_num = 0
        self.tgt_subwords_indices = None
        self.src_ctc_indices = None
        self.src_ctc_prefix_length = 0
        self.tgt_ctc_prefix_length = 0
        self.tgt_units_indices = None
        self.prev_output_tokens_mt = None
        self.tgt_text = ""
        self.mt_decoder_out = None
        self.unit = None
        self.wav = []
        self.post_transcription = ""
        self.unfinished_wav = None
        self.states.reset()
        try:
            self.generator_mt.reset_incremental_states()
            self.ctc_generator.reset_incremental_states()
        except:
            pass

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        state["cfg"].common['user_dir']=args.user_dir
        utils.import_user_module(state["cfg"].common)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        args.global_cmvn = None
        if args.config_yaml is not None:
            task_args.config_yaml = args.config_yaml
            with open(os.path.join(args.data_bin, args.config_yaml), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        self.feature_extractor = OnlineFeatureExtractor(args, config)

        if args.multitask_config_yaml is not None:
            task_args.multitask_config_yaml = args.multitask_config_yaml

        task = tasks.setup_task(task_args)
        self.task = task

        overrides = ast.literal_eval(state["cfg"].common_eval.model_overrides)

        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(filename),
            arg_overrides=overrides,
            task=task,
            suffix=state["cfg"].checkpoint.checkpoint_suffix,
            strict=(state["cfg"].checkpoint.checkpoint_shard_count == 1),
            num_shards=state["cfg"].checkpoint.checkpoint_shard_count,
        )

        chunk_size = args.source_segment_size // 40

        self.models = models

        for model in self.models:
            model.eval()
            model.share_memory()
            if self.gpu:
                model.cuda()
            model.encoder.chunk_size = chunk_size
            chunk_size = min(chunk_size, 16)
            for conv in model.encoder.subsample.conv_layers:
                conv.chunk_size = chunk_size
            for layer in model.encoder.conformer_layers:
                layer.conv_module.depthwise_conv.chunk_size = chunk_size

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

        for k, v in task.multitask_tasks.items():
            self.dict[k] = v.tgt_dict

    # ------------------------------------------------------------------ helpers
    def _detok(self, dictionary, tokens):
        text = "".join([dictionary[c] for c in tokens])
        for a, b in (("_", " "), ("▁", " "), ("<unk>", " "), ("<s>", ""), ("</s>", "")):
            text = text.replace(a, b)
        return text.strip()

    def _tok_strs(self, tokens):
        return [self.generator_mt.tgt_dict[c] for c in tokens]

    def _trace(self, row):
        if self._trace_fh is None:
            return
        self._trace_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._trace_fh.flush()

    def _confidence(self, new_pos, beam_score):
        if new_pos.numel() == 0:
            return 0.0
        if self.conf_type == "min":
            return float(torch.exp(new_pos.min()))
        if self.conf_type == "prod":
            return float(torch.exp(new_pos.sum()))
        return float(torch.exp(new_pos.mean()))

    def _finish_action(self):
        self.states.target_finished = True
        self.reset()
        return WriteAction("", finished=True)

    def _tts_shapes(self, n_new):
        """Model the ideally-cached TTS path for a commit of `n_new` tokens:
        the MT hidden states of previously committed tokens are kept as computed
        at their own commit time (persistent KV cache, fed one token at a time);
        only the new tokens are run against the current encoder output.  T2U and
        the unit CTC decoder are causal, so recomputing them over the cached
        hidden states leaves old units unchanged.  Returns unit counts for the
        new tokens."""
        single_model = self.generator.model.single_model
        mt_decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")
        bos = torch.full((1, 1), self.generator_mt.eos, dtype=torch.long, device=self.device)
        prev = torch.cat((bos, self.committed.to(self.device).long()), dim=-1)
        L = prev.size(-1)
        start = 0 if self.mt_hidden is None else self.mt_hidden.size(0)
        new_h = []
        for j in range(start, L):  # positions not yet cached (incl. bos at first commit)
            h = mt_decoder(
                prev[:, : j + 1], encoder_out=self.encoder_outs[0],
                incremental_state=self.mt_state, features_only=True,
            )[0]  # 1 x 1 x C
            new_h.append(h.transpose(0, 1))
        if new_h:
            x_new = torch.cat(new_h, dim=0)
            if getattr(single_model, "proj", None) is not None:
                x_new = single_model.proj(x_new)
            self.mt_hidden = x_new if self.mt_hidden is None else torch.cat((self.mt_hidden, x_new), 0)
        per_tok, units = self._pipeline_units(self.mt_hidden)
        stable = units[: len(self.units)] == self.units
        new_units = units[len(self.units):] if stable else units
        self.units = units
        out = {
            "units_total": len(units),
            "new_units": len(new_units),
            "units_prefix_stable": bool(stable),
            "units_expanded_total": sum(e for _, e in per_tok),
            "new_units_expanded": sum(e for _, e in per_tok[-n_new:]) if n_new > 0 else 0,
            # per newly committed token: (units, duration-expanded frames)
            "per_token": per_tok[-n_new:] if n_new > 0 else [],
        }
        return out

    def _pipeline_units(self, x):
        """T2U + unit CTC (text-causal mask) + vocoder duration prediction over MT
        hidden states x (T x 1 x C, T = 1 + #tokens, position 0 = bos).
        Returns (per_token, units): per_token[i] = (n_units, expanded_frames) of
        text token i (the bos segment is merged into token 0); units = the full
        collapsed unit sequence."""
        single_model = self.generator.model.single_model
        t2u = single_model.synthesizer_encoder(x, None)
        # text-causal cross-attention (the mask the unit decoder was trained with,
        # k2=0 / n2=1): upsampled position t may only see text tokens <= t // 25.
        # Without it the released agent's cross-attention is bidirectional over the
        # text and earlier units change whenever text is appended.
        fin = self.unit_generator.generate(
            t2u, streaming_config={"src_wait": 0, "src_step": 1}
        )
        org = fin[0][0]["org_tokens"].int().tolist()
        d = self.generator.tgt_dict
        blank = getattr(d, "blank_index", 0)
        skip = {blank, d.pad(), d.eos(), d.bos()}
        up = int(getattr(single_model.decoder, "ctc_upsample_rate", 25))
        T = x.size(0)
        segs, last = [], None
        for i in range(T):
            seg = org[i * up: (i + 1) * up]
            col = [v for j, v in enumerate(seg) if (j == 0 or v != seg[j - 1]) and v not in skip]
            if col and last is not None and col[0] == last and seg[0] == org[i * up - 1]:
                col = col[1:]  # same unit continues across the segment boundary
            if col:
                last = col[-1]
            segs.append(col)
        units = [int(d[c]) for s in segs for c in s]
        durs = []
        if units:
            code = torch.tensor(units, dtype=torch.long, device=self.device).view(1, -1)
            _, dur = self.vocoder({"code": code}, True)
            durs = dur.view(-1).tolist()
        per_tok, k = [], 0
        for i, s in enumerate(segs):
            e = int(sum(durs[k: k + len(s)]))
            k += len(s)
            if i == 0:
                carry = (len(s), e)  # bos segment -> merged into token 0
            elif i == 1:
                per_tok.append((len(s) + carry[0], e + carry[1]))
            else:
                per_tok.append((len(s), e))
        if T == 1:
            per_tok = []
        return per_tok, units

    def _units_for_sequence(self, seq):
        """Non-incremental pipeline on an arbitrary token sequence (1-D tensor) with
        the current encoder output; used to cost speculative (pre-synthesised)
        tokens.  Returns per_token list as in _pipeline_units."""
        single_model = self.generator.model.single_model
        mt_decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")
        bos = torch.full((1, 1), self.generator_mt.eos, dtype=torch.long, device=self.device)
        prev = torch.cat((bos, seq.view(1, -1).to(self.device).long()), dim=-1)
        x = mt_decoder(prev, encoder_out=self.encoder_outs[0], features_only=True)[0]
        x = x.transpose(0, 1)
        if getattr(single_model, "proj", None) is not None:
            x = single_model.proj(x)
        per_tok, _ = self._pipeline_units(x)
        return per_tok

    def _speculate_tokens(self, row, snaps, min_rel, P):
        """Position-free speculation.  mass(t) = max over search steps of the summed
        product of the hypotheses whose not-yet-committed part (relative positions
        >= min_rel) contains t (each hypothesis counts once per token).  Tokens with
        mass >= thr are pre-synthesised once (keyed by token, no position) unless
        already in the table."""
        d = self.generator_mt.tgt_dict
        best = {}  # tok -> (mass, ctx tokens up to the occurrence, abs pos)
        for snap in snaps:
            mass, ctx = {}, {}
            if self.early_norm:
                z = sum(p for _, p in snap) or 1.0
                snap = [(toks, p / z) for toks, p in snap]
            for toks, prod in snap:
                seen = set()
                for j, t in enumerate(toks):
                    if j < min_rel or t in seen:
                        continue
                    seen.add(t)
                    mass[t] = mass.get(t, 0.0) + prod
                    if t not in ctx or prod > ctx[t][0]:
                        ctx[t] = (prod, toks[: j + 1], P + j)
            for t, m in mass.items():
                if t not in best or m > best[t][0]:
                    best[t] = (m, ctx[t][1], ctx[t][2])
        log, new_cost = [], []
        prefix = self.committed[0][:P] if self.committed is not None else None
        for t, (m, ctx_toks, pos) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            if m < self.early_thr:
                continue
            entry = {"pos": pos, "tok": d[t], "mass": round(m, 4)}
            if t in self.spec_tok:
                entry["new"] = False
            else:
                units = expanded = None
                if self.tts_profile:
                    ctx_t = torch.tensor(ctx_toks, dtype=torch.long, device=self.device)
                    seq = ctx_t if prefix is None else torch.cat((prefix.to(self.device), ctx_t))
                    pt = self._units_for_sequence(seq)
                    units, expanded = pt[-1] if pt else (0, 0)
                self.spec_tok[t] = {"units": units, "expanded": expanded}
                entry.update({"new": True, "units": units, "expanded": expanded})
                new_cost.append((pos, units, expanded))
            log.append(entry)
        row["early"] = log
        row["spec_new"] = new_cost
        row["spec_overwritten"] = 0

    def _speculate(self, row, early, min_pos, P, snaps=None):
        """Speculative pre-synthesis bookkeeping.  `early` = (abs_pos, tok, mass, ctx)
        candidates found by this trigger's search from prefix length P (ctx = the
        best beam's new tokens up to and including the candidate).  Positions below
        `min_pos` were really committed at this trigger, so they cannot be reused
        later and are skipped.  A candidate is pre-synthesised (T2U + unit CTC +
        vocoder, costed but not output) unless the same token is already stored at
        that position; a different token overwrites (the old one is wasted)."""
        if self.early_thr is None:
            return
        if self.early_mode == "token":
            return self._speculate_tokens(row, snaps or [], min_pos - P, P)
        log, new_cost, wasted = [], [], 0
        prefix = self.committed[0][:P] if self.committed is not None else None
        for pos, tok, mass, ctx in early:
            if pos < min_pos:
                continue
            old = self.spec.get(pos)
            entry = {"pos": pos, "tok": self.generator_mt.tgt_dict[tok], "mass": round(mass, 4)}
            if old is not None and old["tok"] == tok:
                entry["new"] = False
            else:
                if old is not None:
                    wasted += 1
                units = expanded = None
                if self.tts_profile:
                    seq = ctx if prefix is None else torch.cat((prefix.to(ctx.device), ctx))
                    pt = self._units_for_sequence(seq)
                    units, expanded = pt[-1] if pt else (0, 0)
                self.spec[pos] = {"tok": tok, "units": units, "expanded": expanded}
                entry.update({"new": True, "units": units, "expanded": expanded})
                new_cost.append((pos, units, expanded))
            log.append(entry)
        row["early"] = log
        row["spec_new"] = new_cost
        row["spec_overwritten"] = wasted

    def _consume_spec(self, row, commit_tokens, prefix_len):
        """At a real commit of `commit_tokens` at absolute positions prefix_len..,
        mark which of them were already pre-synthesised (same token at the same
        position) and which still need TTS; drop stale speculative entries."""
        if self.early_thr is None:
            return
        n = int(commit_tokens.numel())
        hits, tts_pos, wasted = [], [], 0
        if self.early_mode == "token":
            for i in range(n):
                if self.spec_tok.pop(int(commit_tokens[i]), None) is not None:
                    hits.append(i)
                else:
                    tts_pos.append(i)
        else:
            for i in range(n):
                e = self.spec.pop(prefix_len + i, None)
                if e is not None and e["tok"] == int(commit_tokens[i]):
                    hits.append(i)
                else:
                    tts_pos.append(i)
                    if e is not None:
                        wasted += 1
            for pos in [p for p in self.spec if p < prefix_len + n]:
                self.spec.pop(pos)
                wasted += 1
        sh = row["shapes"]
        sh["spec_hits"] = len(hits)
        sh["tts_positions"] = tts_pos
        sh["spec_wasted"] = wasted
        if "per_token" in sh:
            pt = sh["per_token"]
            sh["tts_units"] = sum(pt[i][0] for i in tts_pos)
            sh["tts_units_expanded"] = sum(pt[i][1] for i in tts_pos)

    def _stepwise_beam(self, encoder_outs, thr):
        """Beam search from the committed prefix that stops as soon as the top-1
        beam's joint probability of *new* tokens drops below `thr`.

        Returns (best, best_prod, steps, stop, last_top1):
          best      - new tokens of the last step whose top-1 had prod >= thr
          best_prod - that product
          steps     - per-step trace of the top-1 beam (text, prod, eos)
          stop      - "below_thr" | "eos" | "max_len"
          last_top1 - new tokens of the top-1 at the step we stopped on
        """
        single_model = self.generator.model.single_model
        decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")
        eos, pad = self.generator_mt.eos, self.generator_mt.pad
        prefix = self.committed
        P = 0 if prefix is None else int(prefix.size(-1))
        bos = torch.full((1, 1), eos, dtype=torch.long, device=self.device)
        tokens = bos if prefix is None else torch.cat((bos, prefix.to(self.device).long()), -1)
        cum = torch.zeros(1, device=self.device)  # cumulative log-prob of new tokens
        best = tokens.new_zeros((0,))
        best_prod = 1.0
        steps = []
        stop = "max_len"
        new = tokens.new_zeros((0,))
        stepwise_done = False  # top-1 product has dropped below thr (or hit EOS)
        stop_top1 = None  # top-1 new tokens at the step the stepwise decision was made
        steps_real = 0  # search steps needed for the stepwise (real) decision
        early = []  # speculative candidates: (abs_pos, tok, mass, context tokens)
        snaps = []  # token mode: per step, list of (new tokens w/o EOS, prod) of all hypotheses
        finalized = []  # token mode: hypotheses that ended with EOS (kept in later snapshots)
        for k in range(self.max_new):
            B = tokens.size(0)
            enc = self.generator.model.reorder_encoder_out(
                encoder_outs, torch.zeros(B, dtype=torch.long, device=self.device)
            )
            out = decoder(tokens, encoder_out=enc[0])
            lprobs = decoder.get_normalized_probs(
                (out[0][:, -1:, :], None), log_probs=True, sample=None
            )[:, -1, :]
            lprobs[:, pad] = -math.inf
            V = lprobs.size(-1)
            cand = (lprobs + cum.unsqueeze(1)).view(-1)
            top_scores, top_idx = cand.topk(min(self.beam_size, cand.numel()))
            beam_idx = torch.div(top_idx, V, rounding_mode="floor")
            tok_idx = top_idx % V
            tokens = torch.cat((tokens[beam_idx], tok_idx.unsqueeze(1)), dim=-1)
            cum = top_scores
            prod = float(torch.exp(cum[0]))
            new = tokens[0, 1 + P:]
            is_eos = bool(tok_idx[0] == eos)
            step_info = {
                "text": self._detok(self.generator_mt.tgt_dict, new[:-1] if is_eos else new),
                "prod": round(prod, 4),
                "eos": is_eos,
            }
            if not stepwise_done:
                steps_real = k + 1
                if prod < thr:
                    stop = "below_thr"
                    stepwise_done = True
                    stop_top1 = new.clone()  # top-1 at the stepwise decision step
                else:
                    best = new[:-1] if is_eos else new
                    best_prod = prod
                    if is_eos:
                        stop = "eos"
                        stepwise_done = True
                        stop_top1 = new.clone()
            # ---- speculative pre-synthesis: summed beam mass per token at this position
            mass_ok = False
            if self.early_thr is not None:
                live = torch.isfinite(cum)
                best_tok, best_mass, best_beam = None, -1.0, None
                for t in torch.unique(tok_idx[live]).tolist():
                    sel = (tok_idx == t) & live
                    mass = float(torch.exp(torch.logsumexp(cum[sel], 0)))
                    if mass > best_mass:
                        best_tok, best_mass = t, mass
                        best_beam = int(torch.nonzero(sel)[0].item())  # highest-scoring beam with t
                step_info["mass_tok"] = self.generator_mt.tgt_dict[best_tok] if best_tok is not None else None
                step_info["mass"] = round(best_mass, 4)
                if best_tok is not None and best_tok != eos and best_mass >= self.early_thr:
                    mass_ok = True
                    early.append((P + k, best_tok, best_mass, tokens[best_beam, 1 + P:].clone()))
            steps.append(step_info)
            if self.early_thr is not None and self.early_mode == "token":
                live_snap = []
                for b in range(tokens.size(0)):
                    if not torch.isfinite(cum[b]):
                        continue
                    toks_b = tokens[b, 1 + P:].tolist()
                    pb = float(torch.exp(cum[b]))
                    if toks_b and toks_b[-1] == eos:
                        finalized.append((toks_b[:-1], pb))
                    else:
                        live_snap.append((toks_b, pb))
                snaps.append(live_snap + list(finalized))
                if stepwise_done and k + 1 >= self.min_steps:
                    break
            elif stepwise_done and not mass_ok:
                break
            # beams that just emitted EOS cannot be extended
            cum = torch.where(tok_idx == eos, torch.full_like(cum, -math.inf), cum)
            if not torch.isfinite(cum).any():
                break
        return (best, best_prod, steps, stop, (stop_top1 if stop_top1 is not None else new),
                steps_real, early, snaps)

    # ------------------------------------------------------------------ policy
    @torch.inference_mode()
    def policy(self):
        feature = self.feature_extractor(self.states.source)
        if feature.size(0) == 0 and not self.states.source_finished:
            return ReadAction()

        src_ms = len(self.states.source) / self.args.sample_rate * 1000.0
        src_indices = feature.unsqueeze(0)
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long()
        self.encoder_outs = self.generator.model.forward_encoder(
            {"src_tokens": src_indices, "src_lengths": src_lengths}
        )

        # ---- streaming ASR (source CTC) : the trigger --------------------------
        finalized_asr = self.asr_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="source_unigram"
        )
        asr_tokens = finalized_asr[0][0]["tokens"].int()
        asr_text = self._detok(self.dict["source_unigram"], asr_tokens)
        if self.states.source_finished and not self.quiet:
            with open(self.asr_file, "a", encoding="utf-8") as file:
                print(asr_text, file=file)

        asr_count = (
            len(asr_text.split()) if self.asr_trigger == "words" else int(asr_tokens.size(-1))
        )
        if not self.states.source_finished:
            if asr_count < self.asr_count + self.asr_stride:
                return ReadAction()
            self.asr_count = asr_count

        # ---- stepwise mode: thresholded beam search, stop at first drop ----------
        if not self.states.source_finished and self.commit_mode == "stepwise":
            best, best_prod, steps, stop, last_top1, steps_real, early, snaps = self._stepwise_beam(
                self.encoder_outs, self.conf_threshold
            )
            self.n_triggers += 1
            prefix_text = (
                self._detok(self.generator_mt.tgt_dict, self.committed[0])
                if self.committed is not None else ""
            )
            row = {
                "sent": self.sent_idx,
                "trigger": self.n_triggers,
                "src_ms": round(src_ms, 1),
                "source_finished": False,
                "asr": asr_text,
                "prefix": prefix_text,
                "top1": self._detok(self.generator_mt.tgt_dict, best),
                "conf": round(best_prod, 4),
                "stop": stop,
                "steps": steps,
            }
            commit_n = int(best.numel())
            if self.max_commit_tokens > 0:
                commit_n = min(commit_n, self.max_commit_tokens)
            if self.whole_word and commit_n > 0 and stop != "eos":
                # if the rejected next token continues the last committed word
                # (no ▁ boundary), that word is incomplete -> back off to its start
                strs = self._tok_strs(best[:commit_n])
                cont = (
                    last_top1.numel() > commit_n
                    and torch.equal(last_top1[:commit_n], best[:commit_n])
                    and not self.generator_mt.tgt_dict[last_top1[commit_n]].startswith("▁")
                )
                if cont or stop == "max_len":
                    i = commit_n - 1
                    while i >= 0 and not strs[i].startswith("▁"):
                        i -= 1
                    commit_n = max(i, 0)
            row["commit_n"] = commit_n
            prefix_len = 0 if self.committed is None else int(self.committed.size(-1))
            row["shapes"] = {
                "fbank_frames": int(feature.size(0)),
                "enc_frames": int(self.encoder_outs[0]["encoder_out"][0].size(0)),
                "beam_steps": steps_real,
                "beam_steps_total": len(steps),
                # step 1 expands from the single prefix row, later steps from beam_size rows
                "dec_token_fwds": 1 + self.beam_size * max(steps_real - 1, 0),
                "dec_token_fwds_extra": self.beam_size * (len(steps) - steps_real),
                "prefix_len": prefix_len,
                "new_tokens": commit_n,
            }
            if commit_n <= 0:
                row["action"] = "READ"
                self._speculate(row, early, prefix_len, prefix_len, snaps)
                self._trace(row)
                return ReadAction()
            commit = best[:commit_n].unsqueeze(0)
            self.committed = commit if self.committed is None else torch.cat(
                (self.committed, commit), dim=-1
            )
            text = " ".join(self._tok_strs(self.committed[0]))
            new_text = text[len(self.tgt_text):]
            self.tgt_text = text
            row["action"] = "WRITE"
            row["committed_text"] = self._detok(self.generator_mt.tgt_dict, commit[0])
            if self.tts_profile:
                row["shapes"].update(self._tts_shapes(commit_n))
            self._consume_spec(row, commit[0], prefix_len)
            self._speculate(row, early, prefix_len + commit_n, prefix_len, snaps)
            self._trace(row)
            return WriteAction(new_text, finished=False)

        # ---- MT beam search from the committed prefix ---------------------------
        single_model = self.generator.model.single_model
        finalized = self.generator_beam.generate_decoder(
            self.encoder_outs,
            src_indices,
            src_lengths,
            {"id": 1, "net_input": {"src_tokens": src_indices, "src_lengths": src_lengths}},
            self.committed,
            None,
            None,
            aux_task_name=single_model.mt_task_name,
            max_new_tokens=-1,
        )
        hyps = finalized[0]
        top = hyps[0]
        toks = top["tokens"]
        pos = top["positional_scores"]
        if toks[-1] == self.generator_mt.eos:
            toks, pos = toks[:-1], pos[:-1]
        start = 0 if self.committed is None else int(self.committed.size(-1))
        new_toks, new_pos = toks[start:], pos[start:]
        conf = self._confidence(new_pos, top["score"])
        self.n_triggers += 1

        row = {
            "sent": self.sent_idx,
            "trigger": self.n_triggers,
            "src_ms": round(src_ms, 1),
            "source_finished": bool(self.states.source_finished),
            "asr": asr_text,
            "prefix": self._detok(self.generator_mt.tgt_dict, self.committed[0])
            if self.committed is not None else "",
            "top1": self._detok(self.generator_mt.tgt_dict, toks),
            "conf": round(conf, 4),
            "new_tok_probs": [round(float(p), 3) for p in torch.exp(new_pos)],
            "beams": [
                {
                    "text": self._detok(
                        self.generator_mt.tgt_dict,
                        h["tokens"][:-1] if h["tokens"][-1] == self.generator_mt.eos else h["tokens"],
                    ),
                    # same confidence definition as top-1 (over this beam's new tokens)
                    "conf": round(
                        self._confidence(
                            (h["positional_scores"][:-1]
                             if h["tokens"][-1] == self.generator_mt.eos
                             else h["positional_scores"])[start:],
                            h["score"],
                        ),
                        4,
                    ),
                }
                for h in hyps[1:]
            ],
        }

        # ---- decide what to commit ---------------------------------------------
        if self.states.source_finished:
            commit_n = int(new_toks.numel())
        elif self.commit_mode == "all":
            commit_n = int(new_toks.numel()) if conf >= self.conf_threshold else 0
        else:  # prefix: leading run of tokens each with prob >= thr
            if self.conf_type == "prod":  # longest prefix whose running product >= thr
                ok = torch.exp(torch.cumsum(new_pos, 0)) >= self.conf_threshold
            else:
                ok = torch.exp(new_pos) >= self.conf_threshold
            commit_n = int(ok.numel()) if bool(ok.all()) else int((~ok).nonzero()[0].item())

        if not self.states.source_finished:
            if self.max_commit_tokens > 0:
                commit_n = min(commit_n, self.max_commit_tokens)
            if self.whole_word and commit_n > 0:
                end = start + commit_n
                strs = self._tok_strs(toks)
                # if the hypothesis continues after `end` without a word boundary, the
                # word we'd commit is cut in half -> back off to its first subword
                if end < len(strs) and not strs[end].startswith("▁"):
                    j = end
                    while j > start and not strs[j].startswith("▁"):
                        j -= 1
                    commit_n = j - start

        row["commit_n"] = commit_n
        if commit_n <= 0:
            row["action"] = "READ"
            if self.states.source_finished:
                row["spec_leftover"] = len(self.spec) + len(self.spec_tok)
                self.spec, self.spec_tok = {}, {}
            self._trace(row)
            if self.states.source_finished:
                if not self.quiet:
                    with open(self.st_file, "a", encoding="utf-8") as file:
                        print(self._detok(self.generator_mt.tgt_dict,
                                          self.committed[0] if self.committed is not None else []),
                              file=file)
                return self._finish_action()
            return ReadAction()

        commit = toks[start : start + commit_n].unsqueeze(0)
        self.committed = commit if self.committed is None else torch.cat(
            (self.committed, commit), dim=-1
        )
        tokens = self._tok_strs(self.committed[0])
        text = " ".join(tokens)
        new_text = text[len(self.tgt_text):]
        self.tgt_text = text
        row["action"] = "WRITE"
        row["committed_text"] = self._detok(self.generator_mt.tgt_dict, commit[0])
        if self.states.source_finished:
            # final flush: unthresholded beam search over the remainder (n tokens + EOS)
            row["shapes"] = {
                "fbank_frames": int(feature.size(0)),
                "enc_frames": int(self.encoder_outs[0]["encoder_out"][0].size(0)),
                "beam_steps": commit_n + 1,
                "dec_token_fwds": self.beam_size * (commit_n + 1),
                "prefix_len": start,
                "new_tokens": commit_n,
                "final_flush": True,
            }
            if self.tts_profile:
                row["shapes"].update(self._tts_shapes(commit_n))
            self._consume_spec(row, commit[0], start)
            row["spec_leftover"] = len(self.spec) + len(self.spec_tok)  # never used -> wasted
            self.spec, self.spec_tok = {}, {}
        self._trace(row)

        if self.states.source_finished:
            if not self.quiet:
                with open(self.st_file, "a", encoding="utf-8") as file:
                    print(self._detok(self.generator_mt.tgt_dict, self.committed[0]), file=file)
            self.states.target_finished = True
            self.reset()

        return WriteAction(new_text, finished=self.states.target_finished)
