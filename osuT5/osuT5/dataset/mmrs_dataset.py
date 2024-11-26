from __future__ import annotations

import os
import random
from multiprocessing.managers import Namespace
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig
from pandas import Series, DataFrame
from slider import Beatmap, HoldNote
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file, remove_events_of_type
from .osu_parser import OsuParser
from ..tokenizer import Event, EventType, Tokenizer, ContextType

OSZ_FILE_EXTENSION = ".osz"
AUDIO_FILE_NAME = "audio.mp3"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100
context_types_with_kiai = [ContextType.NO_HS, ContextType.GD, ContextType.MAP]


class MmrsDataset(IterableDataset):
    __slots__ = (
        "path",
        "start",
        "end",
        "args",
        "parser",
        "tokenizer",
        "beatmap_files",
        "test",
        "shared",
        "sample_weights",
    )

    def __init__(
            self,
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            subset_ids: Optional[list[int]] = None,
            test: bool = False,
            shared: Namespace = None,
    ):
        """Manage and process MMRS dataset.

        Attributes:
            args: Data loading arguments.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
            subset_ids: List of beatmap set IDs to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self._validate_args(args)
        self.path = Path(args.test_dataset_path if test else args.train_dataset_path)
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.metadata = self._load_metadata()
        if subset_ids is not None:
            self.subset_ids = subset_ids
            self.start = 0
            self.end = len(subset_ids)
        else:
            self.subset_ids = self._beatmap_set_ids_from_metadata()
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.sample_weights = self._get_sample_weights(args.sample_weights_path)

    def _validate_args(self, args: DictConfig):
        if not args.per_track:
            raise ValueError("MMRS dataset requires per_track to be True")
        if args.only_last_beatmap:
            raise ValueError("MMRS dataset does not support only_last_beatmap")

    def _load_metadata(self):
        # Loads the metadata parquet from the dataset path
        df = pd.read_parquet(self.path / "metadata.parquet")
        df["BeatmapIdx"] = df.index
        df.set_index(["BeatmapSetId", "Id"], inplace=True)
        df.sort_index(inplace=True)
        df = df[df["ModeInt"].isin(self.args.gamemodes)]
        return df

    def _beatmap_set_ids_from_metadata(self):
        return self.metadata.index.to_frame()["BeatmapSetId"].unique().tolist()

    @staticmethod
    def _get_sample_weights(sample_weights_path):
        if not os.path.exists(sample_weights_path):
            return None

        # Load the sample weights csv to a dictionary
        with open(sample_weights_path, "r") as f:
            sample_weights = {int(line.split(",")[0]): np.clip(float(line.split(",")[1]), 0.1, 10) for line in
                              f.readlines()}
            # Normalize the weights so the mean is 1
            mean = sum(sample_weights.values()) / len(sample_weights)
            sample_weights = {k: v / mean for k, v in sample_weights.items()}

        return sample_weights

    def __iter__(self):
        subset_ids = self.subset_ids[self.start:self.end].copy()

        if not self.test:
            random.shuffle(subset_ids)

        if self.args.cycle_length > 1 and not self.test:
            return InterleavingBeatmapDatasetIterable(
                subset_ids,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(subset_ids).__iter__()

    def _iterable_factory(self, subset_ids: list[int]):
        return BeatmapDatasetIterable(
            subset_ids,
            self.args,
            self.path,
            self.metadata,
            self.parser,
            self.tokenizer,
            self.test,
            self.shared,
            self.sample_weights,
        )


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            subset_ids: list[int],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(subset_ids) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                subset_ids[i * per_worker: min(len(subset_ids), (i + 1) * per_worker)]
            ).__iter__()
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDatasetIterable:
    __slots__ = (
        "subset_ids",
        "args",
        "path",
        "metadata",
        "parser",
        "tokenizer",
        "test",
        "shared",
        "frame_seq_len",
        "min_pre_token_len",
        "pre_token_len",
        "class_dropout_prob",
        "diff_dropout_prob",
        "add_pre_tokens",
        "add_empty_sequences",
        "sample_weights",
        "gen_start_frame",
        "gen_end_frame",
    )

    def __init__(
            self,
            subset_ids: list[int],
            args: DictConfig,
            path: Path,
            metadata: pd.DataFrame,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
            shared: Namespace,
            sample_weights: dict[int, float] = None,
    ):
        self.subset_ids = subset_ids
        self.args = args
        self.path = path
        self.metadata = metadata
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.sample_weights = sample_weights
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = args.src_seq_len - 1
        self.gen_start_frame = int(round(args.lookback * self.frame_seq_len))
        self.gen_end_frame = int(round((1 - args.lookahead) * self.frame_seq_len))
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.class_dropout_prob = 1 if self.test else args.class_dropout_prob
        self.diff_dropout_prob = 0 if self.test else args.diff_dropout_prob
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Segment audio samples into frames.

        Each frame has `frame_size` audio samples.
        It will also calculate and return the time of each audio frame, in miliseconds.

        Args:
            samples: Audio time-series.

        Returns:
            frames: Audio frames.
            frame_times: Audio frame times.
        """
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_milisecond = (
                self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _create_sequences(
            self,
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            out_context: dict,
            in_context: list[dict],
            extra_data: Optional[dict] = None,
    ) -> list[dict[str, int | npt.NDArray | list[Event]]]:
        """Create frame and token sequences for training/testing.

        Args:
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """

        def get_event_indices(events2: list[Event], event_times2: list[int]) -> tuple[list[int], list[int]]:
            if len(events2) == 0:
                return [], []

            # Corresponding start event index for every audio frame.
            start_indices = []
            event_index = 0

            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)

            # Corresponding end event index for every audio frame.
            end_indices = start_indices[1:] + [len(events2)]

            return start_indices, end_indices

        start_indices, end_indices = {}, {}
        for context in in_context + [out_context]:
            start_indices[context["extra"]["context_type"]], end_indices[
                context["extra"]["context_type"]] = get_event_indices(context["events"], context["event_times"])

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, self.frame_seq_len)
        last_kiai = {}
        # Divide audio frames into splits
        for frame_start_idx in range(offset, n_frames - self.gen_start_frame, self.frame_seq_len):
            frame_end_idx = min(frame_start_idx + self.frame_seq_len, n_frames)

            gen_start_frame = min(frame_start_idx + self.gen_start_frame, n_frames - 1)
            gen_end_frame = min(frame_start_idx + self.gen_end_frame, n_frames)

            event_start_idx = start_indices[out_context["extra"]["context_type"]][frame_start_idx]
            gen_start_idx = start_indices[out_context["extra"]["context_type"]][gen_start_frame]

            frame_pre_idx = max(frame_start_idx - self.frame_seq_len, 0)

            def slice_events(context, frame_start_idx, frame_end_idx):
                if len(context["events"]) == 0:
                    return []
                context_type = context["extra"]["context_type"]
                event_start_idx = start_indices[context_type][frame_start_idx]
                event_end_idx = end_indices[context_type][frame_end_idx - 1]
                return context["events"][event_start_idx:event_end_idx]

            def slice_context(context, frame_start_idx, frame_end_idx):
                return {"events": slice_events(context, frame_start_idx, frame_end_idx)} | context["extra"]

            # Create the sequence
            sequence = {
                           "time": frame_times[frame_start_idx],
                           "frames": frames[frame_start_idx:frame_end_idx],
                           "labels_offset": gen_start_idx - event_start_idx,
                           "out_context": slice_context(out_context, frame_start_idx, gen_end_frame),
                           "in_context": [slice_context(context, frame_start_idx, frame_end_idx) for context in
                                          in_context],
                       } | extra_data

            if self.args.add_pre_tokens or self.args.add_pre_tokens_at_step >= 0:
                sequence["pre_events"] = slice_events(out_context, frame_pre_idx, frame_start_idx)

            def add_last_kiai(sequence_context, context, last_kiai):
                if sequence_context["context_type"] not in [ContextType.NO_HS, ContextType.GD, ContextType.MAP]:
                    return
                if context in last_kiai:
                    sequence_context["last_kiai"] = last_kiai[context]
                else:
                    sequence_context["last_kiai"] = Event(EventType.KIAI, 0)
                # Find the last kiai event in the out context
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.KIAI:
                        last_kiai[context] = event
                        break

            if self.args.add_kiai:
                add_last_kiai(sequence["out_context"], out_context, last_kiai)
                for i, sequence_context in enumerate(sequence["in_context"]):
                    add_last_kiai(sequence_context, in_context[i], last_kiai)

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """

        min_t = self.tokenizer.event_range[EventType.TIME_SHIFT].min_value

        def process(events: list[Event], start_time) -> list[Event] | tuple[list[Event], int]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    t = int((event.value - start_time) * STEPS_PER_MILLISECOND)
                    assert t >= min_t  # TODO: Fix weird unordered events
                    events[i] = Event(EventType.TIME_SHIFT, t)

            return events

        start_time = sequence["time"]
        del sequence["time"]

        sequence["out_context"]["events"] = process(sequence["out_context"]["events"], start_time)

        if "pre_events" in sequence:
            sequence["pre_events"] = process(sequence["pre_events"], start_time)

        for context in sequence["in_context"]:
            context["events"] = process(context["events"], start_time)

        return sequence

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        for context in sequence["in_context"] + [sequence["out_context"]]:
            tokens = torch.empty(len(context["events"]), dtype=torch.long)
            for i, event in enumerate(context["events"]):
                tokens[i] = self.tokenizer.encode(event)
            context["tokens"] = tokens
            special_tokens = []

            if "beatmap_id" in context:
                if self.args.add_gamemode_token:
                    special_tokens.append(self.tokenizer.encode_gamemode(context["extra"]["gamemode"]))

                if self.args.add_style_token:
                    special_tokens.append(self.tokenizer.encode_style_idx(context["beatmap_idx"])
                                          if random.random() >= self.args.class_dropout_prob else self.tokenizer.style_unk)

                if self.args.add_diff_token:
                    special_tokens.append(self.tokenizer.encode_diff(context["difficulty"])
                                          if random.random() >= self.args.diff_dropout_prob else self.tokenizer.diff_unk)

                if self.args.add_mapper_token:
                    special_tokens.append(self.tokenizer.encode_mapper(context["beatmap_id"])
                                          if random.random() >= self.args.mapper_dropout_prob else self.tokenizer.mapper_unk)

                if self.args.add_year_token:
                    special_tokens.append(self.tokenizer.encode_year(context["year"])
                                          if random.random() >= self.args.year_dropout_prob else self.tokenizer.year_unk)

                if self.args.add_cs_token and "circle_size" in context:
                    special_tokens.append(self.tokenizer.encode_cs(context["circle_size"])
                                          if random.random() >= self.args.cs_dropout_prob else self.tokenizer.cs_unk)

                if "keycount" in context:
                    special_tokens.append(self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, context["keycount"])))

                if "hold_note_ratio" in context:
                    special_tokens.append(self.tokenizer.encode_hold_note_ratio(context["hold_note_ratio"])
                                          if random.random() >= self.args.hold_note_ratio_dropout_prob else self.tokenizer.hold_note_ratio_unk)

                if "scroll_speed_ratio" in context:
                    special_tokens.append(self.tokenizer.encode_scroll_speed_ratio(context["scroll_speed_ratio"])
                                          if random.random() >= self.args.scroll_speed_ratio_dropout_prob else self.tokenizer.scroll_speed_ratio_unk)

                if self.args.add_descriptors:
                    special_tokens.extend(self.tokenizer.encode_descriptor(context["beatmap_id"])
                                          if random.random() >= self.args.descriptor_dropout_prob else [self.tokenizer.descriptor_unk])

            if "last_kiai" in context:
                special_tokens.append(self.tokenizer.encode(context["last_kiai"]))

            context["special_tokens"] = special_tokens

        if "pre_events" in sequence:
            pre_tokens = torch.empty(len(sequence["pre_events"]), dtype=torch.long)
            for i, event in enumerate(sequence["pre_events"]):
                pre_tokens[i] = self.tokenizer.encode(event)
            sequence["pre_tokens"] = pre_tokens
            del sequence["pre_events"]

        sequence["beatmap_idx"] = sequence["beatmap_idx"] \
            if random.random() >= self.args.class_dropout_prob else self.tokenizer.num_classes
        # We keep beatmap_idx because it is a model input

        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        """Pad token sequence to a fixed length and split decoder input and labels.

        Pad with `[PAD]` tokens until `tgt_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Prefix the token sequence with the pre_tokens sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        # Count irreducable tokens for SOS/EOS tokens
        stl = 1

        # Count irreducable tokens for all contexts
        for context in sequence["in_context"] + [sequence["out_context"]]:
            if context["add_type"]:
                stl += 2

            stl += len(context["special_tokens"])

        # Count reducible tokens, pre_tokens and context tokens
        num_tokens = len(sequence["out_context"]["tokens"])
        num_pre_tokens = len(sequence["pre_tokens"]) if "pre_tokens" in sequence else 0

        if self.args.max_pre_token_len > 0:
            num_pre_tokens = min(num_pre_tokens, self.args.max_pre_token_len)

        num_other_tokens = sum(len(context["tokens"]) for context in sequence["in_context"])

        # Trim tokens to target sequence length
        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len - 1, num_tokens)
            m = min(self.pre_token_len - stl + 1, num_pre_tokens)
            o = min(self.pre_token_len - m - stl + 1, num_other_tokens)
            si = self.pre_token_len - m - stl + 1 - o
        else:
            # n + m + stl + o + padding = tgt_seq_len
            n = min(self.args.tgt_seq_len - stl - min(self.min_pre_token_len, num_pre_tokens), num_tokens)
            m = min(self.args.tgt_seq_len - stl - n, num_pre_tokens)
            o = min(self.args.tgt_seq_len - stl - n - m, num_other_tokens)
            si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=torch.long)

        def add_special_tokens(context, si):
            for token in context["special_tokens"]:
                input_tokens[si] = token
                si += 1
            return si

        for context in sequence["in_context"]:
            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_sos[context["context_type"]]
                si += 1

            si = add_special_tokens(context, si)

            num_other_tokens_to_add = min(len(context["tokens"]), o)
            input_tokens[si:si + num_other_tokens_to_add] = context["tokens"][:num_other_tokens_to_add]
            si += num_other_tokens_to_add
            o -= num_other_tokens_to_add

            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_eos[context["context_type"]]
                si += 1

        si = add_special_tokens(sequence["out_context"], si)

        if m > 0:
            input_tokens[si:si + m] = sequence["pre_tokens"][-m:]

        tokens = sequence["out_context"]["tokens"]
        labels_offset = sequence["labels_offset"]

        input_tokens[si + m] = self.tokenizer.sos_id
        input_tokens[si + m + 1:si + m + n + 1] = tokens[:n]
        label_tokens[si + m + labels_offset:si + m + n] = tokens[labels_offset:n]
        label_tokens[si + m + n] = self.tokenizer.eos_id

        # Randomize some input tokens
        def randomize_tokens(tokens):
            offset = torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1,
                                   size=tokens.shape)
            return torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (
                    tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                               torch.clamp(tokens + offset,
                                           self.tokenizer.event_start[EventType.TIME_SHIFT],
                                           self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                               tokens)

        if self.args.timing_random_offset > 0:
            input_tokens[si:si + m + n] = randomize_tokens(input_tokens[si:si + m + n])
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.DISTANCE] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.DISTANCE]),
        #                               torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.DISTANCE], self.tokenizer.event_end[EventType.DISTANCE] - 1),
        #                               input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["out_context"]
        del sequence["in_context"]
        del sequence["labels_offset"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

        return sequence

    def _pad_frame_sequence(self, sequence: dict) -> dict:
        """Pad frame sequence with zeros until `frame_seq_len`.

        Frame sequence can be further processed into Mel spectrogram frames,
        which is the input to the transformer encoder.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded frames.
        """
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)

        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(
                self.frame_seq_len,
                frames.shape[-1],
                dtype=frames.dtype,
                device=frames.device,
            )
            padded_frames[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded_frames)
        else:
            sequence["frames"] = torch.flatten(frames)

        return sequence

    def maybe_change_dataset(self):
        if self.shared is None:
            return
        step = self.shared.current_train_step
        if 0 <= self.args.add_empty_sequences_at_step <= step and not self.add_empty_sequences:
            self.add_empty_sequences = True
        if 0 <= self.args.add_pre_tokens_at_step <= step and not self.add_pre_tokens:
            self.add_pre_tokens = True

    def __iter__(self):
        return self._get_next_tracks()

    def _get_difficulty(self, beatmap_metadata: Series, speed: float = 1.0) -> float:
        # StarRating is an array that gives the difficulty for the speeds:
        # 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        # Linearly interpolate between the two closest speeds
        star_ratings = beatmap_metadata["StarRating"]
        speed_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        return np.interp(speed, speed_ratios, star_ratings)  # type: ignore

    def _get_speed_augment(self):
        mi, ma = self.args.dt_augment_range
        return random.random() * (ma - mi) + mi if random.random() < self.args.dt_augment_prob else 1.0

    def _get_hold_note_ratio(self, beatmap: Beatmap) -> float:
        notes = beatmap.hit_objects(circles=True, hold_notes=True, stacking=False)
        hold_note_count = 0
        for note in notes:
            if isinstance(note, HoldNote):
                hold_note_count += 1
        return hold_note_count / len(notes)

    def _get_scroll_speed_ratio(self, beatmap: Beatmap) -> float:
        # Number of scroll speed changes divided by number of distinct hit object times
        notes = beatmap.hit_objects(circles=True, sliders=True, spinners=True, hold_notes=True, stacking=False)
        last_time = -1
        num_note_times = 0
        for note in notes:
            if note.time != last_time:
                num_note_times += 1
                last_time = note.time
        last_scroll_speed = -1
        num_scroll_speed_changes = 0
        for timing_point in beatmap.timing_points:
            if timing_point.parent is None:
                last_scroll_speed = 1
            else:
                scroll_speed = -100 / timing_point.ms_per_beat
                if scroll_speed != last_scroll_speed:
                    num_scroll_speed_changes += 1
                    last_scroll_speed = scroll_speed
        return num_scroll_speed_changes / num_note_times

    def _get_next_tracks(self) -> dict:
        for beatmapset_id in self.subset_ids:
            metadata = self.metadata.loc[beatmapset_id]

            if self.args.add_gd_context and len(metadata) <= 1:
                continue

            if self.args.min_difficulty > 0 and all(beatmap_metadata["DifficultyRating"]
                                                    < self.args.min_difficulty for beatmap_metadata in
                                                    metadata):
                continue

            speed = self._get_speed_augment()
            track_path = self.path / "data" / metadata.iloc[0]["BeatmapSetFolder"]
            audio_path = track_path / metadata.iloc[0]["AudioFile"]
            audio_samples = load_audio_file(audio_path, self.args.sample_rate, speed)

            for i, beatmap_metadata in metadata.iterrows():
                if self.args.min_difficulty > 0 and beatmap_metadata["DifficultyRating"] < self.args.min_difficulty:
                    continue

                for sample in self._get_next_beatmap(audio_samples, i, beatmap_metadata, metadata, speed):
                    yield sample

    def _get_next_beatmap(self, audio_samples, i, beatmap_metadata: Series, set_metadata: DataFrame,
                          speed: float) -> dict:
        context_info = None
        if len(self.args.context_types) > 0:
            # Randomly select a context type with probabilities of context_weights
            context_info = random.choices(self.args.context_types, weights=self.args.context_weights)[0]

            if isinstance(context_info, str):
                context_info = {"out": "map", "in": [context_info]}
            else:
                # It's important to copy the context_info because we will modify it, and we don't want to permanently change the config
                context_info = context_info.copy()

            if "gd" in context_info["in"] and len(set_metadata) <= 1:
                context_info["in"].remove("gd")
            if len(context_info["in"]) == 0:
                context_info["in"].append("none")

        beatmap_path = self.path / "data" / beatmap_metadata["BeatmapSetFolder"] / beatmap_metadata["BeatmapFile"]
        frames, frame_times = self._get_frames(audio_samples)
        osu_beatmap = Beatmap.from_path(beatmap_path)

        def add_special_data(data, beatmap_metadata, beatmap: Beatmap):
            gamemode = beatmap.mode
            data["extra"]["gamemode"] = gamemode
            data["extra"]["beatmap_id"] = beatmap.beatmap_id
            data["extra"]["beatmap_idx"] = beatmap_metadata["BeatmapIdx"]
            data["extra"]["difficulty"] = self._get_difficulty(beatmap_metadata, speed)
            data["extra"]["year"] = beatmap_metadata["SubmittedDate"].year
            if gamemode in [0, 2]:
                data["extra"]["circle_size"] = beatmap.circle_size
            if gamemode == 3:
                data["extra"]["keycount"] = beatmap.circle_size
                data["extra"]["hold_note_ratio"] = self._get_hold_note_ratio(beatmap)
            if gamemode in [1, 3]:
                data["extra"]["scroll_speed_ratio"] = self._get_scroll_speed_ratio(beatmap)

        def get_context(context, add_type=True, force_special_data=False):
            data = {"extra": {"context_type": ContextType(context), "add_type": add_type}}
            if context == "none":
                data["events"], data["event_times"] = [], []
            elif context == "timing":
                data["events"], data["event_times"] = self.parser.parse_timing(osu_beatmap, speed)
            elif context == "no_hs":
                hs_events, hs_event_times = self.parser.parse(osu_beatmap, speed)
                data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                            [EventType.HITSOUND, EventType.VOLUME])
            elif context == "gd":
                other_metadata = set_metadata.drop(i).sample().iloc[0]
                other_beatmap_path = self.path / "data" / other_metadata["BeatmapSetFolder"] / other_metadata[
                    "BeatmapFile"]
                other_beatmap = Beatmap.from_path(other_beatmap_path)
                data["events"], data["event_times"] = self.parser.parse(other_beatmap, speed)
                add_special_data(data, other_metadata, other_beatmap)
            elif context == "map":
                data["events"], data["event_times"] = self.parser.parse(osu_beatmap, speed)
            if force_special_data:
                add_special_data(data, beatmap_metadata, osu_beatmap)
            return data

        extra_data = {
            "beatmap_idx": beatmap_metadata["BeatmapIdx"],
        }

        if self.sample_weights is not None:
            extra_data["sample_weights"] = self.sample_weights.get(osu_beatmap.beatmap_id, 1.0)

        out_context = get_context(context_info["out"], add_type=False, force_special_data=True)

        in_context = []
        for context in context_info["in"]:
            in_context.append(get_context(context))

        if self.args.add_gd_context:
            in_context.append(get_context("gd", False))

        sequences = self._create_sequences(
            frames,
            frame_times,
            out_context,
            in_context,
            extra_data,
        )

        for sequence in sequences:
            self.maybe_change_dataset()
            sequence = self._normalize_time_shifts(sequence)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (
                    sequence["labels"] == LABEL_IGNORE_ID)).all():
                continue
            # if sequence["decoder_input_ids"][self.pre_token_len - 1] != self.tokenizer.pad_id:
            #     continue
            yield sequence
