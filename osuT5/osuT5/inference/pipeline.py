from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from slider import Beatmap
from tqdm import tqdm

from omegaconf import DictConfig

from ..dataset import OsuParser
from ..dataset.data_utils import update_event_times, remove_events_of_type
from ..tokenizer import Event, EventType, Tokenizer, ContextType
from ..model import OsuT

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Pipeline(object):
    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.tgt_seq_len = args.osut5.data.tgt_seq_len
        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.model.spectrogram.hop_length
        self.sample_rate = args.osut5.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.sequence_stride = int(self.samples_per_sequence * (1 - args.lookback - args.lookahead))
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        self.miliseconds_per_stride = self.sequence_stride * MILISECONDS_PER_SECOND / self.sample_rate
        self.lookahead_max_time = (1 - args.lookahead) * self.miliseconds_per_sequence
        self.eos_time = (1 - args.osut5.data.lookahead) * self.miliseconds_per_sequence
        self.center_pad_decoder = args.osut5.data.center_pad_decoder
        self.special_token_len = args.osut5.data.special_token_len
        self.diff_token_index = args.osut5.data.diff_token_index
        self.style_token_index = args.osut5.data.style_token_index
        self.mapper_token_index = args.osut5.data.mapper_token_index
        self.add_descriptors = args.osut5.data.add_descriptors
        self.max_pre_token_len = args.osut5.data.max_pre_token_len
        self.add_pre_tokens = args.osut5.data.add_pre_tokens
        self.add_gd_context = args.osut5.data.add_gd_context
        self.parser = OsuParser(args.osut5, self.tokenizer)
        self.need_beatmap_idx = args.osut5.model.do_style_embed
        self.add_positions = args.osut5.data.add_positions

        if self.add_positions:
            self.position_precision = args.data.position_precision
            x_min, x_max, y_min, y_max = args.data.position_range
            x_min, x_max = x_min / self.position_precision, x_max / self.position_precision
            self.x_count = x_max - x_min + 1

    def generate(
            self,
            model: OsuT,
            sequences: torch.Tensor,
            beatmap_id: int = -1,
            difficulty: float = -1,
            mapper_id: int = -1,
            descriptors=None,
            other_beatmap_path: str = '',
            context_type: ContextType = None,
    ) -> list[Event]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: A list of batched source sequences.
            beatmap_id: Beatmap ID of the desired style.
            difficulty: The desired difficulty in star rating.
            mapper_id: Mapper ID for the style of beatmap.
            descriptors: List of descriptors for the style of beatmap.
            other_beatmap_path: Path to the beatmap file to use as context.
            context_type: Type of context to use for inference.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        if descriptors is None:
            descriptors = ["unknown"] if self.add_descriptors else []

        events = []
        event_times = []

        idx_dict = self.tokenizer.beatmap_idx
        beatmap_idx = torch.tensor([self.tokenizer.num_classes], dtype=torch.long, device=self.device)
        if self.need_beatmap_idx:
            if beatmap_id in idx_dict:
                beatmap_idx = torch.tensor([idx_dict[beatmap_id]], dtype=torch.long, device=self.device)
            else:
                print(f"Beatmap ID {beatmap_id} not found in dataset, using default style.")

        # Prepare special tokens
        special_tokens = torch.empty((1, self.special_token_len + len(descriptors)), dtype=torch.long, device=self.device)

        if self.style_token_index >= 0:
            style_token = self.tokenizer.encode_style(beatmap_id) if beatmap_id != -1 else self.tokenizer.style_unk
            special_tokens[:, self.style_token_index] = style_token
        if self.diff_token_index >= 0:
            diff_token = self.tokenizer.encode_diff(difficulty) if difficulty != -1 else self.tokenizer.diff_unk
            special_tokens[:, self.diff_token_index] = diff_token
        if self.mapper_token_index >= 0:
            mapper_token = self.tokenizer.encode_mapper_id(mapper_id) if mapper_id != -1 else self.tokenizer.mapper_unk
            special_tokens[:, self.mapper_token_index] = mapper_token
        for i, descriptor in enumerate(descriptors):
            special_tokens[:, self.special_token_len + i] = self.tokenizer.encode_descriptor_name(descriptor)

        # Prepare other beatmap context
        other_events, other_event_times = [], []
        other_special_tokens = torch.empty((1, 0), dtype=torch.long, device=self.device)
        if self.add_gd_context or context_type == ContextType.GD or context_type == ContextType.TIMING or context_type == ContextType.NO_HS:
            other_beatmap_path = Path(other_beatmap_path)

            if not other_beatmap_path.is_file():
                raise FileNotFoundError(f"Beatmap file {other_beatmap_path} not found.")

            other_beatmap = Beatmap.from_path(other_beatmap_path)

            if self.add_gd_context or context_type == ContextType.GD:
                other_events = self.parser.parse(other_beatmap)
                other_beatmap_id = other_beatmap.beatmap_id
                other_difficulty = float(other_beatmap.stars())

                if self.add_descriptors:
                    other_descriptor_tokens = self.tokenizer.encode_descriptor(other_beatmap_id)
                else:
                    other_descriptor_tokens = []

                other_special_tokens = torch.empty((1, self.special_token_len + len(other_descriptor_tokens)), dtype=torch.long, device=self.device)

                if self.style_token_index >= 0:
                    other_style_token = self.tokenizer.encode_style(other_beatmap_id)
                    other_special_tokens[:, self.style_token_index] = other_style_token
                if self.diff_token_index >= 0:
                    other_diff_token = self.tokenizer.encode_diff(other_difficulty)
                    other_special_tokens[:, self.diff_token_index] = other_diff_token
                if self.mapper_token_index >= 0:
                    other_mapper_token = self.tokenizer.encode_mapper(other_beatmap_id)
                    other_special_tokens[:, self.mapper_token_index] = other_mapper_token
                for i, descriptor_token in enumerate(other_descriptor_tokens):
                    other_special_tokens[:, self.special_token_len + i] = descriptor_token
            elif context_type == ContextType.NO_HS:
                other_events = self.parser.parse(other_beatmap)
                other_events = remove_events_of_type(other_events, [EventType.HITSOUND, EventType.VOLUME])
            elif context_type == ContextType.TIMING:
                other_events = self.parser.parse_timing(other_beatmap)

        other_events, other_event_times = self._prepare_events(other_events)

        # Prepare context type indicator tokens
        context_sos = torch.tensor([[self.tokenizer.context_sos[context_type]]], dtype=torch.long, device=self.device)\
            if context_type is not None else torch.empty((1, 0), dtype=torch.long, device=self.device)
        context_eos = torch.tensor([[self.tokenizer.context_eos[context_type]]], dtype=torch.long, device=self.device)\
            if context_type is not None else torch.empty((1, 0), dtype=torch.long, device=self.device)

        # Start generation
        for sequence_index, frames in enumerate(tqdm(sequences)):
            # Get tokens of previous frame
            frame_time = sequence_index * self.miliseconds_per_stride

            prev_events = self._get_events_time_range(
                events, event_times, frame_time - self.miliseconds_per_sequence, frame_time) \
                if self.add_pre_tokens else []
            prev_tokens = self._encode(prev_events, frame_time)
            if 0 <= self.max_pre_token_len < prev_tokens.shape[1]:
                prev_tokens = prev_tokens[:, -self.max_pre_token_len:]

            post_events = self._get_events_time_range(
                events, event_times, frame_time, frame_time + self.miliseconds_per_sequence)
            post_tokens = self._encode(post_events, frame_time)
            post_token_length = post_tokens.shape[1]

            context_events = self._get_events_time_range(
                other_events, other_event_times, frame_time,
                frame_time + self.miliseconds_per_sequence)
            context_tokens = self._encode(context_events, frame_time)

            # Get prefix tokens
            prefix = torch.concatenate([context_sos, other_special_tokens, context_tokens, context_eos, special_tokens, prev_tokens], dim=-1)
            if self.center_pad_decoder:
                prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)
            prefix_length = prefix.shape[1]

            # Get tokens
            tokens = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long, device=self.device)
            tokens = torch.concatenate([prefix, tokens, post_tokens], dim=-1)

            frames = frames.to(self.device).unsqueeze(0)
            encoder_outputs = None

            while tokens.shape[-1] < self.tgt_seq_len:
                out = model.forward(
                    frames=frames,
                    decoder_input_ids=tokens,
                    decoder_attention_mask=tokens.ne(self.tokenizer.pad_id),
                    encoder_outputs=encoder_outputs,
                    beatmap_idx=beatmap_idx,
                )
                encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)
                logits = out.logits
                logits = logits[:, -1, :]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, 1)

                tokens = torch.cat([tokens, next_tokens], dim=-1)

                # check if any sentence in batch has reached EOS, mark as finished
                eos_in_sentence = next_tokens == self.tokenizer.eos_id

                # stop preemptively when all sentences have finished
                if eos_in_sentence.all():
                    break

                next_event = self.tokenizer.decode(next_tokens[0].item())
                if (sequence_index != len(sequences) - 1 and
                        next_event.type == EventType.TIME_SHIFT and
                        next_event.value * MILISECONDS_PER_STEP > self.lookahead_max_time):
                    break

            # Trim prefix, SOS, post-tokens, and EOS tokens
            predicted_tokens = tokens[:, prefix_length + 1 + post_token_length:-1]
            result = self._decode(predicted_tokens[0], frame_time)
            events += result
            self._update_event_times(events, event_times, frame_time + self.eos_time)

            # Trim events which are in the lookahead window
            if sequence_index != len(sequences) - 1:
                lookahead_time = frame_time + self.lookahead_max_time
                self._trim_events_after_time(events, event_times, lookahead_time)

        # Rescale and unpack position events
        if self.add_positions:
            events = self._rescale_positions(events)

        return events

    def _prepare_events(self, events: list[Event]) -> tuple[list[Event], list[float]]:
        """Pre-process raw list of events for inference. Calculates event times and removes redundant time shifts."""
        ct = 0
        event_times = []
        for event in events:
            if event.type == EventType.TIME_SHIFT:
                ct = event.value
            event_times.append(ct)

        # Loop through the events in reverse to remove any time shifts that occur before anchor events
        delete_next_time_shift = False
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.TIME_SHIFT and delete_next_time_shift:
                delete_next_time_shift = False
                del events[i]
                del event_times[i]
                continue
            elif events[i].type in [EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR,
                                    EventType.RED_ANCHOR]:
                delete_next_time_shift = True

        return events, event_times

    def _get_events_time_range(self, events: list[Event], event_times: list[float], start_time: float, end_time: float):
        # Look from the end of the list
        s = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < start_time:
                s = i + 1
                break
        e = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < end_time:
                e = i + 1
                break
        return events[s:e]

    def _trim_events_after_time(self, events, event_times, lookahead_time):
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] > lookahead_time:
                del events[i]
                del event_times[i]
            else:
                break

    def _update_event_times(self, events: list[Event], event_times: list[float], end_time: float):
        update_event_times(events, event_times, end_time)

    def _encode(self, events: list[Event], frame_time: float) -> torch.Tensor:
        tokens = torch.empty((1, len(events)), dtype=torch.long)
        for i, event in enumerate(events):
            if event.type == EventType.TIME_SHIFT:
                event = Event(type=event.type, value=int((event.value - frame_time) / MILISECONDS_PER_STEP))
            tokens[0, i] = self.tokenizer.encode(event)
        return tokens.to(self.device)

    def _decode(self, tokens: torch.Tensor, frame_time: float) -> list[Event]:
        """Converts a list of tokens into Event objects and converts to absolute time values.

        Args:
            tokens: List of tokens.
            frame time: Start time of current source sequence.

        Returns:
            events: List of Event objects.
        """
        events = []
        for token in tokens:
            if token == self.tokenizer.eos_id:
                break

            try:
                event = self.tokenizer.decode(token.item())
            except:
                continue

            if event.type == EventType.TIME_SHIFT:
                event.value = frame_time + event.value * MILISECONDS_PER_STEP

            events.append(event)

        return events

    def _filter(
            self, logits: torch.Tensor, top_p: float, filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Filter a distribution of logits using nucleus (top-p) filtering.

        Source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Args:
            logits: logits distribution of shape (batch size, vocabulary size).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Returns:
            logits of top tokens.
        """
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits

    def _rescale_positions(self, events: list[Event]) -> list[Event]:
        new_events = []
        for event in events:
            if event.type == EventType.POS_X or event.type == EventType.POS_Y:
                new_events.append(Event(type=event.type, value=event.value * self.position_precision))
            elif event.type == EventType.POS:
                new_events.append(Event(type=EventType.POS_X, value=(event.value % self.x_count) * self.position_precision))
                new_events.append(Event(type=EventType.POS_Y, value=(event.value // self.x_count) * self.position_precision))
            else:
                new_events.append(event)

        return new_events
