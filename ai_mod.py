from pathlib import Path

import hydra
import numpy as np

from config import InferenceConfig
from inference import prepare_args, get_args_from_beatmap, get_config, load_model
from osuT5.osuT5.dataset.data_utils import get_groups, Group
from osuT5.osuT5.event import EventType, Event
from osuT5.osuT5.inference import Preprocessor, Processor, GenerationConfig
from osuT5.osuT5.inference.server import InferenceClient
from osuT5.osuT5.model import Mapperatorinator


def ai_mod(
        args: InferenceConfig,
        *,
        audio_path: str = None,
        beatmap_path: str = None,
        generation_config: GenerationConfig,
        model: Mapperatorinator | InferenceClient,
        tokenizer,
        verbose=True,
):
    audio_path = args.audio_path if audio_path is None else audio_path
    beatmap_path = args.beatmap_path if beatmap_path is None else beatmap_path

    # Do some validation
    if not Path(audio_path).exists() or not Path(audio_path).is_file():
        raise FileNotFoundError(f"Provided audio file path does not exist: {audio_path}")
    if beatmap_path:
        beatmap_path_obj = Path(beatmap_path)
        if not beatmap_path_obj.exists() or not beatmap_path_obj.is_file():
            raise FileNotFoundError(f"Provided beatmap file path does not exist: {beatmap_path}")
        # Validate beatmap file type
        if beatmap_path_obj.suffix.lower() != '.osu':
            raise ValueError(f"Beatmap file must have .osu extension: {beatmap_path}")

    preprocessor = Preprocessor(args, parallel=False)
    processor = Processor(args, model, tokenizer)

    audio = preprocessor.load(audio_path)
    sequences = preprocessor.segment(audio)

    # Generate logits
    result = processor.ai_mod(
        sequences=sequences,
        generation_config=generation_config,
        beatmap_path=beatmap_path,
        verbose=verbose,
    )

    # Print for every context and every event type, the top 10 events with the highest surprisal
    # Also skip anything below 1 relative suprisal
    if verbose:
        for context in result:
            print(f"Context: {context['context_type']}")
            event_types = sorted(list(set(e.type for e in context['events'])), key=lambda x: x.value)

            groups, group_indices = get_groups(context['events'], event_times=context['event_times'], types_first=args.train.data.types_first)
            # Group indices map each group index to a list of indices of the events in the original list
            # We need the reverse mapping to get the groups for each event
            event_groups: list[int] = [0] * len(context['events'])
            for group_index, indices in enumerate(group_indices):
                for index in indices:
                    event_groups[index] = group_index

            surprisal_events = [
                z for z in zip(
                    range(len(context['events'])),
                    context['event_times'],
                    context['real_events'],
                    context['real'],
                    context['expected_events'],
                    context['expected'],
                    context['surprisals']
                )
            ]

            position_types = [EventType.DISTANCE, EventType.POS_X, EventType.POS_Y, EventType.POS]
            anchor_types = [EventType.RED_ANCHOR, EventType.BEZIER_ANCHOR, EventType.CATMULL_ANCHOR, EventType.PERFECT_ANCHOR]

            for event_type in event_types:
                # Filter suprisal events
                filtered_events = [
                    (i, event_time, event, real, expected_event, expected, surprisal)
                    for i, event_time, event, real, expected_event, expected, surprisal in surprisal_events
                    if (event.type == event_type and
                        surprisal >= 10.0 and
                        not (groups[event_groups[i]].event_type == EventType.SLIDER_END and event.type in position_types) and
                        not (event.type == EventType.TIME_SHIFT and expected_event.type == EventType.TIME_SHIFT and abs(expected_event.value - event.value) <= 10))
                ]

                if not filtered_events:
                    continue

                print(f"  Event Type: {event_type.value}")
                filtered_events.sort(key=lambda x: x[-1], reverse=True)

                for i, event_time, event, real, expected_event, expected, surprisal in filtered_events[:10]:
                    group_type = groups[event_groups[i]].event_type

                    # If the group is an anchor, we want to print the anchor index in the slider
                    if group_type in anchor_types:
                        # Count the number of anchor groups in between this group and the slider head group
                        anchor_index = 2
                        for j in range(event_groups[i] - 1, -1, -1):
                            if groups[j].event_type == EventType.SLIDER_HEAD:
                                break
                            if groups[j].event_type in anchor_types:
                                anchor_index += 1
                        group_type = f"{group_type.value} (Anchor {anchor_index})"
                    else:
                        group_type = group_type.value

                    print(f"    Time: {event_time}, Event: {real}, Suggestion: {expected}, Group: {group_type}, Surprisal: {surprisal:.4f}")


@hydra.main(config_path="configs/inference", config_name="v30", version_base="1.1")
def main(args: InferenceConfig):
    prepare_args(args)

    model, tokenizer = load_model(args.model_path, args.train, args.device, args.max_batch_size, False)

    get_args_from_beatmap(args, tokenizer)
    generation_config, beatmap_config = get_config(args)

    return ai_mod(
        args,
        generation_config=generation_config,
        beatmap_path=args.beatmap_path,
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
