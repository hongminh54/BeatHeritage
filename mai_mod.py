from dataclasses import dataclass
from pathlib import Path
from string import Template

import hydra

from config import InferenceConfig
from inference import prepare_args, get_args_from_beatmap, get_config, load_model
from osuT5.osuT5.dataset.data_utils import get_groups, Group
from osuT5.osuT5.event import EventType, Event, ContextType
from osuT5.osuT5.inference import Preprocessor, Processor, GenerationConfig
from osuT5.osuT5.inference.server import InferenceClient
from osuT5.osuT5.model import Mapperatorinator

different_anchor_type = "Expected anchor type $expected_type instead of $real_type."

# These event types are designed for V30 tokenization
mod_explanations = {
    # Real, Expected
    (EventType.DISTANCE, EventType.DISTANCE): "Expected distance $expected_value to the previous $previous_group instead of $real_value.",
    (EventType.POS_X, EventType.POS_X): "Expected position $expected_value instead of $real_value.",
    (EventType.POS_Y, EventType.POS_Y): "Expected position $expected_value instead of $real_value.",
    (EventType.POS, EventType.POS): "Expected position $expected_value instead of $real_value.",
    (EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR): different_anchor_type,
    (EventType.BEZIER_ANCHOR, EventType.RED_ANCHOR): different_anchor_type,
    (EventType.BEZIER_ANCHOR, EventType.CATMULL_ANCHOR): different_anchor_type,
    (EventType.BEZIER_ANCHOR, EventType.LAST_ANCHOR): different_anchor_type,
    (EventType.PERFECT_ANCHOR, EventType.BEZIER_ANCHOR): different_anchor_type,
    (EventType.PERFECT_ANCHOR, EventType.RED_ANCHOR): different_anchor_type,
    (EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR): different_anchor_type,
    (EventType.PERFECT_ANCHOR, EventType.LAST_ANCHOR): different_anchor_type,
    (EventType.RED_ANCHOR, EventType.BEZIER_ANCHOR): different_anchor_type,
    (EventType.RED_ANCHOR, EventType.PERFECT_ANCHOR): different_anchor_type,
    (EventType.RED_ANCHOR, EventType.CATMULL_ANCHOR): different_anchor_type,
    (EventType.RED_ANCHOR, EventType.LAST_ANCHOR): different_anchor_type,
    (EventType.CATMULL_ANCHOR, EventType.BEZIER_ANCHOR): different_anchor_type,
    (EventType.CATMULL_ANCHOR, EventType.PERFECT_ANCHOR): different_anchor_type,
    (EventType.CATMULL_ANCHOR, EventType.RED_ANCHOR): different_anchor_type,
    (EventType.CATMULL_ANCHOR, EventType.LAST_ANCHOR): different_anchor_type,
    (EventType.LAST_ANCHOR, EventType.BEZIER_ANCHOR): different_anchor_type,
    (EventType.LAST_ANCHOR, EventType.PERFECT_ANCHOR): different_anchor_type,
    (EventType.LAST_ANCHOR, EventType.RED_ANCHOR): different_anchor_type,
    (EventType.LAST_ANCHOR, EventType.CATMULL_ANCHOR): different_anchor_type,
    (EventType.HITSOUND, EventType.HITSOUND): "Expected hitsound $expected_value instead of $real_value.",
    (EventType.VOLUME, EventType.VOLUME): "Expected volume $expected_value instead of $real_value.",
    (EventType.HITSOUND, EventType.NEW_COMBO): "Expected new combo.",
    (EventType.NEW_COMBO, EventType.HITSOUND): "Unexpected new combo.",
    (EventType.HITSOUND, EventType.LAST_ANCHOR): "Expected end of slider repeats.",
    (EventType.CIRCLE, EventType.SLIDER_HEAD): "Expected a slider instead of a circle.",
    (EventType.CIRCLE, EventType.SPINNER): "Expected a spinner instead of a circle.",
    (EventType.SLIDER_HEAD, EventType.CIRCLE): "Expected a circle instead of a slider.",
    (EventType.SLIDER_HEAD, EventType.SPINNER): "Expected a spinner instead of a slider.",
    (EventType.SPINNER, EventType.CIRCLE): "Expected a circle instead of a spinner.",
    (EventType.SPINNER, EventType.SLIDER_HEAD): "Expected a slider instead of a spinner.",
    (EventType.SNAPPING, EventType.SNAPPING): "Expected snapping $expected_value instead of $real_value.",
    (EventType.SNAPPING, EventType.BEAT): "Hit object likely not snapped to a beat.",
    (EventType.SNAPPING, EventType.MEASURE): "Hit object likely not snapped to a beat.",
    (EventType.SNAPPING, EventType.TIMING_POINT): "Hit object likely not snapped to a beat.",
    (EventType.TIME_SHIFT, EventType.CONTROL): "Expected end of beatmap.",
    (EventType.TIME_SHIFT, EventType.TIME_SHIFT): "Expected object at $expected_value instead of $real_value.",
    (EventType.TIME_SHIFT, EventType.DISTANCE): "Expected additional anchors.",
    (EventType.MEASURE, EventType.SNAPPING): "Unexpected new measure.",
    (EventType.MEASURE, EventType.BEAT): "Unexpected new measure.",
    (EventType.TIMING_POINT, EventType.SNAPPING): "Unexpected new timing point.",
    (EventType.TIMING_POINT, EventType.BEAT): "Unexpected new timing point.",
    (EventType.TIMING_POINT, EventType.MEASURE): "Unexpected new timing point.",
}


@dataclass
class Suggestion:
    context_type: ContextType
    index: int
    time: float
    group: Group
    group_str: str
    previous_group_str: str
    next_group: Group
    event: Event
    event_str: str
    expected_event: Event
    expected_event_str: str
    surprisal: float


def type_to_str(event_type: EventType) -> str:
    return event_type.value.replace("_", " ").title()


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

    position_types = [EventType.DISTANCE, EventType.POS_X, EventType.POS_Y, EventType.POS]
    anchor_types = [EventType.RED_ANCHOR, EventType.BEZIER_ANCHOR, EventType.CATMULL_ANCHOR, EventType.PERFECT_ANCHOR]
    hs_types = [EventType.HITSOUND, EventType.VOLUME]
    timing_types = [EventType.BEAT, EventType.MEASURE, EventType.TIMING_POINT]

    # Print for every context and every event type, the top 10 events with the highest surprisal
    # Also skip anything below 1 relative suprisal
    suggestions: list[Suggestion] = []
    for context in result:
        groups, group_indices = get_groups(context['events'], event_times=context['event_times'], types_first=args.train.data.types_first)
        # Group indices map each group index to a list of indices of the events in the original list
        # We need the reverse mapping to get the groups for each event
        event_groups: list[int] = [0] * len(context['events'])
        for group_index, indices in enumerate(group_indices):
            for index in indices:
                event_groups[index] = group_index

        context_suggestions = [
            Suggestion(context['context_type'], *z) for z in zip(
                range(len(context['events'])),
                context['event_times'],
                [groups[event_groups[i]] for i in range(len(context['events']))],
                ["None"] * len(context['events']),
                ["None"] * len(context['events']),
                [groups[event_groups[i] + 1] if event_groups[i] + 1 < len(groups) else None for i in range(len(context['events']))],
                context['events'],
                context['events_str'],
                context['expected_events'],
                context['expected_events_str'],
                context['surprisals']
            )
        ]

        def get_group_str(group_index: int, s: Suggestion) -> str:
            if group_index < 0 or group_index >= len(groups):
                return "None"
            group = groups[group_index]
            if group.event_type == EventType.LAST_ANCHOR and group_index == event_groups[s.index] and s.event.type in hs_types:
                # This group is for a hitsound event on a slider end group, which contains hitsound events for each repeat
                # Find the repeat index this hitsound event corresponds to
                repeat_index = 0
                for j in range(s.index - 1, -1, -1):
                    if context['events'][j].type == EventType.TIME_SHIFT:
                        break
                    if context['events'][j].type == s.event.type:
                        repeat_index += 1

                if repeat_index == 0:
                    return "Slider Body"
                else:
                    return f"Slider Repeat #{repeat_index}"
            elif group.event_type in anchor_types:
                # Count the number of anchor groups in between this group and the slider head group
                anchor_index = 2
                for j in range(group_index - 1, -1, -1):
                    if groups[j].event_type == EventType.SLIDER_HEAD:
                        break
                    if groups[j].event_type in anchor_types:
                        anchor_index += 1
                return f"{type_to_str(group.event_type)} (Anchor #{anchor_index})"
            else:
                return type_to_str(group.event_type)

        # If the group is an anchor, we want to print the anchor index in the slider
        for s in context_suggestions:
            group_index = event_groups[s.index]
            s.group_str = get_group_str(group_index, s)

            # Find the previous group with positions
            for i in range(group_index - 1, -1, -1):
                if groups[i].x is not None:
                    s.previous_group_str = get_group_str(i, s)
                    break

        suggestions.extend(context_suggestions)

    suggestions.sort(key=lambda x: x.surprisal, reverse=True)

    # Filter suggestions
    suggestions = [
        s for s in suggestions
        if (s.surprisal >= 10.0 and
            not (s.group.event_type == EventType.SLIDER_END and s.event.type in position_types) and
            not (s.event.type == EventType.TIME_SHIFT and s.expected_event.type == EventType.TIME_SHIFT and abs(s.expected_event.value - s.event.value) <= 10) and
            not (s.event.type == EventType.SNAPPING and s.expected_event.type in timing_types and s.next_group and abs(s.time - s.next_group.time) < 2))
    ]

    for s in suggestions[:20]:
        explanation_template = mod_explanations.get((s.event.type, s.expected_event.type), "Expected $expected_type $expected_value instead of $real_type $real_value.")
        explanation_template = Template(explanation_template)
        explanation = explanation_template.safe_substitute({
            'expected_value': s.expected_event_str,
            'real_value': s.event_str,
            'expected_type': type_to_str(s.expected_event.type),
            'real_type': type_to_str(s.event.type),
            'group': s.group_str,
            'previous_group': s.previous_group_str,
        })
        print(f"Time: {s.time}, Surprisal: {s.surprisal:.0f}, Group: {s.group_str}: {explanation}")


@hydra.main(config_path="configs/inference", config_name="v30", version_base="1.1")
def main(args: InferenceConfig):
    args.add_to_beatmap = True
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
