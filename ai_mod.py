from pathlib import Path

import hydra
import torch

from config import InferenceConfig
from inference import prepare_args, get_args_from_beatmap, get_config, load_model
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
    result = processor.get_logits(
        sequences=sequences,
        generation_config=generation_config,
        beatmap_path=beatmap_path,
        verbose=verbose,
    )

    for context in result:
        context['surprisals'] = []
        context['suggestions'] = []
        for event, event_time, token, logits in zip(
                context['events'],
                context['event_times'],
                context['tokens'],
                context['logits'],
        ):
            probs = logits.softmax(dim=-1)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1).item()
            surprisal = -torch.log2(probs[token] + 1e-10).item()
            relative_surprisal = surprisal / entropy if entropy > 0 else 0.0
            context['surprisals'].append(relative_surprisal)

            # Get the most likely token
            suggested_token = logits.argmax().item()
            suggested_event = tokenizer.decode(suggested_token)
            context['suggestions'].append(suggested_event)

            # if verbose:
            #     print(f"Event: {event}, Time: {event_time}, Token: {token}, Surprise: {relative_surprisal:.4f}")

    # Print for every context and every event type, the top 10 events with the highest surprisal
    # Also skip anything below 1 relative suprisal
    if verbose:
        for context in result:
            print(f"Context: {context['context_type']}")
            event_types = set(e.type for e in context['events'])
            for event_type in event_types:
                surprisal_events = [
                    z for z in zip(context['events'], context['event_times'], context['suggestions'], context['surprisals']) if z[0].type == event_type and z[-1] >= 1.0
                ]
                if not surprisal_events:
                    continue
                print(f"  Event Type: {event_type.value}")
                surprisal_events.sort(key=lambda x: x[-1], reverse=True)
                for event, event_time, suggested_event, surprisal in surprisal_events[:10]:
                    print(f"    Event: {event}, Time: {event_time}, Suggestion: {suggested_event}, Surprisal: {surprisal:.4f}")


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
