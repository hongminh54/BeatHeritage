import hydra
import torch
from omegaconf import DictConfig

from osuT5.inference import Preprocessor, Pipeline, Postprocessor
from osuT5.tokenizer import Tokenizer
from osuT5.utils import get_config, get_model


@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def main(args: DictConfig):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    t5_config = get_config(args)

    model = get_model(t5_config)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    tokenizer = Tokenizer()
    preprocessor = Preprocessor(args)
    pipeline = Pipeline(args, tokenizer)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(args.audio_path)
    sequences = preprocessor.segment(audio)
    events = pipeline.generate(model, sequences)
    postprocessor.generate(events)


if __name__ == "__main__":
    main()
