"""
RNN Applications - four patterns, real pretrained models
========================================================

    #  RNN pattern     Application               Pretrained model
    ---------------------------------------------------------------------------
    1  one-to-one      Image classification      vit-base-oxford-iiit-pets (37 breeds)
    2  one-to-many     Image captioning          vit-gpt2-image-captioning
    3  many-to-one     Sentiment analysis        distilbert-sst-2-english
    4  many-to-many    Named-entity recognition  bert-base-NER (CoNLL-2003)

Install : pip install torch transformers pillow
Run     : python main.py
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BREED_MODEL = "MaxPowerUnlimited/vit-base-oxford-iiit-pets"
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NER_MODEL = "dslim/bert-base-NER"

CAT_BREEDS = {
    "abyssinian", "bengal", "birman", "bombay", "british shorthair",
    "egyptian mau", "maine coon", "persian", "ragdoll", "russian blue",
    "siamese", "sphynx",
}
ENTITY_NAMES = {"PER": "Person", "LOC": "Location",
                "ORG": "Organization", "MISC": "Miscellaneous"}

_MODELS: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Model loading (lazy: each model is built once, on first use)
# ---------------------------------------------------------------------------
def _load(key: str, factory):
    if key not in _MODELS:
        print("  loading model (first run may take a moment) ...", flush=True)
        _MODELS[key] = factory()
    return _MODELS[key]


def _title(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().title()


# ---------------------------------------------------------------------------
# 1. one-to-one    Image classification  (image -> breed)
# ---------------------------------------------------------------------------
def classify_breed(path: str) -> str:
    from transformers import pipeline
    from PIL import Image

    clf = _load("breed", lambda: pipeline(
        "image-classification", model=BREED_MODEL, top_k=5))
    preds = clf(Image.open(path).convert("RGB"))

    top = preds[0]
    species = "cat" if _title(top["label"]).lower() in CAT_BREEDS else "dog"
    rows = [f"Species : {species}",
            f"Breed   : {_title(top['label'])}  ({top['score'] * 100:.1f}%)",
            "",
            "Top 5 breeds"]
    for p in preds:
        pct = p["score"] * 100
        rows.append(f"  {_title(p['label']):<20}{pct:5.1f}%  {'█' * round(pct / 5)}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 2. one-to-many   Image captioning  (image -> sentence)
# ---------------------------------------------------------------------------
def _build_captioner():
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL).eval()
    return model, ViTImageProcessor.from_pretrained(CAPTION_MODEL), \
        AutoTokenizer.from_pretrained(CAPTION_MODEL)


def caption_image(path: str) -> str:
    import torch
    from PIL import Image

    model, processor, tokenizer = _load("caption", _build_captioner)
    pixels = processor(images=Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixels, max_length=16, num_beams=4)
    return f'Caption : "{tokenizer.decode(ids[0], skip_special_tokens=True).strip()}"'


# ---------------------------------------------------------------------------
# 3. many-to-one   Sentiment analysis  (review -> verdict)
# ---------------------------------------------------------------------------
def analyse_sentiment(text: str) -> str:
    from transformers import pipeline
    analyser = _load("sentiment", lambda: pipeline(
        "sentiment-analysis", model=SENTIMENT_MODEL))
    result = analyser(text)[0]
    return (f"Verdict    : {result['label'].capitalize()}\n"
            f"Confidence : {result['score'] * 100:.1f}%")


# ---------------------------------------------------------------------------
# 4. many-to-many  Named-entity recognition  (sentence -> tag per word)
# ---------------------------------------------------------------------------
def recognise_entities(text: str) -> str:
    from transformers import pipeline
    tagger = _load("ner", lambda: pipeline(
        "ner", model=NER_MODEL, aggregation_strategy="simple"))
    found = tagger(text)
    if not found:
        return "No named entities found (try: Khoa lives in Hanoi and works at Google)."

    # Merge BERT sub-word pieces, then expand each span to full word boundaries.
    merged = []
    for e in found:
        if merged and merged[-1]["group"] == e["entity_group"] and e["start"] == merged[-1]["end"]:
            merged[-1]["end"] = e["end"]
            merged[-1]["score"] = min(merged[-1]["score"], e["score"])
        else:
            merged.append({"group": e["entity_group"], "start": e["start"],
                           "end": e["end"], "score": e["score"]})

    rows, seen = [], set()
    for m in merged:
        start, end = m["start"], m["end"]
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        if (start, end) in seen:
            continue
        seen.add((start, end))
        name = ENTITY_NAMES.get(m["group"], m["group"])
        rows.append(f"{text[start:end]:<16}{name:<14}{m['score'] * 100:5.1f}%")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
RULE = "─" * 60
DEMOS = {
    "1": ("Image classification", "image", classify_breed),
    "2": ("Image captioning", "image", caption_image),
    "3": ("Sentiment analysis", "review", analyse_sentiment),
    "4": ("Named-entity recognition", "sentence", recognise_entities),
}
MENU = f"""
  1   Image classification       one-to-one     image  → breed
  2   Image captioning           one-to-many    image  → sentence
  3   Sentiment analysis         many-to-one    review → Positive / Negative
  4   Named-entity recognition   many-to-many   text   → entity tags
  0   Quit
"""


def _show(title: str, body: str) -> None:
    print(f"\n  {title}\n  {RULE[:52]}")
    for line in body.splitlines():
        print(f"    {line}")
    print()


def _ask_input(kind: str) -> str | None:
    raw = input(f"  Enter {kind}: ").strip()
    if kind == "image":
        raw = os.path.expanduser(raw.strip('"').strip("'"))
        if not os.path.isfile(raw):
            print(f"  File not found: {raw}\n")
            return None
    return raw or None


def _run(title: str, kind: str, action) -> None:
    argument = _ask_input(kind)
    if argument is None:
        return
    try:
        body = action(argument)
    except ImportError as err:
        body = f"Missing package '{err.name}'. Run: pip install torch transformers pillow"
    except Exception as err:                       # noqa: BLE001
        body = f"Could not process the input: {err}"
    _show(title, body)


def main() -> None:
    print("═" * 60)
    print("  RNN APPLICATIONS  ·  four patterns, real pretrained models")
    print("═" * 60)
    try:
        while True:
            print(MENU)
            choice = input("  Select [0-4]: ").strip()
            if choice == "0":
                break
            if choice in DEMOS:
                _run(*DEMOS[choice])
            else:
                print("  Please choose a number from 0 to 4.")
    except (EOFError, KeyboardInterrupt):
        pass
    print("\n  Goodbye.\n")


if __name__ == "__main__":
    main()
