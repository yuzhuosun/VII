"""HuggingFace dataset adapters for VII data sources.

The helpers in this module keep the rest of the project independent from the
source-specific schemas used by upstream HuggingFace datasets.  Each row is
normalized into :class:`vii.types.DatasetSample` with stable fields used by the
pipeline and local inspection scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import Dataset, DatasetDict, DownloadConfig, IterableDataset, IterableDatasetDict, load_dataset

from vii.types import DatasetSample


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """Source-column names used to normalize a HuggingFace dataset row."""

    sample_id: tuple[str, ...] = ("sample_id", "id", "idx", "index")
    image: tuple[str, ...] = ("image_path", "image", "img", "image_file", "file_name", "filepath")
    prompt: tuple[str, ...] = (
        "harmful_video_prompt",
        "unsafe_video_prompt",
        "prompt",
        "text",
        "instruction",
        "question",
        "caption",
    )
    category: tuple[str, ...] = ("category", "risk_category", "concept", "label", "class")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for a supported HuggingFace dataset."""

    key: str
    hf_id: str
    default_split: str = "train"
    cache_dir: str | None = None
    field_mapping: FieldMapping = field(default_factory=FieldMapping)


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "coco_i2v_safetybench": DatasetConfig(
        key="coco_i2v_safetybench",
        hf_id="yonglixiang/COCO-I2VSafetyBench",
        default_split="train",
        field_mapping=FieldMapping(
            sample_id=("sample_id", "id", "image_id", "idx"),
            image=("image_path", "image", "coco_url", "file_name", "filepath"),
            prompt=("harmful_video_prompt", "prompt", "instruction", "question", "caption", "text"),
            category=("category", "risk_category", "safety_category", "label"),
        ),
    ),
    "conceptrisk": DatasetConfig(
        key="conceptrisk",
        hf_id="yonglixiang/ConceptRisk-Repro",
        default_split="train",
        field_mapping=FieldMapping(
            sample_id=("sample_id", "id", "concept_id", "idx"),
            image=("image_path", "image", "img", "file_name", "filepath"),
            prompt=("unsafe_video_prompt", "prompt", "text", "instruction", "caption", "concept"),
            category=("category", "risk_category", "concept", "label", "class"),
        ),
    ),
}


def load_vii_dataset(
    dataset: str,
    split: str | None = None,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    download_config: DownloadConfig | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load a supported VII HuggingFace dataset by short name."""

    config = _get_config(dataset)
    requested_split = None if split == "all" else (split or config.default_split)
    return load_dataset(
        config.hf_id,
        split=requested_split,
        cache_dir=str(cache_dir or config.cache_dir) if (cache_dir or config.cache_dir) else None,
        streaming=streaming,
        download_config=download_config,
    )


def iter_dataset_samples(
    dataset: str,
    split: str | None = None,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    download_config: DownloadConfig | None = None,
) -> Iterable[DatasetSample]:
    """Yield normalized :class:`DatasetSample` objects for a dataset/split."""

    config = _get_config(dataset)
    loaded = load_vii_dataset(dataset, split=split, cache_dir=cache_dir, streaming=streaming, download_config=download_config)
    if isinstance(loaded, (DatasetDict, IterableDatasetDict)):
        for split_name, split_dataset in loaded.items():
            for index, record in enumerate(split_dataset):
                yield normalize_record(record, config, index=index, split=split_name)
    else:
        for index, record in enumerate(loaded):
            yield normalize_record(record, config, index=index, split=split or config.default_split)


def normalize_record(record: Mapping[str, Any], config: DatasetConfig | str, index: int, split: str | None = None) -> DatasetSample:
    """Normalize one source dataset row into the common ``DatasetSample`` shape."""

    dataset_config = _get_config(config) if isinstance(config, str) else config
    mapping = dataset_config.field_mapping
    sample_id = _first_present(record, mapping.sample_id)
    prompt = _first_present(record, mapping.prompt)
    category = _first_present(record, mapping.category)
    image_value = _first_present(record, mapping.image)

    stable_id = str(sample_id if sample_id not in (None, "") else f"{dataset_config.key}-{split or 'unknown'}-{index}")
    metadata = {
        "source_dataset_id": dataset_config.hf_id,
        "source_split": split,
        "source_index": index,
        "source_fields": sorted(record.keys()),
    }
    for key, value in record.items():
        if key not in _flatten_mapping(mapping):
            metadata[key] = _json_safe(value)

    image_path: str | Path = ""
    image: Any | None = None
    if isinstance(image_value, (str, Path)):
        image_path = image_value
    else:
        image = image_value

    return DatasetSample(
        sample_id=stable_id,
        image_path=image_path,
        image=image,
        prompt=str(prompt or ""),
        category=str(category or "uncategorized"),
        source_dataset=dataset_config.key,
        metadata=metadata,
    )


def _get_config(dataset: DatasetConfig | str) -> DatasetConfig:
    if isinstance(dataset, DatasetConfig):
        return dataset
    try:
        return DATASET_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {sorted(DATASET_CONFIGS)}") from exc


def _first_present(record: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _flatten_mapping(mapping: FieldMapping) -> set[str]:
    return set(mapping.sample_id) | set(mapping.image) | set(mapping.prompt) | set(mapping.category)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)
