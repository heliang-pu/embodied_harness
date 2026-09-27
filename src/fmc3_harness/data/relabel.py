"""A copy of a dataset whose subtask texts are the prompt-library texts, for training the executor.

The executor only follows prompts it was trained on word for word, and the harness sends the
library texts. So the executor is trained on a copy in which every annotated subtask label is
replaced by the library text of its call. Memory rows and everything else stay as they are.
Only the `content` of subtask rows changes; the column keeps its Arrow type, whatever
lerobot-annotate wrote.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..skills.registry import SkillRegistry
from .annotations import LANGUAGE_PERSISTENT, SUBTASK_STYLE, data_files
from .labels import LabelMap, normalize_label


def canonical_texts(labels: LabelMap, registry: SkillRegistry) -> dict[str, str]:
    """Normalized label -> library text, for labels mapped to a call the executor runs."""
    texts = {}
    for label, entry in labels.entries.items():
        prompt = registry.resolve_prompt(entry.call) if entry.call is not None else None
        if prompt is not None:
            texts[label] = prompt.text
    return texts


def relabel_dataset(
    source: Path,
    target: Path,
    labels: LabelMap,
    registry: SkillRegistry,
    *,
    copy_videos: bool = False,
) -> dict[str, Any]:
    source, target = source.resolve(), target.resolve()
    if target == source or source in target.parents:
        raise ValueError("write the relabeled copy outside the source dataset")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty; choose a new directory")
    texts = canonical_texts(labels, registry)
    seen: set[str] = set()

    def replacement(text: str) -> str | None:
        seen.add(text)
        label = normalize_label(text)
        if label not in labels.entries:
            raise KeyError(f"subtask label {text!r} is not in the label map; run `fmc3-harness data labels`")
        return texts.get(label)

    target.mkdir(parents=True, exist_ok=True)
    parquet_files = set(data_files(source))
    for entry in sorted(source.iterdir()):
        destination = target / entry.name
        if entry.name == "data":
            for path in sorted(p for p in entry.rglob("*") if p.is_file()):
                out = destination / path.relative_to(entry)
                out.parent.mkdir(parents=True, exist_ok=True)
                if path in parquet_files:
                    _rewrite(path, out, replacement)
                else:
                    shutil.copy2(path, out)
        elif entry.name == "videos" and not copy_videos:
            destination.symlink_to(entry, target_is_directory=True)
        elif entry.is_dir():
            shutil.copytree(entry, destination, symlinks=True)
        else:
            shutil.copy2(entry, destination)

    replaced = sorted(text for text in seen if normalize_label(text) in texts)
    kept = sorted(text for text in seen if normalize_label(text) not in texts)
    return {
        "files": len(parquet_files),
        "labels_replaced": len(replaced),
        "labels_kept": kept,
        "videos": "copied" if copy_videos else "linked",
    }


def _rewrite(path: Path, out: Path, replacement: Callable[[str], str | None]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    index = table.schema.get_field_index(LANGUAGE_PERSISTENT)
    if index < 0:
        raise ValueError(f"{path} has no {LANGUAGE_PERSISTENT} column; run lerobot-annotate first")
    column = table.column(index)
    chunks = [_replace_subtasks(chunk, replacement) for chunk in column.chunks]
    table = table.set_column(index, table.schema.field(index), pa.chunked_array(chunks, type=column.type))
    _write_by_episode(table, out)


def _replace_subtasks(chunk: Any, replacement: Callable[[str], str | None]) -> Any:
    """Rebuild one list<struct> chunk with new subtask contents; offsets and other fields are reused."""
    import pyarrow as pa
    import pyarrow.compute as pc

    values = chunk.values  # every row of every frame; chunk.offsets index into it
    struct_type = values.type
    if not pa.types.is_struct(struct_type):
        return chunk  # list<null>: a shard whose episodes have no rows at all
    content_index = struct_type.get_field_index("content")
    if content_index < 0 or struct_type.get_field_index("style") < 0:
        return chunk
    content = values.field(content_index)
    is_subtask = pc.fill_null(pc.equal(values.field("style"), SUBTASK_STYLE), False)
    replacements = {}
    for text in pc.unique(pc.filter(content, is_subtask)).to_pylist():
        if text is None:
            continue
        new_text = replacement(text)
        if new_text is not None and new_text != text:
            replacements[text] = new_text
    if not replacements:
        return chunk
    positions = pc.index_in(content, value_set=pa.array(list(replacements), type=content.type))
    replaced = pc.take(pa.array(list(replacements.values()), type=content.type), positions)
    new_content = pc.if_else(pc.and_(is_subtask, pc.is_valid(positions)), replaced, content)
    fields = [struct_type.field(i) for i in range(struct_type.num_fields)]
    children = [new_content if i == content_index else values.field(i) for i in range(len(fields))]
    struct = pa.StructArray.from_arrays(
        children, fields=fields, mask=values.is_null() if values.null_count else None
    )
    return type(chunk).from_arrays(
        chunk.offsets, struct, type=chunk.type, mask=chunk.is_null() if chunk.null_count else None
    )


def _write_by_episode(table: Any, path: Path) -> None:
    """One row group per episode, as LeRobot writes its shards, via a temp file."""
    import pyarrow.parquet as pq

    episode_ids = table.column("episode_index").to_pylist()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with pq.ParquetWriter(temporary, table.schema, compression="snappy", use_dictionary=True) as writer:
        start = 0
        for row in range(1, len(episode_ids) + 1):
            if row == len(episode_ids) or episode_ids[row] != episode_ids[start]:
                writer.write_table(table.slice(start, row - start))
                start = row
    temporary.replace(path)
