"""Planner steps: annotated decisions mapped onto skill calls, as the harness would have made them."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.types import SkillCall
from .annotations import AnnotatedEpisode
from .labels import LabelMap


@dataclass(frozen=True)
class Step:
    episode_index: int
    index: int
    frame: int
    t: float
    call: SkillCall | None  # None: the task is done
    labels: tuple[str, ...]  # the annotated spans this step covers
    duration_s: float
    memory_in: str  # what the planner wrote at its previous step; empty at the first
    memory_out: str

    def target(self) -> dict[str, object]:
        """The decision the planner should output, in the form parse_decision() returns."""
        if self.call is None:
            return {"done": True, "memory": self.memory_out}
        return {"skill": self.call.skill, "args": dict(self.call.args), "memory": self.memory_out}


@dataclass
class StepStats:
    episodes: int = 0
    unannotated: int = 0  # episodes without subtask rows
    spans: int = 0
    skipped: int = 0  # spans whose label maps to null
    merged: int = 0  # spans folded into the previous step because they map to the same call


def episode_steps(episode: AnnotatedEpisode, labels: LabelMap, stats: StepStats | None = None) -> list[Step]:
    """Map one episode's decisions to planner steps.

    Consecutive spans that map to the same call become one step, since the harness runs a
    call until its verifier fires. Memory is chained the way the harness chains it: each step
    receives the memory written at the previous emitted step.
    """
    stats = stats if stats is not None else StepStats()
    stats.episodes += 1
    if not episode.decisions:
        stats.unannotated += 1
        return []
    steps: list[Step] = []
    memory = ""
    skipped_since_last = False
    for decision in episode.decisions:
        if decision.label is None:
            done = Step(
                episode.episode_index,
                len(steps),
                decision.frame,
                decision.t,
                None,
                (),
                0.0,
                memory,
                decision.memory,
            )
            steps.append(done)
            break
        stats.spans += 1
        call = labels.lookup(decision.label).call
        if call is None:
            stats.skipped += 1
            skipped_since_last = True
            continue
        if steps and not skipped_since_last and steps[-1].call == call:
            last = steps[-1]
            steps[-1] = replace(
                last, labels=(*last.labels, decision.label), duration_s=last.duration_s + decision.duration_s
            )
            stats.merged += 1
            continue
        steps.append(
            Step(
                episode_index=episode.episode_index,
                index=len(steps),
                frame=decision.frame,
                t=decision.t,
                call=call,
                labels=(decision.label,),
                duration_s=decision.duration_s,
                memory_in=memory,
                memory_out=decision.memory,
            )
        )
        memory = decision.memory
        skipped_since_last = False
    return steps
