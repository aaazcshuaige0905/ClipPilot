from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


TJob = TypeVar("TJob")
TResult = TypeVar("TResult")


@dataclass(frozen=True)
class ParallelJobResult(Generic[TJob, TResult]):
    job: TJob
    result: TResult | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.result is not None


def run_bounded_jobs(
    jobs: list[TJob],
    worker: Callable[[TJob], TResult],
    *,
    max_workers: int,
) -> list[ParallelJobResult[TJob, TResult]]:
    """Execute independent blocking jobs with bounded concurrency and isolated failures."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if not jobs:
        return []

    completed: list[ParallelJobResult[TJob, TResult]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        future_to_job = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                completed.append(ParallelJobResult(job=job, result=future.result()))
            except Exception as exc:  # isolated job failure is returned to the orchestrator
                completed.append(ParallelJobResult(job=job, error=exc))
    return completed
