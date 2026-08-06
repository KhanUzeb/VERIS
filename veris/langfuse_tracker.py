"""Pushes per-case traces to Langfuse so failures/regressions can be
drilled into beyond what the terminal report shows. Degrades gracefully —
if no key pair is set, this becomes a no-op rather than a hard failure,
so the harness still runs locally without a Langfuse account.

Compatible with Langfuse Python SDK v3/v4 (`start_as_current_observation`)
with a fallback to the legacy `trace()` API for older installs.

Langfuse traces are scored using its native `score` concept (0-1, one per
metric) rather than stuffing everything into free-text outputs, so the
per-case DeepEval/RAGAS scores show up as first-class filterable/sortable
columns in the Langfuse UI, not buried in a JSON blob."""
from __future__ import annotations

from .config import SETTINGS
from .schemas import RunReport


class LangfuseTracker:
    def __init__(self):
        self.enabled = SETTINGS.langfuse_enabled
        self._client = None
        if self.enabled:
            try:
                from langfuse import Langfuse
                # v3/v4 use base_url; older SDKs used host — try both.
                try:
                    self._client = Langfuse(
                        public_key=SETTINGS.langfuse_public_key,
                        secret_key=SETTINGS.langfuse_secret_key,
                        base_url=SETTINGS.langfuse_host,
                    )
                except TypeError:
                    self._client = Langfuse(
                        public_key=SETTINGS.langfuse_public_key,
                        secret_key=SETTINGS.langfuse_secret_key,
                        host=SETTINGS.langfuse_host,
                    )
            except Exception:
                self.enabled = False

    def log_run(self, report: RunReport) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            for cr in report.case_results:
                tags = ["veris", report.run_id]
                if cr.regressed:
                    tags.append("regressed")
                if not cr.passed_min_score:
                    tags.append("failed")

                metadata = {
                    "run_id": report.run_id,
                    "git_sha": report.git_sha,
                    "regression_detail": cr.regression_detail,
                    "latency_ms": cr.latency_ms,
                    "tags": tags,
                }
                input_payload = {"input": cr.input}
                output_payload = {"actual_output": cr.actual_output}

                if hasattr(self._client, "start_as_current_observation"):
                    with self._client.start_as_current_observation(
                        name=f"eval:{cr.case_id}",
                        as_type="span",
                        input=input_payload,
                        metadata=metadata,
                    ) as span:
                        span.update(output=output_payload)
                        for metric_name, score in cr.scores.items():
                            span.score(name=metric_name, value=score, data_type="NUMERIC")
                else:
                    # Legacy SDK (v2): client.trace(...) / trace.score(...)
                    trace = self._client.trace(
                        name=f"eval:{cr.case_id}",
                        input=input_payload,
                        output=output_payload,
                        metadata=metadata,
                        tags=tags,
                    )
                    for metric_name, score in cr.scores.items():
                        trace.score(name=metric_name, value=score)

            self._client.flush()
        except Exception as e:
            # Tracing is advisory — never let a Langfuse outage fail the gate.
            print(f"[langfuse] trace push failed, continuing: {e}")
