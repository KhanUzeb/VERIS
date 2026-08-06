from __future__ import annotations

from ..judge_model import GroqEvalModel


def _build_ragas_llm(judge: GroqEvalModel):
    """Adapt GroqEvalModel into something RAGAS accepts.

    Prefer LangchainLLMWrapper when langchain_core is available; otherwise
    fall back to a minimal BaseRagasLLM that delegates to judge.generate().
    Passing GroqEvalModel (a DeepEval wrapper) directly to evaluate() is not
    supported and fails at runtime."""
    try:
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
        from ragas.llms import LangchainLLMWrapper
        from typing import Any, Optional

        class _JudgeLangChainLLM(LLM):
            """Thin LangChain LLM around the shared judge."""

            @property
            def _llm_type(self) -> str:
                return "veris-judge"

            def _call(
                self,
                prompt: str,
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                return judge.generate(prompt)

        return LangchainLLMWrapper(_JudgeLangChainLLM())
    except ImportError:
        pass

    # Fallback: implement BaseRagasLLM directly.
    from ragas.llms.base import BaseRagasLLM
    from langchain_core.outputs import Generation, LLMResult

    class _JudgeRagasLLM(BaseRagasLLM):
        def generate_text(self, prompt, n: int = 1, temperature: float = 0.01,
                          stop=None, callbacks=None) -> LLMResult:
            text = judge.generate(prompt.to_string() if hasattr(prompt, "to_string") else str(prompt))
            return LLMResult(generations=[[Generation(text=text)] for _ in range(n)])

        async def agenerate_text(self, prompt, n: int = 1, temperature=0.01,
                                 stop=None, callbacks=None) -> LLMResult:
            return self.generate_text(prompt, n=n, temperature=temperature or 0.01,
                                      stop=stop, callbacks=callbacks)

        def is_finished(self, response: LLMResult) -> bool:
            try:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if not gen.text or not gen.text.strip():
                            return False
                return True
            except Exception:
                return False

    return _JudgeRagasLLM()


def score_ragas(case, actual_output: str) -> dict[str, float]:
    """RAGAS needs a HuggingFace Dataset even for a single row — that's
    the API. context_precision/recall are skipped if no retrieval context
    exists, since they're undefined without it.

    Imports are deliberately deferred to call time, not module load time.
    ragas pulls in a wide optional-dependency surface (langchain_community
    integrations for providers you'll never use) and a broken one there
    shouldn't take down the whole veris import — including
    deepeval-only workflows that never call this function."""
    if not case.retrieval_context:
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, context_precision, context_recall
    except ImportError as e:
        raise ImportError(
            f"ragas scoring requires the ragas + datasets packages and their "
            f"dependency chain to import cleanly: {e}"
        ) from e

    row = {
        "question": [case.input],
        "answer": [actual_output],
        "contexts": [case.retrieval_context],
        "ground_truth": [case.expected_output or ""],
    }
    ds = Dataset.from_dict(row)

    judge = GroqEvalModel()
    llm = _build_ragas_llm(judge)
    metrics = [context_precision, context_recall]
    if case.expected_output:
        metrics.append(answer_correctness)

    result = evaluate(ds, metrics=metrics, llm=llm)
    df = result.to_pandas()
    out: dict[str, float] = {}
    for m in ("context_precision", "context_recall", "answer_correctness"):
        if m in df.columns:
            val = df[m].iloc[0]
            out[m] = float(val) if val == val else 0.0  # NaN guard
    return out
