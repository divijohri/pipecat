#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A judge backed by TypeSafe's Jev, a decision model with typed answers.

Jev is a classifier, not an LLM. You give it some state (here, the
conversation) and a question with a fixed set of answers, and it gives back
a probability for each answer. It can never answer outside those options,
and it is fast (a few hundred milliseconds) and cheap, but it can't say why
it chose an answer. The questions go through
:class:`~pipecat.classifiers.jev.JevClassifier`, and this module wraps that
in the same interface as the LLM judge,
:class:`~pipecat.evals.judge.EvalJudge`, so the harness can't tell them
apart.

When it's used:

Only when a scenario asks for it: :func:`~pipecat.evals.base_judge.judge_from_config`
builds this judge for ``service: typesafe`` in the ``judge.eval:`` block.
Any other block, or none, gets the LLM judge. All the options::

    judge:
      eval:
        service: typesafe
        model: jev-latest        # optional; the Jev model
        endpoint: https://...    # optional; the API's base URL
        explainer:               # optional; the LLM judge that gives reasons
          service: ollama        #   (the default LLM judge if omitted,
          model: gemma4:12b      #   none if set to false)
        explain_below: 0.75      # optional; see "The explainer" below
        allow_continue: true     # optional; false judges a reply yes or no only

The API key comes from ``TYPESAFE_API_KEY``. Because the explainer is on by
default, a Jev judge still needs the explainer's model running (Ollama, for
the default) unless the block sets ``explainer: false``.

What Jev is asked:

The harness asks three kinds of question, and each is a Jev request:

- A reply, for an ``eval:`` on a scripted turn. The state holds the
  conversation so far and, separately, the bot's latest reply. The answers
  are ``yes`` (the reply meets the criterion), ``no`` (it's a real answer
  that doesn't, including a reply that waits for the user instead), and
  ``continue`` (the bot is still working toward its answer: it only
  greeted, said it's checking, or the reply is still arriving). On
  ``continue`` the harness waits for more of the reply and asks again. A
  suite whose judged replies are all final answers, with nothing for the bot
  to fetch first, sets ``allow_continue: false``, and a reply is then yes or
  no; a reply that is still arriving is still judged again as more of it
  comes.
- A function call, for an ``eval:`` on a ``function_call``. The state holds
  the call's name and arguments and the conversation as context. The
  answer is yes or no.
- A whole simulation, at the end of the run. The goal is one yes/no
  question over the whole conversation, tool calls included. Each bot turn
  is asked about on its own, once per criterion, with only the conversation
  before that turn as context. The answers are ``meets``, ``fails``, and
  ``not_applicable`` (the criterion only covers some situation, like "when
  the time is taken, apologise", and this turn isn't in it); only ``fails``
  fails the turn. All of a run's questions go out at once.

How each question to Jev is written:

- The reply or function call being judged is sent separately from the
  conversation before it (``latest_bot_reply`` or ``call``), so Jev knows
  which part to judge.
- Every possible answer is one of Jev's options. "The bot hasn't answered
  yet" is the option ``continue``, and "this criterion doesn't apply to this
  turn" is the option ``not_applicable``. Don't write these as rules in the
  instructions instead: Jev follows an instruction for every criterion, even
  where it doesn't fit. A rule like "a criterion about some situation passes
  when that situation doesn't come up" also passes turns that never state a
  price against "the reply states a price".
- The instructions hold only the question and the criterion.

How answers become verdicts:

Each verdict carries a confidence from 0 to 1, and its reason starts out as
Jev's probabilities (for example ``Jev: P(yes)=0.97``):

- A reply's verdict is Jev's chosen answer, with Jev's own confidence in it.
- A yes/no question (a function call, a goal) is ``yes`` when Jev's
  probability of yes is at least 0.5; its confidence is the probability of
  the answer given.
- A turn's confidence is the probability of the verdict given: for a pass,
  the probability of ``meets`` and ``not_applicable`` together.

Each question's verdict is cached by its criterion and state, so asking the
same question about the same conversation twice costs one request.

The explainer:

Jev gives no reasons, so the explainer, an LLM judge sharing this judge's
conversation, supplies them. It's asked for:

- every ``no``, on a reply, a function call, a turn or a goal;
- every verdict Jev is unsure of, meaning a confidence below
  ``explain_below`` (0.75 by default). When a verdict is close to 50/50, Jev
  can give a different answer on another run, so these are worth a second
  look;
- never a ``continue``, which is only a request to wait for more text.

``explain_below: 0`` limits it to the ``no`` verdicts, and a value above 1
sends it every verdict (useful for comparing the two judges).

The explainer is asked the same question and makes its own judgement, which
costs a full LLM call. For a simulation, it judges the whole run once if any
verdict needs a reason, and its reasons go only to the verdicts that need
one. Either way, the explainer never changes a verdict: Jev's stands. If
the explainer agrees, its reason is used, followed by Jev's probabilities.
If it disagrees, the reason says so ("the explainer judged yes: ..."). If
it agrees without giving a reason, Jev's probabilities are the reason.

When a request fails:

Jev's client retries a question Jev was too busy to answer. A question that
times out (2.5 s by default), can't connect, or is refused is asked once
more. A second failure, or an answer that isn't of the type asked for, fails
the question: a reply or a function call gets a ``no`` with the reason
"judge call failed", and each verdict of a failed simulation question is a
``none`` (the verdict the harness reports as not given).

The connection:

Questions go over one HTTP/2 connection, which the judge opens as soon as
it's created (with a request that runs no model) and keeps open between
questions, so a question doesn't wait for a new connection and a
simulation's questions share one. The session closes the judge when the run
ends, which closes the connection. A judge given a classifier of its own
uses that classifier's connection and leaves it alone.
"""

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from loguru import logger

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.jev import JevClassifier
from pipecat.classifiers.jev_client import JevClient
from pipecat.evals.base_judge import BaseEvalJudge, JudgeVerdict, RunVerdicts
from pipecat.evals.judge import NO_REASON, EvalJudge
from pipecat.evals.services import DEFAULT_JEV_MODEL

DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"

_R = TypeVar("_R")

_TRANSCRIPTION_NOTE = (
    "The bot's text may be an automatic speech-to-text transcription: judge its intended "
    "spoken meaning, never its spelling ('for' may mean 'four', 'to' may mean 'two')."
)

_REPLY_OUTCOMES = {
    "yes": "The bot has given its answer, and the answer satisfies the criterion.",
    "no": (
        "The bot has given its answer, and the answer does not satisfy the criterion. A reply "
        "that waits for the user (asking them to take their time or to go on) instead of "
        "giving what the criterion asks for is a no."
    ),
    "continue": (
        "The bot is still working toward its answer: it only greets, says it is checking or "
        "looking something up and will report back, or the reply is an obviously incomplete "
        "fragment."
    ),
}

# A run's per-turn outcomes. Jev can't tell by itself whether a criterion only
# applies in some situation ("when the time is taken, apologises"), so "doesn't
# apply" is an outcome of its own rather than a rule in the instructions.
_TURN_OUTCOMES = {
    "meets": "The reply does what the criterion asks.",
    "fails": "The criterion applies to this reply, and the reply does not do what it asks.",
    "not_applicable": (
        "The criterion only asks something of replies in a particular situation (it says "
        "'when', 'if', or similar), and that situation does not arise in this reply."
    ),
}


class JevEvalJudge(BaseEvalJudge):
    """Judges with Jev, and asks an LLM judge to explain the verdicts that need a reason.

    Args:
        api_key: The TypeSafe API key; ``TYPESAFE_API_KEY`` if omitted.
        model: The Jev model.
        base_url: The TypeSafe API's base URL.
        client: The HTTP client to send requests with; one is created if
            omitted, and closed with the judge.
        timeout: Seconds to wait for each answer. A request that times out,
            fails to connect, or gets a server error is retried once, at once,
            on a new connection.
        explainer: An LLM judge asked for the reason behind a ``no`` or an
            unsure verdict, or ``None`` to report Jev's probabilities alone.
        explain_below: A ``yes`` less sure than this is explained too.
        allow_continue: Whether a reply may be judged ``continue``; when
            ``False``, a reply is ``yes`` or ``no``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_JEV_MODEL,
        base_url: str = DEFAULT_JEV_BASE_URL,
        classifier: JevClassifier | None = None,
        timeout: float = 2.5,
        explainer: EvalJudge | None = None,
        explain_below: float = 0.75,
        allow_continue: bool = True,
    ):
        """Initialize the judge.

        Args:
            api_key: The TypeSafe API key; ``TYPESAFE_API_KEY`` if omitted.
            model: The Jev model.
            base_url: The TypeSafe API's base URL.
            classifier: The Jev classifier to ask; one is created if omitted,
                and cleaned up with the judge.
            timeout: Seconds to wait for each answer. A question that times
                out or fails to connect is asked once more.
            explainer: An LLM judge asked for the reason behind a ``no`` or an
                unsure verdict, or ``None`` to report Jev's probabilities alone.
            explain_below: A ``yes`` less sure than this is explained too.
            allow_continue: Whether a reply may be judged ``continue``; when
                ``False``, a reply is ``yes`` or ``no``.

        Raises:
            ValueError: If the judge makes its own classifier and there is no
                API key.
        """
        super().__init__(allow_continue=allow_continue)
        self._owns_classifier = classifier is None
        if classifier is None:
            api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
            if not api_key:
                raise ValueError("The Jev judge needs an API key: set TYPESAFE_API_KEY.")
            classifier = self._new_classifier(api_key, model, base_url.rstrip("/"), timeout)
        self._classifier = classifier
        self._warm_task: asyncio.Task | None = None
        if self._owns_classifier:
            try:
                self._warm_task = asyncio.get_running_loop().create_task(self._warm())
            except RuntimeError:
                pass  # no running loop: the first question opens the connection
        # The explainer judges the same conversation, so it shares this one.
        if explainer is not None:
            explainer._transcript = self._transcript
        self._explainer = explainer
        self._explain_below = explain_below
        self._reply_outcomes = (
            _REPLY_OUTCOMES
            if self._allow_continue
            else {k: v for k, v in _REPLY_OUTCOMES.items() if k != "continue"}
        )
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}

    @classmethod
    def from_config(cls, judge_config: dict | None) -> "JevEvalJudge":
        """Build a Jev judge from a scenario's ``judge.eval:`` block.

        Args:
            judge_config: Mapping with ``service: typesafe`` and optional keys
                ``model``, ``endpoint`` (the API's base URL), ``explainer``
                (an LLM judge block, as ``judge.eval:`` takes; the default LLM
                judge when omitted, none when ``false``), ``explain_below`` and
                ``allow_continue``.

        Returns:
            A configured JevEvalJudge.
        """
        config = judge_config or {}
        allow_continue = config.get("allow_continue", True) is not False
        explainer_config = config.get("explainer", {})
        # The explainer is asked the same questions, so it follows the same rule.
        explainer = (
            None
            if explainer_config is False
            else EvalJudge.from_config(
                {**(explainer_config or {}), "allow_continue": allow_continue}
            )
        )
        return cls(
            model=config.get("model") or DEFAULT_JEV_MODEL,
            base_url=config.get("endpoint") or DEFAULT_JEV_BASE_URL,
            explainer=explainer,
            explain_below=float(config.get("explain_below", 0.75)),
            allow_continue=allow_continue,
        )

    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, in the conversation so far.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            Jev's ``yes``, ``no`` or, unless ``allow_continue`` is off,
            ``continue``, cached by criterion and conversation. A final
            verdict that needs a reason is explained.
        """
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        latest = entries.pop()["content"] if entries and entries[-1]["role"] == "bot" else ""
        state = {"conversation": _conversation(entries), "latest_bot_reply": latest}
        key = _cache_key("reply", criterion, state)
        if key not in self._cache:
            answers = await self._choices(
                state,
                {
                    "verdict": (
                        "Does `latest_bot_reply`, the bot's most recent reply, following "
                        f"`conversation`, satisfy this criterion? Criterion: "
                        f"{_sentence(criterion)} {_TRANSCRIPTION_NOTE}",
                        self._reply_outcomes,
                    )
                },
            )
            if answers is None:
                verdict = _failed("no")
            else:
                verdict = _reply_verdict(answers["verdict"])
                if verdict.verdict != "continue":
                    verdict = await self._explain(verdict, lambda j: j.evaluate(criterion))
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_call(self, name: str, args: dict | None, criterion: str) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, in the conversation so far.

        Args:
            name: The function's name.
            args: The call's arguments.
            criterion: Natural-language description of what the call should be.

        Returns:
            Jev's ``yes`` or ``no``, cached by call, criterion and
            conversation, and explained when it needs a reason.
        """
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        state = {
            "conversation": _conversation(entries),
            "call": {"name": name, "arguments": args or {}},
        }
        key = _cache_key("call", criterion, state)
        if key not in self._cache:
            answer = await self._yes_no(
                state,
                "Does the bot's function `call`, judged by its name and arguments, satisfy "
                f"this criterion? Criterion: {_sentence(criterion)} `conversation` is context "
                f"only. {_TRANSCRIPTION_NOTE}",
            )
            if answer is None:
                verdict = _failed("no")
            else:
                verdict = await self._explain(
                    _yes_no_verdict(answer),
                    lambda j: j.evaluate_call(name, args, criterion),
                )
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_run(self, criteria: dict[str, str], success: str) -> RunVerdicts:
        """Judge the whole conversation: every bot turn on every criterion, and the goal.

        The goal is asked over the whole conversation. Each bot turn is its
        own request, judged in the light of the conversation before it, with
        the turn itself as ``latest_bot_reply``; the requests run
        concurrently. When any verdict needs a reason, the explainer judges
        the run once and its reasons go with Jev's verdicts.

        Args:
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order. A failed request is a ``none`` for each verdict it asked for.
        """
        entries = _numbered_turns(self._transcript)
        names = list(criteria)
        key = _cache_key("run", criteria, success, entries)
        if key in self._run_cache:
            return self._run_cache[key]

        goal_instructions = (
            f"Does the conversation as a whole achieve this goal? Goal: {_sentence(success)} "
            "A `tool` entry is a function the bot called at that point; a completed "
            "call is stronger evidence of an action than the bot saying it did it. "
            f"{_TRANSCRIPTION_NOTE}"
        )
        turn_questions = {
            name: (
                "`latest_bot_reply` is the bot's reply following `conversation`. How does it "
                f"stand against this criterion? Criterion: {_sentence(criteria[name])} "
                f"{_TRANSCRIPTION_NOTE}",
                _TURN_OUTCOMES,
            )
            for name in names
        }
        turn_states = [
            {"conversation": _conversation(entries[:i]), "latest_bot_reply": entry["content"]}
            for i, entry in enumerate(entries)
            if entry["role"] == "bot"
        ]

        # The goal, and every criterion on each turn: one request each, all at once.
        goal_answer, turn_answers = await asyncio.gather(
            self._yes_no({"conversation": _conversation(entries)}, goal_instructions),
            asyncio.gather(
                *(
                    self._choices(state, turn_questions)
                    for state in (turn_states if turn_questions else [])
                )
            ),
        )

        failed = _failed("none")
        verdicts = RunVerdicts(
            goal=_yes_no_verdict(goal_answer) if goal_answer else failed,
            turns={
                name: [
                    _turn_verdict(answers[name]) if answers else failed for answers in turn_answers
                ]
                for name in names
            },
        )
        verdicts = await self._explain_run(verdicts, criteria, success)
        self._run_cache[key] = verdicts
        return verdicts

    async def close(self) -> None:
        """Clean up the classifier the judge created."""
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        if self._owns_classifier:
            await self._classifier.cleanup()
            # The judge gave the classifier this client, so it leaves it open.
            await self._classifier.client.close()

    def _new_classifier(
        self, api_key: str, model: str, base_url: str, timeout: float
    ) -> JevClassifier:
        """The judge's own classifier, over a connection of its own."""
        return JevClassifier(
            client=JevClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
        )

    async def _warm(self) -> None:
        """Open the connection ahead of the first question."""
        try:
            await self._classifier.client.connect()
        except ClassifierError as e:
            logger.debug(f"Jev judge couldn't open its connection early: {e}")

    def _needs_reason(self, verdict: JudgeVerdict) -> bool:
        """Whether a verdict is worth explaining: a ``no``, or an unsure ``yes``."""
        unsure = verdict.confidence is not None and verdict.confidence < self._explain_below
        return verdict.verdict == "no" or unsure

    async def _explain(
        self,
        verdict: JudgeVerdict,
        ask: Callable[[EvalJudge], Awaitable[JudgeVerdict]],
    ) -> JudgeVerdict:
        """The verdict with the explainer's reason for it, when it needs one and there is an explainer."""
        if self._explainer is None or not self._needs_reason(verdict):
            return verdict
        return _explained(verdict, await ask(self._explainer))

    async def _explain_run(
        self, verdicts: RunVerdicts, criteria: dict[str, str], success: str
    ) -> RunVerdicts:
        """The run's verdicts, those that need a reason given the explainer's; it judges the run once."""
        everything = [verdicts.goal, *(v for vs in verdicts.turns.values() for v in vs)]
        if self._explainer is None or not any(self._needs_reason(v) for v in everything):
            return verdicts
        explanation = await self._explainer.evaluate_run(criteria, success)

        def explain(verdict: JudgeVerdict, reasoned: JudgeVerdict | None) -> JudgeVerdict:
            if reasoned is None or reasoned.verdict == "none" or not self._needs_reason(verdict):
                return verdict
            return _explained(verdict, reasoned)

        turns: dict[str, list[JudgeVerdict]] = {}
        for name, turn_verdicts in verdicts.turns.items():
            reasoned = explanation.turns.get(name, [])
            turns[name] = [
                explain(verdict, reasoned[i] if i < len(reasoned) else None)
                for i, verdict in enumerate(turn_verdicts)
            ]
        return RunVerdicts(goal=explain(verdicts.goal, explanation.goal), turns=turns)

    async def _yes_no(self, state, instructions: str) -> YesNoResult | None:
        """Jev's probability of yes, or ``None`` when the question failed."""
        answers = await self._ask(
            state,
            lambda: self._classifier.yes_no(
                state, {"answer": YesNoQuestion(instructions=instructions)}
            ),
        )
        return answers["answer"] if answers else None

    async def _choices(
        self, state, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceResult] | None:
        """Jev's chosen option for each question, or ``None`` when they failed.

        The questions share the state, so they go in one request.
        """
        return await self._ask(
            state,
            lambda: self._classifier.choice(
                state,
                {
                    name: ChoiceQuestion(instructions=instructions, options=dict(options))
                    for name, (instructions, options) in questions.items()
                },
            ),
        )

    async def _ask(self, state, ask: Callable[[], Awaitable[_R]]) -> "_R | None":
        """The classifier's answer, asked once more if it failed, or ``None``.

        Jev's client retries a busy answer itself; this covers a question that
        timed out or couldn't connect, which it reports as a failure.
        """
        if self._warm_task is not None and not self._warm_task.done():
            await asyncio.shield(self._warm_task)
        logger.debug(f"Jev judge asking over state:\n{json.dumps(state)}")
        for attempt in (1, 2):
            try:
                return await ask()
            except ClassifierError as e:
                if attempt == 1:
                    logger.warning(f"Jev judge question failed, asking again: {e}")
                else:
                    logger.error(f"Jev judge question failed: {e}")
        return None


def _numbered_turns(transcript: Iterable[dict]) -> list[dict]:
    """The conversation with each bot reply joined into one numbered turn.

    A bot turn is a run of reply segments with nothing else between them.

    Args:
        transcript: The judge's conversation entries.

    Returns:
        The entries, each a dict with a ``role`` of ``user``, ``bot`` or
        ``tool`` and the ``content``, a bot entry also carrying its ``turn``
        (from 1).
    """
    entries: list[dict] = []
    turn = 0
    for entry in transcript:
        if entry["role"] == "assistant":
            if entries and entries[-1]["role"] == "bot":
                entries[-1]["content"] += f" {entry['content']}"
                continue
            turn += 1
            entries.append({"role": "bot", "turn": turn, "content": entry["content"]})
        else:
            entries.append({"role": entry["role"], "content": entry["content"]})
    return entries


def _cache_key(*parts) -> str:
    """Hash a question's parts (its kind, criterion, and state) for cache lookup."""
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _conversation(entries: list[dict]) -> list[dict]:
    """The conversation as Jev's state carries it: a speaker, the text, and a bot turn's number."""
    return [
        {"speaker": e["role"], **({"turn": e["turn"]} if "turn" in e else {}), "text": e["content"]}
        for e in entries
    ]


def _reply_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A reply's verdict: Jev's choice of ``yes``, ``no`` or ``continue``."""
    return JudgeVerdict(
        verdict=answer.label,
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=answer.confidence,
    )


def _yes_no_verdict(answer: YesNoResult) -> JudgeVerdict:
    """A yes/no verdict from Jev's probability of yes."""
    probability = answer.probability
    return JudgeVerdict(
        verdict="yes" if probability >= 0.5 else "no",
        reason=f"Jev: P(yes)={probability:.2f}",
        raw_response=answer.model_dump_json(),
        confidence=max(probability, 1 - probability),
    )


def _turn_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A turn's verdict: a ``no`` only when the criterion applies to the turn and fails.

    Its confidence is the probability of the verdict given: for a pass, the
    probability of either outcome that passes.
    """
    fails = answer.probabilities.get("fails", 0.0)
    passed = answer.label != "fails"
    return JudgeVerdict(
        verdict="yes" if passed else "no",
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=1 - fails if passed else fails,
    )


def _sentence(text: str) -> str:
    """``text`` ending in punctuation, so the instruction after it reads as a new sentence."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _probabilities(probabilities: dict[str, float]) -> str:
    """Jev's probability for each outcome, as a verdict's reason."""
    return "Jev: " + ", ".join(f"P({k})={v:.2f}" for k, v in probabilities.items())


def _failed(verdict: str) -> JudgeVerdict:
    """The verdict a failed Jev request gives."""
    return JudgeVerdict(verdict=verdict, reason="judge call failed", raw_response="")


def _explained(verdict: JudgeVerdict, explanation: JudgeVerdict) -> JudgeVerdict:
    """Jev's verdict with the explainer's reason, noting when the explainer disagreed."""
    if explanation.verdict == verdict.verdict:
        given = explanation.reason and explanation.reason != NO_REASON
        reason = f"{explanation.reason} ({verdict.reason})" if given else verdict.reason
    else:
        reason = (
            f"{verdict.reason}; the explainer judged {explanation.verdict}: {explanation.reason}"
        )
    return JudgeVerdict(
        verdict=verdict.verdict,
        reason=reason,
        raw_response=verdict.raw_response,
        confidence=verdict.confidence,
    )
