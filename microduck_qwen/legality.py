"""Action legality: reject a decoded skill that contradicts the skill contract, then fall back to the best legal skill.

Harness layer between the model and the executor. The rules come from the contract the model is trained and prompted
with (`agent.agent_prompt` and the state classes of the training data), never from the expected label of a task:

  * `done` means "everything finished", so it needs at least one verified skill.
  * `return_home` needs a recorded home; without one the contract says `clarify`.
  * `clarify` means "the referent or parameter is missing". In the training data it appears only at the start and only
    for instructions without a target ("去那边", "Go over there"). With a home recorded it is therefore illegal after a
    verified skill, and illegal at the start when the instruction names a supported target colour.
  * The prompt forbids repeating a verified skill: no second explore/search for the colour explore just found, no second
    approach of the same colour, no second return_home/bypass_return.

Deliberately not rules: `stop` vs `done` after an approach (both are legal terminals; which one the instruction asks for
is a semantic question, not a contract violation) and `unsupported` after move/turn (legitimate when the instruction
still contains an unimplemented skill; the executor feedback line addresses that slip).

Termination rules (`MICRODUCK_TERMINATION=1`, separate flag, checked alone or together with the rules above): in the
lang-v2 training data a verified return_home/bypass_return is followed by `done` in 455 of 455 states, and a verified move/turn
is never followed by the identical move/turn (turn -> a different turn occurs 102 times). So after a verified return only
`done` or `unsupported` (an unimplemented remainder) is legal, and repeating the identical verified move/turn is illegal.
The frozen base model breaks exactly these in 47 of its 61 failures on seeds 99200/99300 (run `frozen`).

When a decision is illegal or unparseable, every action type is scored by the model's own log-probability of the prefix
`{"action":"<type>` and the output is regenerated with the best-scoring legal prefix forced. The choice is the model's,
restricted to the legal set; no rule picks the replacement directly.
"""

import os
import re

from .skills import parse_action

ENABLED = os.environ.get("MICRODUCK_LEGALITY", "0") != "0"
TERMINATION = os.environ.get("MICRODUCK_TERMINATION", "0") != "0"
ACTIVE = ENABLED or TERMINATION
ACTIONS = ("move", "turn", "approach", "search", "explore", "return_home", "stop", "done", "clarify", "unsupported",
           "bypass_return")
COLOUR = re.compile(r"红|绿|蓝|黄|red|green|blue|yellow", re.IGNORECASE)
MAX_ATTEMPTS = 3


def prefix(name):
    """The start of the model's output for an action type, exactly as the adapter writes it (compact JSON)."""
    return '{"action":"' + name


def violation(action, instruction, memory):
    """Why `action` breaks the skill contract in this state, or None when it is legal."""
    name = action["action"]
    completed = memory.get("completed") or []
    last = completed[-1] if completed else None
    home = bool(memory.get("home_recorded"))
    if name == "done" and last is None:
        return "done before any skill was verified"
    if name == "return_home" and not home:
        return "return_home without a recorded home (contract: clarify)"
    if name == "clarify" and home:
        if last is not None:
            return "clarify after a verified skill"
        if COLOUR.search(instruction):
            return "clarify although the instruction names a target colour"
    if last is not None:
        same_colour = action.get("color") is not None and action.get("color") == last.get("color")
        if last["action"] == "explore" and name in ("explore", "search") and same_colour:
            return "search again for the colour explore already found"
        if last["action"] == "approach" and name == "approach" and same_colour:
            return "repeat the verified approach"
        if last["action"] in ("return_home", "bypass_return") and name in ("return_home", "bypass_return"):
            return "repeat the verified return"
    return None


def termination_violation(action, instruction, memory):
    """Why `action` fails to end a finished task (termination rules), or None."""
    completed = memory.get("completed") or []
    if not completed:
        return None
    last, name = completed[-1], action["action"]
    if last["action"] in ("return_home", "bypass_return") and name not in ("done", "unsupported"):
        return "only done after a verified return"
    if last["action"] in ("move", "turn") and action == last:
        return "repeat the identical verified move/turn"
    return None


def active_check(action, instruction, memory):
    """The rule sets switched on by the environment flags, in order."""
    for on, check in ((ENABLED, violation), (TERMINATION, termination_violation)):
        reason = check(action, instruction, memory) if on else None
        if reason is not None:
            return reason
    return None


def type_legal(name, instruction, memory, check=violation):
    """Whether some action of this type can be legal here (colour-specific rules are checked on the regenerated action)."""
    probe = {"action": name}
    if name in ("approach", "search", "explore"):
        probe["color"] = None
    return check(probe, instruction, memory) is None


def enforce(result, instruction, memory, score, regenerate, check=violation):
    """Return `result` unchanged when its action is legal; otherwise the best legal regeneration with a `legality` record.

    `score(names) -> {name: log-probability of prefix(name)}` and `regenerate(name) -> raw text starting with prefix(name)`
    are supplied by the planner, so this function stays model-free and testable.
    """
    action = result.get("action")
    reason = ("unparseable: " + str(result.get("error"))) if action is None else check(action, instruction, memory)
    if reason is None:
        return result
    scores = score(ACTIONS)
    original = action["action"] if action else None
    record = {"violation": reason, "original_raw": result.get("raw"), "original_action": action, "scores": scores,
              "top_scored": max(scores, key=scores.get), "attempts": []}
    if original is not None:
        record["top_scored_is_original"] = record["top_scored"] == original
    legal = sorted((name for name in ACTIONS if type_legal(name, instruction, memory, check)), key=scores.get, reverse=True)
    fixed = {key: value for key, value in result.items() if key not in ("action", "error", "raw")}
    fixed["raw"], fixed["error"] = result.get("raw"), "no legal regeneration: " + reason
    for name in legal[:MAX_ATTEMPTS]:
        raw = regenerate(name)
        attempt = {"name": name, "raw": raw}
        record["attempts"].append(attempt)
        try:
            candidate = parse_action(raw)
        except (ValueError, TypeError, KeyError) as error:
            attempt["error"] = str(error)
            continue
        problem = check(candidate, instruction, memory) if candidate["action"] == name else "forced prefix not kept"
        if problem is not None:
            attempt["error"] = problem
            continue
        fixed.pop("error")
        fixed["raw"], fixed["action"] = raw, candidate
        record["chosen"] = name
        break
    fixed["legality"] = record
    return fixed
