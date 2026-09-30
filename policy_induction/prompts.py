"""Prompts for the heuristic-generation LLM and the Jev question template.

The generation LLM writes investor-style heuristics, drawing on its own domain
knowledge as well as the labelled samples it is shown. Jev then applies each
heuristic to every sample through ``DEFAULT_JEV_TEMPLATE`` (or a task-specific
template), answering whether the case is positive when viewed through that
heuristic. The logistic regression learns how much to trust each heuristic.
"""

GEN_SYSTEM = """\
You write investor heuristics for predicting a binary outcome. Each heuristic
is one short sentence an experienced investor would use to judge a case, e.g.
"Founders who previously built and sold a company are more likely to succeed."
A separate model applies each heuristic to every case, and a logistic
regression learns how much to trust each one.

Good heuristics are:
- general: drawn from domain knowledge and true of many cases, not one sample
- focused: one signal each, grounded in the case's fields (e.g. `profile`)
- clean: no quotes, names or exact numbers taken from the samples

Return JSON: {"policies": [...]}
"""

SEED_PROMPT = """\
Task: {task}
Fields: {fields}

Write {n} diverse heuristics. Labelled examples:

YES:
{yes_block}

NO:
{no_block}
"""

BOOST_PROMPT = """\
Task: {task}
Fields: {fields}

Current heuristics (weight: + favours YES, - favours NO):
{rules_block}

The model gets these {label} cases wrong:
{hard_block}

and these {label} cases right:
{contrast_block}

Write {n} new heuristics that would catch the missed cases, generalising
beyond them. Do not repeat or reword the heuristics listed above.
"""

# How each heuristic is put to Jev, as one yes/no question per sample. Must
# contain {policy}; may contain {task}.
DEFAULT_JEV_TEMPLATE = (
    "Task: {task}\n"
    "Heuristic (guidance, not a strict rule): {policy}\n"
    "Considering this heuristic along with the full case, is the answer YES?"
)
