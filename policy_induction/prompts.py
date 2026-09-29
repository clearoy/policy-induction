"""Prompts for the rule-generation LLM.

Rules are judged by Jev, which reads each rule literally, one at a time, and is
weak at arithmetic, multi-hop logic and compound conditions. The writing
constraints below are therefore not style advice: a rule that breaks them is
scored unreliably. The same constraints also make rules generalise, because a
single observable condition cannot memorise a particular training row.
"""

RULE_WRITING_RULES = """\
Each rule is a single statement that is either true or false of ONE sample.

Write every rule so that:
1. It describes exactly one observable condition. No "and", "or", "unless",
   or nested conditions. If you want two conditions, write two rules.
2. It states a condition, NOT a verdict. Never write "then YES", "likely to
   succeed", or anything about the label. The model learns from the data
   whether a condition points towards YES or NO.
3. It refers to the sample's fields by name in backticks, e.g. `description`.
4. It can be judged from the sample's own content in a second by a
   knowledgeable reader. Avoid arithmetic, counting, date comparison and
   precise numeric thresholds.
5. It describes a general pattern. Never quote a sample, and never name a
   specific person, company, place or number that appears in a sample.
6. It is phrased positively, so that "true" means the condition is present.
"""

GEN_SYSTEM = f"""\
You design features for an interpretable binary classifier. Each feature is a
natural-language rule. A separate model judges, for every sample, the
probability that the rule is true; a logistic regression then learns how much
each rule matters.

{RULE_WRITING_RULES}
Return only the rules, as a JSON object {{"rules": [..]}}.
"""

SEED_PROMPT = """\
TASK:
{task}

SAMPLE FIELDS: {fields}

Below are labelled samples. Propose {n} diverse rules capturing the signals
that distinguish YES samples from NO samples. Cover different aspects of the
samples rather than several wordings of one idea.

YES SAMPLES:
{yes_block}

NO SAMPLES:
{no_block}
"""

BOOST_PROMPT = """\
TASK:
{task}

SAMPLE FIELDS: {fields}

The current model uses the rules below. For each rule: its weight (positive
pushes towards YES, negative towards NO) and how often it holds.

CURRENT RULES:
{rules_block}

The model gets the following {label} samples WRONG: it gives them a low
probability of being {label}. Next to them are {label} samples it gets RIGHT.

{label} SAMPLES THE MODEL MISSES (with the model's P({label})):
{hard_block}

{label} SAMPLES THE MODEL GETS RIGHT (with the model's P({label})):
{contrast_block}

Propose {n} NEW rules that separate the missed samples from the correctly
handled ones, capturing signals the current rules do not already express.
Do not restate or reword a current rule.
"""
