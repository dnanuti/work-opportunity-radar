# Work Opportunity Radar

A beginner-friendly companion to the **Data & AI: Build Your Own Work Opportunity
Radar** talk. The demo follows **Amina**, a fictional recent software-development
graduate in Nairobi, then lets any candidate replace her profile without changing the
code.

The central question is not “can we call a model?” It is:

> Which opportunities should this candidate investigate, what evidence supports the
> recommendation, and what is still unknown?

The default demo is fictional, deterministic, and offline. No account, key, paid
service, or live job site is required.

## The talk and the code

| Talk idea | What the demo makes visible |
|---|---|
| AI starts before the model | Candidate goals, fields, labels, exclusions, and success are explicit |
| Structured / semi-structured / unstructured data | Provider JSON becomes a table while descriptions remain free text |
| Raw and trusted storage | Original payload + provenance are preserved; cleaned records are written separately |
| Classification is not truth | Posts become `likely_opportunity`, `uncertain`, or `not_current` with reasons |
| Real does not mean relevant | Classification is candidate-independent; ranking uses the active candidate profile |
| Machine learning and evaluation | A labelled experiment shows duplicate leakage, false positives, and false negatives |
| Plausible is not true | A vague prompt invents facts; grounded output is schema-validated |
| Evidence, not instructions | The final card shows source, verification date, match reasons, gaps, uncertainty, and next step |
| Continuous feedback | Candidate judgement is stored as reviewable feedback, not automatic ground truth |

## Run the complete demo

Install Python **3.10-3.13**, download or clone the repository, and open a terminal in
this folder.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py demo
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py demo
```

`run.py` is the supported cross-platform launcher. `Makefile` and `run.sh` are optional
Unix shortcuts.

## Use your own candidate profile

The easiest local path is the six-question setup wizard:

```text
Windows:       .venv\Scripts\python run.py configure
macOS/Linux:   .venv/bin/python run.py configure
```

It creates the git-ignored `profiles/candidate.json`. Future `demo`, `radar`, `prompt`,
and `chatgpt` commands use it automatically. To switch back to Amina, delete that local
file. You can also copy `profiles/candidate.example.json` and edit it.

Only add information needed for matching: a name or nickname, broad location, skills,
target roles, preferred locations, and preferred levels. Do **not** add a CV, email,
phone number, home address, protected traits, or API keys.

To use another profile without creating a local file, set `RADAR_PROFILE` to a JSON file
path before running the command.

## Use the two guided notebooks

```text
Windows:       .venv\Scripts\python run.py notebook
macOS/Linux:   .venv/bin/python run.py notebook
```

Jupyter opens the `notebooks` folder. Work through just two files:

1. `01_data_foundations.ipynb` - collect, preserve, inspect, measure, and clean the data.
2. `02_work_opportunity_radar.ipynb` - customize the candidate, classify and rank posts,
   evaluate the ML example, use ChatGPT safely, and review an evidence card.

The second notebook can run independently, so a candidate who only wants the finished
radar does not have to complete the data lesson first.

## Plug in ChatGPT - no API key

This is a manual copy/paste bridge; it does not automate the ChatGPT website.

```text
Windows:       .venv\Scripts\python run.py chatgpt
macOS/Linux:   .venv/bin/python run.py chatgpt
```

The command:

1. creates `output/chatgpt_prompt.txt` from the active profile’s top-ranked post;
2. asks you to paste that prompt into ChatGPT;
3. accepts the JSON reply back in the terminal (finish with a line containing `END`);
4. validates the reply locally before showing an evidence card and draft message.

If you prefer files instead of the interactive flow:

```text
python run.py prompt
# paste output/chatgpt_prompt.txt into ChatGPT
# save only the reply as output/chatgpt_response.json
python run.py validate output/chatgpt_response.json
```

The validator accepts plain JSON or a fenced `json` block. A fluent reply is still not
proof: verify the original source and every consequential claim before applying.

## Optional OpenAI API integration

This mode is for developers who specifically want a programmatic model call. A ChatGPT
subscription and OpenAI API usage are billed separately. The official OpenAI guidance
is linked here: [ChatGPT and API billing are separate](https://help.openai.com/en/articles/8156019-how-can-i-move-my-chatgpt-subscription-to-the-api).

1. Install the optional package:

```text
Windows:       .venv\Scripts\python -m pip install -r requirements-openai.txt
macOS/Linux:   .venv/bin/python -m pip install -r requirements-openai.txt
```

2. Change `llm.provider` in `config.yaml` from `offline` to `openai`.

3. Set the key for the current terminal session - never put it in a notebook or file:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "your-key"
```

```bash
# macOS / Linux
export OPENAI_API_KEY="your-key"
```

4. Run `python run.py demo` using the platform-specific interpreter shown above.

The integration uses the Responses API and a Pydantic Structured Output. If the key,
package, network, or response is unavailable, it reports the issue and falls back to the
offline teaching stub. See the official [Responses API text guide](https://developers.openai.com/api/docs/guides/text) and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Job data sources

The safe default is `sample`, a bundled fictional feed. In `config.yaml`, or in the
second notebook's customize cell, choose:

- `sample` - deterministic, fictional, offline;
- `africa_ats` - configured public employer job-board APIs;
- `remotive` / `remoteok` - public remote-job feeds, filtered for Africa reachability;
- `csv` / `json` - your own exported or permitted public dataset.

The project does not scrape LinkedIn, Jobberman, BrighterMonday, logged-in pages, or
sites without clear permission. Publicly visible does not automatically mean permitted
to scrape. Live sources fall back to the offline sample rather than breaking a talk.

## Try to break it

- **Starter:** customize the candidate and add one city or skill alias.
- **Builder:** add a transparent ranking signal and explain it in `why`.
- **Explorer:** add a permitted public API/ATS adapter, offline fixture, and test.

Every change should answer: **What data changed? What decision changed? How would we
know whether it helped a real person?**

Run the checks with:

```text
Windows:       .venv\Scripts\python run.py test
macOS/Linux:   .venv/bin/python run.py test
```

## Responsible-use rules

- Preserve provenance, freshness, missing fields, and the original link.
- Never infer or rank on protected traits.
- Treat generated text as a draft for human review.
- Keep uncertain cases visible rather than forcing a confident label.
- Measure false positives and false negatives by human cost, not accuracy alone.
- Review feedback before using it as a training label.
