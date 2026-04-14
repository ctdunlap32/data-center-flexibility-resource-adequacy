# Tomorrow's first task — pick up here

## State at end of Apr 13 session

- Repo created, committed, pushed to GitHub ✓
- Directory structure in place (notebooks/, data/, outputs/, docs/, paper/, archive/) ✓
- Source-of-truth files archived ✓
- LICENSE, README.md, requirements.txt, .gitignore created ✓
- First regression baseline captured: Cross_BA v5 ✓
  - archive/executed_v5.ipynb
  - archive/executed_v5.md (human-readable form)

## Three regression baselines still to capture

### 1. Bartlett v18 (first priority tomorrow)

**Problem**: The .ipynb version you made today (with the _pillar3 validation fix)
is NOT in the repo. Only the .py file is.

**First task**: find the .ipynb. Try searching C:\Users\dunla\ for
`bartlett_analysis_v18.ipynb`. If it exists, copy it into archive/.
If it doesn't exist anymore, you'll need to recreate it: open a fresh
notebook, copy-paste the contents of archive/bartlett_analysis_v18.py
cell by cell, re-apply the _pillar3 fix (delete the two _check blocks
in the validation cell that reference _pillar3_committed_mw and
_pillar3_accredited_mw), save.

**Then nbconvert it**, from inside the archive/ directory:

    cd archive
    jupyter nbconvert --to notebook --execute bartlett_analysis_v18.ipynb --output executed_bartlett_v18.ipynb --ExecutePreprocessor.kernel_name=python3
    jupyter nbconvert --to markdown executed_bartlett_v18.ipynb

**Watch for**: cell 3 loads three CSVs (DA LMP, RT LMP, load). Their paths
are hardcoded in the notebook. If those paths are broken, nbconvert fails
at cell 3 with FileNotFoundError. Fix the paths in the notebook, re-run.
The Cross_BA v5 process worked because its data loading was already
correct.

### 2. Azure trace notebook (after Bartlett v18)

**File**: archive/P1_Azure_Trace_Analysis.ipynb

**Known issue**: The data loading cell hardcodes paths to the Azure CSVs.
Earlier today, the fix was to use the `r` prefix and point at:
C:\Users\dunla\OneDrive\Documents\Bartlett Fellowship\Thesis\Data\Azure_Traces\

**Check first**: does the version in archive/ have the fix, or the
original broken path? Open in VS Code and check the CODE_PATH and
CONV_PATH variables in cell 3. If they point at the Hopper directory
(the old path), update to the Azure_Traces directory. If they point
at Azure_Traces already, you're good.

**Then nbconvert**:

    cd archive
    jupyter nbconvert --to notebook --execute P1_Azure_Trace_Analysis.ipynb --output executed_azure.ipynb --ExecutePreprocessor.kernel_name=python3
    jupyter nbconvert --to markdown executed_azure.ipynb

**Expect this to take 15–30 minutes** to run. It loads 44M rows across
two CSVs. Don't babysit it, just let it run.

### 3. BurstGPT validation script (after Azure)

**File**: archive/burstgpt_validation.py

**This is a script, not a notebook**. nbconvert doesn't apply. Instead:

    python archive\burstgpt_validation.py > archive\regression_baseline_burstgpt.txt 2>&1

The `> archive\...` captures stdout to a file. The `2>&1` captures stderr
into the same file. You end with a text file containing everything the
script printed.

**Expect**: the script auto-downloads BurstGPT data from GitHub on first
run. Takes a few minutes plus whatever the bandwidth is.

## Once all four baselines are captured

Commit them:

    git add archive\
    git commit -m "Capture all four regression baselines before migration"
    git push

## Then start Phase 3

Open Repository_Migration_Guide.md (saved from yesterday's conversation)
and go to Part D, Phase 3 (Build notebook 01 — Empirical Evidence).

## Important reminders

- Use Anaconda Prompt, not PowerShell. PowerShell has paste issues.
- Always type `--output` with no space. Never `-- output`.
- Always pass `--ExecutePreprocessor.kernel_name=python3` to nbconvert.
- When in doubt, `git status` first.
- When debugging nbconvert errors, don't try to fix the notebook blind.
  Report what you see and diagnose first.

## Session-end numbers to verify tomorrow

The Cross_BA v5 baseline (executed_v5.md) should contain these figures
from the paper. Eyeball-check the file at some point tomorrow to
confirm they're there:

- 99.0% dynamic availability (ComEd, 198/200 stress hours)
- 52.0% mean single-facility commitment depth (500 MW)
- 26.5% empirical fleet commitment depth
- Per-GW sweep: 48.7%/39.0%/31.5%/24.6%/22.2% at 1/3/5/10/15 GW

If any of these are missing, the nbconvert run didn't complete all the
way through and we need to re-run.
