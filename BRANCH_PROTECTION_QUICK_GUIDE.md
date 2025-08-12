# Branch Protection Quick Setup Guide

## 🛡️ GitHub Branch Protection voor Main Branch

### Stap 1: GitHub Repository Settings
1. Ga naar je GitHub repository
2. Klik op **Settings** tab
3. Ga naar **Branches** in de sidebar
4. Klik **Add rule**

### Stap 2: Branch Protection Rule
**Branch name pattern:** `main`

### Stap 3: Required Settings
✅ **Require a pull request before merging**
- Require approvals: **1**
- Dismiss stale reviews when new commits are pushed: **checked**

✅ **Require status checks to pass before merging**
- Require branches to be up to date before merging: **checked**
- **Required status checks** (selecteer deze 3):
  - `CI / test (python-3.11)`
  - `CI / lint (python-3.11)`  
  - `CI / security`

✅ **Require conversation resolution before merging**

✅ **Restrict pushes that create files in a directory** (optioneel)

### Stap 4: Save Rule
Klik **Create** om de branch protection rule op te slaan.

## ✅ .replit Configuration Status

De huidige `.replit` configuratie is **OPTIMAL** - geen wijzigingen nodig:

```toml
[[workflows.workflow]]
name = "UVMultiService" 
author = "agent"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "bash -c \"uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)\""
waitForPort = 5000
```

**Services draaiend:**
- ✅ Dashboard op port 5000
- ✅ API op port 8001  
- ✅ Metrics op port 8000

## 🧪 Test Branch Protection

Na setup, test de protection:
1. `git checkout -b test-protection`
2. Maak kleine wijziging
3. `git push origin test-protection`
4. Maak PR in GitHub
5. Verifieer dat merge geblokkeerd is zonder groene CI
6. Wacht op CI success, voeg reviewer toe
7. Merge na approval + groene checks

---

**Status:** Alle PR bundles (CI + Config + API) geïmplementeerd  
**Next:** Branch protection setup in GitHub repository settings