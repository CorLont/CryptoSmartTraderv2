# Branch Protection Setup Guide

## 🛡️ GitHub Branch Protection voor Main Branch

### Waarom Branch Protection?
Met onze enterprise CI/CD pipeline (GitHub Actions) in plaats zorgt branch protection ervoor dat:
- Alleen code met succesvolle CI tests wordt gemerged
- Code review verplicht is voor kritieke wijzigingen
- Main branch altijd in stabiele staat blijft
- Productie deployment veilig en betrouwbaar is

### Stap-voor-stap Setup in GitHub

#### 1. Navigeer naar Repository Settings
```
https://github.com/{username}/{repository}/settings/branches
```

#### 2. Add Branch Protection Rule
- Klik "Add rule"
- Branch name pattern: `main`

#### 3. Configureer Protection Settings

##### ✅ Require a pull request before merging
- **Require approvals**: 1 (minimaal voor enterprise code)
- **Dismiss stale reviews**: ✅ (zorgt voor up-to-date reviews)
- **Require review from code owners**: ✅ (indien CODEOWNERS bestaat)

##### ✅ Require status checks to pass before merging
- **Require branches to be up to date**: ✅
- **Status checks to require**:
  - `CI / test (python-3.11)` (onze pytest suite)
  - `CI / lint (python-3.11)` (ruff + black + mypy)
  - `CI / security` (pip-audit vulnerability scan)

##### ✅ Require conversation resolution before merging
- Zorgt ervoor dat alle PR comments zijn afgehandeld

##### ✅ Restrict pushes that create files in a directory
- Optional: Restrict direct pushes to `src/` directory

#### 4. Advanced Protections (Enterprise Optionaal)

##### 🔒 Require signed commits
- Verhoogt security voor enterprise deployment
- Vereist GPG key setup voor ontwikkelaars

##### 🔒 Restrict force pushes  
- Voorkomt accidentele history rewrites
- Behoudt audit trail voor compliance

##### 🔒 Do not allow bypassing the above settings
- Geldt ook voor admins
- Enterprise security best practice

### Configuratie voor onze CI Pipeline

#### Status Checks die Required moeten zijn:
```yaml
# Onze .github/workflows/ci.yml produceert deze checks:
✅ CI / test (python-3.11)     # pytest met 85%+ coverage
✅ CI / lint (python-3.11)     # ruff/black/mypy linting  
✅ CI / security               # pip-audit security scan
```

#### PR Workflow met Branch Protection:
1. **Create Feature Branch**: `git checkout -b feat/nieuwe-feature`
2. **Develop & Test**: Lokale ontwikkeling met tests
3. **Push Branch**: `git push origin feat/nieuwe-feature`
4. **Create PR**: Via GitHub interface
5. **Automatic CI**: GitHub Actions draait automatisch
6. **Code Review**: Reviewer controleert wijzigingen
7. **Green CI**: Alle status checks moeten slagen
8. **Merge**: Alleen mogelijk met groene CI + approval

### Configuratie Voorbeeld

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "CI / test (python-3.11)",
      "CI / lint (python-3.11)", 
      "CI / security"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

### Emergency Procedures

#### Hotfix Process
Voor kritieke production fixes:
1. Create hotfix branch from main: `git checkout -b hotfix/critical-fix`
2. Minimal fix implementatie
3. Fast-track PR met expedited review
4. Emergency merge met admin override (indien nodig)
5. Post-deployment monitoring

#### Bypass Procedures (Admin Only)
```bash
# Alleen voor echte emergencies
# Admin kan tijdelijk protection disablen voor kritieke fix
# Direct na fix: protection weer enablen
```

### Testing Branch Protection

#### Simuleer Protected Merge:
1. Create test branch: `git checkout -b test/protection`
2. Maak kleine wijziging
3. Push branch: `git push origin test/protection`
4. Create PR via GitHub
5. Verifieer dat merge geblokkeerd is zonder groene CI
6. Wacht op CI completion
7. Voeg reviewer toe
8. Test merge na approval + groene CI

### Monitoring & Compliance

#### Protection Status Monitoring
- GitHub webhooks voor protection events
- Audit logs voor protection bypasses
- Regular review van protection settings
- Compliance reporting voor enterprise governance

#### Metrics to Track
- PR merge tijd (met vs zonder protection)
- CI failure rate (pre vs post protection)
- Code review coverage
- Protection bypass frequency

---

**Implementation Date:** 2025-01-11  
**CI Pipeline:** GitHub Actions met UV + pytest + ruff + mypy + security  
**Status Checks:** test, lint, security (Python 3.11)  
**Enterprise Ready:** ✅ Comprehensive protection voor production deployment