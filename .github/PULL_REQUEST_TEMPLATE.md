# Pull Request

## Beschrijving
<!-- Korte beschrijving van wat deze PR doet -->

## Type Wijziging
<!-- Vink aan wat van toepassing is -->

- [ ] 🐛 Bug fix (non-breaking change die een issue oplost)
- [ ] ✨ Nieuwe feature (non-breaking change die functionaliteit toevoegt)
- [ ] 💥 Breaking change (fix of feature die bestaande functionaliteit beïnvloedt)
- [ ] 📚 Documentatie update
- [ ] 🔧 Refactoring (geen nieuwe features of bug fixes)
- [ ] ⚡ Performance verbetering
- [ ] 🧪 Tests toevoegen of bijwerken
- [ ] 🔒 Security fix

## Checklist
<!-- Alle items moeten aangevinkt zijn voordat de PR kan worden gemerged -->

### Code Kwaliteit
- [ ] Code volgt project style guidelines (ruff, black, mypy clean)
- [ ] Zelf-review uitgevoerd van eigen code
- [ ] Code is goed gedocumenteerd met docstrings
- [ ] Geen console.log, print statements of debug code achtergelaten
- [ ] Import statements zijn clean en georganiseerd

### Tests
- [ ] Tests toegevoegd die de wijziging dekken
- [ ] Alle nieuwe en bestaande tests slagen
- [ ] Coverage blijft > 85% (check artifacts)
- [ ] Edge cases zijn getest
- [ ] Integration tests bijgewerkt indien nodig

### Documentatie
- [ ] README.md bijgewerkt indien interface wijzigingen
- [ ] API documentatie bijgewerkt indien endpoints gewijzigd
- [ ] CHANGELOG.md entry toegevoegd met juiste versie
- [ ] Inline commentaar toegevoegd voor complexe logic
- [ ] replit.md bijgewerkt bij architectuur wijzigingen

### Security & Performance
- [ ] Geen hardcoded secrets of API keys
- [ ] Input validation toegevoegd waar nodig
- [ ] Geen performance regressies geïntroduceerd
- [ ] Security implications overwogen
- [ ] Database migrations safe voor productie (indien van toepassing)

### Trading System Specifiek
- [ ] Data integrity checks toegevoegd/behouden
- [ ] Risk management regels niet omzeild
- [ ] 80% confidence gate respecteerd
- [ ] Geen synthetic/dummy data geïntroduceerd
- [ ] ML model changes hebben proper validation

## Screenshots/Demo
<!-- Voor UI changes, voeg screenshots toe -->

## Related Issues
<!-- Link naar gerelateerde GitHub issues -->
Fixes #(issue nummer)

## Deployment Notes
<!-- Speciale instructies voor deployment indien nodig -->

## Reviewer Checklist
<!-- Voor reviewers -->
- [ ] Code review voltooid
- [ ] Tests draaien lokaal
- [ ] Documentatie review
- [ ] Security review indien applicable