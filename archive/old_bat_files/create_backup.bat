@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Backup System
echo ==========================================
echo.

echo 📦 Creating system backup...
python scripts/backup_restore.py backup --include-logs --output-dir backups
if %errorlevel% neq 0 (
    echo ❌ Backup failed
    pause
    exit /b 1
)

echo.
echo ✅ Backup completed successfully!
echo 📁 Backup location: backups/
echo.
pause
