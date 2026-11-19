@echo off
"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_duel_dashboard.ps1" -ReuseRun "4090abada3654f30a9dba79d963f795c" -Open
if errorlevel 1 pause
