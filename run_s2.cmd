@echo off
rem System 2 training that survives crashes: resume from the last save until the final eval has written its metrics.
rem ponytail: 5 tries, then it gives up so a real bug can't loop forever.
cd /d "%~dp0"
for /l %%i in (1,1,5) do (
  python -u train_s2.py --hours 6.5 --n-text 150 --n-img 100 --resume >> train_s2.log 2>> train_s2.err
  if exist checkpoints_s2\best\metrics.json goto :eof
  echo restart %%i after a crash >> train_s2.log
  timeout /t 60 /nobreak >nul
)
