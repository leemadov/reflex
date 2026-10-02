@echo off
rem Reflex desktop app (built-in browser + chat + local models). Starts the model server itself if none is running.
start "" "%~dp0desktop\node_modules\electron\dist\electron.exe" "%~dp0desktop"
