@echo off
rem start.bat — NORMEX launcher (Windows)
rem
rem   start.bat                                  start the live dashboard
rem   start.bat dashboard --camera-url http://192.168.1.100:8080/video
rem   start.bat setup                            verify datasets / create demo
rem   start.bat benchmark                        run the MVTec AD benchmark
rem   start.bat capture --out datasets\mygood --camera-url http://IP:8080/video
rem
rem Everything is delegated to launch.py so Windows and Linux behave the same.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo error: no Python interpreter found ^(create .venv or install python^) 1>&2
        exit /b 1
    )
    set "PY=python"
)

"%PY%" launch.py %*
exit /b %errorlevel%
