@echo off
:: start.bat — NORMEX Production Deploy
:: Full pipeline: IP Camera → Training → Inspection → Dashboard
:: Works on Windows

cd /d "C:\home\naman\project\main_folder" || exit /b 1

echo ============================================
echo  NORMEX — Production Deploy
echo ============================================
echo.

:: Step 1: Check and Setup Datasets
echo == Step 1: Dataset Setup ==
echo.

:: Ensure MVTec AD screw category exists
if not exist datasets\mvtec_ad\screw (
    echo extracting MVTec AD screw category...
    md datasets\mvtec_ad
    cd C:\home\naman\Downloads
    unzip -o archive.zip -d datasets\mvtec_ad\screw > nul
)

:: Ensure IP camera dataset exists
if not exist datasets\ipcam (
    echo creating IP camera dataset folders...
    md datasets\ipcam\good
    md datasets\ipcam\defect
    md datasets\ipcam\new_good
)

:: Ensure demo dataset exists
if not exist datasets\demo (
    echo generating synthetic demo dataset...
    python scripts\make_demo_dataset.py --out datasets\demo
)

echo datasets ready
echo.

:: Step 2: Check for Live IP Camera
echo == Step 2: IP Camera Check ==
echo.

set IP_FOUND=0
:: Try common phone IP camera ports
for /f "tokens=*" %%P in ('curl -s -o nul -w "%%{http_code}" "http://192.168.1.100:8080/video" 2^>nul') do (
    if %%P == 200 (
        echo Found IP camera at 192.168.1.100:8080
        set IP_FOUND=1
    )
)

if %IP_FOUND% equ 0 (
    echo No IP camera detected on default network
    echo Will use synthetic demo dataset
    set IP_CAMERA=0
) else (
    echo IP camera ready for capture
    set IP_CAMERA=1
)
echo.

:: Step 3: Capture from IP Camera or Use Synthetic
echo == Step 3: Data Capture ==
echo.

if %IP_CAMERA% equ 1 (
    echo Capturing from IP camera...
    python -c "
from module_1_acquisition.camera import Camera
try:
    cam = Camera(url='http://192.168.1.100:8080/video', warmup=10)
    info = cam.open()
    print Camera opened: %info.source% %info.width%x%info.height%
    md datasets\ipcam\good
    for /L %i in (1,1,30) do (
        frame = cam.read()
        call save_frame frame datasets\ipcam\good prefix=good
        timeout /t 1 /nobreak > nul
        echo Captured %i/30
    )
    cam.close()
    echo IP camera capture complete
except Exception as e (
    echo IP camera capture failed: %e%
    echo Falling back to synthetic dataset
    set IP_CAMERA=0
)
"
) else (
    echo Using existing IP camera dataset frames
)
echo.

:: Step 4: Train on All Available Datasets
echo == Step 4: Training Phase ==
echo.

echo Training on MVTec AD (screw category)...
C:\home\naman\.venv\Scripts\python main.py train --product screw --images datasets\mvtec_ad\screw\train\good 2>&1 | find "total"

echo Training on IP camera good images...
C:\home\naman\.venv\Scripts\python main.py train --product ipcam_good --images datasets\ipcam\good 2>&1 | find "total"

echo Training on synthetic demo data...
C:\home\naman\.venv\Scripts\python main.py train --product demo --images datasets\demo\good 2>&1 | find "total"
echo.

:: Step 5: Dashboard
echo == Step 5: Dashboard & Summary ==
echo.

echo Generating comprehensive summary...
C:\home\naman\.venv\Scripts\python scripts\summarize_inspect.py --good results\demo\good.jsonl --defect results\demo\defect.jsonl 2^>nul | find "AUROC"

echo.
echo ---

:: Step 6: Launch Dashboard
echo == Step 6: Launch Dashboard ==
echo.

echo NORMEX Dashboard opening in browser...
echo.

if exist C:\Program Files\Internet Explorer\iexplore.exe (
    start "" "C:\Program Files\Internet Explorer\iexplore.exe" dashboard.html
) else if exist "\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "\Program Files (x86)\Google\Chrome\Application\chrome.exe" dashboard.html
) else (
    echo Open dashboard.html manually in your browser
)

echo.
echo ============================================
echo  NORMEX — Production Session Complete
echo ============================================
echo.
pause