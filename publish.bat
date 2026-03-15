@echo off
echo ============================================================
echo 🚀 AMF-Core PyPI Publishing Script
echo ============================================================

echo.
echo [1/3] Checking prerequisites...
python -m pip install --upgrade pip build twine

echo.
echo [2/3] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist amf_core.egg-info rmdir /s /q amf_core.egg-info

echo.
echo [3/3] Building the package (sdist ^& wheel)...
python -m build

echo.
echo ============================================================
echo ✅ Build complete! Ready to upload to PyPI.
echo ============================================================
echo.
echo You will need your PyPI username and password (or API token).
echo If using an API token, set username to __token__
echo.
set /p proceed="Do you want to upload to PyPI now? (y/n): "
if /i "%proceed%"=="y" (
    echo.
    echo ⬆️ Uploading via Twine...
    python -m twine upload dist/*
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo 🎉 Congratulations! amf-core is now live on PyPI.
        echo Anyone can now run: pip install amf-core
    ) else (
        echo.
        echo ❌ Upload failed. Please check your credentials and try again.
    )
) else (
    echo.
    echo 🛑 Upload cancelled. You can manually upload later by running:
    echo python -m twine upload dist/*
)

pause
