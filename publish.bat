@echo off
REM ============================================================
REM  AMF-Core v0.2.0 — Publish to PyPI + GitHub
REM
REM  Usage:
REM    publish.bat <PYPI_TOKEN>
REM
REM  GitHub push uses the token stored in git credential manager.
REM  To set it once:
REM    git config --global credential.helper manager
REM    (Windows will prompt for GitHub token on first push)
REM ============================================================

SET VERSION=0.2.0
SET GITHUB_USER=jadcrypto
SET REPO=amf-core

SET PYPI_TOKEN=%1

IF "%PYPI_TOKEN%"=="" (
    echo.
    echo  ERROR: PyPI token not provided.
    echo  Usage: publish.bat ^<PYPI_TOKEN^>
    echo.
    exit /b 1
)

echo.
echo  ===========================================================
echo    AMF-Core v%VERSION% — Publishing
echo  ===========================================================
echo.

REM ── 1. Clean ─────────────────────────────────────────────────
echo  [1/4] Cleaning build artifacts...
IF EXIST dist         rmdir /s /q dist
IF EXIST build        rmdir /s /q build
IF EXIST amf_core.egg-info rmdir /s /q amf_core.egg-info
IF EXIST amf-core.egg-info rmdir /s /q amf-core.egg-info
echo        Done.

REM ── 2. Build ─────────────────────────────────────────────────
echo.
echo  [2/4] Building wheel and sdist...
python -m build
IF ERRORLEVEL 1 (
    echo  ERROR: Build failed. Run: pip install build
    exit /b 1
)
echo        Done.

REM ── 3. PyPI upload ───────────────────────────────────────────
echo.
echo  [3/4] Uploading to PyPI...
python -m twine upload dist/* ^
    --username __token__ ^
    --password %PYPI_TOKEN% ^
    --non-interactive ^
    --skip-existing
IF ERRORLEVEL 1 (
    echo  ERROR: PyPI upload failed.
    exit /b 1
)
echo        Done.
echo        https://pypi.org/project/amf-core/

REM ── 4. GitHub ────────────────────────────────────────────────
echo.
echo  [4/4] Pushing to GitHub...
git add -A
git commit -m "release: v%VERSION% — AMFEngine molecular inference"
git tag -a "v%VERSION%" -m "AMF-Core v%VERSION%" 2>nul || echo  Tag already exists, skipping.
git remote set-url origin https://github.com/%GITHUB_USER%/%REPO%.git
git push origin main --force-with-lease
git push origin "v%VERSION%"
IF ERRORLEVEL 1 (
    echo  WARNING: Git push incomplete. Check credentials.
) ELSE (
    echo        Done.
    echo        https://github.com/%GITHUB_USER%/%REPO%
)

echo.
echo  ===========================================================
echo    Published: amf-core v%VERSION%
echo    PyPI  : https://pypi.org/project/amf-core/
echo    GitHub: https://github.com/%GITHUB_USER%/%REPO%
echo  ===========================================================
echo.
