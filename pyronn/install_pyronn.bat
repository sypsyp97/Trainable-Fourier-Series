@echo off
setlocal enabledelayedexpansion

REM Get the directory where the batch file is located
SET "SCRIPT_DIR=%~dp0"
SET "ORIGINAL_DIR=%CD%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"
if %ERRORLEVEL% neq 0 (
    echo Failed to change to script directory
    pause
    exit /b %ERRORLEVEL%
)

echo Installing PYRO-NN and dependencies from %CD%...

REM Step 1: Create virtual environment
echo Creating virtual environment...
call conda create --name pyronn_env python=3.12 -y
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Step 2: Activate the virtual environment
echo Activating virtual environment...
call conda activate pyronn_env
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Step 3: Install CUDA toolkit
echo Installing CUDA toolkit...
call conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
if %ERRORLEVEL% neq 0 (
    echo Failed to install CUDA toolkit
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Step 4: Install PyTorch and associated libraries
echo Installing PyTorch 2.5.0...
call pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% neq 0 (
    echo Failed to install PyTorch
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Step 5: Install additional Python packages
echo Installing numpy, matplotlib, and h5py...
call pip install numpy matplotlib h5py
if %ERRORLEVEL% neq 0 (
    echo Failed to install additional Python packages
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Step 6: Install PYRO-NN
echo Installing PYRO-NN...
if not exist "wheel\pyronn-0.3.2-cp312-cp312-win_amd64.whl" (
    echo Error: PYRO-NN wheel file not found in wheel directory
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b 1
)
call pip install wheel\pyronn-0.3.2-cp312-cp312-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo Failed to install PYRO-NN
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b %ERRORLEVEL%
)

REM Return to original directory
cd /d "%ORIGINAL_DIR%"

echo Installation completed successfully!
echo Press any key to exit...
pause >nul
endlocal