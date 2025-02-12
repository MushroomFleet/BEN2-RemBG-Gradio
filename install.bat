@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"
echo Installing Repo...
echo Installing dependencies...
pip install -r requirements.txt
echo Setup complete!
pause