@REM copy notebooks
echo Copying notebooks2...

set "BASEDIR=%~dp0"
set "SRC=%BASEDIR%..\application\SSCValidationHeligoland\"
set "DST=%BASEDIR%source\notebooks\_temp\"


if not exist "%SRC%\" (
  echo ERROR: Source folder does not exist: "%SRC%"
  echo Current working dir: "%CD%"
  exit /b 1
)

REM Copy all ipynb files recursively
for /r "%SRC%" %%F in (*.ipynb) do (
  copy /Y "%%F" "%DST%\"
)



echo Done Copying notebooks.

sphinx-apidoc -o source/api -f ../lidalign

call make clean
call make html


echo Removing copied notebooks...



REM Remove all files in _temp
for /r "%DST%" %%F in (*.ipynb) do (
  del "%%F"
)
