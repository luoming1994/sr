echo "creating txt names"
set dir=F:\Data\SR\test_data\Set14
set obj=F:\Data\SR\test.txt
pushd %dir%
for /f %%j in ('dir /b') do echo %dir%\%%j>>%obj%
echo "create txt names done.."
pause
