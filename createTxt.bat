echo "creating txt names"
set dir=F:\Data\SR\train_data\291
set obj=F:\Data\SR\train.txt
pushd %dir%
for /f %%j in ('dir /b') do echo %dir%\%%j>>%obj%
echo "create txt names done.."
pause