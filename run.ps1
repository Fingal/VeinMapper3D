
rm -Force generate_line.cp311-win_amd64.pyd
rm -Force cone_calculation.cp311-win_amd64.pyd
rm -Force convert_to_tiff_smooth.cp311-win_amd64.pyd
cd .\cython_files
python .\setup.py build_ext --inplace
python .\setup_cone.py build_ext --inplace
python .\setup_convert.py build_ext --inplace
cd ..
copy .\cython_files\generate_line.cp311-win_amd64.pyd generate_line.cp311-win_amd64.pyd
copy .\cython_files\cone_calculation.cp311-win_amd64.pyd cone_calculation.cp311-win_amd64.pyd
copy .\cython_files\convert_to_tiff_smooth.cp311-win_amd64.pyd convert_to_tiff_smooth.cp311-win_amd64.pyd
C:/ProgramData/Anaconda3/python.exe ./mainApp.py 
exit
