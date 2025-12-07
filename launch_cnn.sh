cd build
./cuda_cnn
cd ..
echo "Comparaison avec pytorch :"
python bench_cnn_pytorch.py