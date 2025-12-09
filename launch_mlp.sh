cd build
./cuda_mlp
cd ..
echo "Comparaison avec pytorch :"
python bench_mlp_pytorch.py