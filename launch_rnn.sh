cd build
./cuda_rnn
echo "---------------- v2 ------------------"
./cuda_rnn2
cd ..
echo "Comparaison avec pytorch :"
python bench_rnn_pytorch.py