rm -R output
mkdir output
rm gpu-vh
make clean
make
#./gpu-vh --config rhic-conf/ -o output -h
