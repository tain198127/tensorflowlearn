Download mnist dataset and extract in **1 second!**

## For Caffe users

create `$CAFFE/data/mnist/get_mnist_fast.sh`:

```bash
#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

wget https://gitee.com/aczz/mnist/repository/archive/master.zip
unzip master.zip

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    gunzip -c mnist/${fname}.gz > ${fname} # will keep original files
done
```

## For general usage

```
git clone https://gitee.com/aczz/mnist

bash ./extract.sh

make && ./a.out
# or use cmake:
#mkdir build && cd build && cmake .. && make

python gen_path.py
```

