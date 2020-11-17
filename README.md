
This demo will show you how to use pytorch c++ inference with libtorch.


### 1. test environment

- `ubuntu 18.04`
- `python 3.8.3`
- `pytorch 1.7.0`
- `libtorch 1.7.0`
- `opencv 4.5.0`

### 2. dependencies

you must install these dependencies before build the demo.

- [libtorch](https://pytorch.org/)
- [opencv](https://github.com/opencv/opencv)
- [gflags](https://github.com/gflags/gflags)
- [json](https://github.com/nlohmann/json)

### 3. build

```bash
git clone https://github.com/lookenwu/pytorchcxx.git
cd pytorchcxx
python main.py
mkdir build && cd build
# you must edit CMakeLists.txt to append /path/to/libtorch/share/cmake to CMAKE_PREFIX_PATH before you run cmake
cmake ..
make
./torchcxx
./torchcxx --label_path=/path/to/imagenet_json_label --img_path=/path/to/image --model_path=/path/to/model --topk=5
```

output `top k` result

```txt
Top-0 label name: ["n02134084","ice_bear"], probability: 99.4734%
Top-1 label name: ["n02132136","brown_bear"], probability: 0.362484%
Top-2 label name: ["n02111500","Great_Pyrenees"], probability: 0.0443497%
```
