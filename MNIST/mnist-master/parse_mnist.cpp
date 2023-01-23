/*
* Read mnist image and labels, save as bmp images
*
* Modified from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
*
* Compile:
*      clang++ parse_mnist.cpp `pkg-config --libs --flags opencv4`
*/

#include <string>
#include <opencv2/opencv.hpp>

#include "pixel_benchmark.h"

#define LOGD(fmt, ...) fprintf(stdout, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)

#if __linux__ || __APPLE__
#include <sys/stat.h>
#include <unistd.h>
#elif _MSC_VER
#include <direct.h>
#endif


static void pixel_mkdir(const char* dirname) {
#if __linux__ || __APPLE__
    if (0 == access(dirname, W_OK)) {
        LOGD("Directory %s already exists\n", dirname);
    }
    else {
        if (0 != mkdir(dirname, 0744)) {
            LOGE("Failed to create directory %s\n", dirname);
        }
        else {
            LOGD("Directory %s was successfully created\n", dirname);
        }
    }
#elif _MSC_VER
    if (_mkdir(dirname) == 0) {
        LOGD("Directory %s was successfully created\n", dirname);
    }
    else {
        if (errno == EEXIST) {
            LOGD("Problem creating directory %s, already exists\n", dirname);
        }
        else if (errno == ENOENT) {
            LOGD("Problem creating directory %s, Path was not found\n", dirname);
        }
    }
#else
    PIXEL_LOGE("%s not implemented yet!\n", __FUNCTION__);
#endif
}

// 日常用的PC CPU是Intel处理器，用小端格式
// 把大端数据转换为我们常用的小端数据
uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int read_and_save(const std::string& mnist_img_path, const std::string& mnist_label_path, const std::string& save_dir)
{
    // 以二进制格式读取mnist数据库中的图像文件和标签文件
    FILE* fin_image = fopen(mnist_img_path.c_str(), "rb");
    if (fin_image==NULL)
    {
        LOGE("open mnist image file error!\n");
        return 1;
    }
    FILE* fin_label = fopen(mnist_label_path.c_str(), "rb");
    if (fin_label==NULL)
    {
        LOGE("open mnist label file error!\n");
        return 2;
    }

    uint32_t magic; // 文件中的魔术数(magic number)
    uint32_t num_items;// mnist图像集文件中的图像数目
    uint32_t num_label;// mnist标签集文件中的标签数目
    uint32_t rows;// 图像的行数
    uint32_t cols;// 图像的列数
    // 读魔术数
    fread(&magic, 4, 1, fin_image);
    magic = swap_endian(magic);
    if (magic != 2051)
    {
        LOGE("this is not the mnist image file\n");
        return 3;
    }
    fread(&magic, 4, 1, fin_label);
    magic = swap_endian(magic);
    if (magic != 2049)
    {
        LOGE("this is not the mnist label file\n");
        return 4;
    }
    // 读图像/标签数
    fread(&num_items, 4, 1, fin_image);
    num_items = swap_endian(num_items);
    fread(&num_label, 4, 1, fin_label);
    num_label = swap_endian(num_label);
    // 判断两种标签数是否相等
    if (num_items != num_label)
    {
        LOGE("the image file and label file are not a pair\n");
        return 5;
    }
    // 读图像行数、列数
    fread(&rows, 4, 1, fin_image);
    rows = swap_endian(rows);
    fread(&cols, 4, 1, fin_image);
    cols = swap_endian(cols);

    // 读取图像
    char* pixels = new char[rows * cols];
    cv::Mat image(rows, cols, CV_8UC1, (uchar*)pixels);
    char label;
    char save_pth[256];
    int size = rows * cols;

    double t_start, t_cost, parse_cost = 0, write_cost = 0;
    //for (int i = 0; i != num_items; i++)
    for (int i = 0; i != 1000; i++)
    {
        t_start = pixel_get_current_time();
        fread(pixels, size, 1, fin_image);
        fread(&label, 4, 1, fin_label);
        t_cost = pixel_get_current_time() - t_start;
        parse_cost += t_cost;

        t_start = pixel_get_current_time();
        sprintf(save_pth, "%s/%d_%04d.bmp", save_dir.c_str(), (int)label, i);
        cv::imwrite(save_pth, image);
        t_cost = pixel_get_current_time() - t_start;
        write_cost += t_cost;
    }
    printf("parse_cost %.4lf ms\n", parse_cost);
    printf("write_cost %.4lf ms\n", write_cost);
    // windows: parse_cost 3   ms, write cost 3000 ms
    // macosx:  parse_cost 1.7 ms, write cost 340  ms
    // linux:   parse_cost 1   ms, write cost 95   ms

    delete[] pixels;

    fclose(fin_image);
    fclose(fin_label);

    return 0;
}

int main()
{
    // 注意：请确保原始mnist文件存在、路径正确
    // 并且确保保存的目录已经存在
#ifdef _MSC_VER
    //std::string mnist_dir = "D:/dev/mnist";
    std::string mnist_dir = "C:/mnist";
#else
    std::string mnist_dir = ".";
#endif
    std::string train_image_path = mnist_dir + "/train-images-idx3-ubyte";
    std::string train_label_path = mnist_dir + "/train-labels-idx1-ubyte";
    std::string test_image_path = mnist_dir + "/test-images-idx3-ubyte";
    std::string test_label_path = mnist_dir + "/test-labels-idx1-ubyte";

    std::string train_save_dir = mnist_dir + "/train";
    std::string test_save_dir = mnist_dir + "/test";
    pixel_mkdir(train_save_dir.c_str());
    pixel_mkdir(test_save_dir.c_str());

    read_and_save(test_image_path, test_label_path, test_save_dir);
    LOGD("parsing test image & label done\n");
    read_and_save(train_image_path, train_label_path, train_save_dir);
    LOGD("parsing train image & label done\n");

    return 0;
}
