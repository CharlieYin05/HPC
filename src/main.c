# include <inttypes.h>      // C 标准库头文件：固定宽度整数的输入输出格式
#include <stdint.h>         // C 标准库头文件：定义固定宽度整数类型
#include <stdio.h>          // C 标准库头文件：输入输出函数
#include <stdlib.h>         // C 标准库头文件：通用工具函数，如内存分配、程序退出等
#include "toy_hash.h"       // 自定义头文件：声明 toy_hash 函数

// Read an entire PDF and calculate its toy hash.
static int hash_file(const char *path, uint64_t *hash)
{
    FILE *file = fopen(path, "rb");     //read the pdf file in binary mode

    // Check if the file was opened successfully.
    if (file == NULL) {
        perror(path);
        return 0;
    }

    // Determine file size，用于分配内存空间
    if (fseek(file, 0, SEEK_END) != 0) {    //把文件读取位置移动到末尾
        fprintf(stderr, "Failed to seek: %s\n", path);
        fclose(file);
        return 0;
    }

    // Get the current file position, which is the size of the file.
    long length = ftell(file);

    // Check if the file size is valid and reset the file position to the beginning.
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Invalid file or file size: %s\n", path);
        fclose(file);
        return 0;
    }

    // Allocate memory for the PDF.
    unsigned char *buffer = malloc((size_t)length); //buffer 指针的第一个值指向分配的内存块的第一个地址

    if (buffer == NULL) {
        fprintf(stderr, "Out of memory.\n");
        fclose(file);
        return 0;
    }

    // Read all bytes.
    size_t count = fread(buffer, 1, (size_t)length, file);
    fclose(file);

    // 检查是否读完整个文件，如果没有，释放内存并返回错误。
    if (count != (size_t)length) {
        fprintf(stderr, "Failed to read entire file: %s\n", path);
        free(buffer);
        return 0;
    }

    // Calculate the 48-bit hash.
    *hash = toy_hash(buffer, (size_t)length);

    free(buffer);
    return 1;
}

int main(int argc, char *argv[])
{
    // Check if the correct number of command-line arguments is provided.
    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s file_a.pdf file_b.pdf\n",
                argv[0]);
        return 1;
    }

    // Declare variables to hold the hash values for both files.
    uint64_t hash_a, hash_b;

    // Calculate the hashes for both files. If either fails, exit with an error.
    if (!hash_file(argv[1], &hash_a) || !hash_file(argv[2], &hash_b)) {
        return 1;
    }

    printf("Hash A: %012" PRIx64 "\n", hash_a);
    printf("Hash B: %012" PRIx64 "\n", hash_b);

    printf("Result: %s\n",
           hash_a == hash_b ? "MATCH" : "DIFFERENT");

    return 0;
}