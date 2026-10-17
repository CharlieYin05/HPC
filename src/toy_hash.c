#include "toy_hash.h"

uint64_t toy_hash(const unsigned char *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    while (n--) {
        h ^= *p++;
        h *= 0x100000001b3ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h & 0x0000ffffffffffffULL;
}
