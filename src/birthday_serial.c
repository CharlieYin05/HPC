#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "toy_hash.h"

#define NONCE_OFFSET 16
#define STUDENT_ID_OFFSET 45
#define DEFAULT_MAX_NONCES 50000000ULL

typedef struct {
    unsigned char *bytes;
    size_t length;
} Pdf;

// A key is nonzero: (48-bit hash << 2) | (file_id + 1).
// File IDs are 0 (A) and 1 (B). 16 bytes per occupied/empty slot.
typedef struct {
    uint64_t key;
    uint64_t nonce;
} Entry;

typedef struct {
    Entry *slots;
    size_t capacity;
    size_t used;
} HashTable;

static double now_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int load_pdf(const char *path, Pdf *pdf)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }

    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return 0; }
    long size = ftell(fp);
    if (size < 54 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Invalid or too-small PDF: %s\n", path);
        fclose(fp);
        return 0;
    }

    pdf->length = (size_t)size;
    pdf->bytes = malloc(pdf->length);
    if (!pdf->bytes) { fclose(fp); return 0; }

    size_t got = fread(pdf->bytes, 1, pdf->length, fp);
    fclose(fp);
    if (got != pdf->length ||
        memcmp(pdf->bytes, "%PDF-1.4\n%NONCE=", 16) != 0 ||
        pdf->bytes[32] != '\n' ||
        memcmp(pdf->bytes + 33, "%STUDENT_ID=", 12) != 0 ||
        pdf->bytes[53] != '\n') {
        fprintf(stderr, "Read error or unexpected header: %s\n", path);
        free(pdf->bytes);
        pdf->bytes = NULL;
        return 0;
    }
    return 1;
}

static void set_nonce(unsigned char *pdf, uint64_t nonce)
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 15; i >= 0; --i) {
        pdf[NONCE_OFFSET + i] = (unsigned char)hex[nonce & 15ULL];
        nonce >>= 4;
    }
}

static uint64_t hash_trial(Pdf *pdf, uint64_t nonce)
{
    set_nonce(pdf->bytes, nonce);
    return toy_hash(pdf->bytes, pdf->length);
}

// Mix a 48-bit hash into an evenly-distributed table index.
static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int table_resize(HashTable *table, size_t new_capacity)
{
    Entry *new_slots = calloc(new_capacity, sizeof(*new_slots));
    if (!new_slots) return 0;

    for (size_t i = 0; i < table->capacity; ++i) {
        Entry old = table->slots[i];
        if (old.key == 0) continue;
        size_t index = (size_t)mix64(old.key >> 2) & (new_capacity - 1);
        while (new_slots[index].key != 0)
            index = (index + 1) & (new_capacity - 1);
        new_slots[index] = old;
    }
    free(table->slots);
    table->slots = new_slots;
    table->capacity = new_capacity;
    return 1;
}

// Returns 1 when a cross-file match is found, 0 otherwise, -1 on allocation failure.
static int table_check(HashTable *table, uint64_t hash, uint64_t nonce,
                       int file_id, uint64_t *matching_nonce)
{
    if ((table->used + 1) * 10 >= table->capacity * 7) {
        if (table->capacity > SIZE_MAX / 2 ||
            !table_resize(table, table->capacity * 2))
            return -1;
    }
    size_t index = (size_t)mix64(hash) & (table->capacity - 1);
    for (;;) {
        Entry *slot = &table->slots[index];
        if (slot->key == 0) {
            slot->key = (hash << 2) | (uint64_t)(file_id + 1);
            slot->nonce = nonce;
            ++table->used;
            return 0;
        }
        if ((slot->key >> 2) == hash) {
            int existing_file = (int)(slot->key & 3ULL) - 1;
            if (existing_file != file_id) {
                *matching_nonce = slot->nonce;
                return 1;
            }
            return 0; // Same-file hash; cross-file collisions only.
        }
        index = (index + 1) & (table->capacity - 1);
    }
}

static int save_pdf(const char *path, const Pdf *pdf)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 0; }
    size_t count = fwrite(pdf->bytes, 1, pdf->length, fp);
    int close_ok = (fclose(fp) == 0);
    int ok = (count == pdf->length && close_ok);
    if (!ok) fprintf(stderr, "Could not write %s\n", path);
    return ok;
}

int main(int argc, char *argv[])
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s file_a.pdf file_b.pdf STUDENT_ID [max_nonces_per_file]\n", argv[0]);
        return 1;
    }

    if (strlen(argv[3]) != 8 || strspn(argv[3], "0123456789") != 8) {
        fprintf(stderr, "STUDENT_ID must be exactly 8 digits.\n");
        return 1;
    }
    uint64_t max_nonces = DEFAULT_MAX_NONCES;
    if (argc == 5) {
        char *end = NULL;
        unsigned long long parsed = strtoull(argv[4], &end, 10);
        if (!argv[4][0] || *end != '\0' || parsed == 0) {
            fprintf(stderr, "max_nonces_per_file must be a positive integer.\n");
            return 1;
        }
        max_nonces = (uint64_t)parsed;
    }

    Pdf a = {0}, b = {0};
    if (!load_pdf(argv[1], &a) || !load_pdf(argv[2], &b)) {
        free(a.bytes);
        free(b.bytes);
        return 1;
    }
    memcpy(a.bytes + STUDENT_ID_OFFSET, argv[3], 8);
    memcpy(b.bytes + STUDENT_ID_OFFSET, argv[3], 8);

    HashTable table = {0};
    if (!table_resize(&table, 1U << 18)) {
        fprintf(stderr, "Failed to allocate hash table.\n");
        free(a.bytes);
        free(b.bytes);
        return 1;
    }

    double start = now_seconds();
    uint64_t nonce_a = 0, nonce_b = 0, collision_hash = 0;
    uint64_t trials = 0;
    int found = 0;

    for (uint64_t nonce = 0; nonce < max_nonces; ++nonce) {
        uint64_t other_nonce;
        uint64_t h_a = hash_trial(&a, nonce);
        ++trials;
        int result = table_check(&table, h_a, nonce, 0, &other_nonce);
        if (result < 0) { fprintf(stderr, "Hash table allocation failed.\n"); break; }
        if (result == 1) {
            nonce_a = nonce;
            nonce_b = other_nonce;
            collision_hash = h_a;
            found = 1;
            break;
        }

        uint64_t h_b = hash_trial(&b, nonce);
        ++trials;
        result = table_check(&table, h_b, nonce, 1, &other_nonce);
        if (result < 0) { fprintf(stderr, "Hash table allocation failed.\n"); break; }
        if (result == 1) {
            nonce_a = other_nonce;
            nonce_b = nonce;
            collision_hash = h_b;
            found = 1;
            break;
        }
        if ((nonce + 1) % 1000000ULL == 0)
            fprintf(stderr, "Tried %" PRIu64 " candidates (%.1f s)\n",
                    trials, now_seconds() - start);
    }
    double elapsed = now_seconds() - start;
    printf("Trials: %" PRIu64 ", search time: %.3f s, throughput: %.0f hashes/s\n",
           trials, elapsed, elapsed > 0 ? trials / elapsed : 0.0);

    int exit_code = 2;
    if (found) {
        uint64_t verify_a = hash_trial(&a, nonce_a);
        uint64_t verify_b = hash_trial(&b, nonce_b);
        printf("Nonce A: %016" PRIx64 "\nNonce B: %016" PRIx64 "\n",
               nonce_a, nonce_b);
        printf("Collision hash: %012" PRIx64 "\n", collision_hash);
        if (verify_a == verify_b && verify_a == collision_hash) {
            printf("Verification: MATCH\n");
            if (save_pdf("output/serial_a.pdf", &a) &&
                save_pdf("output/serial_b.pdf", &b)) {
                printf("Saved output/serial_a.pdf and output/serial_b.pdf\n");
                exit_code = 0;
            } else {
                exit_code = 1;
            }
        } else {
            fprintf(stderr, "Verification FAILED.\n");
            exit_code = 1;
        }
    } else {
        fprintf(stderr, "No collision found within the trial limit.\n");
    }
    free(table.slots);
    free(a.bytes);
    free(b.bytes);
    return exit_code;
}