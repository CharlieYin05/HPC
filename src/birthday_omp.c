/* Stage 3: OpenMP birthday attack.
 * Parallel work: nonce generation + PDF hashing.
 * Single-thread merge: shared collision table (simple, race-free baseline).
 * Build: gcc-16 -std=c11 -O2 -Wall -Wextra -fopenmp \
 *          src/birthday_omp.c src/toy_hash.c -o birthday_omp
 */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toy_hash.h"

#define NONCE_OFFSET 16
#define STUDENT_ID_OFFSET 45
#define DEFAULT_MAX_NONCES UINT64_C(50000000)
#define DEFAULT_BATCH 16384U

/* A PDF is a byte array. Each worker has its OWN byte arrays. */
typedef struct { unsigned char *bytes; size_t length; } Pdf;
typedef struct { Pdf a, b; } Worker;

typedef struct {
    uint64_t hash;
    uint64_t nonce;
    int file_id;                 /* 0 = A, 1 = B */
} Candidate;

typedef struct { uint64_t nonce_a, nonce_b, hash; } Collision;

/* key = (48-bit hash << 3) | file marker (1=A, 2=B) | matched flag (4)
 * key == 0 marks an empty slot. "Matched" prevents counting the same
 * collision hash twice, even when searching for K > 1 collisions. */
typedef struct { uint64_t key, nonce; } Entry;
typedef struct { Entry *slots; size_t capacity, used; } HashTable;

static int parse_positive(const char *s, uint64_t *value)
{
    if (!s[0] || s[0] == '-') return 0;
    errno = 0;
    char *end;
    unsigned long long x = strtoull(s, &end, 10);
    if (errno || *end || x == 0) return 0;
    *value = (uint64_t)x;
    return 1;
}

static int load_pdf(const char *path, Pdf *pdf)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return 0; }
    long length = ftell(fp);
    if (length < 54 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Invalid PDF size: %s\n", path);
        fclose(fp);
        return 0;
    }
    pdf->length = (size_t)length;
    pdf->bytes = malloc(pdf->length);
    if (!pdf->bytes) { fclose(fp); return 0; }
    size_t count = fread(pdf->bytes, 1, pdf->length, fp);
    fclose(fp);
    if (count != pdf->length ||
        memcmp(pdf->bytes, "%PDF-1.4\n%NONCE=", 16) != 0 ||
        pdf->bytes[32] != '\n' ||
        memcmp(pdf->bytes + 33, "%STUDENT_ID=", 12) != 0 ||
        pdf->bytes[53] != '\n') {
        fprintf(stderr, "Invalid header/read error: %s\n", path);
        free(pdf->bytes);
        pdf->bytes = NULL;
        return 0;
    }
    return 1;
}

static void set_nonce(unsigned char *bytes, uint64_t nonce)
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 15; i >= 0; --i) {
        bytes[NONCE_OFFSET + i] = (unsigned char)hex[nonce & 15ULL];
        nonce >>= 4;
    }
}

static uint64_t hash_nonce(Pdf *pdf, uint64_t nonce)
{
    set_nonce(pdf->bytes, nonce);
    return toy_hash(pdf->bytes, pdf->length);
}

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int table_resize(HashTable *t, size_t new_capacity)
{
    if (!new_capacity || new_capacity > SIZE_MAX / sizeof(Entry)) return 0;
    Entry *slots = calloc(new_capacity, sizeof(*slots));
    if (!slots) return 0;
    for (size_t i = 0; i < t->capacity; ++i) {
        Entry old = t->slots[i];
        if (!old.key) continue;
        size_t index = (size_t)mix64(old.key >> 3) & (new_capacity - 1);
        while (slots[index].key) index = (index + 1) & (new_capacity - 1);
        slots[index] = old;
    }
    free(t->slots);
    t->slots = slots;
    t->capacity = new_capacity;
    return 1;
}

/* Return: 1 = NEW cross-file collision, 0 = none/duplicate, -1 = OOM. */
static int table_check(HashTable *t, Candidate c, Collision *collision)
{
    if (t->used >= t->capacity - t->capacity / 3) {
        if (t->capacity > SIZE_MAX / 2 || !table_resize(t, t->capacity * 2))
            return -1;
    }
    size_t index = (size_t)mix64(c.hash) & (t->capacity - 1);
    for (;;) {
        Entry *e = &t->slots[index];
        if (!e->key) {
            e->key = (c.hash << 3) | (uint64_t)(c.file_id + 1);
            e->nonce = c.nonce;
            ++t->used;
            return 0;
        }
        if ((e->key >> 3) == c.hash) {
            if (e->key & 4ULL) return 0; /* already counted */
            int old_file = (int)(e->key & 3ULL) - 1;
            if (old_file == c.file_id) return 0;
            collision->hash = c.hash;
            collision->nonce_a = c.file_id == 0 ? c.nonce : e->nonce;
            collision->nonce_b = c.file_id == 1 ? c.nonce : e->nonce;
            e->key |= 4ULL;
            return 1;
        }
        index = (index + 1) & (t->capacity - 1);
    }
}

static int save_pdf(const char *path, const Pdf *pdf)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 0; }
    size_t written = fwrite(pdf->bytes, 1, pdf->length, fp);
    int closed = (fclose(fp) == 0);
    if (written != pdf->length || !closed) {
        fprintf(stderr, "Failed to save %s\n", path);
        return 0;
    }
    return 1;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s A.pdf B.pdf STUDENT_ID [max_nonces_per_file] [options]\n"
        "Options:\n"
        "  --threads N      OpenMP threads (default: OMP_NUM_THREADS/runtime)\n"
        "  --batch N        candidate hashes per batch (default: 16384)\n"
        "  --collisions K   distinct collision hashes to find (default: 1)\n"
        "  --benchmark N    process exactly N candidate hashes, no early stop\n"
        "Examples:\n"
        "  %s data/example_a.pdf data/example_b.pdf 00000000 50000000 --threads 4\n"
        "  %s data/example_a.pdf data/example_b.pdf 00000000 --threads 4 --benchmark 2000000\n",
        prog, prog, prog);
}

int main(int argc, char **argv)
{
    int exit_code = 1;
    Pdf a = {0}, b = {0};
    Worker *workers = NULL;
    Candidate *batch = NULL;
    Collision *collisions = NULL;
    HashTable table = {0};
    uint64_t max_nonces = DEFAULT_MAX_NONCES, requested_k = 1;
    uint64_t bench_trials = 0;
    size_t batch_size = DEFAULT_BATCH;
    int threads = omp_get_max_threads();
    int benchmark = 0;

    if (argc < 4 || strlen(argv[3]) != 8 || strspn(argv[3], "0123456789") != 8) {
        usage(argv[0]);
        return 1;
    }
    int arg = 4;
    if (arg < argc && argv[arg][0] != '-') {
        if (!parse_positive(argv[arg++], &max_nonces)) { usage(argv[0]); return 1; }
    }
    while (arg < argc) {
        if (arg + 1 >= argc) { usage(argv[0]); return 1; }
        uint64_t number;
        if (!parse_positive(argv[arg + 1], &number)) { usage(argv[0]); return 1; }
        if (strcmp(argv[arg], "--threads") == 0 && number <= INT_MAX)
            threads = (int)number;
        else if (strcmp(argv[arg], "--batch") == 0 && number <= SIZE_MAX / sizeof(Candidate))
            batch_size = (size_t)number;
        else if (strcmp(argv[arg], "--collisions") == 0 && number <= SIZE_MAX / sizeof(Collision))
            requested_k = number;
        else if (strcmp(argv[arg], "--benchmark") == 0) {
            benchmark = 1;
            bench_trials = number;
        } else { usage(argv[0]); return 1; }
        arg += 2;
    }
    if (!benchmark && max_nonces > UINT64_MAX / 2) {
        fprintf(stderr, "max_nonces_per_file is too large.\n");
        return 1;
    }
    if (threads > 1024) {
        fprintf(stderr, "Please use <= 1024 OpenMP threads.\n");
        return 1;
    }
    if (!load_pdf(argv[1], &a) || !load_pdf(argv[2], &b)) goto cleanup;
    memcpy(a.bytes + STUDENT_ID_OFFSET, argv[3], 8);
    memcpy(b.bytes + STUDENT_ID_OFFSET, argv[3], 8);

    /* Setup is excluded from timed search. */
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    workers = calloc((size_t)threads, sizeof(*workers));
    if (!workers) goto oom;
    for (int tid = 0; tid < threads; ++tid) {
        workers[tid].a.length = a.length;
        workers[tid].b.length = b.length;
        workers[tid].a.bytes = malloc(a.length);
        workers[tid].b.bytes = malloc(b.length);
        if (!workers[tid].a.bytes || !workers[tid].b.bytes) goto oom;
        memcpy(workers[tid].a.bytes, a.bytes, a.length);
        memcpy(workers[tid].b.bytes, b.bytes, b.length);
    }
    batch = malloc(batch_size * sizeof(*batch));
    if (!batch || !table_resize(&table, 1U << 18)) goto oom;
    if (!benchmark) {
        collisions = calloc((size_t)requested_k, sizeof(*collisions));
        if (!collisions) goto oom;
    }

    uint64_t limit = benchmark ? bench_trials : max_nonces * 2;
    uint64_t generated = 0, checked = 0, distinct_found = 0;
    int failed = 0;
    double start = omp_get_wtime();

    while (generated < limit && !failed && (benchmark || distinct_found < requested_k)) {
        uint64_t left = limit - generated;
        size_t count = left < batch_size ? (size_t)left : batch_size;
        uint64_t base = generated;

        /* Each iteration writes to a UNIQUE batch[i].
         * Each thread modifies only its OWN PDF buffers: no data race. */
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            uint64_t trial = base + (uint64_t)i;
            int file_id = (int)(trial & 1ULL);
            uint64_t nonce = trial >> 1;
            Worker *w = &workers[omp_get_thread_num()];
            Pdf *pdf = file_id == 0 ? &w->a : &w->b;
            batch[i].hash = hash_nonce(pdf, nonce);
            batch[i].nonce = nonce;
            batch[i].file_id = file_id;
        }
        generated += count;

        /* The entire batch is finished now (implicit OpenMP barrier).
         * ONE thread merges results into the table in Serial trial order.
         * This avoids locks and keeps the Stage 2 collision reproducible. */
        for (size_t i = 0; i < count; ++i) {
            Collision c;
            int result = table_check(&table, batch[i], &c);
            ++checked;
            if (result < 0) { failed = 1; break; }
            if (result == 1) {
                if (!benchmark) {
                    collisions[distinct_found] = c;
                    ++distinct_found;
                    if (distinct_found == requested_k) break;
                } else {
                    ++distinct_found;
                }
            }
        }
        if (generated / 2000000 != (generated - count) / 2000000)
            fprintf(stderr, "Generated %" PRIu64 " candidates (%.1f s)\n",
                    generated, omp_get_wtime() - start);
    }
    double elapsed = omp_get_wtime() - start;
    printf("Threads: %d, batch size: %zu\n", threads, batch_size);
    printf("Trials: %" PRIu64 ", candidates checked: %" PRIu64
           ", search time: %.3f s, throughput: %.0f hashes/s\n",
           generated, checked, elapsed,
           elapsed > 0 ? (double)generated / elapsed : 0.0);

    if (failed) {
        fprintf(stderr, "Collision table allocation failed.\n");
        goto cleanup;
    }
    if (benchmark) {
        printf("Benchmark completed: %" PRIu64 " candidate hashes, "
               "%" PRIu64 " distinct collision hashes observed.\n",
               generated, distinct_found);
        exit_code = 0;
        goto cleanup;
    }
    if (distinct_found < requested_k) {
        fprintf(stderr, "Found %" PRIu64 "/%" PRIu64
                " distinct collisions before trial limit.\n", distinct_found, requested_k);
        exit_code = 2;
        goto cleanup;
    }

    /* Independently verify each solution, then save corresponding PDFs. */
    for (uint64_t i = 0; i < distinct_found; ++i) {
        Collision c = collisions[i];
        uint64_t verify_a = hash_nonce(&a, c.nonce_a);
        uint64_t verify_b = hash_nonce(&b, c.nonce_b);
        printf("Collision #%" PRIu64 "\n", i + 1);
        printf("Nonce A: %016" PRIx64 "\nNonce B: %016" PRIx64
               "\nCollision hash: %012" PRIx64 "\n",
               c.nonce_a, c.nonce_b, c.hash);
        if (verify_a != verify_b || verify_a != c.hash) {
            fprintf(stderr, "Verification FAILED for collision #%" PRIu64 "\n", i + 1);
            goto cleanup;
        }
        printf("Verification: MATCH\n");
        char path_a[128], path_b[128];
        if (requested_k == 1) {
            strcpy(path_a, "output/omp_a.pdf");
            strcpy(path_b, "output/omp_b.pdf");
        } else {
            snprintf(path_a, sizeof(path_a), "output/omp_%03" PRIu64 "_a.pdf", i + 1);
            snprintf(path_b, sizeof(path_b), "output/omp_%03" PRIu64 "_b.pdf", i + 1);
        }
        if (!save_pdf(path_a, &a) || !save_pdf(path_b, &b)) goto cleanup;
        printf("Saved %s and %s\n", path_a, path_b);
    }
    exit_code = 0;
    goto cleanup;

oom:
    fprintf(stderr, "Out of memory preparing Stage 3.\n");
cleanup:
    free(collisions);
    free(table.slots);
    free(batch);
    if (workers) {
        for (int i = 0; i < threads; ++i) {
            free(workers[i].a.bytes);
            free(workers[i].b.bytes);
        }
    }
    free(workers);
    free(a.bytes);
    free(b.bytes);
    return exit_code;
}
