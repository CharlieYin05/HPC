/* Stage 4 — pure MPI birthday attack.
 * All ranks compute non-overlapping candidate trials.
 * owner(hash) = hash % nranks; MPI_Alltoallv routes candidates to owners.
 * Every rank maintains its own hash table and discovers cross-file collisions.
 * MPI_Allreduce gives the same stop decision to all ranks after each batch.
 * Build: mpicc -std=c11 -O2 -Wall -Wextra src/birthday_mpi.c \
 *              src/toy_hash.c -o birthday_mpi
 */
#include <mpi.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toy_hash.h"

#define NONCE_OFFSET 16
#define STUDENT_ID_OFFSET 45
#define DEFAULT_MAX_NONCES UINT64_C(50000000)
#define DEFAULT_BATCH 16384U

typedef struct { unsigned char *bytes; size_t length; } Pdf;
typedef struct { uint64_t hash, nonce; int file_id; } Candidate;
typedef struct { uint64_t hash, nonce_a, nonce_b; } Collision;
/* Low bits of key: 1=A / 2=B, bit 2=already matched. 0=empty. */
typedef struct { uint64_t key, nonce; } Entry;
typedef struct { Entry *slots; size_t capacity, used; } HashTable;

static void die(int rank, const char *message) {
    fprintf(stderr, "[rank %d] %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static int parse_positive(const char *s, uint64_t *out) {
    if (!*s || *s == '-') return 0;
    errno = 0;
    char *end;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || *end || v == 0) return 0;
    *out = (uint64_t)v;
    return 1;
}

static int load_pdf(const char *path, Pdf *pdf) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return 0; }
    long len = ftell(fp);
    if (len < 54 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return 0; }
    pdf->length = (size_t)len;
    pdf->bytes = malloc(pdf->length);
    if (!pdf->bytes) { fclose(fp); return 0; }
    size_t nread = fread(pdf->bytes, 1, pdf->length, fp);
    fclose(fp);
    if (nread != pdf->length ||
        memcmp(pdf->bytes, "%PDF-1.4\n%NONCE=", 16) != 0 ||
        pdf->bytes[32] != '\n' ||
        memcmp(pdf->bytes + 33, "%STUDENT_ID=", 12) != 0 ||
        pdf->bytes[53] != '\n') {
        fprintf(stderr, "Invalid PDF header: %s\n", path);
        free(pdf->bytes); pdf->bytes = NULL;
        return 0;
    }
    return 1;
}

static void set_nonce(unsigned char *bytes, uint64_t nonce) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 15; i >= 0; --i) {
        bytes[NONCE_OFFSET + i] = (unsigned char)hex[nonce & 15ULL];
        nonce >>= 4;
    }
}

static uint64_t hash_nonce(Pdf *pdf, uint64_t nonce) {
    set_nonce(pdf->bytes, nonce);
    return toy_hash(pdf->bytes, pdf->length);
}

static int save_pdf(const char *path, const Pdf *pdf) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 0; }
    size_t n = fwrite(pdf->bytes, 1, pdf->length, fp);
    int ok = fclose(fp) == 0;
    return n == pdf->length && ok;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int table_resize(HashTable *t, size_t cap) {
    if (!cap || cap > SIZE_MAX / sizeof(Entry)) return 0;
    Entry *slots = calloc(cap, sizeof(Entry));
    if (!slots) return 0;
    for (size_t i = 0; i < t->capacity; ++i) {
        Entry e = t->slots[i];
        if (!e.key) continue;
        size_t j = (size_t)mix64(e.key >> 3) & (cap - 1);
        while (slots[j].key) j = (j + 1) & (cap - 1);
        slots[j] = e;
    }
    free(t->slots);
    t->slots = slots; t->capacity = cap;
    return 1;
}

/* Return: 1 = newly discovered cross-file hash; 0 = no new match; -1 = OOM. */
static int table_check(HashTable *t, Candidate c, Collision *out) {
    if (t->used >= t->capacity - t->capacity / 3) {
        if (t->capacity > SIZE_MAX / 2 || !table_resize(t, t->capacity * 2))
            return -1;
    }
    size_t j = (size_t)mix64(c.hash) & (t->capacity - 1);
    for (;;) {
        Entry *e = &t->slots[j];
        if (!e->key) {
            e->key = (c.hash << 3) | (uint64_t)(c.file_id + 1);
            e->nonce = c.nonce;
            ++t->used;
            return 0;
        }
        if ((e->key >> 3) == c.hash) {
            if (e->key & 4ULL) return 0; /* Already counted this hash. */
            int old_file = (int)(e->key & 3ULL) - 1;
            if (old_file == c.file_id) return 0;
            out->hash = c.hash;
            out->nonce_a = c.file_id == 0 ? c.nonce : e->nonce;
            out->nonce_b = c.file_id == 1 ? c.nonce : e->nonce;
            e->key |= 4ULL;
            return 1;
        }
        j = (j + 1) & (t->capacity - 1);
    }
}

/* Explicit MPI struct types, avoiding assumptions about C struct padding. */
static MPI_Datatype make_candidate_type(void) {
    MPI_Datatype dtype, resized;
    int blocks[3] = {1, 1, 1};
    MPI_Aint offsets[3] = {
        (MPI_Aint)offsetof(Candidate, hash),
        (MPI_Aint)offsetof(Candidate, nonce),
        (MPI_Aint)offsetof(Candidate, file_id)
    };
    MPI_Datatype types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_INT};
    MPI_Type_create_struct(3, blocks, offsets, types, &dtype);
    MPI_Type_create_resized(dtype, 0, (MPI_Aint)sizeof(Candidate), &resized);
    MPI_Type_free(&dtype);
    return resized;
}

static MPI_Datatype make_collision_type(void) {
    MPI_Datatype dtype;
    int blocks[3] = {1, 1, 1};
    MPI_Aint offsets[3] = {
        (MPI_Aint)offsetof(Collision, hash),
        (MPI_Aint)offsetof(Collision, nonce_a),
        (MPI_Aint)offsetof(Collision, nonce_b)
    };
    MPI_Datatype types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_UINT64_T};
    MPI_Type_create_struct(3, blocks, offsets, types, &dtype);
    MPI_Type_commit(&dtype);
    return dtype;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: mpirun -np P %s A.pdf B.pdf STUDENT_ID [max_nonces_per_file] [options]\n"
        "Options:\n"
        "  --batch B       Candidate hashes per rank per communication round (default 16384)\n"
        "  --collisions K  Find K different collision hashes (default 1)\n"
        "  --benchmark N   Execute exactly N candidate hash trials globally, no early stop\n"
        "Examples:\n"
        "  mpirun -np 4 %s data/example_a.pdf data/example_b.pdf 00000000 50000000\n"
        "  mpirun -np 4 %s data/example_a.pdf data/example_b.pdf 00000000 --batch 4096 --benchmark 2000000\n",
        prog, prog, prog);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) die(-1, "Required MPI thread support is unavailable");
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    uint64_t max_nonces = DEFAULT_MAX_NONCES, target_k = 1, benchmark_n = 0;
    uint64_t batch_u64 = DEFAULT_BATCH;
    int benchmark = 0;
    if (argc < 4 || strlen(argv[3]) != 8 || strspn(argv[3], "0123456789") != 8) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    int arg = 4;
    if (arg < argc && argv[arg][0] != '-') {
        if (!parse_positive(argv[arg++], &max_nonces)) die(rank, "Invalid max_nonces");
    }
    while (arg < argc) {
        uint64_t n;
        if (arg + 1 >= argc || !parse_positive(argv[arg + 1], &n))
            die(rank, "Invalid option or missing numeric value");
        if (strcmp(argv[arg], "--batch") == 0) batch_u64 = n;
        else if (strcmp(argv[arg], "--collisions") == 0) target_k = n;
        else if (strcmp(argv[arg], "--benchmark") == 0) {
            benchmark = 1; benchmark_n = n;
        } else die(rank, "Unknown option");
        arg += 2;
    }
    if (max_nonces > UINT64_MAX / 2) die(rank, "max_nonces is too large");
    if (target_k > (uint64_t)(INT_MAX / nranks)) die(rank, "K too large for MPI_Gatherv");
    if (batch_u64 > (uint64_t)INT_MAX / (uint64_t)nranks || batch_u64 > SIZE_MAX / sizeof(Candidate))
        die(rank, "Batch size too large for MPI_Alltoallv");
    const int batch_size = (int)batch_u64;
    const uint64_t stride = (uint64_t)nranks * batch_u64;
    const uint64_t limit = benchmark ? benchmark_n : 2 * max_nonces;

    Pdf a = {0}, b = {0};
    if (!load_pdf(argv[1], &a) || !load_pdf(argv[2], &b))
        die(rank, "Cannot load PDF input files");
    memcpy(a.bytes + STUDENT_ID_OFFSET, argv[3], 8);
    memcpy(b.bytes + STUDENT_ID_OFFSET, argv[3], 8);

    Candidate *local = malloc((size_t)batch_size * sizeof(Candidate));
    Candidate *sendbuf = malloc((size_t)batch_size * sizeof(Candidate));
    Candidate *recvbuf = malloc(sizeof(Candidate)); /* valid buffer even with zero receives */
    size_t recv_capacity = 1;
    int *sendcounts = calloc((size_t)nranks, sizeof(int));
    int *recvcounts = calloc((size_t)nranks, sizeof(int));
    int *sdispls = calloc((size_t)nranks, sizeof(int));
    int *rdispls = calloc((size_t)nranks, sizeof(int));
    int *positions = calloc((size_t)nranks, sizeof(int));
    HashTable table = {0};
    Collision *found = benchmark ? NULL : calloc((size_t)target_k, sizeof(Collision));
    if (!local || !sendbuf || !recvbuf || !sendcounts || !recvcounts || !sdispls ||
        !rdispls || !positions || (!benchmark && !found) ||
        !table_resize(&table, 1U << 18))
        die(rank, "Out of memory during initial allocation");

    MPI_Datatype cand_type = make_candidate_type();
    MPI_Datatype coll_type = make_collision_type();
    MPI_Type_commit(&cand_type);

    uint64_t generated_local = 0, found_local = 0, found_global = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (uint64_t base = 0; base < limit; ) {
        /* rank r computes trial indexes [base+r*batch, base+(r+1)*batch). */
        uint64_t offset = (uint64_t)rank * batch_u64;
        uint64_t remain = limit - base;
        size_t count = 0;
        if (offset < remain) {
            uint64_t available = remain - offset;
            count = (size_t)(available < batch_u64 ? available : batch_u64);
        }
        memset(sendcounts, 0, (size_t)nranks * sizeof(int));
        for (size_t i = 0; i < count; ++i) {
            uint64_t trial = base + offset + (uint64_t)i;
            int file_id = (int)(trial & 1ULL);
            uint64_t nonce = trial >> 1;
            uint64_t hash = hash_nonce(file_id == 0 ? &a : &b, nonce);
            local[i] = (Candidate){hash, nonce, file_id};
            int owner = (int)(hash % (uint64_t)nranks);
            ++sendcounts[owner];
        }
        generated_local += (uint64_t)count;

        /* Pack records in destination-rank order. */
        int next = 0;
        for (int r = 0; r < nranks; ++r) {
            sdispls[r] = next;
            positions[r] = next;
            next += sendcounts[r];
        }
        for (size_t i = 0; i < count; ++i) {
            int owner = (int)(local[i].hash % (uint64_t)nranks);
            sendbuf[positions[owner]++] = local[i];
        }

        MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT,
                     MPI_COMM_WORLD);
        int recv_total = 0;
        for (int r = 0; r < nranks; ++r) {
            rdispls[r] = recv_total;
            recv_total += recvcounts[r];
        }
        if ((size_t)recv_total > recv_capacity) {
            Candidate *larger = realloc(recvbuf, (size_t)recv_total * sizeof(Candidate));
            if (!larger) die(rank, "Out of memory during receive buffer growth");
            recvbuf = larger;
            recv_capacity = (size_t)recv_total;
        }
        /* Every rank must call all collectives, even if its local count is 0. */
        MPI_Alltoallv(sendbuf, sendcounts, sdispls, cand_type,
                      recvbuf, recvcounts, rdispls, cand_type, MPI_COMM_WORLD);

        for (int i = 0; i < recv_total; ++i) {
            Collision c;
            int result = table_check(&table, recvbuf[i], &c);
            if (result < 0) die(rank, "Out of memory growing distributed hash table");
            if (result == 1 && !benchmark && found_local < target_k)
                found[found_local++] = c;
        }
        /* Identical global decision on every process avoids collective deadlock. */
        if (!benchmark) {
            MPI_Allreduce(&found_local, &found_global, 1, MPI_UINT64_T,
                          MPI_SUM, MPI_COMM_WORLD);
            if (found_global >= target_k) break;
        }
        if (limit - base <= stride) break; /* avoid overflow */
        base += stride;
    }
    double local_seconds = MPI_Wtime() - start;
    double seconds = 0;
    uint64_t total_generated = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&generated_local, &total_generated, 1, MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);
    int status = 0;
    if (rank == 0) {
        printf("MPI ranks: %d, batch per rank: %d\n", nranks, batch_size);
        printf("Trials: %" PRIu64 ", search time: %.3f s, throughput: %.0f hashes/s\n",
               total_generated, seconds, seconds > 0 ? (double)total_generated / seconds : 0.0);
        if (benchmark) {
            printf("Benchmark completed: %" PRIu64 " total candidate hashes (global).\n",
                   total_generated);
        } else if (found_global < target_k) {
            fprintf(stderr, "Found %" PRIu64 "/%" PRIu64
                    " distinct collisions before trial limit.\n", found_global, target_k);
            status = 2;
        }
    }

    if (!benchmark && found_global >= target_k) {
        /* Gather only final collision metadata, NOT all hash table entries. */
        int my_count = (int)found_local;
        int *counts = rank == 0 ? calloc((size_t)nranks, sizeof(int)) : NULL;
        int *displs = rank == 0 ? calloc((size_t)nranks, sizeof(int)) : NULL;
        if (rank == 0 && (!counts || !displs)) die(rank, "Out of memory for gather counts");
        MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        Collision *all = NULL;
        if (rank == 0) {
            int total = 0;
            for (int r = 0; r < nranks; ++r) {
                displs[r] = total;
                total += counts[r];
            }
            all = malloc((size_t)(total ? total : 1) * sizeof(Collision));
            if (!all) die(rank, "Out of memory for gathered collisions");
        }
        MPI_Gatherv(found, my_count, coll_type, all, counts, displs,
                    coll_type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            uint64_t written = 0;
            for (int r = 0; r < nranks && written < target_k; ++r) {
                for (int i = 0; i < counts[r] && written < target_k; ++i) {
                    Collision c = all[displs[r] + i];
                    uint64_t ha = hash_nonce(&a, c.nonce_a);
                    uint64_t hb = hash_nonce(&b, c.nonce_b);
                    printf("Collision #%" PRIu64 "\n", written + 1);
                    printf("Nonce A: %016" PRIx64 "\nNonce B: %016" PRIx64
                           "\nCollision hash: %012" PRIx64 "\n",
                           c.nonce_a, c.nonce_b, c.hash);
                    if (ha != hb || ha != c.hash) {
                        fprintf(stderr, "Verification: FAILED\n");
                        status = 1;
                        break;
                    }
                    printf("Verification: MATCH\n");
                    char outa[128], outb[128];
                    if (target_k == 1) {
                        strcpy(outa, "output/mpi_a.pdf");
                        strcpy(outb, "output/mpi_b.pdf");
                    } else {
                        snprintf(outa, sizeof outa, "output/mpi_%03" PRIu64 "_a.pdf", written+1);
                        snprintf(outb, sizeof outb, "output/mpi_%03" PRIu64 "_b.pdf", written+1);
                    }
                    if (!save_pdf(outa, &a) || !save_pdf(outb, &b)) {
                        fprintf(stderr, "Could not write solved PDFs (create output/ first).\n");
                        status = 1;
                        break;
                    }
                    printf("Saved %s and %s\n", outa, outb);
                    ++written;
                }
                if (status) break;
            }
            if (written != target_k && status == 0) status = 1;
        }
        free(all); free(counts); free(displs);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&coll_type);
    MPI_Type_free(&cand_type);
    free(found); free(table.slots); free(positions); free(rdispls);
    free(sdispls); free(recvcounts); free(sendcounts);
    free(recvbuf); free(sendbuf); free(local);
    free(a.bytes); free(b.bytes);
    MPI_Finalize();
    return status;
}
