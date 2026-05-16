#include "hnswlib/hnswlib.h"
#include "hnswlib_wrapper.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const size_t kMaxDim = 32;

// --- float32 ↔ int8 quantization ---
// All values are in [-1, 1] (clamped). Scale = 127.

static inline int8_t f32_to_i8(float f) {
    float v = f;
    if (v < -1.0f) v = -1.0f;
    if (v >  1.0f) v =  1.0f;
    int q = (int)(v * 127.0f + (v >= 0 ? 0.5f : -0.5f));
    if (q < -127) q = -127;
    if (q >  127) q =  127;
    return (int8_t)q;
}

// --- L2 space for int8 storage ---

static size_t g_dim = 0;

// Squared L2 distance with int8 storage.
// (a-b) fits in int16, (a-b)^2 fits in int32, sum of up to kMaxDim*16384 fits easily in int32.
static float l2_i8(const void* a, const void* b, const void* param) {
    size_t qty = *(const size_t*)param;
    const int8_t* pa = (const int8_t*)a;
    const int8_t* pb = (const int8_t*)b;
    int32_t res = 0;
    for (size_t i = 0; i < qty; i++) {
        int32_t d = (int32_t)pa[i] - (int32_t)pb[i];
        res += d * d;
    }
    return (float)res;
}

class L2SpaceI8 : public hnswlib::SpaceInterface<float> {
    size_t dim_;
public:
    explicit L2SpaceI8(size_t dim) : dim_(dim) {}
    size_t get_data_size() override { return dim_ * sizeof(int8_t); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return l2_i8; }
    void* get_dist_func_param() override { return &dim_; }
};

// --- Global state ---
static hnswlib::HierarchicalNSW<float>* g_index   = nullptr;
static L2SpaceI8*                        g_space   = nullptr;
static unsigned char*                    g_labels  = nullptr;  // 0=legit, 1=fraud

// Track mmap'd region so we can detach hnswlib's pointers before it tries to
// free() them in clear()/destructor (which would crash on mmap pages).
static void*  g_mmap_addr = nullptr;
static size_t g_mmap_size = 0;

extern "C" {

void hnsw_init(int dim, int max_elements, int M, int ef_construction) {
    delete g_index;
    delete g_space;
    free(g_labels);
    g_dim    = (size_t)dim;
    assert(g_dim <= kMaxDim);
    g_space  = new L2SpaceI8(g_dim);
    g_index  = new hnswlib::HierarchicalNSW<float>(
        g_space, (size_t)max_elements, (size_t)M, (size_t)ef_construction);
    g_labels = (unsigned char*)calloc((size_t)max_elements, 1);
}

void hnsw_add(float* vec, int label, int id) {
    assert((size_t)id < g_index->max_elements_);
    int8_t q[kMaxDim];
    for (size_t i = 0; i < g_dim; i++) {
        q[i] = f32_to_i8(vec[i]);
    }
    g_labels[(size_t)id] = (unsigned char)label;
    g_index->addPoint((void*)q, (size_t)id);
}

void hnsw_set_ef(int ef) {
    g_index->setEf((size_t)ef);
}

int hnsw_search(float* query, int k, unsigned char* out_labels) {
    int8_t q[kMaxDim];
    for (size_t i = 0; i < g_dim; i++) {
        q[i] = f32_to_i8(query[i]);
    }
    auto result = g_index->searchKnn((void*)q, (size_t)k);
    int count = (int)result.size();
    for (int i = count - 1; i >= 0; i--) {
        auto top = result.top();
        out_labels[i] = g_labels[top.second];
        result.pop();
    }
    return count;
}

int hnsw_save(const char* path) {
    if (!g_index) return -1;
    std::string base(path);
    try {
        g_index->saveIndex(base + ".graph");
    } catch (...) {
        return -1;
    }
    size_t n = g_index->cur_element_count;
    FILE* f = fopen((base + ".labels").c_str(), "wb");
    if (!f) return -1;
    size_t w = fwrite(g_labels, 1, n, f);
    fclose(f);
    return (w == n) ? 0 : -1;
}

int hnsw_load(const char* path, int dim, int max_elements) {
    std::string base(path);
    delete g_index;
    delete g_space;
    free(g_labels);
    g_index = nullptr;
    g_space = nullptr;
    g_labels = nullptr;

    g_dim = (size_t)dim;
    assert(g_dim <= kMaxDim);
    g_space = new L2SpaceI8(g_dim);
    try {
        g_index = new hnswlib::HierarchicalNSW<float>(
            g_space, base + ".graph", false, (size_t)max_elements);
    } catch (...) {
        delete g_space;
        g_space = nullptr;
        return -1;
    }

    // (Memory cleanup of label_lookup_/link_list_locks_ was tried and removed.
    // Even though search doesn't touch them, freeing them correlated with
    // worse stress results — possibly because hnswlib touches them via paths
    // I haven't audited. Leave alone.)

    size_t n = g_index->cur_element_count;
    g_labels = (unsigned char*)calloc((size_t)max_elements, 1);
    if (!g_labels) return -1;

    FILE* f = fopen((base + ".labels").c_str(), "rb");
    if (!f) return -1;
    size_t r = fread(g_labels, 1, n, f);
    fclose(f);
    if (r != n) return -1;

    return (int)n;
}

// hnsw_load_mmap: load index where data_level0_memory_ and per-element
// linkLists_ live in a read-only mmap'd region of the file. The Linux page
// cache shares those pages across processes that map the same inode — so two
// api containers reading the same image layer share physical memory.
//
// Format (from hnswlib's saveIndex):
//   header (104 bytes):
//     size_t offsetLevel0_, max_elements_, cur_element_count,
//            size_data_per_element_, label_offset_, offsetData_;        (48)
//     int maxlevel_; unsigned int enterpoint_node_;                     ( 8)
//     size_t maxM_, maxM0_, M_;                                         (24)
//     double mult_; size_t ef_construction_;                            (16)
//   data_level0_memory_:    cur_element_count * size_data_per_element_ bytes
//   per-element linklists:  for each i: uint32 size, then size bytes
int hnsw_load_mmap(const char* path, int dim, int max_elements) {
    std::string base(path);
    std::string graph_path = base + ".graph";

    int fd = ::open(graph_path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (::fstat(fd, &st) < 0) { ::close(fd); return -1; }
    size_t file_size = (size_t)st.st_size;

    void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) return -1;
    ::madvise(map, file_size, MADV_RANDOM);

    // Parse header in-place from the mapping.
    const char* p = (const char*)map;
    size_t offsetLevel0_;        std::memcpy(&offsetLevel0_, p, 8); p += 8;
    size_t max_elements_field;   std::memcpy(&max_elements_field, p, 8); p += 8;
    size_t cur_element_count;    std::memcpy(&cur_element_count, p, 8); p += 8;
    size_t size_data_per_element_; std::memcpy(&size_data_per_element_, p, 8); p += 8;
    size_t label_offset_;        std::memcpy(&label_offset_, p, 8); p += 8;
    size_t offsetData_;          std::memcpy(&offsetData_, p, 8); p += 8;
    int maxlevel_;               std::memcpy(&maxlevel_, p, 4); p += 4;
    unsigned int enterpoint_node_; std::memcpy(&enterpoint_node_, p, 4); p += 4;
    size_t maxM_;                std::memcpy(&maxM_, p, 8); p += 8;
    size_t maxM0_;               std::memcpy(&maxM0_, p, 8); p += 8;
    size_t M_;                   std::memcpy(&M_, p, 8); p += 8;
    double mult_;                std::memcpy(&mult_, p, 8); p += 8;
    size_t ef_construction_;     std::memcpy(&ef_construction_, p, 8); p += 8;
    (void)offsetLevel0_; (void)label_offset_; (void)offsetData_;

    const char* level0_data = p;
    size_t level0_bytes = cur_element_count * size_data_per_element_;
    p += level0_bytes;

    // Tear down any prior state. If the prior state was mmap'd, detach its
    // pointers from hnswlib before delete to avoid free()ing mmap memory.
    if (g_index && g_mmap_addr) {
        g_index->data_level0_memory_ = nullptr;
        for (size_t i = 0; i < g_index->cur_element_count; i++) {
            g_index->linkLists_[i] = nullptr;
            g_index->element_levels_[i] = 0;
        }
    }
    delete g_index; delete g_space; free(g_labels);
    if (g_mmap_addr) ::munmap(g_mmap_addr, g_mmap_size);
    g_index = nullptr; g_space = nullptr; g_labels = nullptr;
    g_mmap_addr = map; g_mmap_size = file_size;

    g_dim = (size_t)dim;
    assert(g_dim <= kMaxDim);
    g_space = new L2SpaceI8(g_dim);

    // Construct an empty index with the right shape. This mallocs
    // data_level0_memory_ and linkLists_ that we'll override below.
    g_index = new hnswlib::HierarchicalNSW<float>(
        g_space, (size_t)max_elements_field, M_, ef_construction_);

    // Free the constructor's malloc'd buffer and point at our mmap region.
    ::free(g_index->data_level0_memory_);
    g_index->data_level0_memory_ = const_cast<char*>(level0_data);

    g_index->cur_element_count.store(cur_element_count);
    g_index->maxlevel_ = maxlevel_;
    g_index->enterpoint_node_ = (hnswlib::tableint)enterpoint_node_;
    g_index->mult_ = mult_;
    g_index->revSize_ = 1.0 / mult_;

    // Per-element linklists for higher levels. Each is (u32 size, size bytes).
    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize;
        std::memcpy(&linkListSize, p, sizeof(unsigned int));
        p += sizeof(unsigned int);
        if (linkListSize == 0) {
            g_index->element_levels_[i] = 0;
            g_index->linkLists_[i] = nullptr;
        } else {
            g_index->element_levels_[i] = (int)(linkListSize / g_index->size_links_per_element_);
            // Point into mmap'd region — no malloc, no copy.
            g_index->linkLists_[i] = const_cast<char*>(p);
            p += linkListSize;
        }
    }

    // Labels live in a separate small file — load normally (3MB).
    g_labels = (unsigned char*)calloc((size_t)max_elements, 1);
    if (!g_labels) return -1;
    FILE* f = fopen((base + ".labels").c_str(), "rb");
    if (!f) return -1;
    size_t r = fread(g_labels, 1, cur_element_count, f);
    fclose(f);
    if (r != cur_element_count) return -1;

    return (int)cur_element_count;
}

} // extern "C"
