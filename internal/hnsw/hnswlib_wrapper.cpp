#include "hnswlib/hnswlib.h"
#include "hnswlib_wrapper.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdint>

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

} // extern "C"
