#include "hnswlib/hnswlib.h"
#include "hnswlib_wrapper.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static const size_t kMaxDim = 32;

// --- float16 ↔ float32 conversion ---
// All our values are in [-1, 1] — well within float16 precision.

static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t s = (uint16_t)((x >> 16) & 0x8000u);
    int e = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0)  return s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint16_t)e << 10) | (uint16_t)(m >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = (h & 0x3ffu) << 13;
    if (e == 0)  return *reinterpret_cast<float*>(&s);       // ±0 or subnormal → 0
    if (e == 31) { uint32_t r = s | 0x7f800000u | m; return *reinterpret_cast<float*>(&r); }
    uint32_t r = s | ((e + 112u) << 23) | m;                 // e+112 = e-15+127
    return *reinterpret_cast<float*>(&r);
}

// --- Custom L2 space for float16 storage ---

static size_t g_dim = 0;

static float l2_f16(const void* a, const void* b, const void* param) {
    size_t qty = *(const size_t*)param;
    const uint16_t* pa = (const uint16_t*)a;
    const uint16_t* pb = (const uint16_t*)b;
    float res = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        float da = f16_to_f32(pa[i]);
        float db = f16_to_f32(pb[i]);
        float d  = da - db;
        res += d * d;
    }
    return res;
}

class L2SpaceF16 : public hnswlib::SpaceInterface<float> {
    size_t dim_;
public:
    explicit L2SpaceF16(size_t dim) : dim_(dim) {}
    size_t get_data_size() override { return dim_ * sizeof(uint16_t); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return l2_f16; }
    void* get_dist_func_param() override { return &dim_; }
};

// --- Global state ---
static hnswlib::HierarchicalNSW<float>* g_index   = nullptr;
static L2SpaceF16*                       g_space   = nullptr;
static unsigned char*                    g_labels  = nullptr;  // 0=legit, 1=fraud

extern "C" {

void hnsw_init(int dim, int max_elements, int M, int ef_construction) {
    delete g_index;
    delete g_space;
    free(g_labels);
    g_dim    = (size_t)dim;
    assert(g_dim <= kMaxDim);
    g_space  = new L2SpaceF16(g_dim);
    g_index  = new hnswlib::HierarchicalNSW<float>(
        g_space, (size_t)max_elements, (size_t)M, (size_t)ef_construction);
    g_labels = (unsigned char*)calloc((size_t)max_elements, 1);
}

void hnsw_add(float* vec, int label, int id) {
    // Convert float32 → float16 before handing off to hnswlib
    assert((size_t)id < g_index->max_elements_);
    uint16_t f16[kMaxDim];
    for (size_t i = 0; i < g_dim; i++) {
        f16[i] = f32_to_f16(vec[i]);
    }
    g_labels[(size_t)id] = (unsigned char)label;
    g_index->addPoint((void*)f16, (size_t)id);
}

void hnsw_set_ef(int ef) {
    g_index->setEf((size_t)ef);
}

int hnsw_search(float* query, int k, unsigned char* out_labels) {
    // Convert query float32 → float16
    uint16_t f16[kMaxDim];
    for (size_t i = 0; i < g_dim; i++) {
        f16[i] = f32_to_f16(query[i]);
    }
    auto result = g_index->searchKnn((void*)f16, (size_t)k);
    int count = (int)result.size();
    for (int i = count - 1; i >= 0; i--) {
        auto top = result.top();
        out_labels[i] = g_labels[top.second];
        result.pop();
    }
    return count;
}

} // extern "C"
