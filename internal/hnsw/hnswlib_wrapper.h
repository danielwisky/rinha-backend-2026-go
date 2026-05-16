#ifdef __cplusplus
extern "C" {
#endif

void hnsw_init(int dim, int max_elements, int M, int ef_construction);
void hnsw_add(float* vec, int label, int id);
void hnsw_set_ef(int ef);
int  hnsw_search(float* query, int k, unsigned char* out_labels);

// Persistence — writes <path>.graph (hnswlib) and <path>.labels (uint8 array).
// Returns 0 on success, -1 on failure.
int  hnsw_save(const char* path);

// Loads a previously saved index. Returns the number of elements loaded, or -1 on failure.
// max_elements caps capacity; pass the same value used at build time.
int  hnsw_load(const char* path, int dim, int max_elements);

// Loads via mmap — data_level0 and per-element linklists live in a read-only
// mapping of the file. Two processes mapping the same inode share the page
// cache. Returns element count or -1.
int  hnsw_load_mmap(const char* path, int dim, int max_elements);

#ifdef __cplusplus
}
#endif
