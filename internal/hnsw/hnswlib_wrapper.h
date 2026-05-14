#ifdef __cplusplus
extern "C" {
#endif

void hnsw_init(int dim, int max_elements, int M, int ef_construction);
void hnsw_add(float* vec, int label, int id);
void hnsw_set_ef(int ef);
int  hnsw_search(float* query, int k, unsigned char* out_labels);

#ifdef __cplusplus
}
#endif
