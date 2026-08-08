#include <torch/extension.h>

std::vector<torch::Tensor> popkv_pack_cuda(torch::Tensor keys, int64_t token_capacity);
void popkv_pack_into_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset);
torch::Tensor popkv_score_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens);
void popkv_query_encode_out_cuda(
    torch::Tensor queries, torch::Tensor q_sign, torch::Tensor q_mag,
    torch::Tensor q_low, torch::Tensor q_delta, int64_t dim_group_size);
std::vector<torch::Tensor> popkv_pack_dim_cuda(
    torch::Tensor keys, int64_t token_capacity, int64_t dim_group_size);
void popkv_pack_into_dim_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset,
    int64_t dim_group_size);
torch::Tensor popkv_score_dim_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens, int64_t dim_group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack", &popkv_pack_cuda);
    m.def("pack_into", &popkv_pack_into_cuda);
    m.def("score", &popkv_score_cuda);
    m.def("query_encode_out", &popkv_query_encode_out_cuda);
    m.def("pack_dim", &popkv_pack_dim_cuda);
    m.def("pack_into_dim", &popkv_pack_into_dim_cuda);
    m.def("score_dim", &popkv_score_dim_cuda);
}
