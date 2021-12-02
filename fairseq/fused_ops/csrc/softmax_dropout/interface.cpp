#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input, 
                               float                dropout_prob
                                                  );

torch::Tensor bwd_cuda(
                        int heads,
                        torch::Tensor const& output_grads, 
                        torch::Tensor const& softmax_results,
                        torch::Tensor const& dropout_mask,
                        float                dropout_prob
                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(input.type().scalarType() == at::ScalarType::Half || input.type().scalarType() == at::ScalarType::BFloat16 || input.type().scalarType() == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");


  return fwd_cuda(
                                 is_training,
                                 heads, 
                                 input, 
                                 dropout_prob
                                );
}

torch::Tensor bwd(int heads,
                torch::Tensor const& output_grads, 
                torch::Tensor const& softmax_results,
                torch::Tensor const& dropout_mask,
                float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half || output_grads.type().scalarType()      == at::ScalarType::BFloat16 || output_grads.type().scalarType()      == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");
  AT_ASSERTM(softmax_results.type().scalarType()   == at::ScalarType::Half || softmax_results.type().scalarType()   == at::ScalarType::BFloat16 || softmax_results.type().scalarType()   == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");

  return bwd_cuda(
                heads,
                output_grads,
                softmax_results, 
                dropout_mask, 
                dropout_prob
                );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwd, "softmax dropout -- Forward.");
  m.def("backward", &bwd, "softmax dropout -- Backward.");
}
