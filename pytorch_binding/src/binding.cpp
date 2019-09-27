#include <iostream>
#include <vector>

#include <numeric>

#include "ctc.h"

#include <torch/extension.h>
#ifdef WARPCTC_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    extern THCState* state;
#endif

int cpu_ctc(torch::Tensor probs,
            torch::Tensor grads,
            torch::Tensor labels,
            torch::Tensor label_sizes,
            torch::Tensor sizes,
            int minibatch_size,
            torch::Tensor costs,
            int blank_label) {

    int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads
    options.blank_label = blank_label;

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes = 0;
    //void* cpu_workspace = 0;
    switch (probs.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        get_workspace_size(label_sizes.data<int>(), sizes.data<int>(),
                           probs_size, minibatch_size,
                           options, &cpu_size_bytes);

        float* cpu_workspace = (float*) new char[cpu_size_bytes];

        compute_ctc_loss(probs.data<float>(), grads.data<float>(),
                         labels.data<int>(), label_sizes.data<int>(),
                         sizes.data<int>(), probs_size,
                         minibatch_size, costs.data<float>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        }
        return 0;
      case torch::ScalarType::Double:
        {
        get_workspace_size_f64(label_sizes.data<int>(), sizes.data<int>(),
                           probs_size, minibatch_size,
                           options, &cpu_size_bytes);

        double* cpu_workspace = (double*) new char[cpu_size_bytes];

        compute_ctc_loss_f64(probs.data<double>(), grads.data<double>(),
                         labels.data<int>(), label_sizes.data<int>(),
                         sizes.data<int>(), probs_size,
                         minibatch_size, costs.data<double>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        }
        return 0;
        break;
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsuported data type" << std::endl;
    }
    return -1;
}
#ifdef WARPCTC_ENABLE_GPU
int gpu_ctc(torch::Tensor probs,
           torch::Tensor grads,
           torch::Tensor labels,
           torch::Tensor label_sizes,
           torch::Tensor sizes,
           int minibatch_size,
           torch::Tensor costs,
           int blank_label) {

    int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.blank_label = blank_label;
    options.stream = THCState_getCurrentStream(state);

    size_t gpu_size_bytes = 0;
    switch (probs.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        get_workspace_size(label_sizes.data<int>(), sizes.data<int>(),
                          probs_size, minibatch_size,
                          options, &gpu_size_bytes);

        int device_id = probs.get_device();
        cudaSetDevice(device_id);
        void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

        compute_ctc_loss(probs.data<float>(), grads.data<float>(),
                        labels.data<int>(), label_sizes.data<int>(),
                        sizes.data<int>(), probs_size,
                        minibatch_size, costs.data<float>(),
                        gpu_workspace, options);

        THCudaFree(state, gpu_workspace);
        }
        return 0;
      case torch::ScalarType::Double:
        {
        get_workspace_size_f64(label_sizes.data<int>(), sizes.data<int>(),
                          probs_size, minibatch_size,
                          options, &gpu_size_bytes);

        int device_id = probs.get_device();
        cudaSetDevice(device_id);
        void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

        compute_ctc_loss_f64(probs.data<double>(), grads.data<double>(),
                        labels.data<int>(), label_sizes.data<int>(),
                        sizes.data<int>(), probs_size,
                        minibatch_size, costs.data<double>(),
                        gpu_workspace, options);

        THCudaFree(state, gpu_workspace);
        }
        return 0;
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsuported data type" << std::endl;
    }
    return -1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_ctc", &cpu_ctc, "CTC Loss function with cpu");
#ifdef WARPCTC_ENABLE_GPU
    m.def("gpu_ctc", &gpu_ctc, "CTC Loss function with gpu");
#endif
}
