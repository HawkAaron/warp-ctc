#include <cmath>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

#include <chrono>

#include <iostream>

#include <ctc.h>

#include "test.h"

bool run_test(int B, int T, int L, int A, int num_threads) {
    std::mt19937 gen(2);

    int len = B * T * A;
    float * acts = genActs(len);

    std::vector<std::vector<int>> labels;
    std::vector<int> sizes;

    for (int mb = 0; mb < B; ++mb) {
        labels.push_back(genLabels(A, L));
        sizes.push_back(T);
    }

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(B);

    float * grads = new float[len];

    ctcOptions options{};
    options.loc = CTC_CPU;
    options.num_threads = num_threads;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), sizes.data(),
                                     A, sizes.size(), options,
                                     &cpu_alloc_bytes),
                    "Error: get_workspace_size in run_test");
    
    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    // average time
    std::vector<float> time;
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        throw_on_error(compute_ctc_loss(acts, grads,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        A,
                                        B,
                                        costs.data(),
                                        ctc_cpu_workspace,
                                        options),
                        "Error: compute_ctc_loss (0) in run_test");
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time.push_back(elapsed.count() * 1000);
        std::cout << "compute_ctc_loss elapsed time: " << elapsed.count() * 1000 << " ms\n";
    }

    float sum = 0;
    for (int i = 0; i < 10; ++i) {
        sum += time[i];
    }
    std::cout << "average 10 time cost: " << sum / time.size() << " ms\n";

    float cost = std::accumulate(costs.begin(), costs.end(), 0.);

    free(ctc_cpu_workspace);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Arguments: <Batch size> <Time step> <Label length> <Alphabet size>\n";
        return 1;
    }

    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int L = atoi(argv[3]);
    int A = atoi(argv[4]);
    std::cout << "Arguments: " \
                << "\nBatch size: " << B \
                << "\nTime step: " << T \
                << "\nLabel length: " << L \
                << "\nAlphabet size: " << A \
                << std::endl;
    
    int num_threads = 1;
    if (argc >= 6) {
        num_threads = atoi(argv[5]);
        std::cout << "Num threads: " << num_threads << std::endl;
    }

    run_test(B, T, L, A, num_threads);
}