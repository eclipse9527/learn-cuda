load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "global_memory_bandwidth_lib",
    srcs = ["global_memory_bandwidth.cu"],
    deps = ["//utils"],
)

cc_binary(
    name = "global_memory_bandwidth",
    deps = [":global_memory_bandwidth_lib"],
)

cuda_library(
    name = "cpu_gpu_bandwidth_lib",
    srcs = ["cpu_gpu_bandwidth.cu"],
    deps = ["//utils"],
)

cc_binary(
    name = "cpu_gpu_bandwidth",
    deps = [":cpu_gpu_bandwidth_lib"],
)
