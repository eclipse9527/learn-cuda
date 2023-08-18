load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "mat_transpose_lib",
    srcs = ["mat_transpose.cu"],
    deps = ["//utils"],
)

cc_binary(
    name = "mat_transpose",
    deps = [":mat_transpose_lib"],
)
