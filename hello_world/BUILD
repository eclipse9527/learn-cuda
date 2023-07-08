load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "hello_world_lib",
    srcs = ["hello_world.cu"],
    deps = ["//utils"],
)

cc_binary(
    name = "hello_world",
    deps = [":hello_world_lib"],
)
