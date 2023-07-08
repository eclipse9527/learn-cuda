load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cuda_library(
    name = "cuda_event",
    hdrs = ["cuda_event.h"],
)

cc_library(
    name = "utils",
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_event",
        ":macros",
    ],
)
