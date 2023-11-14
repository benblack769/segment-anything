# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import ImageEncoderOnnxModel

import os
import argparse
import warnings

import onnx
import sys
from onnx import onnx_pb as onnx_proto
import numpy as np
from collections import deque


try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM image encoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--use-preprocess",
    action="store_true",
    help="Whether to preprocess the image by resizing, standardizing, etc.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
    gelu_approximate: bool = False,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = ImageEncoderOnnxModel(
        model=sam,
        use_preprocess=use_preprocess,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = sam.image_encoder.img_size
    if use_preprocess:
        dummy_input = {
            "input_image": torch.randn((image_size, image_size, 3), dtype=torch.float)
        }
        # dynamic_axes = {
        #     "input_image": {0: "image_height", 1: "image_width"},
        # }
    else:
        dummy_input = {
            "input_image": torch.randn(
                (1, 3, image_size, image_size), dtype=torch.float
            )
        }
    dynamic_axes = None

    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        if False and model_type == "vit_h":
            output_dir, output_file = os.path.split(output)
            if output_dir:
                os.makedirs(output_dir, mode=0o777, exist_ok=True)
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                output,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        else:
            with open(output, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_input.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=list(dummy_input.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_input.items()}
        providers = ["CPUExecutionProvider"]

        if model_type == "vit_h":
            session_option = onnxruntime.SessionOptions()
            ort_session = onnxruntime.InferenceSession(output, providers=providers)
            param_file = os.listdir(output_dir)
            param_file.remove(output_file)
            for i, layer in enumerate(param_file):
                with open(os.path.join(output_dir, layer), "rb") as fp:
                    weights = np.frombuffer(fp.read(), dtype=np.float32)
                    weights = onnxruntime.OrtValue.ortvalue_from_numpy(weights)
                    session_option.add_initializer(layer, weights)
        else:
            
            sess_options = onnxruntime.SessionOptions()
            # Set graph optimization level
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            sess_options.optimized_model_filepath = "optimized_model.onnx"

            ort_session = onnxruntime.InferenceSession(output, sess_options, providers=providers)

        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")

        onnx_model_bf16 = convert_fp32_to_bf16(onnx.load("optimized_model.onnx"))
        onnx.save(onnx_model_bf16, f"{output}.bf16.onnx")
        print("FP32 to bf16 conversion successful.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


def convert_np_to_bf16(np_data):
    # mask out the lower 16 fractional bits
    buffer_data = np_data.tobytes()
    int_data = np.frombuffer(buffer_data,dtype="uint32")
    # print(int_data)

    masked_data = np.bitwise_and(int_data, np.uint32(0xffff0000)) 
    buffer_data = masked_data.tobytes()
    bf16_data = np.frombuffer(buffer_data, dtype="float32")
    return bf16_data

def mask_tensor_values(tensor):
    if tensor.float_data:
        bf16_data = convert_np_to_bf16(np.array(tensor.float_data))
        # int_list = _npfloat16_to_int(float16_data)
        # tensor.int32_data[:] = int_list
        tensor.float_data[:] = bf16_data
        # if bf16_data.size > 1000:
        #     print(bf16_data.shape)
    # convert raw_data (bytes type)
    if tensor.raw_data:
        # convert n.raw_data to float
        float32_list = np.frombuffer(tensor.raw_data, dtype='float32')
        # convert float to float16
        # if float32_list.size > 1000:
        #     print(float32_list.shape)
        float16_list = convert_np_to_bf16(float32_list)
        # convert float16 to bytes and write back to raw_data
        tensor.raw_data = float16_list.tobytes()

    # data = np.array(tensor)
    return tensor

def convert_fp32_to_bf16(model):
    queue = deque()
    queue.append(model)
    while queue:
        q = queue.popleft()
        if isinstance(q, onnx_proto.ModelProto):
            queue.append(q.graph)
        # if q is model.graph, push q.node.attribute (AttributeProto)
        if isinstance(q, onnx_proto.GraphProto):
            for n in q.node:
                for attr in n.attribute:
                    queue.append(attr)
        # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
        # and process node.attribute.t and node.attribute.tensors (TensorProto)
        if isinstance(q, onnx_proto.AttributeProto):
            queue.append(q.g)
            for n in q.graphs:
                queue.append(n) 
            if q.t.data_type == onnx_proto.TensorProto.FLOAT:
                q.t.CopyFrom(mask_tensor_values(q.t))
            for n in q.tensors:
                if n.data_type == onnx_proto.TensorProto.FLOAT:
                    n = mask_tensor_values(n)
        # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
        if isinstance(q, onnx_proto.GraphProto):
            for n in q.initializer:  # TensorProto type
                if n.data_type == onnx_proto.TensorProto.FLOAT:
                    n = mask_tensor_values(n)


    return model


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")