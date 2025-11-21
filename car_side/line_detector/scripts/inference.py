#!/usr/bin/env python3

import engine_utils
import numpy as np
import tensorrt as trt
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, trt_engine_path, batch_size=1):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """
        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None
        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        # print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))
        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)
        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()
        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine)

    def infer(self, full_img, output_shapes):
        """Infers model on given image.
        Args:
            image_path (str): image to run object detection model on
        """

        np.copyto(self.inputs[0], full_img.ravel())

        self.do_inference(
            self.context, bindings=self.bindings,
            stream=self.stream)

        outputs = [
            output.reshape(shape)
            for output, shape in zip(self.outputs, output_shapes)
        ]
        return outputs

    def do_inference(self, context, bindings, stream, batch_size=1):
        """Execute asynchronous inference on the TensorRT context.

        Args:
            context: TensorRT execution context.
            bindings: List of binding indices for inputs/outputs.
            stream: CUDA stream for asynchronous execution.
            batch_size (int): Number of images in the batch.
        """
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
