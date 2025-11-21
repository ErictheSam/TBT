#!/usr/bin/env python3

# Utility functions for building/saving/loading TensorRT Engine
import pycuda.driver as cuda
import tensorrt as trt


def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.

    This function is similar to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        mem = cuda.pagelocked_empty(size, dtype)
        bindings.append(int(mem.base.get_device_pointer()))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(mem)
        else:
            outputs.append(mem)
    return inputs, outputs, bindings, stream

def load_engine(trt_runtime, engine_path):
    """Loads a TensorRT engine from a serialized file.

    Args:
        trt_runtime (trt.Runtime): TensorRT runtime instance
        engine_path (str): Path to the serialized engine file

    Returns:
        trt.ICudaEngine: The deserialized TensorRT engine
    """
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
