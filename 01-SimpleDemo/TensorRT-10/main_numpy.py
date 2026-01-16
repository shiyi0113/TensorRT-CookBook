import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

trt_file = Path("model.trt")
input_tensor_name = "inputT0"
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)  # 准备测试数据

def run():
    #------------------构建期--------------------
    logger = trt.Logger(trt.Logger.ERROR)                                       # Create Logger, available level: VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR
    # 将TRT模型读入内存，即 engine_bytes。有的话直接读，没有的话进行构建。
    if trt_file.exists():                                                       # Load engine from file and skip building process if it existed
        with open(trt_file, "rb") as f:
            engine_bytes = f.read()  # 将模型读入内存（Host MEM）
        if not engine_bytes:
            raise ValueError("Fail getting serialized engine.")
        print(f"Succeed getting serialized engine:{len(engine_bytes)/1024**2:.2f}MB")
    else:
        # 初始化
        # logger
        # logger = trt.Logger(trt.Logger.ERROR)
        # builder
        builder = trt.Builder(logger)                                           
        # config
        config = builder.create_builder_config()                
        # network
        network = builder.create_network() #创建空网络 

        # 编辑
        profile = builder.create_optimization_profile()# Dynamic-Shape模式的核心  创建优化配置文件
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)     # Set workspace for the building process (all GPU memory is used by default)
        input_tensor = network.add_input(input_tensor_name, trt.float32, [-1, -1, -1])  # 定义输入张量：[-1, -1, -1] 意味着三维均不固定（动态）
        profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])  # 设置输入尺寸的范围：最小(min), 常见(opt), 最大(max)。
        config.add_optimization_profile(profile)# 将配置好的 Profile 注册到配置对象中
        identity_layer = network.add_identity(input_tensor)      # 添加一个恒等层。这里是最简单网络的演示
        network.mark_output(identity_layer.get_output(0))        # 设置输出层为刚刚的恒等层
        
        # serialized_network
        engine_bytes = builder.build_serialized_network(network, config) 
        if engine_bytes == None:
            print("Fail building engine")
            return
        print("Succeed building engine")
        with open(trt_file, "wb") as f:                                         # Save the serialized network as binaray file
            f.write(engine_bytes)
            print(f"Succeed saving engine ({trt_file})")
    #-------------------------------------------

    #------------------执行期--------------------
    # engine
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)   
    if engine == None:
        print("Fail getting engine for inference")
        return
    print("Succeed getting engine for inference")
    # context
    context = engine.create_execution_context()
    # 遍历所有输入和输出张量的名称
    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    # 为动态形状模型的输入张量设置本次推理的具体尺寸
    context.set_input_shape(input_tensor_name, data.shape)                      
    # 用于调试：打印各张量的类型、编译期形状（含 -1）和当前运行期确定的形状。
    for name in tensor_name_list:                                               
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")
    # 内存管理
    buffer = OrderedDict()                                                      
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        # 计算该张量在显存中所需的字节数。
        n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
        # 分配内存和显存
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]
    # 将输入数据转为内存连续的 NumPy 数组。
    buffer[input_tensor_name][0] = np.ascontiguousarray(data)                
    # 地址绑定：告诉 TensorRT：张量 name 对应的数据在 GPU 的 buffer[name][1] 这个地址上。
    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])                       
    # Host -> Device
    for name in tensor_name_list:                                              
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpy(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    # 异步执行推理
    context.execute_async_v3(0)                                                 
    # Device -> Host
    for name in tensor_name_list:                                               
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    # 打印结果
    for name in tensor_name_list:
        print(name)
        print(buffer[name][0])
    # 释放显存
    for _, device_buffer, _ in buffer.values(): 
        cudart.cudaFree(device_buffer)

if __name__ == "__main__":
    os.system("rm -rf *.trt")  # 删除目录下所有".trt"文件

    run()    # 创建TRT模型并推理
    run()    # 加载TRT模型并推理

    print("Finish")
