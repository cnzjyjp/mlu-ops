import numpy as np
import pytest
import bangpy
import bangpy.eager as eg
from stest import Stest as op
from bangpy.common import load_op_by_type
from stest import DTYPES, KERNEL_NAME, TARGET_LIST


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 2, 1, 1, 1000, 1),
    ]
)

@pytest.mark.parametrize(
    "dtype",
    DTYPES,
)
def test_stest(target, shape, dtype):
    if target not in TARGET_LIST:
        return

    nram_size = 0
    if target == "mlu370-s4":
        nram_size = 768 * 1024
        IO_BANDWIDTH = 307.2 * 2**30  # MLU370-s4: 307.2GB/s
    else:
        nram_size = 512 * 1024
        IO_BANDWIDTH = 2**40  # MLU290: 1024GB/s

    data_in0 = np.random.uniform(low=-100, high=100, size=shape)
    data_in1 = np.random.uniform(low=-100, high=100, size=shape)
    data_out = np.add(data_in0,data_in1)
    # data_out = data_in0 + data_in1
    dev = bangpy.device(0)
    data_in0_dev = bangpy.Array(data_in0.flatten().astype(dtype.as_numpy_dtype), dev)
    data_in1_dev = bangpy.Array(data_in1.flatten().astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_in0.flatten().shape, dtype.as_numpy_dtype), dev)
    data_total = len(data_in0.flatten())

    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    f1(
        data_in0_dev,
        data_in1_dev,
        data_out_dev,
        data_total,
    )

    evaluator = f1.time_evaluator(number=1, repeat=1, min_repeat_ms=0)
    latency = evaluator(data_in0_dev, data_in1_dev, data_out_dev, data_total,).median* 1e3
    print("Shape:", shape)
    print("dtype:", dtype)
    print("Hardware time : %f ms" % latency)

    # io_efficiency
    theory_io_size = data_total * dtype.bytes
    io_efficiency = 1000 * theory_io_size / (latency * IO_BANDWIDTH)
    print("theory_io_size : %f GB" % (theory_io_size / (2**30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%\n")

    data_out = data_out.flatten()
    data_out_dev = data_out_dev.numpy().flatten()
    diff = np.abs(data_out - data_out_dev)
    data_out = np.abs(data_out)
    maxdiff3 = 0
    if dtype == bangpy.float16:
        th = 1e-4
    elif dtype == bangpy.float32:
        th = 1e-6
    for i in range(10):
        print(data_out_dev[i])
    for i in range(data_total-10,data_total):
        print(data_out_dev[i])
    # for i, data in enumerate(data_out):
    #     if data > th:
    #         diff3 = diff[i] / data
    #     else:
    #         diff3 = diff[i]
    #     if diff3 > maxdiff3:
    #         maxdiff3 = diff3
    # assert maxdiff3 < 0.05


    
