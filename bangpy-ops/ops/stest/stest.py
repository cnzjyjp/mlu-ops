import bangpy
from bangpy.common.dtypes import DType
from bangpy.script import tcp, build_module, ty

# DTYPES = [bangpy.float16, bangpy.float32]
DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "stest"

class Stest(object):
    def __init__(self, dtype: ty.string) -> None:
        self.dtype = dtype
        self.dtype_size = DType(dtype).bytes

    def stest_body(
        self,
        c: ty.Buffer("nram"),
        a: ty.Buffer("nram"),
        b: ty.Buffer("nram"),
    ) -> None:
        tcp.add(c,a,b)

    def main(
        self,
        input0: ty.handle,
        input1: ty.handle,
        output: ty.handle,
        buffer_size: ty.int32,
    ) -> None:

        buffer_in0 = tcp.match_buffer(input0, [buffer_size], dtype=self.dtype)
        buffer_in1 = tcp.match_buffer(input1, [buffer_size], dtype=self.dtype)
        buffer_out = tcp.match_buffer(output, [buffer_size], dtype=self.dtype)

        tgt = tcp.target()
        task_num = tgt.cluster_num * tgt.core_num

        data_each_task = buffer_size // task_num
        data_rest = buffer_size % task_num
        single_buffer_size = (
                (tgt.nram_size - 10 * 1024) // 6 // 128 * 128 
            )
        data_each_buffer = single_buffer_size // self.dtype_size
        loop_num = data_each_task // data_each_buffer
        rest = data_each_task % data_each_buffer

        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                
                start = cluster_id * 4 * data_each_task
                for i in range(loop_num, pipeline = True):
                    buffer_as = tcp.alloc_buffer(
                            [4*data_each_buffer], dtype=self.dtype, scope="sram"
                    )  
                    buffer_bs = tcp.alloc_buffer(
                            [4*data_each_buffer], dtype=self.dtype, scope="sram"
                    ) 
                    buffer_cs = tcp.alloc_buffer(
                            [4*data_each_buffer], dtype=self.dtype, scope="sram"
                    ) 
                    buffer_a = tcp.alloc_buffer(
                            [data_each_buffer], dtype=self.dtype, scope="nram"
                    )  
                    buffer_b = tcp.alloc_buffer(
                            [data_each_buffer], dtype=self.dtype, scope="nram"
                    ) 
                    buffer_c = tcp.alloc_buffer(
                            [data_each_buffer], dtype=self.dtype, scope="nram"
                    )  
                    with tcp.block("data_copy"):                   
                        tcp.memcpy(buffer_as[0:4 * data_each_buffer], buffer_in0[start+ 4 * i * data_each_buffer: start + 4 * (i+1) * data_each_buffer])
                        tcp.memcpy(buffer_bs[0:4 * data_each_buffer], buffer_in1[start+ 4 * i * data_each_buffer: start + 4 * (i+1) * data_each_buffer])
                    with tcp.block("data_copy"):                   
                        tcp.memcpy(buffer_a[0: data_each_buffer], buffer_as[core_id*data_each_buffer : (core_id+1)*data_each_buffer])
                        tcp.memcpy(buffer_b[0: data_each_buffer], buffer_bs[core_id*data_each_buffer : (core_id+1)*data_each_buffer])
                    with tcp.block("compute"):
                        self.stest_body(buffer_c,buffer_a,buffer_b)
                    with tcp.block("data_copy"):
                        tcp.memcpy(buffer_cs[core_id*data_each_buffer : (core_id+1)*data_each_buffer], buffer_c[0:data_each_buffer])
                    with tcp.block("data_copy"):
                        tcp.memcpy(buffer_out[start+ 4 * i * data_each_buffer: start + 4 * (i+1) * data_each_buffer], buffer_cs[0:4*data_each_buffer])

                # if rest > 0:
                #     # buffer_as = tcp.alloc_buffer(
                #     #         [data_each_buffer], dtype=self.dtype, scope="sram"
                #     # )  
                #     # buffer_bs = tcp.alloc_buffer(
                #     #         [data_each_buffer], dtype=self.dtype, scope="sram"
                #     # ) 
                #     # buffer_cs = tcp.alloc_buffer(
                #     #         [data_each_buffer], dtype=self.dtype, scope="sram"
                #     # )
                #     buffer_a = tcp.alloc_buffer(
                #             [data_each_buffer], dtype=self.dtype, scope="nram"
                #     )  
                #     buffer_b = tcp.alloc_buffer(
                #             [data_each_buffer], dtype=self.dtype, scope="nram"
                #     ) 
                #     buffer_c = tcp.alloc_buffer(
                #             [data_each_buffer], dtype=self.dtype, scope="nram"
                #     )
                #     tcp.memcpy(buffer_a[0:rest], buffer_in0[start+ loop_num * data_each_buffer: start+ loop_num * data_each_buffer+rest])
                #     tcp.memcpy(buffer_b[0:rest], buffer_in1[start+ loop_num * data_each_buffer: start+ loop_num * data_each_buffer+rest])
                #     self.stest_body(buffer_c,buffer_a,buffer_b)
                #     tcp.memcpy(buffer_out[start+ loop_num * data_each_buffer: start + loop_num * data_each_buffer+rest], buffer_c[0:rest])

               
                # if data_rest > 0:
                #     if (task_id == task_num - 1):  
                #         start = buffer_size - data_rest
                #         buffer_a = tcp.alloc_buffer(
                #             [128], dtype=self.dtype, scope="nram"
                #         )
                #         buffer_b = tcp.alloc_buffer(
                #             [128], dtype=self.dtype, scope="nram"
                #         )
                #         buffer_c = tcp.alloc_buffer(
                #             [128], dtype=self.dtype, scope="nram"
                #         )
                #         tcp.memcpy(buffer_a[0:data_rest], buffer_in0[start:start +  data_rest])
                #         tcp.memcpy(buffer_b[0:data_rest], buffer_in1[start:start +  data_rest])
                #         self.stest_body(
                #             buffer_c[0:128],
                #             buffer_a[0:128],
                #             buffer_b[0:128],
                #         )
                #         tcp.memcpy(buffer_out[start:start +  data_rest], buffer_c[0:data_rest])
                    
@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_stest(dtype=None, target=None):
    # build a executable module
    func = build_module.build(Stest(dtype.name), target_tag=target, name=KERNEL_NAME)
    return func
