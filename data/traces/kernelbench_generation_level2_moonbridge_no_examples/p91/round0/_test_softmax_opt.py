import avelang
import avelang.language as al

@avelang.jit
def softmax_channel_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32, C: al.constexpr, H: al.i32, W: al.i32,
):
    input_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)
    output_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)
    block_hw = al.block_id(0)
    b = al.block_id(1)
    tid = al.thread_id(0)
    c = tid
    h = block_hw // W
    w = block_hw % W
    val = al.convert(input_t[b, c, h, w], al.f32)
    shared = al.make_shared((C,), al.f32)
    
    max_val = val
    other = al.shuffle_down(max_val, 32, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 16, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 8, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 4, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 2, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 1, 64)
    if other > max_val:
        max_val = other
    
    warp_id = tid // 64
    lane_id = tid % 64
    if lane_id == 0:
        shared[warp_id] = max_val
    al.syncthreads()
    
    num_warps = C // 64
    if tid == 0:
        global_max = shared[0]
        for w in al.range(1, num_warps):
            if shared[w] > global_max:
                global_max = shared[w]
        shared[0] = global_max
    al.syncthreads()
    
    global_max = shared[0]
    shifted = al.exp(val - global_max)
    
    sum_val = shifted
    sum_val = sum_val + al.shuffle_down(sum_val, 32, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 16, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 8, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 4, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 2, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 1, 64)
    
    if lane_id == 0:
        shared[warp_id] = sum_val
    al.syncthreads()
    
    if tid == 0:
        global_sum = shared[0]
        for w in al.range(1, num_warps):
            global_sum = global_sum + shared[w]
        shared[0] = global_sum
    al.syncthreads()
    
    global_sum = shared[0]
    result = shifted / global_sum
    output_t[b, c, h, w] = al.convert(result, al.bf16)
