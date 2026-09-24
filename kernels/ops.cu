// These FP32 kernels favor a transparent correctness baseline over peak throughput.
// One block handles a row/head. No scores matrix is materialized in attention.
extern "C" __global__ void rms(const float* x, const float* w, float* y,
                               int dim, float eps) {
    __shared__ float partial[256];
    int row = blockIdx.x, lane = threadIdx.x;
    float sum = 0;
    for (int d = lane; d < dim; d += blockDim.x) {
        float a = x[row * dim + d]; sum += a * a;
    }
    partial[lane] = sum; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / dim + eps);
    for (int d = lane; d < dim; d += blockDim.x)
        y[row * dim + d] = x[row * dim + d] * scale * w[d];
}

extern "C" __global__ void rotate(const float* x, const int* positions,
                                  float* y, int heads, int dim, float theta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2, total = gridDim.y; // batch carried by blockIdx.y
    int batch = blockIdx.y;
    if (index >= heads * half) return;
    int head = index / half, d = index % half;
    int base = (batch * heads + head) * dim;
    float angle = positions[batch] * powf(theta, -2.0f * d / dim);
    float c = cosf(angle), s = sinf(angle);
    float a = x[base+d], b = x[base+d+half];
    y[base+d] = a*c-b*s; y[base+d+half] = b*c+a*s;
}

extern "C" __global__ void paged_attention(
    const float* q, const float* k, const float* v, const int* tables,
    const int* lengths, float* out, int width, int page_size,
    int query_heads, int kv_heads, int dim) {
    // Exactly 256 threads, dim <= 256. Each thread owns one output coordinate.
    __shared__ float dot[256];
    __shared__ float m, l, alpha, beta;
    int lane = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
    int kv_head = head / (query_heads / kv_heads);
    int qbase = (row * query_heads + head) * dim;
    float accumulator = 0;
    if (lane == 0) { m = -1.0f/0.0f; l = 0; }
    __syncthreads();
    for (int token = 0; token < lengths[row]; ++token) {
        int page = tables[row * width + token / page_size];
        long long base = ((long long)page * page_size + token % page_size)
                         * kv_heads * dim + kv_head * dim;
        dot[lane] = lane < dim ? q[qbase+lane] * k[base+lane] : 0;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (lane < stride) dot[lane] += dot[lane+stride];
            __syncthreads();
        }
        if (lane == 0) {
            float score = dot[0] * rsqrtf((float)dim);
            float next_m = fmaxf(m, score);
            alpha = expf(m-next_m); beta = expf(score-next_m);
            l = alpha*l + beta; m = next_m;
        }
        __syncthreads();
        if (lane < dim) accumulator = alpha*accumulator + beta*v[base+lane];
        // alpha/beta cannot be overwritten until every lane has consumed them.
        __syncthreads();
    }
    if (lane < dim) out[qbase+lane] = accumulator / l;
}
