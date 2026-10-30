// FP16/BF16 storage with FP32 reduction and softmax accumulation.
// One block handles a row/head. No scores matrix is materialized in attention.
extern "C" __global__ void rms(const scalar_t* x, const scalar_t* w, scalar_t* y,
                               int dim, float eps) {
    __shared__ float partial[256];
    int row = blockIdx.x, lane = threadIdx.x;
    float sum = 0;
    for (int d = lane; d < dim; d += blockDim.x) {
        float a = read_scalar(x[row * dim + d]); sum += a * a;
    }
    partial[lane] = sum; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / dim + eps);
    for (int d = lane; d < dim; d += blockDim.x)
        y[row * dim + d] = write_scalar(read_scalar(x[row * dim + d]) * scale * read_scalar(w[d]));
}

extern "C" __global__ void rotate(const scalar_t* x, const int* positions,
                                  scalar_t* y, int heads, int dim, float theta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int batch = blockIdx.y;
    if (index >= heads * half) return;
    int head = index / half, d = index % half;
    int base = (batch * heads + head) * dim;
    float angle = positions[batch] * powf(theta, -2.0f * d / dim);
    float c = cosf(angle), s = sinf(angle);
    float a = read_scalar(x[base+d]), b = read_scalar(x[base+d+half]);
    y[base+d] = write_scalar(a*c-b*s); y[base+d+half] = write_scalar(b*c+a*s);
}

// Split a long context into independent CTA reductions, then merge their
// (maximum, denominator, normalized output) triples using log-sum-exp weights.
extern "C" __global__ void split_attention(
    const scalar_t* q, const scalar_t* k, const scalar_t* v, const int* tables,
    const int* lengths, float* partial, float* stats, int width, int page_size,
    int query_heads, int kv_heads, int dim, int splits) {
    __shared__ float dot[256];
    __shared__ float m,l,alpha,beta;
    int lane=threadIdx.x,row=blockIdx.x,head=blockIdx.y,split=blockIdx.z;
    int kh=head/(query_heads/kv_heads), qb=(row*query_heads+head)*dim;
    int chunk=(lengths[row]+splits-1)/splits;
    int begin=split*chunk,end=min(begin+chunk,lengths[row]);
    int part=(row*query_heads+head)*splits+split;
    float accumulator=0;
    if(lane==0){m=-1.0f/0.0f;l=0;}
    __syncthreads();
    for(int t=begin;t<end;++t){
        int page=tables[row*width+t/page_size];
        long long base=((long long)page*page_size+t%page_size)*kv_heads*dim+kh*dim;
        dot[lane]=lane<dim?read_scalar(q[qb+lane])*read_scalar(k[base+lane]):0;
        __syncthreads();
        for(int stride=128;stride;stride>>=1){
            if(lane<stride)dot[lane]+=dot[lane+stride];
            __syncthreads();
        }
        if(lane==0){
            float score=dot[0]*rsqrtf((float)dim), next=fmaxf(m,score);
            alpha=expf(m-next);beta=expf(score-next);l=alpha*l+beta;m=next;
        }
        __syncthreads();
        if(lane<dim)accumulator=alpha*accumulator+beta*read_scalar(v[base+lane]);
        __syncthreads();
    }
    if(lane<dim)partial[part*dim+lane]=l>0?accumulator/l:0;
    if(lane==0){stats[part*2]=m;stats[part*2+1]=l;}
}

extern "C" __global__ void merge_attention(const float* partial,const float* stats,
                                           scalar_t* out,int heads,int dim,int splits){
    int d=threadIdx.x,row=blockIdx.x,head=blockIdx.y;
    if(d>=dim)return;
    int base=(row*heads+head)*splits;
    float maximum=-1.0f/0.0f;
    for(int s=0;s<splits;++s)maximum=fmaxf(maximum,stats[(base+s)*2]);
    float denominator=0,numerator=0;
    for(int s=0;s<splits;++s){
        float l=stats[(base+s)*2+1];
        float weight=l>0?expf(stats[(base+s)*2]-maximum)*l:0;
        denominator+=weight;numerator+=weight*partial[(base+s)*dim+d];
    }
    out[(row*heads+head)*dim+d]=write_scalar(numerator/denominator);
}

extern "C" __global__ void paged_attention(
    const scalar_t* q, const scalar_t* k, const scalar_t* v, const int* tables,
    const int* lengths, scalar_t* out, int width, int page_size,
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
        dot[lane] = lane < dim ? read_scalar(q[qbase+lane]) * read_scalar(k[base+lane]) : 0;
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
        if (lane < dim) accumulator = alpha*accumulator + beta*read_scalar(v[base+lane]);
        // alpha/beta cannot be overwritten until every lane has consumed them.
        __syncthreads();
    }
    if (lane < dim) out[qbase+lane] = write_scalar(accumulator / l);
}
