// Ada-compatible execution kernels. There is no Hopper TMA/WGMMA dependency.
extern "C" __global__ void embedding(const int* ids,const scalar_t* table,scalar_t* out,int dim){
    int d=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y;
    if(d<dim)out[row*dim+d]=write_scalar(read_scalar(table[(long long)ids[row]*dim+d]));
}

extern "C" __global__ void swiglu(const scalar_t* gate,const scalar_t* up,scalar_t* out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float g=read_scalar(gate[i]);out[i]=write_scalar((g/(1+expf(-g)))*read_scalar(up[i]));}
}

extern "C" __global__ void residual_norm(
    const scalar_t* x,const scalar_t* residual,const scalar_t* weight,scalar_t* sum,scalar_t* out,int dim,float eps){
    __shared__ float reduce[256];
    int lane=threadIdx.x,row=blockIdx.x;float squares=0;
    for(int d=lane;d<dim;d+=256){
        float value=read_scalar(write_scalar(read_scalar(x[row*dim+d])+read_scalar(residual[row*dim+d])));
        sum[row*dim+d]=write_scalar(value);squares+=value*value;
    }
    reduce[lane]=squares;__syncthreads();
    for(int s=128;s;s>>=1){if(lane<s)reduce[lane]+=reduce[lane+s];__syncthreads();}
    float scale=rsqrtf(reduce[0]/dim+eps);
    for(int d=lane;d<dim;d+=256)out[row*dim+d]=write_scalar(read_scalar(sum[row*dim+d])*scale*read_scalar(weight[d]));
}

// Combine K's RoPE conversion with paged K/V writes. Slots are token-slot
// indices into the physical pool, not logical sequence positions.
extern "C" __global__ void rope_scatter(
    const scalar_t* k,const scalar_t* v,const int* positions,const int* slots,
    scalar_t* cache_k,scalar_t* cache_v,int heads,int dim,float theta){
    int i=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;
    if(i>=heads*dim)return;
    int head=i/dim,d=i%dim,half=dim/2,pair=d%half;
    float angle=positions[row]*powf(theta,-2.0f*pair/dim);
    int base=(row*heads+head)*dim;
    float a=read_scalar(k[base+pair]),b=read_scalar(k[base+pair+half]);
    float value=d<half?a*cosf(angle)-b*sinf(angle):b*cosf(angle)+a*sinf(angle);
    long long target=(long long)slots[row]*heads*dim+i;
    cache_k[target]=write_scalar(value);cache_v[target]=write_scalar(read_scalar(v[row*heads*dim+i]));
}

extern "C" __global__ void add_residual(const scalar_t* x,const scalar_t* r,scalar_t* out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=write_scalar(read_scalar(x[i])+read_scalar(r[i]));
}

// Each warp owns one query head. Registers hold up to 8 coordinates per lane;
// shuffle reductions replace CTA-wide shared-memory barriers for every token.
extern "C" __global__ void warp_attention(
    const scalar_t* q,const scalar_t* k,const scalar_t* v,const int* tables,const int* lengths,
    scalar_t* partial,float* stats,int width,int page_size,int heads,int kv_heads,int dim,int splits){
    int lane=threadIdx.x,head=blockIdx.x*blockDim.y+threadIdx.y;
    int row=blockIdx.y,split=blockIdx.z;
    if(head>=heads)return; // whole warp exits together
    int kh=head/(heads/kv_heads),qb=(row*heads+head)*dim;
    int chunk=(lengths[row]+splits-1)/splits;
    int begin=split*chunk,end=min(begin+chunk,lengths[row]);
    float acc[8]={0,0,0,0,0,0,0,0};
    float query[8];
    #pragma unroll
    for(int j=0;j<8;++j){int d=lane+j*32;query[j]=d<dim?read_scalar(q[qb+d]):0;}
    float maximum=-1.0f/0.0f,denominator=0;
    for(int t=begin;t<end;++t){
        int page=tables[row*width+t/page_size];
        long long base=((long long)page*page_size+t%page_size)*kv_heads*dim+kh*dim;
        float dot=0;
        #pragma unroll
        for(int j=0;j<8;++j){int d=lane+j*32;if(d<dim)dot+=query[j]*read_scalar(k[base+d]);}
        for(int delta=16;delta;delta>>=1)dot+=__shfl_down_sync(0xffffffff,dot,delta);
        float score=__shfl_sync(0xffffffff,dot,0)*rsqrtf((float)dim);
        float next=fmaxf(maximum,score),alpha=expf(maximum-next),beta=expf(score-next);
        denominator=alpha*denominator+beta;maximum=next;
        #pragma unroll
        for(int j=0;j<8;++j){int d=lane+j*32;if(d<dim)acc[j]=alpha*acc[j]+beta*read_scalar(v[base+d]);}
    }
    int part=(row*heads+head)*splits+split;
    #pragma unroll
    for(int j=0;j<8;++j){int d=lane+j*32;if(d<dim)partial[part*dim+d]=write_scalar(denominator>0?acc[j]/denominator:0);}
    if(lane==0){stats[part*2]=maximum;stats[part*2+1]=denominator;}
}
