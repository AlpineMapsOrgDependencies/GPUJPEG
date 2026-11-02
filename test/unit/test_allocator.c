#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Counting device allocator: coders must free everything they allocated through it with their stream, must not
 * allocate in steady state and must not leak if an allocation fails. Stream-ordered allocation is used if supported.
 */

enum {
    ALLOC_MAX_LIVE = 128,
};

#define ALLOC_CHECK(cond)                                                                                              \
    if ( !(cond) ) {                                                                                                   \
        fprintf(stderr, "%s:%d: allocator check '%s' failed!\n", __FILE__, __LINE__, #cond);                           \
        abort();                                                                                                       \
    }

struct alloc_state
{
    cudaStream_t stream; ///< expected stream
    bool async;          ///< cudaMallocAsync/cudaFreeAsync instead of cudaMalloc/cudaFree
    int fail_at;         ///< fail the n-th allocation (counted from 1), 0 for never
    int allocs;
    int errors; ///< unknown pointer freed or a wrong stream
    void* live[ALLOC_MAX_LIVE];
    int live_count;
};

static uint8_t alloc_image[256 * 256 * 3];

static void*
alloc_device_alloc(void* user_data, size_t size, cudaStream_t stream)
{
    struct alloc_state* s = user_data;
    s->errors += stream != s->stream;
    void* ptr = NULL;
    if ( ++s->allocs == s->fail_at || s->live_count == ALLOC_MAX_LIVE ) {
        return NULL;
    }
#if CUDART_VERSION >= 11020
    if ( s->async && cudaMallocAsync(&ptr, size, stream) != cudaSuccess ) {
        ptr = NULL;
    }
#endif
    if ( !s->async && cudaMalloc(&ptr, size) != cudaSuccess ) {
        ptr = NULL;
    }
    if ( ptr != NULL ) {
        s->live[s->live_count++] = ptr;
    }
    return ptr;
}

static void
alloc_device_free(void* user_data, void* ptr, cudaStream_t stream)
{
    struct alloc_state* s = user_data;
    s->errors += stream != s->stream;
    for ( int i = 0; i < s->live_count; ++i ) {
        if ( s->live[i] == ptr ) {
            s->live[i] = s->live[--s->live_count];
#if CUDART_VERSION >= 11020
            if ( s->async ) {
                cudaFreeAsync(ptr, stream);
                return;
            }
#endif
            cudaFree(ptr);
            return;
        }
    }
    s->errors += 1;
}

/// creates coders, codes 128x128 and then 256x256 (1 + steady_calls times); @retval true if everything succeeded
static bool
alloc_sequence(struct alloc_state* s, int steady_calls, int* allocs_before_steady)
{
    struct gpujpeg_allocator allocator = {alloc_device_alloc, alloc_device_free, s};
    struct gpujpeg_decoder_init_parameters param_dec = gpujpeg_decoder_default_init_parameters();
    param_dec.stream = s->stream;
    param_dec.verbose = GPUJPEG_LL_QUIET;
    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create_with_allocator(s->stream, &allocator);
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create_with_allocator(&param_dec, &allocator);
    bool ok = encoder != NULL && decoder != NULL;
    for ( int i = 0; ok && i < 2 + steady_calls; ++i ) {
        if ( i == 2 ) {
            *allocs_before_steady = s->allocs;
        }
        struct gpujpeg_parameters param = gpujpeg_default_parameters();
        param.verbose = GPUJPEG_LL_QUIET;
        struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
        param_image.width = param_image.height = i == 0 ? 128 : 256;
        struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(alloc_image);
        uint8_t* jpeg = NULL;
        size_t jpeg_len = 0;
        struct gpujpeg_decoder_output output;
        gpujpeg_decoder_output_set_default(&output);
        ok = gpujpeg_encoder_encode(encoder, &param, &param_image, &input, &jpeg, &jpeg_len) == 0 &&
             gpujpeg_decoder_decode(decoder, jpeg, jpeg_len, &output) == 0;
    }
    if ( decoder != NULL ) {
        gpujpeg_decoder_destroy(decoder);
    }
    if ( encoder != NULL ) {
        gpujpeg_encoder_destroy(encoder);
    }
    return ok;
}

void
test_allocator();

void
test_allocator()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    for ( unsigned i = 0; i < sizeof alloc_image; ++i ) {
        alloc_image[i] = (uint8_t)(i / 3 % 256 + (i * 2654435761U >> 27));
    }
    struct alloc_state s = {.async = false};
    ALLOC_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) == cudaSuccess);
#if CUDART_VERSION >= 11020
    int device = 0;
    int pools = 0;
    void* probe = NULL;
    cudaGetDevice(&device);
    s.async = cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess && pools &&
              cudaMallocAsync(&probe, 1, s.stream) == cudaSuccess && cudaFreeAsync(probe, s.stream) == cudaSuccess;
    (void)cudaGetLastError();
#endif

    int allocs_before_steady = 0;
    ALLOC_CHECK(alloc_sequence(&s, 5, &allocs_before_steady));
    ALLOC_CHECK(s.allocs > 0 && s.allocs == allocs_before_steady); // nothing allocated in steady state
    ALLOC_CHECK(s.live_count == 0 && s.errors == 0);

    // a failing allocation at creation or growth must not leak
    for ( int n = 1; n <= s.allocs; ++n ) {
        struct alloc_state f = {.stream = s.stream, .async = s.async, .fail_at = n};
        ALLOC_CHECK(!alloc_sequence(&f, 0, NULL));
        ALLOC_CHECK(f.live_count == 0 && f.errors == 0);
    }

    struct gpujpeg_allocator half = {alloc_device_alloc, NULL, &s};
    ALLOC_CHECK(gpujpeg_encoder_create_with_allocator(s.stream, &half) == NULL);
    cudaStreamDestroy(s.stream);
    printf("Ok (%d allocations%s)\n", s.allocs, s.async ? ", stream-ordered" : "");
}
