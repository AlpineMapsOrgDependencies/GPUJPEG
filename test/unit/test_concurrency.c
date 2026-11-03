#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

/*
 * Coders in separate threads, each with an own non-blocking stream, must produce the same results as a single coder.
 * Streams are delayed before each call (host function) so that queued work lags behind the host; quality (thus
 * quantization tables) changes each iteration, size once (buffer growth) and one worker recreates its coders.
 */

enum {
    CONC_ITERS = 6,
    CONC_MAX_SIZE = 256,
};

struct conc_worker
{
    bool recreate; ///< recreate the coders every 3rd iteration (and once destroy right after the creation)
    bool delay;    ///< delay the stream before each call
    uint64_t hash[CONC_ITERS][2];
    bool failed;
    thrd_t thread;
};

static uint8_t conc_image[CONC_MAX_SIZE * CONC_MAX_SIZE * 3];
static bool conc_delay_supported = true;

static uint64_t
conc_hash(const uint8_t* data, size_t len)
{
    uint64_t hash = 14695981039346656037ULL;
    for ( size_t i = 0; i < len; ++i ) {
        hash = (hash ^ data[i]) * 1099511628211ULL;
    }
    return hash;
}

static void CUDART_CB
conc_sleep(void* unused)
{
    (void)unused;
    thrd_sleep(&(struct timespec){.tv_nsec = 2 * 1000 * 1000}, NULL);
}

static void
conc_create(struct conc_worker* w, cudaStream_t stream, struct gpujpeg_encoder** encoder,
            struct gpujpeg_decoder** decoder)
{
    struct gpujpeg_decoder_init_parameters param = gpujpeg_decoder_default_init_parameters();
    param.stream = stream;
    param.verbose = GPUJPEG_LL_QUIET;
    *encoder = gpujpeg_encoder_create(stream);
    *decoder = gpujpeg_decoder_create_with_params(&param);
    w->failed |= *encoder == NULL || *decoder == NULL;
}

static int
conc_worker_run(void* arg)
{
    struct conc_worker* w = arg;
    struct gpujpeg_parameters p420 = gpujpeg_default_parameters();
    gpujpeg_parameters_chroma_subsampling(&p420, GPUJPEG_SUBSAMPLING_420);
    gpujpeg_subsampling_get_name(p420.comp_count, p420.sampling_factor); // thread-local buffer
    cudaStream_t stream = NULL;
    if ( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ) {
        w->failed = true;
        return 1;
    }
    struct gpujpeg_encoder* encoder = NULL;
    struct gpujpeg_decoder* decoder = NULL;
    conc_create(w, stream, &encoder, &decoder);
    for ( int it = 0; it < CONC_ITERS && !w->failed; ++it ) {
        if ( w->recreate && it % 3 == 2 ) {
            gpujpeg_decoder_destroy(decoder);
            gpujpeg_encoder_destroy(encoder);
            conc_create(w, stream, &encoder, &decoder);
            if ( it == 2 ) { // destroy right after creation
                struct gpujpeg_encoder* e = NULL;
                struct gpujpeg_decoder* d = NULL;
                conc_create(w, stream, &e, &d);
                gpujpeg_decoder_destroy(d);
                gpujpeg_encoder_destroy(e);
            }
        }
        struct gpujpeg_parameters param = gpujpeg_default_parameters();
        param.verbose = GPUJPEG_LL_QUIET;
        param.quality = it % 2 == 0 ? 90 : 60;
        struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
        param_image.width = param_image.height = it < CONC_ITERS / 2 ? CONC_MAX_SIZE / 2 : CONC_MAX_SIZE;
        struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(conc_image);
        uint8_t* jpeg = NULL;
        size_t jpeg_len = 0;
        if ( w->delay ) {
            cudaLaunchHostFunc(stream, conc_sleep, NULL);
        }
        if ( gpujpeg_encoder_encode(encoder, &param, &param_image, &input, &jpeg, &jpeg_len) != 0 ) {
            w->failed = true;
            break;
        }
        w->hash[it][0] = conc_hash(jpeg, jpeg_len);
        if ( w->delay ) {
            cudaLaunchHostFunc(stream, conc_sleep, NULL);
        }
        struct gpujpeg_decoder_output output;
        gpujpeg_decoder_output_set_default(&output);
        if ( gpujpeg_decoder_decode(decoder, jpeg, jpeg_len, &output) != 0 ) {
            w->failed = true;
            break;
        }
        w->hash[it][1] = conc_hash(output.data, output.data_size);
    }
    gpujpeg_decoder_destroy(decoder);
    gpujpeg_encoder_destroy(encoder);
    cudaStreamDestroy(stream);
    return 0;
}

void
test_concurrency();

void
test_concurrency()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    for ( unsigned i = 0; i < sizeof conc_image; ++i ) {
        conc_image[i] = (uint8_t)(i / 3 % CONC_MAX_SIZE + (i * 2654435761U >> 27));
    }
    // a host function in a stream may be unsupported (eg. by a GPU simulator)
    cudaStream_t stream = NULL;
    if ( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
         cudaLaunchHostFunc(stream, conc_sleep, NULL) != cudaSuccess ) {
        conc_delay_supported = false;
        (void)cudaGetLastError();
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    struct conc_worker reference = {.recreate = false};
    conc_worker_run(&reference);
    if ( reference.failed ) {
        fprintf(stderr, "single-threaded run failed!\n");
        abort();
    }
    struct gpujpeg_parameters p444 = gpujpeg_default_parameters();
    gpujpeg_parameters_chroma_subsampling(&p444, GPUJPEG_SUBSAMPLING_444);
    const char* name = gpujpeg_subsampling_get_name(p444.comp_count, p444.sampling_factor);
    struct conc_worker workers[2] = {
        {.recreate = true,  .delay = conc_delay_supported},
        {.recreate = false, .delay = conc_delay_supported},
    };
    for ( int i = 0; i < 2; ++i ) {
        if ( thrd_create(&workers[i].thread, conc_worker_run, &workers[i]) != thrd_success ) {
            printf("-- (cannot create threads)\n");
            for ( int j = 0; j < i; ++j ) {
                thrd_join(workers[j].thread, NULL);
            }
            return;
        }
    }
    for ( int i = 0; i < 2; ++i ) {
        thrd_join(workers[i].thread, NULL);
        if ( workers[i].failed || memcmp(workers[i].hash, reference.hash, sizeof reference.hash) != 0 ) {
            fprintf(stderr, "worker %d differs from the single-threaded run!\n", i);
            abort();
        }
    }
    if ( strcmp(name, "4:4:4") != 0 ) { // overwritten by workers if not thread-local
        fprintf(stderr, "subsampling name changed by another thread to %s!\n", name);
        abort();
    }
    printf("Ok%s\n", conc_delay_supported ? "" : " (without stream delays)");
}
