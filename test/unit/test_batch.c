#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "../../src/gpujpeg_decoder_internal.h"
#include "../../src/gpujpeg_encoder_internal.h"

/*
 * gpujpeg_decoder_decode_batch(): every image equals gpujpeg_decoder_decode() of it by a new decoder (pixels without
 * padding); batch rules, malformed headers (also for the single-image API), arguments and limits, reuse with growth
 * and allocation failures, table consistency after a failed batch and completion on a delayed stream.
 *
 * gpujpeg_encoder_encode_batch(): every JPEG equals gpujpeg_encoder_encode() of the image by a new encoder (but the
 * Exif DateTime) and the JPEGs are stored one after another; arguments, unsupported options, metadata and batch
 * limits leaving the results unchanged, reuse with growth and allocation failures, single images in between.
 */

#define TB_CHECK(cond)                                                                                                 \
    if ( !(cond) ) {                                                                                                   \
        fprintf(stderr, "%s:%d: batch check '%s' failed!\n", __FILE__, __LINE__, #cond);                               \
        abort();                                                                                                       \
    }

enum {
    TB_MAX_W = 256,
    TB_MAX_H = 256,
    TB_MAX_COUNT = 17,
    TB_GUARD = 64, ///< sentinel bytes after a custom host output
    TB_SENTINEL = 0xA5,
    TB_ALLOC_MAX_LIVE = 128,
};

struct tb_jpeg
{
    uint8_t* data;
    size_t size;
};

struct tb_desc
{
    int width;
    int height;
    gpujpeg_sampling_factor_t subsampling; ///< GPUJPEG_SUBSAMPLING_400 for grayscale
    int interleaved;
    int restart_interval;
    int quality;                                   ///< 0 for 75
    enum gpujpeg_color_space color_space_internal; ///< 0 for the default
};

struct tb_format
{
    enum gpujpeg_color_space color_space;
    enum gpujpeg_pixel_format pixel_format;
    const char* alignment;
};

/// decoded image
struct tb_image
{
    uint8_t* data;
    size_t size;
    struct gpujpeg_image_parameters param_image;
};

static const struct tb_format tb_default_format = {GPUJPEG_CS_DEFAULT, GPUJPEG_PIXFMT_AUTODETECT, "0"};
/// non-interleaved 4:2:0 with 32 + 2 * 8 segments (GPU Huffman decoder)
static const struct tb_desc tb_base = {64, 64, GPUJPEG_SUBSAMPLING_420, 0, 2, 0, 0};
static uint8_t tb_pixels[TB_MAX_W * TB_MAX_H * 3];
static const struct tb_jpeg* tb_oracle; ///< images decoded by tb_single() in tb_batch_equal() if not NULL
static cudaStream_t tb_delay_stream;    ///< delayed right before batch decoding in tb_batch_equal() if not NULL
static const struct tb_format* tb_fail_format = &tb_default_format; ///< output format in tb_batch_fail()

/// fills tb_pixels with content given by seed
static void
tb_fill(int seed)
{
    for ( unsigned i = 0; i < sizeof tb_pixels; ++i ) {
        tb_pixels[i] = (uint8_t)(i / 3 % TB_MAX_W * (2 * seed + 1) + (i * 2654435761U >> 26) + 37 * seed);
    }
}

/// encoding of images of a description (contents of seeds 0 to count - 1 for batches)
struct tb_enc_case
{
    struct tb_desc desc;
    int count;
    enum gpujpeg_pixel_format pixel_format; ///< 0 for the default
    enum gpujpeg_color_space color_space;   ///< 0 for the default
    int width_padding;
    bool device;         ///< batch input in device memory
    size_t stride_extra; ///< batch input_stride is the image size plus this, 0 for input_stride 0
    int segment_info;
    const char* const* options; ///< encoder options and their values, NULL-terminated (or NULL)
    size_t exif_ifd;            ///< offset of the Exif IFD in the TIFF data of Exif metadata, 0 for no Exif
    bool custom_date;           ///< an Exif tag option replaces the generated DateTime
};

static void
tb_enc_params(const struct tb_enc_case* c, struct gpujpeg_parameters* param,
              struct gpujpeg_image_parameters* param_image)
{
    const struct tb_desc* d = &c->desc;
    *param = gpujpeg_default_parameters();
    param->verbose = GPUJPEG_LL_QUIET;
    param->quality = d->quality != 0 ? d->quality : 75;
    param->interleaved = d->interleaved;
    param->restart_interval = d->restart_interval;
    param->segment_info = c->segment_info;
    *param_image = gpujpeg_default_image_parameters();
    param_image->width = d->width;
    param_image->height = d->height;
    param_image->width_padding = c->width_padding;
    if ( d->subsampling == GPUJPEG_SUBSAMPLING_400 ) {
        param_image->pixel_format = GPUJPEG_U8;
        param_image->color_space = GPUJPEG_YCBCR_JPEG;
    }
    else {
        gpujpeg_parameters_chroma_subsampling(param, d->subsampling);
    }
    if ( d->color_space_internal != 0 ) {
        param->color_space_internal = d->color_space_internal;
    }
    if ( c->pixel_format != 0 ) {
        param_image->pixel_format = c->pixel_format;
    }
    if ( c->color_space != 0 ) {
        param_image->color_space = c->color_space;
    }
}

/// gpujpeg_encoder_encode() of a host image
static struct tb_jpeg
tb_enc_single(struct gpujpeg_encoder* encoder, const struct tb_enc_case* c, uint8_t* image)
{
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    tb_enc_params(c, &param, &param_image);
    struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(image);
    uint8_t* out = NULL;
    size_t size = 0;
    TB_CHECK(gpujpeg_encoder_encode(encoder, &param, &param_image, &input, &out, &size) == 0);
    struct tb_jpeg j = {malloc(size), size};
    TB_CHECK(j.data != NULL);
    memcpy(j.data, out, size);
    return j;
}

/// encodes an image with content given by seed
static struct tb_jpeg
tb_encode(struct gpujpeg_encoder* encoder, const struct tb_desc* d, int seed)
{
    tb_fill(seed);
    const struct tb_enc_case c = {.desc = *d, .count = 1};
    return tb_enc_single(encoder, &c, tb_pixels);
}

static void
tb_encode_set(struct gpujpeg_encoder* encoder, const struct tb_desc* d, int count, struct tb_jpeg* jpegs)
{
    for ( int i = 0; i < count; ++i ) {
        jpegs[i] = tb_encode(encoder, d, i);
    }
}

static void
tb_free_set(struct tb_jpeg* jpegs, int count)
{
    for ( int i = 0; i < count; ++i ) {
        free(jpegs[i].data);
    }
}

/// copy of j with remove bytes at offset replaced by insert
static struct tb_jpeg
tb_splice(const struct tb_jpeg* j, size_t offset, size_t remove, const uint8_t* insert, size_t insert_size)
{
    struct tb_jpeg r = {malloc(j->size - remove + insert_size), j->size - remove + insert_size};
    TB_CHECK(r.data != NULL && offset + remove <= j->size);
    memcpy(r.data, j->data, offset);
    if ( insert_size > 0 ) {
        memcpy(r.data + offset, insert, insert_size);
    }
    memcpy(r.data + offset + insert_size, j->data + offset + remove, j->size - offset - remove);
    return r;
}

static struct tb_jpeg
tb_copy(const struct tb_jpeg* j)
{
    return tb_splice(j, 0, 0, NULL, 0);
}

/// size of the marker segment at pos (with the marker)
static size_t
tb_length(const struct tb_jpeg* j, size_t pos)
{
    return 2 + ((size_t)j->data[pos + 2] << 8 | j->data[pos + 3]);
}

/// offset of the n-th marker with the code, 0 if none
static size_t
tb_find(const struct tb_jpeg* j, uint8_t code, int n)
{
    size_t pos = 2; // after SOI
    while ( pos + 4 <= j->size && j->data[pos] == 0xFF ) {
        const uint8_t marker = j->data[pos + 1];
        if ( marker == code && n-- == 0 ) {
            return pos;
        }
        if ( marker == 0xD9 ) {
            break;
        }
        pos += tb_length(j, pos);
        if ( marker == 0xDA ) { // entropy-coded data up to the next marker other than RSTn
            while ( pos + 1 < j->size &&
                    (j->data[pos] != 0xFF || j->data[pos + 1] == 0 || (j->data[pos + 1] & 0xF8) == 0xD0) ) {
                pos++;
            }
        }
    }
    return 0;
}

/// restart markers of the n-th scan: offsets (up to max) and count, *end is the offset of the marker after the scan
static int
tb_restarts(const struct tb_jpeg* j, int scan, size_t* offsets, int max, size_t* end)
{
    size_t pos = tb_find(j, 0xDA, scan);
    TB_CHECK(pos != 0);
    int count = 0;
    for ( pos += tb_length(j, pos); pos + 1 < j->size; pos++ ) {
        if ( j->data[pos] != 0xFF || j->data[pos + 1] == 0 ) {
            continue;
        }
        if ( (j->data[pos + 1] & 0xF8) != 0xD0 ) {
            break;
        }
        if ( count < max ) {
            offsets[count] = pos;
        }
        count++;
        pos++;
    }
    *end = pos;
    return count;
}

/// copy of j with the marker segment at pos inserted before the n-th marker with the code
static struct tb_jpeg
tb_insert_segment(const struct tb_jpeg* j, size_t pos, uint8_t index, uint8_t code, int n)
{
    uint8_t segment[2 + 65535];
    const size_t length = tb_length(j, pos);
    memcpy(segment, j->data + pos, length);
    segment[4] = index; // table class and index of DQT/DHT
    const size_t at = tb_find(j, code, n);
    TB_CHECK(at != 0);
    return tb_splice(j, at, 0, segment, length);
}

static struct gpujpeg_decoder*
tb_decoder(cudaStream_t stream, const struct gpujpeg_allocator* allocator)
{
    struct gpujpeg_decoder_init_parameters param = gpujpeg_decoder_default_init_parameters();
    param.stream = stream;
    param.verbose = GPUJPEG_LL_QUIET;
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create_with_allocator(&param, allocator);
    TB_CHECK(decoder != NULL);
    return decoder;
}

static void
tb_set_format(struct gpujpeg_decoder* decoder, const struct tb_format* f)
{
    gpujpeg_decoder_set_output_format(decoder, f->color_space, f->pixel_format);
    TB_CHECK(gpujpeg_decoder_set_option(decoder, GPUJPEG_DEC_OPT_ALIGNMENT_BYTES_INT, f->alignment) == 0);
}

/// gpujpeg_decoder_decode() by a new decoder
static struct tb_image
tb_single(const struct tb_jpeg* j, const struct tb_format* f)
{
    struct gpujpeg_decoder* decoder = tb_decoder(NULL, NULL);
    tb_set_format(decoder, f);
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    TB_CHECK(gpujpeg_decoder_decode(decoder, j->data, j->size, &output) == 0);
    struct tb_image image = {malloc(output.data_size), output.data_size, output.param_image};
    TB_CHECK(image.data != NULL);
    memcpy(image.data, output.data, output.data_size);
    gpujpeg_decoder_destroy(decoder);
    return image;
}

/// compares the pixels (not the line padding)
static bool
tb_same(const struct tb_image* expected, const uint8_t* actual)
{
    const struct gpujpeg_image_parameters* p = &expected->param_image;
    if ( gpujpeg_pixel_format_is_planar(p->pixel_format) ) {
        return memcmp(expected->data, actual, expected->size) == 0;
    }
    const size_t pitch = expected->size / p->height;
    for ( int y = 0; y < p->height; ++y ) {
        if ( memcmp(expected->data + y * pitch, actual + y * pitch, pitch - p->width_padding) != 0 ) {
            return false;
        }
    }
    return true;
}

static void CUDART_CB
tb_sleep(void* unused)
{
    (void)unused;
    thrd_sleep(&(struct timespec){.tv_nsec = 20 * 1000 * 1000}, NULL);
}

/// batch decoding to the output type equals tb_single() of each image
static void
tb_batch_equal(struct gpujpeg_decoder* decoder, const struct tb_jpeg* jpegs, int count, const struct tb_format* f,
               enum gpujpeg_decoder_output_type type, size_t stride_extra, enum gpujpeg_huffman_route route)
{
    struct tb_image expected[TB_MAX_COUNT];
    uint8_t* images[TB_MAX_COUNT];
    size_t sizes[TB_MAX_COUNT];
    for ( int i = 0; i < count; ++i ) {
        expected[i] = tb_single(tb_oracle != NULL ? &tb_oracle[i] : &jpegs[i], f);
        images[i] = jpegs[i].data;
        sizes[i] = jpegs[i].size;
    }
    const size_t size = expected[0].size;
    const size_t stride = size + stride_extra;
    const size_t extent = (count - 1) * stride + size;
    const size_t padding = expected[0].param_image.width_padding;
    uint8_t* custom = NULL;
    struct gpujpeg_decoder_output output;
    memset(&output, 0, sizeof output);
    gpujpeg_decoder_output_set_default(&output);
    output.type = type;
    if ( type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) {
        custom = malloc(extent + TB_GUARD);
        TB_CHECK(custom != NULL);
        memset(custom, TB_SENTINEL, extent + TB_GUARD);
        output.data = custom;
    }
    if ( type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER ) {
        TB_CHECK(cudaMalloc((void**)&custom, extent) == cudaSuccess);
        output.data = custom;
    }
    tb_set_format(decoder, f);
    if ( tb_delay_stream != NULL ) {
        TB_CHECK(cudaLaunchHostFunc(tb_delay_stream, tb_sleep, NULL) == cudaSuccess);
    }
    int index = -2;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, count, images, sizes, stride_extra == 0 ? 0 : stride, &output,
                                          &index) == GPUJPEG_NOERR);
    TB_CHECK(index == -1 && output.data_size == size && output.metadata == decoder->batch.metadata);
    TB_CHECK(decoder->huffman_route == route);
    TB_CHECK(custom == NULL || output.data == custom);
    uint8_t* host = output.data;
    if ( type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER || type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER ) {
        host = malloc(extent);
        TB_CHECK(host != NULL && cudaMemcpy(host, output.data, extent, cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    for ( int i = 0; i < count; ++i ) {
        TB_CHECK(tb_same(&expected[i], host + i * stride));
        free(expected[i].data);
    }
    if ( type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) { // last line padding and after the images not written
        for ( size_t i = extent - padding; i < extent + TB_GUARD; ++i ) {
            TB_CHECK(custom[i] == TB_SENTINEL);
        }
        free(custom);
    }
    if ( host != output.data ) {
        free(host);
    }
    if ( type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER ) {
        cudaFree(custom);
    }
}

/// batch of jpegs with jpegs[index] replaced fails with rc, error index and reason (if not NONE); the custom output
/// and its descriptor are unchanged
static void
tb_batch_fail(struct gpujpeg_decoder* decoder, const struct tb_jpeg* jpegs, int count, int index,
              const struct tb_jpeg* replacement, int rc, int error_index, enum gpujpeg_batch_reason reason)
{
    uint8_t* images[TB_MAX_COUNT];
    size_t sizes[TB_MAX_COUNT];
    for ( int i = 0; i < count; ++i ) {
        images[i] = jpegs[i].data;
        sizes[i] = jpegs[i].size;
    }
    if ( replacement != NULL ) {
        images[index] = replacement->data;
        sizes[index] = replacement->size;
    }
    static uint8_t custom[TB_MAX_COUNT * TB_MAX_W * TB_MAX_H * 3];
    memset(custom, TB_SENTINEL, sizeof custom);
    struct gpujpeg_decoder_output output;
    memset(&output, 0, sizeof output);
    gpujpeg_decoder_output_set_custom(&output, custom);
    tb_set_format(decoder, tb_fail_format);
    int actual_index = -2;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, count, images, sizes, 0, &output, &actual_index) == rc);
    TB_CHECK(actual_index == error_index);
    TB_CHECK(reason == GPUJPEG_BATCH_REASON_NONE || decoder->batch.reason == reason);
    TB_CHECK(output.type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER && output.data == custom && output.data_size == 0 &&
             output.metadata == NULL && output.param_image.width == 0);
    for ( size_t i = 0; i < sizeof custom; ++i ) {
        TB_CHECK(custom[i] == TB_SENTINEL);
    }
}

/// a malformed image fails as image 1 of a batch and with gpujpeg_decoder_decode()
static void
tb_malformed(struct gpujpeg_decoder* decoder, const struct tb_jpeg* jpegs, struct tb_jpeg bad)
{
    tb_batch_fail(decoder, jpegs, 2, 1, &bad, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_NONE);
    struct gpujpeg_decoder* single = tb_decoder(NULL, NULL);
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    TB_CHECK(gpujpeg_decoder_decode(single, bad.data, bad.size, &output) != 0);
    gpujpeg_decoder_destroy(single);
    free(bad.data);
}

static void
tb_equivalence(struct gpujpeg_encoder* encoder, struct gpujpeg_decoder* decoder)
{
    const struct tb_format aligned = {GPUJPEG_CS_DEFAULT, GPUJPEG_PIXFMT_AUTODETECT, "256"};
    const struct tb_format planar = {GPUJPEG_NONE, GPUJPEG_PIXFMT_NATIVE, "0"};
    const struct
    {
        struct tb_desc desc;
        int count;
        const struct tb_format* format;
        enum gpujpeg_decoder_output_type type;
        size_t stride_extra;
        enum gpujpeg_huffman_route route;
    } cases[] = {
        {{64, 64, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0},
         4,  &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER, 100,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
        {{200, 36, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0},
         3,  &aligned,
         GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER,      7,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
        {{72, 40, GPUJPEG_SUBSAMPLING_422, 0, 3, 0, 0},
         5,  &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER,    0,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
        {{8, 8, GPUJPEG_SUBSAMPLING_400, 0, 8, 0, 0},
         17, &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER,        0,
         GPUJPEG_HUFFMAN_ROUTE_CPU},
        {{64, 64, GPUJPEG_SUBSAMPLING_400, 0, 1, 0, 0},
         2,  &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER, 0,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
        {{128, 64, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0},
         2,  &planar,
         GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER,    3,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
        {{64, 64, GPUJPEG_SUBSAMPLING_444, 1, 0, 0, 0},
         3,  &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER,      0,
         GPUJPEG_HUFFMAN_ROUTE_CPU},
        {{64, 64, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0},
         1,  &tb_default_format,
         GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER, 0,
         GPUJPEG_HUFFMAN_ROUTE_GPU},
    };
    for ( unsigned c = 0; c < sizeof cases / sizeof cases[0]; ++c ) {
        struct tb_jpeg jpegs[TB_MAX_COUNT];
        tb_encode_set(encoder, &cases[c].desc, cases[c].count, jpegs);
        tb_batch_equal(decoder, jpegs, cases[c].count, cases[c].format, cases[c].type, cases[c].stride_extra,
                       cases[c].route);
        // also by a new decoder (zeroed unused Huffman tables for the GPU table setup)
        struct gpujpeg_decoder* fresh = tb_decoder(NULL, NULL);
        tb_batch_equal(fresh, jpegs, cases[c].count, cases[c].format, cases[c].type, cases[c].stride_extra,
                       cases[c].route);
        gpujpeg_decoder_destroy(fresh);
        tb_free_set(jpegs, cases[c].count);
    }
    tb_set_format(decoder, &tb_default_format);
}

/// differences from the first image that are rejected or allowed
static void
tb_homogeneity(struct gpujpeg_encoder* encoder, struct gpujpeg_decoder* decoder)
{
    struct tb_jpeg base[3];
    tb_encode_set(encoder, &tb_base, 3, base);

    const struct
    {
        struct tb_desc desc;
        enum gpujpeg_batch_reason reason;
    } variants[] = {
        {{72, 64, GPUJPEG_SUBSAMPLING_420, 0, 2, 0, 0},                   GPUJPEG_BATCH_REASON_LAYOUT},
        {{64, 64, GPUJPEG_SUBSAMPLING_444, 0, 2, 0, 0},                   GPUJPEG_BATCH_REASON_LAYOUT},
        {{64, 64, GPUJPEG_SUBSAMPLING_420, 0, 4, 0, 0},                   GPUJPEG_BATCH_REASON_LAYOUT},
        {{64, 64, GPUJPEG_SUBSAMPLING_420, 1, 2, 0, 0},                   GPUJPEG_BATCH_REASON_LAYOUT},
        {{64, 64, GPUJPEG_SUBSAMPLING_420, 0, 2, 0, GPUJPEG_YCBCR_BT709}, GPUJPEG_BATCH_REASON_LAYOUT},
        {{64, 64, GPUJPEG_SUBSAMPLING_420, 0, 2, 30, 0},                  GPUJPEG_BATCH_REASON_TABLES},
    };
    for ( unsigned v = 0; v < sizeof variants / sizeof variants[0]; ++v ) {
        struct tb_jpeg other = tb_encode(encoder, &variants[v].desc, 1);
        tb_batch_fail(decoder, base, 3, 1, &other, GPUJPEG_ERR_BATCH_MISMATCH, 1, variants[v].reason);
        tb_batch_fail(decoder, base, 3, 2, &other, GPUJPEG_ERR_BATCH_MISMATCH, 2, variants[v].reason);
        free(other.data);
    }

    const size_t sof = tb_find(&base[1], 0xC0, 0);
    const size_t dht = tb_find(&base[1], 0xC4, 0);
    const size_t sos = tb_find(&base[1], 0xDA, 0);
    const size_t sos_cb = tb_find(&base[1], 0xDA, 1);
    TB_CHECK(sof != 0 && dht != 0 && sos != 0 && sos_cb != 0);
    struct tb_jpeg j = tb_copy(&base[1]); // component ID of Cb (frame and scan)
    j.data[sof + 13] = 5;
    j.data[sos_cb + 5] = 5;
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_MISMATCH, 1, GPUJPEG_BATCH_REASON_COMPONENTS);
    free(j.data);
    j = tb_copy(&base[1]); // Huffman table selectors of Y
    j.data[sos + 6] = 0x11;
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_MISMATCH, 1, GPUJPEG_BATCH_REASON_COMPONENTS);
    free(j.data);
    j = tb_copy(&base[1]); // Huffman table content (2 symbols of luminance DC swapped)
    const uint8_t symbol = j.data[dht + 21];
    j.data[dht + 21] = j.data[dht + 22];
    j.data[dht + 22] = symbol;
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_MISMATCH, 1, GPUJPEG_BATCH_REASON_TABLES);
    free(j.data);

    // allowed: comment, orientation (SPIFF), SOF1, unused table, identical redefinition of a used table after its use
    struct tb_jpeg allowed[3] = {
        base[0], {NULL, 0},
         base[2]
    };
    const uint8_t com[] = {0xFF, 0xFE, 0x00, 0x05, 'a', 'b', 0};
    allowed[1] = tb_splice(&base[1], 2, 0, com, sizeof com);
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    free(allowed[1].data);
    struct gpujpeg_encoder* oriented = gpujpeg_encoder_create(NULL);
    TB_CHECK(oriented != NULL);
    TB_CHECK(gpujpeg_encoder_set_option(oriented, GPUJPEG_ENC_OPT_METADATA, "orientation=90") == 0);
    allowed[1] = tb_encode(oriented, &tb_base, 1);
    gpujpeg_encoder_destroy(oriented);
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    TB_CHECK(!decoder->batch.metadata[0].vals[GPUJPEG_METADATA_ORIENTATION].set &&
             decoder->batch.metadata[1].vals[GPUJPEG_METADATA_ORIENTATION].set &&
             decoder->batch.metadata[1].vals[GPUJPEG_METADATA_ORIENTATION].orient.rotation == 1);
    free(allowed[1].data);
    allowed[1] = tb_copy(&base[1]);
    allowed[1].data[sof + 1] = 0xC1;
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    free(allowed[1].data);
    allowed[1] = tb_insert_segment(&base[1], tb_find(&base[1], 0xC4, 3), 0x12, 0xDA, 0); // AC table 2 (unused)
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    free(allowed[1].data);
    // between scans (gpujpeg_decoder_decode() rejects these markers there): same AC table 0, same quantization
    // table 0, comment, same restart interval
    tb_oracle = base;
    const uint8_t dri[] = {0xFF, 0xDD, 0x00, 0x04, 0x00, 0x02};
    for ( int i = 0; i < 4; ++i ) {
        if ( i < 2 ) {
            allowed[1] = tb_insert_segment(&base[1], tb_find(&base[1], i == 0 ? 0xC4 : 0xDB, i == 0 ? 1 : 0),
                                           i == 0 ? 0x10 : 0x00, 0xDA, 1);
        }
        else {
            allowed[1] = tb_splice(&base[1], tb_find(&base[1], 0xDA, 1), 0, i == 2 ? com : dri,
                                   i == 2 ? sizeof com : sizeof dri);
        }
        tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                       GPUJPEG_HUFFMAN_ROUTE_GPU);
        free(allowed[1].data);
    }
    const uint8_t empty_com[] = {0xFF, 0xFE, 0x00, 0x02};
    allowed[1] = tb_splice(&base[1], tb_find(&base[1], 0xDA, 1), 0, empty_com, sizeof empty_com);
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    free(allowed[1].data);
    tb_oracle = NULL;

    for ( int length = 0; length < 2; ++length ) {
        const uint8_t short_com[] = {0xFF, 0xFE, 0x00, (uint8_t)length};
        j = tb_splice(&base[1], tb_find(&base[1], 0xDA, 1), 0, short_com, sizeof short_com);
        tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_NONE);
        free(j.data);
    }

    // tables: not defined by the image (abbreviated), changed after use, changed and restored
    j = tb_copy(&base[1]);
    for ( size_t pos; (pos = tb_find(&j, 0xC4, 0)) != 0; ) {
        struct tb_jpeg stripped = tb_splice(&j, pos, tb_length(&j, pos), NULL, 0);
        free(j.data);
        j = stripped;
    }
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_TABLE_SCOPE);
    free(j.data);
    j = tb_insert_segment(&base[1], tb_find(&base[1], 0xC4, 3), 0x10, 0xDA, 1); // chroma AC as AC table 0
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_TABLE_SCOPE);
    struct tb_jpeg restored = tb_insert_segment(&j, tb_find(&j, 0xC4, 1), 0x10, 0xDA, 1); // original after it
    tb_batch_fail(decoder, base, 3, 1, &restored, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_TABLE_SCOPE);
    free(restored.data);
    free(j.data);
    j = tb_insert_segment(&base[1], tb_find(&base[1], 0xDB, 1), 0x00, 0xDA, 1); // chroma as quantization table 0
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_TABLE_SCOPE);
    free(j.data);

    // scans: equal non-canonical order in all images, duplicate component, missing component, not sequential
    struct tb_jpeg swapped[3];
    for ( int i = 0; i < 3; ++i ) {
        swapped[i] = tb_copy(&base[i]);
        swapped[i].data[tb_find(&swapped[i], 0xDA, 0) + 5] = 2;
        swapped[i].data[tb_find(&swapped[i], 0xDA, 1) + 5] = 1;
    }
    tb_batch_fail(decoder, swapped, 3, 0, NULL, GPUJPEG_ERR_BATCH_UNSUPPORTED, 0, GPUJPEG_BATCH_REASON_SCANS);
    tb_free_set(swapped, 3);
    j = tb_copy(&base[1]);
    j.data[tb_find(&j, 0xDA, 2) + 5] = 1;
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_SCANS);
    free(j.data);
    const size_t sos_cr = tb_find(&base[1], 0xDA, 2);
    j = tb_splice(&base[1], sos_cr, base[1].size - 2 - sos_cr, NULL, 0);
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERR_BATCH_UNSUPPORTED, 1, GPUJPEG_BATCH_REASON_SCANS);
    free(j.data);
    j = tb_copy(&base[1]);
    j.data[sos + 8] = 62; // Se
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_SCANS);
    free(j.data);

    // restart segments of a scan: missing, compensated by the next scan, wrong order; trailing empty one allowed
    size_t rst[64];
    size_t end = 0;
    const int rst_y = tb_restarts(&base[1], 0, rst, 64, &end);
    TB_CHECK(rst_y == 31);
    struct tb_jpeg missing = tb_splice(&base[1], rst[rst_y - 1], 2, NULL, 0);
    tb_batch_fail(decoder, base, 3, 1, &missing, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_SEGMENTS);
    const int rst_cb = tb_restarts(&missing, 1, rst, 64, &end);
    const uint8_t extra[] = {0xFF, (uint8_t)(0xD0 | ((missing.data[rst[rst_cb - 1] + 1] + 1) & 7)), 0x00};
    j = tb_splice(&missing, end, 0, extra, sizeof extra);
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_SEGMENTS);
    free(j.data);
    free(missing.data);
    j = tb_copy(&base[1]);
    tb_restarts(&j, 0, rst, 64, &end);
    j.data[rst[0] + 1] = 0xD2;
    tb_batch_fail(decoder, base, 3, 1, &j, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_SEGMENTS);
    free(j.data);
    const int rst_cr = tb_restarts(&base[1], 2, rst, 64, &end);
    const uint8_t trailing[] = {0xFF, (uint8_t)(0xD0 | ((base[1].data[rst[rst_cr - 1] + 1] + 1) & 7))};
    allowed[1] = tb_splice(&base[1], end, 0, trailing, sizeof trailing);
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    free(allowed[1].data);

    // APP13 (segment info) not used by batches, even with more headers than the single-image reader holds
    uint8_t app13[120 * 7];
    for ( int i = 0; i < 120; ++i ) {
        const uint8_t header[] = {0xFF, 0xED, 0x00, 0x05, 0x00, 0x00, 0x08};
        memcpy(app13 + 7 * i, header, sizeof header);
    }
    allowed[1] = tb_splice(&base[1], 2, 0, app13, sizeof app13);
    tb_oracle = base; // gpujpeg_decoder_decode() would overflow its segment info headers
    tb_batch_equal(decoder, allowed, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    tb_oracle = NULL;
    free(allowed[1].data);

    tb_free_set(base, 3);
}

/// malformed headers and data: errors (no undefined behavior) for batches and single images
static void
tb_malformed_headers(struct gpujpeg_encoder* encoder, struct gpujpeg_decoder* decoder)
{
    struct tb_jpeg base[2];
    tb_encode_set(encoder, &tb_base, 2, base);
    const size_t dqt = tb_find(&base[1], 0xDB, 0);
    const size_t dht = tb_find(&base[1], 0xC4, 0);
    const size_t sof = tb_find(&base[1], 0xC0, 0);
    const size_t sos = tb_find(&base[1], 0xDA, 0);
    struct tb_jpeg j = tb_copy(&base[1]);
    j.data[dqt + 4] = 0x05; // Tq
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[dht + 4] = 0x07; // Th
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[dht + 19] = j.data[dht + 20] = 200; // symbol count
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[dht + 5] = 2; // luminance DC: 2 codes of 1 bit (incl. all-ones code) instead of 2 codes of 3 bits
    j.data[dht + 7] -= 2;
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[sof + 3] -= 3; // SOF too short for the components
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[sof + 11] = 0x01; // zero horizontal sampling factor
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[sof + 9] = 2; // 2 components
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[sof + 5] = j.data[sof + 6] = 0; // height 0 (DNL)
    tb_malformed(decoder, base, j);
    j = tb_copy(&base[1]);
    j.data[sos + 6] = 0x44; // Huffman table 4
    tb_malformed(decoder, base, j);
    const uint8_t app2[] = {0xFF, 0xE2, 0x00, 0x00}; // invalid length
    tb_malformed(decoder, base, tb_splice(&base[1], 2, 0, app2, sizeof app2));
    tb_free_set(base, 2);

    // more entropy-coded data than the buffer of a new decoder holds (not the larger ones a used decoder keeps)
    const struct tb_desc gray = {8, 8, GPUJPEG_SUBSAMPLING_400, 0, 0, 0, 0};
    tb_encode_set(encoder, &gray, 2, base);
    static uint8_t zeros[10000];
    const size_t gray_sos = tb_find(&base[1], 0xDA, 0);
    struct gpujpeg_decoder* small = tb_decoder(NULL, NULL);
    tb_malformed(small, base, tb_splice(&base[1], gray_sos + tb_length(&base[1], gray_sos), 0, zeros, sizeof zeros));
    TB_CHECK(small->coder.data_compressed_allocated_size < sizeof zeros);
    gpujpeg_decoder_destroy(small);
    // batch limits for the image dimensions (before allocations), also for 1 image
    j = tb_copy(&base[0]);
    const size_t gray_sof = tb_find(&j, 0xC0, 0);
    memset(j.data + gray_sof + 5, 0xFF, 4); // 65535x65535
    tb_batch_fail(decoder, &j, 1, 0, NULL, GPUJPEG_ERROR, -1, GPUJPEG_BATCH_REASON_NONE);
    free(j.data);
    tb_free_set(base, 2);
}

static void
tb_arguments(struct gpujpeg_encoder* encoder, struct gpujpeg_decoder* decoder)
{
    struct tb_jpeg base[3];
    tb_encode_set(encoder, &tb_base, 3, base);
    uint8_t* images[3] = {base[0].data, base[1].data, base[2].data};
    size_t sizes[3] = {base[0].size, base[1].size, base[2].size};
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    int index = -2;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 0, images, sizes, 0, &output, &index) == GPUJPEG_ERROR &&
             index == -1);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, GPUJPEG_BATCH_MAX_COUNT + 1, images, sizes, 0, &output, &index) ==
             GPUJPEG_ERROR);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, NULL, sizes, 0, &output, NULL) == GPUJPEG_ERROR);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, NULL, 0, &output, NULL) == GPUJPEG_ERROR);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, 0, NULL, NULL) == GPUJPEG_ERROR);
    TB_CHECK(gpujpeg_decoder_decode_batch(NULL, 3, images, sizes, 0, &output, &index) == GPUJPEG_ERROR && index == -1);
    images[2] = NULL;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, 0, &output, &index) == GPUJPEG_ERROR &&
             index == 2);
    images[2] = base[2].data;
    sizes[1] = 3;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, 0, &output, &index) == GPUJPEG_ERROR &&
             index == 1);
    sizes[1] = base[1].size;
    const size_t size = 64 * 64 * 3;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, size - 1, &output, &index) == GPUJPEG_ERROR &&
             index == -1);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, SIZE_MAX / 2, &output, &index) == GPUJPEG_ERROR &&
             index == -1);
    output.type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, 0, &output, &index) ==
                 GPUJPEG_ERR_BATCH_UNSUPPORTED &&
             index == -1);
    gpujpeg_decoder_output_set_custom(&output, NULL);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 3, images, sizes, 0, &output, &index) == GPUJPEG_ERROR);
    gpujpeg_decoder_output_set_default(&output);

    // invalid requested output format or alignment, an unsupported alignment (custom output untouched)
    const struct tb_format invalid[] = {
        {GPUJPEG_CS_DEFAULT,           GPUJPEG_PIXFMT_NONE,           "0"      },
        {GPUJPEG_CS_DEFAULT,           (enum gpujpeg_pixel_format)42, "0"      },
        {(enum gpujpeg_color_space)42, GPUJPEG_PIXFMT_AUTODETECT,     "0"      },
        {GPUJPEG_CS_DEFAULT,           GPUJPEG_PIXFMT_AUTODETECT,     "-1"     },
        {GPUJPEG_CS_DEFAULT,           GPUJPEG_PIXFMT_AUTODETECT,     "2097152"},
    };
    for ( unsigned i = 0; i < sizeof invalid / sizeof invalid[0]; ++i ) {
        tb_fail_format = &invalid[i];
        tb_batch_fail(decoder, base, 3, 0, NULL, i < 4 ? GPUJPEG_ERROR : GPUJPEG_ERR_BATCH_UNSUPPORTED, -1,
                      GPUJPEG_BATCH_REASON_NONE);
    }
    tb_fail_format = &tb_default_format;
    tb_set_format(decoder, &tb_default_format);

    // flipping: rejected for a batch (also of 1 image), not for a single image
    TB_CHECK(gpujpeg_decoder_set_option(decoder, GPUJPEG_DEC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_TRUE) == 0);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 1, images, sizes, 0, &output, &index) ==
                 GPUJPEG_ERR_BATCH_UNSUPPORTED &&
             index == -1);
    TB_CHECK(gpujpeg_decoder_decode(decoder, images[0], sizes[0], &output) == 0);
    TB_CHECK(gpujpeg_decoder_set_option(decoder, GPUJPEG_DEC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_FALSE) == 0);
    tb_batch_equal(decoder, base, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU); // after the flipped single image

    // odd-width packed 4:2:2 output
    const struct tb_desc odd = {63, 64, GPUJPEG_SUBSAMPLING_422, 1, 1, 0, 0};
    struct tb_jpeg odd_jpeg = tb_encode(encoder, &odd, 0);
    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_YCBCR_JPEG, GPUJPEG_422_U8_P1020);
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, 1, &odd_jpeg.data, &odd_jpeg.size, 0, &output, &index) ==
                 GPUJPEG_ERR_BATCH_UNSUPPORTED &&
             index == -1 && decoder->batch.reason == GPUJPEG_BATCH_REASON_DOMAIN);
    free(odd_jpeg.data);
    tb_set_format(decoder, &tb_default_format);

    // batch limits of the count (compressed data of all images) before allocations, then usable again
    uint8_t** many_images = malloc(GPUJPEG_BATCH_MAX_COUNT * sizeof *many_images);
    size_t* many_sizes = malloc(GPUJPEG_BATCH_MAX_COUNT * sizeof *many_sizes);
    TB_CHECK(many_images != NULL && many_sizes != NULL);
    const struct tb_desc large = {256, 256, GPUJPEG_SUBSAMPLING_444, 1, 8, 0, 0};
    struct tb_jpeg large_jpeg = tb_encode(encoder, &large, 0);
    for ( int i = 0; i < GPUJPEG_BATCH_MAX_COUNT; ++i ) {
        many_images[i] = large_jpeg.data;
        many_sizes[i] = large_jpeg.size;
    }
    TB_CHECK(gpujpeg_decoder_decode_batch(decoder, GPUJPEG_BATCH_MAX_COUNT, many_images, many_sizes, 0, &output,
                                          &index) == GPUJPEG_ERROR &&
             index == -1);
    free(large_jpeg.data);
    free(many_images);
    free(many_sizes);
    tb_batch_equal(decoder, base, 3, &tb_default_format, GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER, 0,
                   GPUJPEG_HUFFMAN_ROUTE_GPU);
    tb_free_set(base, 3);
}

/// tables committed by a failed batch are those a following single image inherits (also on the device)
static void
tb_failed_tables(struct gpujpeg_encoder* encoder)
{
    struct tb_jpeg base[3];
    tb_encode_set(encoder, &tb_base, 3, base);
    struct gpujpeg_decoder* decoder = tb_decoder(NULL, NULL);
    const size_t sos = tb_find(&base[1], 0xDA, 0);
    struct tb_jpeg truncated = tb_splice(&base[1], sos + 40, base[1].size - sos - 40, NULL, 0);
    tb_batch_fail(decoder, base, 2, 1, &truncated, GPUJPEG_ERROR, 1, GPUJPEG_BATCH_REASON_NONE);
    free(truncated.data);
    TB_CHECK(decoder->batch.quant_dirty == 0 && decoder->batch.huffman_dirty == 0);
    struct tb_jpeg abbreviated = tb_copy(&base[2]);
    for ( uint8_t code = 0xC4; code != 0; code = code == 0xC4 ? 0xDB : 0 ) {
        for ( size_t pos; (pos = tb_find(&abbreviated, code, 0)) != 0; ) {
            struct tb_jpeg stripped = tb_splice(&abbreviated, pos, tb_length(&abbreviated, pos), NULL, 0);
            free(abbreviated.data);
            abbreviated = stripped;
        }
    }
    struct tb_image expected = tb_single(&base[2], &tb_default_format);
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    TB_CHECK(gpujpeg_decoder_decode(decoder, abbreviated.data, abbreviated.size, &output) == 0);
    TB_CHECK(decoder->huffman_route == GPUJPEG_HUFFMAN_ROUTE_GPU && tb_same(&expected, output.data));
    free(expected.data);
    free(abbreviated.data);
    gpujpeg_decoder_destroy(decoder);
    tb_free_set(base, 3);
}

struct tb_alloc
{
    cudaStream_t stream;
    int fail_at; ///< fail the n-th allocation (counted from 1), 0 for never
    int allocs;
    int errors;
    void* live[TB_ALLOC_MAX_LIVE];
    int live_count;
};

static void*
tb_alloc_device(void* user_data, size_t size, cudaStream_t stream)
{
    struct tb_alloc* s = user_data;
    s->errors += stream != s->stream;
    void* ptr = NULL;
    if ( ++s->allocs == s->fail_at || s->live_count == TB_ALLOC_MAX_LIVE || cudaMalloc(&ptr, size) != cudaSuccess ) {
        return NULL;
    }
    s->live[s->live_count++] = ptr;
    return ptr;
}

static void
tb_free_device(void* user_data, void* ptr, cudaStream_t stream)
{
    struct tb_alloc* s = user_data;
    s->errors += stream != s->stream;
    for ( int i = 0; i < s->live_count; ++i ) {
        if ( s->live[i] == ptr ) {
            s->live[i] = s->live[--s->live_count];
            cudaFree(ptr);
            return;
        }
    }
    s->errors += 1;
}

/// counts 4, 2, 4, single, 7, 3, other geometry; growth allocation failures on a used decoder are retried
static void
tb_reuse(struct gpujpeg_encoder* encoder)
{
    struct tb_jpeg jpegs[7];
    tb_encode_set(encoder, &tb_base, 7, jpegs);
    struct tb_alloc s = {0};
    struct gpujpeg_allocator allocator = {tb_alloc_device, tb_free_device, &s};
    struct gpujpeg_decoder* decoder = tb_decoder(NULL, &allocator);
    const enum gpujpeg_decoder_output_type cuda = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
    tb_batch_equal(decoder, jpegs, 4, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    const int allocs_4 = s.allocs;
    tb_batch_equal(decoder, jpegs, 2, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    tb_batch_equal(decoder, jpegs, 4, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    struct tb_image expected = tb_single(&jpegs[3], &tb_default_format);
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    TB_CHECK(gpujpeg_decoder_decode(decoder, jpegs[3].data, jpegs[3].size, &output) == 0);
    TB_CHECK(tb_same(&expected, output.data));
    free(expected.data);
    TB_CHECK(s.allocs == allocs_4); // capacity reused
    tb_batch_equal(decoder, jpegs, 7, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    const int growth_allocs = s.allocs - allocs_4;
    TB_CHECK(growth_allocs > 0);
    tb_batch_equal(decoder, jpegs, 3, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    TB_CHECK(s.allocs == allocs_4 + growth_allocs);
    struct tb_jpeg other[2];
    const struct tb_desc other_desc = {96, 48, GPUJPEG_SUBSAMPLING_422, 1, 2, 0, 0}; // 18 segments
    tb_encode_set(encoder, &other_desc, 2, other);
    tb_batch_equal(decoder, other, 2, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_CPU);
    tb_batch_equal(decoder, jpegs, 4, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
    gpujpeg_decoder_destroy(decoder);
    TB_CHECK(s.live_count == 0 && s.errors == 0);

    // each allocation of the growth fails once on a used decoder, a retry succeeds
    for ( int n = 1; n <= growth_allocs; ++n ) {
        struct tb_alloc f = {0};
        allocator.user_data = &f;
        decoder = tb_decoder(NULL, &allocator);
        tb_batch_equal(decoder, jpegs, 4, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
        f.fail_at = f.allocs + n;
        tb_batch_fail(decoder, jpegs, 7, 0, NULL, GPUJPEG_ERROR, -1, GPUJPEG_BATCH_REASON_NONE);
        if ( n % 2 == 0 ) { // other geometry after the failure
            tb_batch_equal(decoder, other, 2, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_CPU);
        }
        tb_batch_equal(decoder, jpegs, 7, &tb_default_format, cuda, 0, GPUJPEG_HUFFMAN_ROUTE_GPU);
        gpujpeg_decoder_destroy(decoder);
        TB_CHECK(f.live_count == 0 && f.errors == 0);
    }
    tb_free_set(other, 2);
    tb_free_set(jpegs, 7);
}

/// warmed delayed-call pixel check, read by the legacy default stream (not synchronized with the decoder streams)
/// Pageable table uploads may consume the initial delay, so this case alone does not test final-wait omission.
/// The two decoders have different tables and are used one after another (not concurrently).
/// @retval false if a stream delay is not supported (eg. by a simulator), the case is then skipped
static bool
tb_completion(struct gpujpeg_encoder* encoder)
{
    cudaStream_t probe = NULL;
    TB_CHECK(cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking) == cudaSuccess);
    const cudaError_t delay_status = cudaLaunchHostFunc(probe, tb_sleep, NULL);
    TB_CHECK(delay_status == cudaSuccess || delay_status == cudaErrorNotSupported);
    (void)cudaGetLastError();
    TB_CHECK(cudaStreamSynchronize(probe) == cudaSuccess);
    TB_CHECK(cudaStreamDestroy(probe) == cudaSuccess);
    if ( delay_status == cudaErrorNotSupported ) {
        return false;
    }
    struct tb_desc desc[2] = {tb_base, tb_base};
    desc[0].quality = 90;
    desc[1].quality = 40;
    for ( int d = 0; d < 2; ++d ) {
        struct tb_jpeg jpegs[2][4];
        cudaStream_t stream[2];
        struct gpujpeg_decoder* decoder[2];
        for ( int i = 0; i < 2; ++i ) {
            tb_encode_set(encoder, &desc[(d + i) % 2], 4, jpegs[i]);
            TB_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking) == cudaSuccess);
            decoder[i] = tb_decoder(stream[i], NULL);
        }
        for ( int i = 0; i < 2; ++i ) { // warm-up: layout, tables and buffers of the decoders
            tb_batch_equal(decoder[i], jpegs[i], 4, &tb_default_format, GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER, 0,
                           GPUJPEG_HUFFMAN_ROUTE_GPU);
        }
        for ( int i = 0; i < 2; ++i ) {
            tb_delay_stream = stream[i];
            tb_batch_equal(decoder[i], jpegs[i], 4, &tb_default_format, GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER, 0,
                           GPUJPEG_HUFFMAN_ROUTE_GPU);
            tb_delay_stream = NULL;
        }
        for ( int i = 0; i < 2; ++i ) {
            gpujpeg_decoder_destroy(decoder[i]);
            cudaStreamDestroy(stream[i]);
            tb_free_set(jpegs[i], 4);
        }
    }
    return true;
}

/// encoder with the options of a case
static struct gpujpeg_encoder*
tb_enc_create(const struct tb_enc_case* c, const struct gpujpeg_allocator* allocator)
{
    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create_with_allocator(NULL, allocator);
    TB_CHECK(encoder != NULL);
    for ( int i = 0; c->options != NULL && c->options[i] != NULL; i += 2 ) {
        TB_CHECK(gpujpeg_encoder_set_option(encoder, c->options[i], c->options[i + 1]) == 0);
    }
    return encoder;
}

/// big-endian value of bytes
static uint32_t
tb_be(const uint8_t* data, int bytes)
{
    uint32_t value = 0;
    for ( int i = 0; i < bytes; ++i ) {
        value = value << 8 | data[i];
    }
    return value;
}

/// TIFF data of Exif metadata in an APP1 marker right after SOI (big-endian as written by the encoder), NULL if none
static const uint8_t*
tb_tiff(const uint8_t* jpeg, size_t size, size_t* tiff_size)
{
    if ( size < 22 || tb_be(jpeg + 2, 2) != 0xFFE1 || memcmp(jpeg + 6, "Exif\0\0MM\0\x2A", 10) != 0 ) {
        return NULL;
    }
    const size_t length = tb_be(jpeg + 4, 2);     // with the length field, the Exif identifier and its padding
    if ( length < 8 + 10 || 4 + length > size ) { // TIFF header and IFD entry count
        return NULL;
    }
    *tiff_size = length - 8;
    return jpeg + 12;
}

/// offset of the entry of a tag in the IFD at offset ifd of TIFF data, 0 if none
static size_t
tb_ifd_entry(const uint8_t* tiff, size_t size, size_t ifd, unsigned tag)
{
    if ( ifd < 8 || ifd + 2 > size ) {
        return 0;
    }
    const size_t count = tb_be(tiff + ifd, 2);
    for ( size_t entry = ifd + 2; entry < ifd + 2 + 12 * count && entry + 12 <= size; entry += 12 ) {
        if ( tb_be(tiff + entry, 2) == tag ) {
            return entry;
        }
    }
    return 0;
}

/// offset of the value of an IFD entry in TIFF data and its size, 0 if not within the data
static size_t
tb_ifd_value(const uint8_t* tiff, size_t size, size_t entry, size_t* value_size)
{
    static const unsigned type_sizes[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8}; // TIFF 6.0 types 1 to 10
    const unsigned type = tb_be(tiff + entry + 2, 2);
    *value_size =
        (type < sizeof type_sizes / sizeof type_sizes[0] ? type_sizes[type] : 0) * (size_t)tb_be(tiff + entry + 4, 4);
    const size_t value = *value_size <= 4 ? entry + 8 : tb_be(tiff + entry + 8, 4);
    return *value_size != 0 && value <= size && *value_size <= size - value ? value : 0;
}

/// Exif metadata of a JPEG: the Exif IFD pointer targets the byte after the 0th IFD at 8 and its values (stored one
/// after another), where the Exif IFD holds version 2.30 and the image size; 0th IFD tags ascending
/// @return offset of the Exif IFD in the TIFF data, 0 for no or other Exif metadata
static size_t
tb_exif_ifd(const uint8_t* jpeg, size_t size, int width, int height)
{
    size_t tiff_size = 0;
    const uint8_t* tiff = tb_tiff(jpeg, size, &tiff_size);
    if ( tiff == NULL || tb_be(tiff + 4, 4) != 8 ) {
        return 0;
    }
    const size_t count = tb_be(tiff + 8, 2);
    size_t end = 8 + 2 + 12 * count + 4;
    unsigned previous = 0;
    for ( size_t entry = 8 + 2; entry < 8 + 2 + 12 * count; entry += 12 ) {
        size_t value_size = 0;
        if ( entry + 12 > tiff_size || tb_be(tiff + entry, 2) <= previous ||
             tb_ifd_value(tiff, tiff_size, entry, &value_size) == 0 ) {
            return 0;
        }
        previous = tb_be(tiff + entry, 2);
        end += value_size > 4 ? value_size : 0;
    }
    const size_t pointer = tb_ifd_entry(tiff, tiff_size, 8, 0x8769);
    const size_t version = tb_ifd_entry(tiff, tiff_size, end, 0x9000);
    const size_t x = tb_ifd_entry(tiff, tiff_size, end, 0xA002);
    const size_t y = tb_ifd_entry(tiff, tiff_size, end, 0xA003);
    if ( pointer == 0 || tb_be(tiff + pointer + 2, 2) != 4 || tb_be(tiff + pointer + 4, 4) != 1 ||
         tb_be(tiff + pointer + 8, 4) != end || version == 0 || memcmp(tiff + version + 8, "0230", 4) != 0 || x == 0 ||
         tb_be(tiff + x + 8, 2) != (uint32_t)width || y == 0 || tb_be(tiff + y + 8, 2) != (uint32_t)height ) {
        return 0;
    }
    return end;
}

/// offset of the value of the 0th IFD DateTime (20 characters with NUL) of the Exif metadata of a JPEG, 0 if none
static size_t
tb_exif_date(const uint8_t* jpeg, size_t size)
{
    size_t tiff_size = 0;
    size_t value_size = 0;
    const uint8_t* tiff = tb_tiff(jpeg, size, &tiff_size);
    const size_t entry = tiff != NULL ? tb_ifd_entry(tiff, tiff_size, 8, 0x0132) : 0;
    const size_t value = entry != 0 ? tb_ifd_value(tiff, tiff_size, entry, &value_size) : 0;
    return value != 0 && value_size == 20 ? (size_t)(tiff - jpeg) + value : 0;
}

/// JPEG equal to the expected one, but the value of a generated Exif DateTime (written when encoded)
static bool
tb_enc_same(const struct tb_jpeg* expected, const uint8_t* actual, size_t size, bool generated_date)
{
    const size_t date = generated_date ? tb_exif_date(expected->data, expected->size) : 0;
    const size_t rest = date != 0 ? date + 19 : 0;
    return size == expected->size && (!generated_date || (date != 0 && tb_exif_date(actual, size) == date)) &&
           memcmp(expected->data, actual, date) == 0 && memcmp(expected->data + rest, actual + rest, size - rest) == 0;
}

/// batch encoding of a case equals tb_enc_single() of each image by a new encoder but a generated Exif DateTime (that
/// of the first image in all), Exif metadata are as expected, the JPEGs are stored one after another and device input
/// is unchanged
static void
tb_enc_batch_equal(struct gpujpeg_encoder* encoder, const struct tb_enc_case* c)
{
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    tb_enc_params(c, &param, &param_image);
    const size_t size = gpujpeg_image_calculate_size(&param_image);
    const size_t stride = size + c->stride_extra;
    const size_t extent = (c->count - 1) * stride + size;
    TB_CHECK(c->count <= TB_MAX_COUNT && size <= sizeof tb_pixels);
    uint8_t* host = malloc(extent);
    TB_CHECK(host != NULL);
    memset(host, TB_SENTINEL, extent);
    struct tb_jpeg expected[TB_MAX_COUNT];
    for ( int i = 0; i < c->count; ++i ) {
        tb_fill(i);
        memcpy(host + i * stride, tb_pixels, size);
        struct gpujpeg_encoder* single = tb_enc_create(c, NULL);
        expected[i] = tb_enc_single(single, c, host + i * stride);
        gpujpeg_encoder_destroy(single);
    }
    uint8_t* device = NULL;
    struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(host);
    if ( c->device ) {
        TB_CHECK(cudaMalloc((void**)&device, extent) == cudaSuccess &&
                 cudaMemcpy(device, host, extent, cudaMemcpyHostToDevice) == cudaSuccess);
        input = gpujpeg_encoder_input_gpu_image(device);
    }
    uint8_t* images[TB_MAX_COUNT];
    size_t sizes[TB_MAX_COUNT];
    TB_CHECK(gpujpeg_encoder_encode_batch(encoder, &param, &param_image, c->count, &input,
                                          c->stride_extra != 0 ? stride : 0, images, sizes) == GPUJPEG_NOERR);
    const bool generated_date = c->exif_ifd != 0 && !c->custom_date;
    const size_t date = generated_date ? tb_exif_date(images[0], sizes[0]) : 0;
    for ( int i = 0; i < c->count; ++i ) {
        TB_CHECK(tb_enc_same(&expected[i], images[i], sizes[i], generated_date));
        TB_CHECK(i == 0 || images[i] == images[i - 1] + sizes[i - 1]);
        TB_CHECK(date == 0 || memcmp(images[i] + date, images[0] + date, 19) == 0);
        TB_CHECK(tb_exif_ifd(images[i], sizes[i], c->desc.width, c->desc.height) == c->exif_ifd &&
                 tb_exif_ifd(expected[i].data, expected[i].size, c->desc.width, c->desc.height) == c->exif_ifd);
        free(expected[i].data);
    }
    if ( device != NULL ) {
        uint8_t* copy = malloc(extent);
        TB_CHECK(copy != NULL && cudaMemcpy(copy, device, extent, cudaMemcpyDeviceToHost) == cudaSuccess);
        TB_CHECK(memcmp(copy, host, extent) == 0);
        free(copy);
        TB_CHECK(cudaFree(device) == cudaSuccess);
    }
    free(host);
}

/// batch encoding whose result arrays are unchanged if it fails
/// @return the result
static int
tb_enc_try(struct gpujpeg_encoder* encoder, const struct gpujpeg_parameters* param,
           const struct gpujpeg_image_parameters* param_image, int count, const struct gpujpeg_encoder_input* input,
           size_t input_stride)
{
    uint8_t* images[TB_MAX_COUNT];
    size_t sizes[TB_MAX_COUNT];
    for ( int i = 0; i < TB_MAX_COUNT; ++i ) {
        images[i] = tb_pixels + i;
        sizes[i] = SIZE_MAX - i;
    }
    const int rc = gpujpeg_encoder_encode_batch(encoder, param, param_image, count, input, input_stride, images, sizes);
    for ( int i = 0; rc != GPUJPEG_NOERR && i < TB_MAX_COUNT; ++i ) {
        TB_CHECK(images[i] == tb_pixels + i && sizes[i] == SIZE_MAX - i);
    }
    return rc;
}

/// batch encoding fails with rc, the result arrays are unchanged
static void
tb_enc_fail(struct gpujpeg_encoder* encoder, const struct gpujpeg_parameters* param,
            const struct gpujpeg_image_parameters* param_image, int count, const struct gpujpeg_encoder_input* input,
            size_t input_stride, int rc)
{
    TB_CHECK(tb_enc_try(encoder, param, param_image, count, input, input_stride) == rc);
}

/// the oracle rejects a changed custom DateTime and a changed Exif IFD pointer, not a changed generated DateTime
static void
tb_enc_oracle(const struct tb_enc_case* c)
{
    struct gpujpeg_encoder* encoder = tb_enc_create(c, NULL);
    tb_fill(0);
    struct tb_jpeg j = tb_enc_single(encoder, c, tb_pixels);
    gpujpeg_encoder_destroy(encoder);
    struct tb_jpeg changed = tb_copy(&j);
    const size_t date = tb_exif_date(j.data, j.size);
    TB_CHECK(date != 0 && tb_enc_same(&j, changed.data, changed.size, !c->custom_date));
    changed.data[date + 3] ^= 1; // last digit of the year
    TB_CHECK(tb_enc_same(&j, changed.data, changed.size, !c->custom_date) == !c->custom_date);
    changed.data[date + 3] ^= 1;
    size_t tiff_size = 0;
    const uint8_t* tiff = tb_tiff(j.data, j.size, &tiff_size);
    const size_t pointer = (size_t)(tiff - j.data) + tb_ifd_entry(tiff, tiff_size, 8, 0x8769) + 8;
    changed.data[pointer + 3] ^= 2;
    TB_CHECK(!tb_enc_same(&j, changed.data, changed.size, !c->custom_date) &&
             tb_exif_ifd(changed.data, changed.size, c->desc.width, c->desc.height) == 0 &&
             tb_exif_ifd(j.data, j.size, c->desc.width, c->desc.height) == c->exif_ifd);
    free(changed.data);
    free(j.data);
}

static void
tb_enc_equivalence(struct gpujpeg_encoder* encoder)
{
    const char* const exif[] = {GPUJPEG_ENC_OPT_HDR, GPUJPEG_ENC_HDR_VAL_EXIF, GPUJPEG_ENC_OPT_METADATA,
                                "orientation=90", NULL};
    const char* const exif_tag[] = {GPUJPEG_ENC_OPT_EXIF_TAG, "270:ASCII=batch", NULL};
    const char* const date[] = {GPUJPEG_ENC_OPT_EXIF_TAG, "306:ASCII=2000:01:01 00:00:00", NULL};
    const char* const tags[] = {GPUJPEG_ENC_OPT_EXIF_TAG,
                                "270:ASCII=first",
                                GPUJPEG_ENC_OPT_EXIF_TAG,
                                "305:ASCII=second",
                                GPUJPEG_ENC_OPT_EXIF_TAG,
                                "42016:ASCII=unique",
                                NULL};
    const char* const exif_tags[] = {GPUJPEG_ENC_OPT_EXIF_TAG, "37510:UNDEFINED=comment", GPUJPEG_ENC_OPT_EXIF_TAG,
                                     "42016:ASCII=unique", NULL};
    const char* const orientation[] = {GPUJPEG_ENC_OPT_METADATA, "orientation=180", NULL};
    const char* const pinned[] = {GPUJPEG_ENC_OPT_OUT, GPUJPEG_ENC_OUT_VAL_PINNED, NULL};
    // Exif IFD after the 0th IFD at 8 (2 + 12 bytes per entry + 4) and its values longer than 4 bytes (resolutions
    // 2 * 8, DateTime 20, custom ones)
    const struct tb_enc_case generated_date = {
        .desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0},
          .count = 3, .options = exif, .exif_ifd = 134
    };
    const struct tb_enc_case custom_date = {
        .desc = {64, 64, GPUJPEG_SUBSAMPLING_422, 1, 2, 0, 0},
        .count = 2,
        .options = date,
        .exif_ifd = 134,
        .custom_date = true
    };
    const struct tb_enc_case cases[] = {
        // packed or strided host or device input, interleaved or not, partial MCUs, padded lines, 1 image
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0}, .count = 4},
        {.desc = {200, 36, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0}, .count = 3, .device = true, .stride_extra = 7},
        {.desc = {72, 40, GPUJPEG_SUBSAMPLING_422, 0, 3, 0, 0}, .count = 5, .width_padding = 5, .stride_extra = 100},
        {.desc = {8, 8, GPUJPEG_SUBSAMPLING_400, 0, 8, 0, 0}, .count = 17, .device = true},
        {.desc = {256, 256, GPUJPEG_SUBSAMPLING_444, 1, 2, 0, 0}, .count = 4, .device = true},
        {.desc = {256, 256, GPUJPEG_SUBSAMPLING_420, 0, 1, 0, 0}, .count = 2},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0}, .count = 1},
        // without restart markers (CPU Huffman coder)
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_444, 1, 0, 0, 0}, .count = 3, .stride_extra = 9},
        {.desc = {48, 24, GPUJPEG_SUBSAMPLING_420, 0, 0, 0, 0}, .count = 2, .device = true},
        {.desc = {40, 24, GPUJPEG_SUBSAMPLING_400, 0, 0, 0, 0},
         .count = 3,
         .width_padding = 3,
         .device = true,
         .stride_extra = 1},
        // pixel formats (also copied planes), color spaces (Adobe, SPIFF), automatic restart interval, quality
        {.desc = {64, 32, GPUJPEG_SUBSAMPLING_422, 1, 2, 0, 0},
         .count = 3,
         .pixel_format = GPUJPEG_422_U8_P1020,
         .color_space = GPUJPEG_YCBCR_BT601,
         .device = true},
        {.desc = {64, 48, GPUJPEG_SUBSAMPLING_420, 0, 1, 0, GPUJPEG_YCBCR_BT601},
         .count = 2,
         .pixel_format = GPUJPEG_420_U8_P0P1P2,
         .color_space = GPUJPEG_YCBCR_BT601,
         .stride_extra = 3},
        {.desc = {48, 32, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0},
         .count = 3,
         .pixel_format = GPUJPEG_444_U8_P0P1P2,
         .device = true,
         .stride_extra = 5},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_444, 1, 4, 90, GPUJPEG_RGB}, .count = 2},
        {.desc = {80, 48, GPUJPEG_SUBSAMPLING_420, 1, RESTART_AUTO, 0, GPUJPEG_YCBCR_BT709},
         .count = 2,
         .device = true},
        // next to unsupported ones: packed 4:2:2 rows padded by 4, copied odd-width and converted even-width planar
        // 4:2:2, interleaved grayscale with sampling 1x1
        {.desc = {64, 32, GPUJPEG_SUBSAMPLING_422, 1, 2, 0, 0},
         .count = 2,
         .pixel_format = GPUJPEG_422_U8_P1020,
         .color_space = GPUJPEG_YCBCR_BT601,
         .width_padding = 4,
         .stride_extra = 2},
        {.desc = {63, 32, GPUJPEG_SUBSAMPLING_422, 0, 1, 0, GPUJPEG_YCBCR_BT601},
         .count = 3,
         .pixel_format = GPUJPEG_422_U8_P0P1P2,
         .color_space = GPUJPEG_YCBCR_BT601,
         .device = true},
        {.desc = {64, 32, GPUJPEG_SUBSAMPLING_422, 1, 1, 0, 0},
         .count = 2,
         .pixel_format = GPUJPEG_422_U8_P0P1P2,
         .color_space = GPUJPEG_YCBCR_BT601},
        {.desc = {40, 24, GPUJPEG_SUBSAMPLING_400, 1, 2, 0, 0}, .count = 2},
        // segment info, header and metadata options, pinned output
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 1, 2, 0, 0}, .count = 3, .segment_info = 1},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 0, 2, 0, 0}, .count = 3, .device = true, .segment_info = 1},
        generated_date,
        custom_date,
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_444, 1, 0, 0, 0},
         .count = 2,
         .device = true,
         .options = exif_tag,
         .exif_ifd = 134 + 12 + 6},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0},
         .count = 3,
         .device = true,
         .options = tags,
         .exif_ifd = 134 + 2 * 12 + 6 + 7},
        {.desc = {48, 32, GPUJPEG_SUBSAMPLING_444, 0, 0, 0, 0}, .count = 2, .options = exif_tags, .exif_ifd = 134},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0}, .count = 2, .options = orientation},
        {.desc = {64, 64, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0}, .count = 3, .device = true, .options = pinned},
    };
    for ( unsigned c = 0; c < sizeof cases / sizeof cases[0]; ++c ) {
        struct gpujpeg_encoder* fresh = tb_enc_create(&cases[c], NULL);
        tb_enc_batch_equal(fresh, &cases[c]);
        tb_enc_batch_equal(fresh, &cases[c]); // reused
        gpujpeg_encoder_destroy(fresh);
        if ( cases[c].options == NULL ) { // after the other geometries
            tb_enc_batch_equal(encoder, &cases[c]);
        }
    }
    tb_enc_oracle(&generated_date);
    tb_enc_oracle(&custom_date);
}

/// invalid arguments and parameters, unsupported inputs and options, Exif metadata exceeding APP1 and batch limits
/// fail before results; the encoder is usable again
static void
tb_enc_arguments(struct gpujpeg_encoder* encoder)
{
    const struct tb_enc_case base = {.desc = tb_base, .count = 3};
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    tb_enc_params(&base, &param, &param_image);
    const struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(tb_pixels); // 3 images fit
    tb_enc_fail(NULL, &param, &param_image, 3, &input, 0, GPUJPEG_ERROR);
    tb_enc_fail(encoder, NULL, &param_image, 3, &input, 0, GPUJPEG_ERROR);
    tb_enc_fail(encoder, &param, NULL, 3, &input, 0, GPUJPEG_ERROR);
    tb_enc_fail(encoder, &param, &param_image, 3, NULL, 0, GPUJPEG_ERROR);
    uint8_t* images[3] = {tb_pixels, tb_pixels, tb_pixels};
    size_t sizes[3] = {1, 2, 3};
    TB_CHECK(gpujpeg_encoder_encode_batch(encoder, &param, &param_image, 3, &input, 0, NULL, sizes) == GPUJPEG_ERROR &&
             sizes[0] == 1 && sizes[1] == 2 && sizes[2] == 3);
    TB_CHECK(gpujpeg_encoder_encode_batch(encoder, &param, &param_image, 3, &input, 0, images, NULL) == GPUJPEG_ERROR &&
             images[0] == tb_pixels && images[1] == tb_pixels && images[2] == tb_pixels);
    tb_enc_fail(encoder, &param, &param_image, 0, &input, 0, GPUJPEG_ERROR);
    tb_enc_fail(encoder, &param, &param_image, GPUJPEG_BATCH_MAX_COUNT + 1, &input, 0, GPUJPEG_ERROR);
    struct gpujpeg_encoder_input other = input;
    other.image = NULL;
    tb_enc_fail(encoder, &param, &param_image, 3, &other, 0, GPUJPEG_ERROR);
    other = input;
    other.type = (enum gpujpeg_encoder_input_type)42;
    tb_enc_fail(encoder, &param, &param_image, 3, &other, 0, GPUJPEG_ERROR);
    other.type = GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE;
    tb_enc_fail(encoder, &param, &param_image, 3, &other, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
    const size_t size = gpujpeg_image_calculate_size(&param_image);
    tb_enc_fail(encoder, &param, &param_image, 3, &input, size - 1, GPUJPEG_ERROR);
    tb_enc_fail(encoder, &param, &param_image, 3, &input, SIZE_MAX / 2, GPUJPEG_ERROR);

    // parameters (each invalid, the others valid)
    for ( int i = 0; i < 7; ++i ) {
        struct gpujpeg_parameters p = param;
        p.quality = i == 0 ? 101 : p.quality;
        p.restart_interval = i == 1 ? 0x10000 : i == 2 ? -2 : p.restart_interval;
        p.interleaved = i == 3 ? 2 : p.interleaved;
        p.comp_count = i == 4 ? 2 : p.comp_count;
        p.sampling_factor[1].horizontal = i == 5 ? 3 : p.sampling_factor[1].horizontal; // 2 not its multiple
        p.color_space_internal = i == 6 ? (enum gpujpeg_color_space)42 : p.color_space_internal;
        tb_enc_fail(encoder, &p, &param_image, 3, &input, 0, GPUJPEG_ERROR);
    }
    for ( int i = 0; i < 7; ++i ) {
        struct gpujpeg_image_parameters p = param_image;
        p.width = i == 0 ? 0 : i == 1 ? 65536 : p.width;
        p.height = i == 2 ? 0 : p.height;
        p.width_padding = i == 3 ? -1 : p.width_padding;
        p.pixel_format = i == 4 ? GPUJPEG_PIXFMT_NONE : i == 5 ? (enum gpujpeg_pixel_format)42 : p.pixel_format;
        p.color_space = i == 6 ? (enum gpujpeg_color_space)42 : p.color_space;
        tb_enc_fail(encoder, &param, &p, 3, &input, 0, GPUJPEG_ERROR);
    }
    // padded planar input, 4:2:2 packed input with odd width or rows padded by 2
    struct gpujpeg_image_parameters unsupported = param_image;
    unsupported.pixel_format = GPUJPEG_420_U8_P0P1P2;
    unsupported.width_padding = 2;
    tb_enc_fail(encoder, &param, &unsupported, 3, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
    unsupported = param_image;
    unsupported.pixel_format = GPUJPEG_422_U8_P1020;
    unsupported.width = 63;
    tb_enc_fail(encoder, &param, &unsupported, 3, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
    unsupported.width = 64;
    unsupported.width_padding = 2;
    tb_enc_fail(encoder, &param, &unsupported, 3, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);

    // effective parameters reaching unsupported single-image paths, each followed by a batch: interleaved grayscale
    // with sampling 2x2 (1 and 2 images), converted odd-width planar 4:2:2, GPUJPEG_YUV input (supported only if the
    // library defines ENABLE_YUV)
    const struct tb_enc_case gray = {
        .desc = {16, 16, GPUJPEG_SUBSAMPLING_400, 1, 1, 0, 0},
          .count = 2
    };
    struct gpujpeg_parameters domain_param;
    struct gpujpeg_image_parameters domain_param_image;
    tb_enc_params(&gray, &domain_param, &domain_param_image);
    domain_param.comp_count = 1;
    domain_param.sampling_factor[0].horizontal = domain_param.sampling_factor[0].vertical = 2;
    for ( int count = 1; count <= 2; ++count ) {
        tb_enc_fail(encoder, &domain_param, &domain_param_image, count, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
        tb_enc_batch_equal(encoder, &gray);
    }
    const struct tb_enc_case odd = {
        .desc = {63, 32, GPUJPEG_SUBSAMPLING_422, 1, 1, 0, 0},
        .count = 2,
        .pixel_format = GPUJPEG_422_U8_P0P1P2,
        .color_space = GPUJPEG_YCBCR_BT601
    };
    tb_enc_params(&odd, &domain_param, &domain_param_image);
    tb_enc_fail(encoder, &domain_param, &domain_param_image, 2, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
    tb_enc_batch_equal(encoder, &base);
    const struct tb_enc_case yuv = {.desc = tb_base, .count = 2, .color_space = GPUJPEG_YUV};
    tb_enc_params(&yuv, &domain_param, &domain_param_image);
    const int yuv_rc = tb_enc_try(encoder, &domain_param, &domain_param_image, 2, &input, 0);
    TB_CHECK(yuv_rc == GPUJPEG_ERR_BATCH_UNSUPPORTED || yuv_rc == GPUJPEG_NOERR);
    if ( yuv_rc == GPUJPEG_NOERR ) {
        tb_enc_batch_equal(encoder, &yuv);
    }
    tb_enc_batch_equal(encoder, &base);

    // flipping, channel remapping (encoders with several Exif tags in both IFDs, released afterwards)
    const char* const tags[] = {GPUJPEG_ENC_OPT_EXIF_TAG, "270:ASCII=first",        GPUJPEG_ENC_OPT_EXIF_TAG,
                                "305:ASCII=second",       GPUJPEG_ENC_OPT_EXIF_TAG, "37510:UNDEFINED=comment",
                                GPUJPEG_ENC_OPT_EXIF_TAG, "42016:ASCII=unique",     NULL};
    const struct tb_enc_case tagged = {.desc = tb_base, .count = 1, .options = tags};
    for ( int i = 0; i < 2; ++i ) {
        struct gpujpeg_encoder* optioned = tb_enc_create(&tagged, NULL);
        TB_CHECK((i == 0 ? gpujpeg_encoder_set_option(optioned, GPUJPEG_ENC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_TRUE)
                         : gpujpeg_encoder_set_option(optioned, GPUJPEG_ENC_OPT_CHANNEL_REMAP, "210")) == 0);
        tb_enc_fail(optioned, &param, &param_image, 1, &input, 0, GPUJPEG_ERR_BATCH_UNSUPPORTED);
        gpujpeg_encoder_destroy(optioned);
    }

    // Exif: a custom ASCII value of n characters takes 233 + n bytes of APP1 (65535 at most); an empty UNDEFINED one
    // is not written
    const size_t prefix = strlen("270:ASCII=");
    char* tag = malloc(prefix + 65303 + 1);
    TB_CHECK(tag != NULL);
    memcpy(tag, "270:ASCII=", prefix);
    for ( size_t n = 65302; n <= 65303; ++n ) {
        memset(tag + prefix, 'x', n);
        tag[prefix + n] = '\0';
        const char* const options[] = {GPUJPEG_ENC_OPT_EXIF_TAG, tag, NULL};
        const struct tb_enc_case exif = {
            .desc = {256, 256, GPUJPEG_SUBSAMPLING_420, 1, 8, 0, 0},
            .count = 2,
            .options = options,
            .exif_ifd = 134 + 12 + n + 1
        };
        struct gpujpeg_encoder* exif_encoder = tb_enc_create(&exif, NULL);
        if ( n == 65302 ) {
            tb_enc_batch_equal(exif_encoder, &exif);
            tb_fill(0);
            struct tb_jpeg j = tb_enc_single(exif_encoder, &exif, tb_pixels);
            TB_CHECK(tb_length(&j, 2) == 2 + 0xFFFF);
            free(j.data);
        }
        else {
            struct gpujpeg_parameters exif_param;
            struct gpujpeg_image_parameters exif_param_image;
            tb_enc_params(&exif, &exif_param, &exif_param_image);
            tb_enc_fail(exif_encoder, &exif_param, &exif_param_image, 1, &input, 0, GPUJPEG_ERROR);
        }
        gpujpeg_encoder_destroy(exif_encoder);
    }
    free(tag);
    const char* const empty_tag[] = {GPUJPEG_ENC_OPT_EXIF_TAG, "270:UNDEFINED=", NULL};
    const struct tb_enc_case empty = {.desc = tb_base, .count = 3, .options = empty_tag};
    struct gpujpeg_encoder* empty_encoder = tb_enc_create(&empty, NULL);
    tb_enc_fail(empty_encoder, &param, &param_image, 3, &input, 0, GPUJPEG_ERROR);
    gpujpeg_encoder_destroy(empty_encoder);
    // two tags in each IFD exceeding APP1 together (each fits), released afterwards
    static const char* const prefixes[] = {"270:ASCII=", "305:ASCII=", "37510:UNDEFINED=", "42016:ASCII="};
    char* values[4];
    const char* several_tags[2 * 4 + 1] = {NULL};
    for ( int i = 0; i < 4; ++i ) {
        const size_t length = strlen(prefixes[i]);
        values[i] = malloc(length + 20000 + 1);
        TB_CHECK(values[i] != NULL);
        memcpy(values[i], prefixes[i], length);
        memset(values[i] + length, 'a' + i, 20000);
        values[i][length + 20000] = '\0';
        several_tags[2 * i] = GPUJPEG_ENC_OPT_EXIF_TAG;
        several_tags[2 * i + 1] = values[i];
    }
    const struct tb_enc_case several = {.desc = tb_base, .count = 3, .options = several_tags};
    struct gpujpeg_encoder* several_encoder = tb_enc_create(&several, NULL);
    tb_enc_fail(several_encoder, &param, &param_image, 3, &input, 0, GPUJPEG_ERROR);
    gpujpeg_encoder_destroy(several_encoder);
    for ( int i = 0; i < 4; ++i ) {
        free(values[i]);
    }

    // batch limits (compressed data of all images) before allocations, then usable again
    const struct tb_enc_case large = {
        .desc = {256, 256, GPUJPEG_SUBSAMPLING_444, 1, 1, 0, 0},
          .count = 1
    };
    struct gpujpeg_parameters large_param;
    struct gpujpeg_image_parameters large_param_image;
    tb_enc_params(&large, &large_param, &large_param_image);
    tb_enc_fail(encoder, &large_param, &large_param_image, GPUJPEG_BATCH_MAX_COUNT, &input, 0, GPUJPEG_ERROR);
    tb_enc_batch_equal(encoder, &base);
}

/// counts 2, 4, 3, a single image, other geometry, 4; growth allocation failures on a used encoder are retried
static void
tb_enc_reuse(void)
{
    struct tb_enc_case c = {.desc = tb_base, .count = 2};
    struct tb_alloc s = {0};
    struct gpujpeg_allocator allocator = {tb_alloc_device, tb_free_device, &s};
    struct gpujpeg_encoder* encoder = tb_enc_create(&c, &allocator);
    tb_enc_batch_equal(encoder, &c);
    const int allocs_2 = s.allocs;
    c.count = 4;
    tb_enc_batch_equal(encoder, &c);
    const int growth_allocs = s.allocs - allocs_2;
    TB_CHECK(growth_allocs > 0);
    c.count = 3;
    tb_enc_batch_equal(encoder, &c);
    struct tb_jpeg reused = tb_encode(encoder, &tb_base, 5); // single image on the batch layout, as by a new one
    struct gpujpeg_encoder* fresh = tb_enc_create(&c, NULL);
    struct tb_jpeg expected = tb_encode(fresh, &tb_base, 5);
    gpujpeg_encoder_destroy(fresh);
    TB_CHECK(tb_enc_same(&expected, reused.data, reused.size, false));
    free(expected.data);
    free(reused.data);
    TB_CHECK(s.allocs == allocs_2 + growth_allocs); // capacity reused
    const struct tb_enc_case other = {
        .desc = {96, 48, GPUJPEG_SUBSAMPLING_422, 1, 2, 0, 0},
          .count = 2
    };
    tb_enc_batch_equal(encoder, &other);
    c.count = 4;
    tb_enc_batch_equal(encoder, &c);
    gpujpeg_encoder_destroy(encoder);
    TB_CHECK(s.live_count == 0 && s.errors == 0);

    // each allocation of the growth fails once on a used encoder, a retry succeeds
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    tb_enc_params(&c, &param, &param_image);
    const struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(tb_pixels);
    for ( int n = 1; n <= growth_allocs; ++n ) {
        struct tb_alloc f = {0};
        allocator.user_data = &f;
        c.count = 2;
        encoder = tb_enc_create(&c, &allocator);
        tb_enc_batch_equal(encoder, &c);
        f.fail_at = f.allocs + n;
        tb_enc_fail(encoder, &param, &param_image, 4, &input, 0, GPUJPEG_ERROR);
        if ( n % 2 == 0 ) { // other geometry after the failure
            tb_enc_batch_equal(encoder, &other);
        }
        c.count = 4;
        tb_enc_batch_equal(encoder, &c);
        gpujpeg_encoder_destroy(encoder);
        TB_CHECK(f.live_count == 0 && f.errors == 0);
    }
}

/// a flipped single image on a batch layout equals that of a new encoder; the next batch is not affected by what
/// flipping left in the padding of the first image
static void
tb_enc_flip(void)
{
    const struct tb_enc_case c = {
        .desc = {200, 36, GPUJPEG_SUBSAMPLING_420, 1, 1, 0, 0},
          .count = 3
    };
    const char* const flip[] = {GPUJPEG_ENC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_TRUE, NULL};
    struct tb_enc_case flipped = c;
    flipped.options = flip;
    struct gpujpeg_encoder* encoder = tb_enc_create(&c, NULL);
    tb_enc_batch_equal(encoder, &c);
    TB_CHECK(gpujpeg_encoder_set_option(encoder, GPUJPEG_ENC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_TRUE) == 0);
    tb_fill(7);
    struct tb_jpeg j = tb_enc_single(encoder, &c, tb_pixels);
    TB_CHECK(encoder->coder.image_capacity == 3 && encoder->coder.data_padding_dirty);
    struct gpujpeg_encoder* fresh = tb_enc_create(&flipped, NULL);
    struct tb_jpeg expected = tb_enc_single(fresh, &flipped, tb_pixels);
    gpujpeg_encoder_destroy(fresh);
    TB_CHECK(tb_enc_same(&expected, j.data, j.size, false));
    free(expected.data);
    free(j.data);
    TB_CHECK(gpujpeg_encoder_set_option(encoder, GPUJPEG_ENC_OPT_FLIPPED_BOOL, GPUJPEG_VAL_FALSE) == 0);
    tb_enc_batch_equal(encoder, &c);
    TB_CHECK(encoder->coder.image_capacity == 3 && !encoder->coder.data_padding_dirty);
    gpujpeg_encoder_destroy(encoder);
}

void
test_batch();

void
test_batch()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(NULL);
    TB_CHECK(encoder != NULL);
    struct gpujpeg_decoder* decoder = tb_decoder(NULL, NULL);
    tb_equivalence(encoder, decoder);
    tb_homogeneity(encoder, decoder);
    tb_malformed_headers(encoder, decoder);
    tb_arguments(encoder, decoder);
    gpujpeg_decoder_destroy(decoder);
    tb_failed_tables(encoder);
    tb_reuse(encoder);
    const bool delayed = tb_completion(encoder);
    tb_enc_equivalence(encoder);
    tb_enc_arguments(encoder);
    gpujpeg_encoder_destroy(encoder);
    tb_enc_reuse();
    tb_enc_flip();
    printf("Ok%s\n", delayed ? "" : " (delayed completion case skipped, stream host functions not supported)");
}
