#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/gpujpeg_decoder_internal.h"
#include "../../src/gpujpeg_huffman_cpu_decoder.h"
#include "../../src/gpujpeg_huffman_gpu_decoder.h"

/*
 * GPU Huffman decoding without restart markers (gpujpeg_huffman_gpu_decoder_resync_decode()). JPEGs are written
 * here from generated coefficients (one interleaved baseline scan, only the used Huffman tables) and the decoded
 * coefficients, including padding blocks, must be exactly those of the CPU Huffman decoder, which must be the
 * generated ones. The route taken is checked. Small chunk, sequence and repair limits (private decoder parameters)
 * cover empty chunks, stuffed zeros at chunk starts, partial sequences and repair exhaustion.
 */

// gpujpeg_table.c (not declared in a header)
void
gpujpeg_table_huffman_encoder_compute(struct gpujpeg_table_huffman_encoder* table);

#define RS_CHECK(cond)                                                                                                 \
    if ( !(cond) ) {                                                                                                   \
        fprintf(stderr, "%s:%d: Huffman resync check '%s' failed!\n", __FILE__, __LINE__, #cond);                      \
        abort();                                                                                                       \
    }

/// route expected by rs_decode(): RESYNC, or RESYNC_RETRY if not synchronized
#define RS_ROUTE_RESYNC_ANY ((enum gpujpeg_huffman_route) - 1)

enum {
    RS_MIN_BYTES = 16384, ///< gpujpeg_huffman_gpu_resync_param::min_bytes of the tested decoders
};

enum rs_padding {
    RS_PAD_ONES,       ///< standard
    RS_PAD_ZEROS,      ///< not standard, retried on CPU (if there are padding bits)
    RS_PAD_EXTRA_BYTE, ///< standard padding followed by a zero byte, retried on CPU
};

struct rs_jpeg_desc
{
    int width;
    int height;
    int comp_count;
    int sampling[3][2]; ///< horizontal, vertical
    bool reversed;      ///< symbol order of the Annex K tables reversed (other code lengths, 16-bit end of block)
    bool shared_table;  ///< chroma uses table 0 too (parsing does not tell the MCU phase)
    bool sof1;
    bool dht_after_scan;
    enum rs_padding padding;
    unsigned seed;
    int ac_limit; ///< zig-zag coefficients from this index on are zero (0: as generated)
};

struct rs_jpeg
{
    uint8_t* data;
    size_t size;
    size_t scan_begin;
    size_t scan_size;
    int pad_bits;
    int block_count;
    int16_t* blocks; ///< zig-zag coefficients (absolute DC) of the blocks in scan order, NULL if not valid
};

struct rs_buf
{
    uint8_t* data;
    size_t size;
    size_t cap;
};

struct rs_bits
{
    struct rs_buf* buf;
    unsigned acc;
    int count;
};

static void
rs_byte(struct rs_buf* b, unsigned value)
{
    if ( b->size == b->cap ) {
        b->cap = b->cap == 0 ? 65536 : 2 * b->cap;
        b->data = realloc(b->data, b->cap);
        RS_CHECK(b->data != NULL);
    }
    b->data[b->size++] = (uint8_t)value;
}

static void
rs_word(struct rs_buf* b, unsigned value)
{
    rs_byte(b, value >> 8);
    rs_byte(b, value & 0xFF);
}

/// writes len bits of code MSB first with byte stuffing
static void
rs_put(struct rs_bits* w, unsigned code, int len)
{
    for ( int i = len - 1; i >= 0; --i ) {
        w->acc = (w->acc << 1) | ((code >> i) & 1U);
        if ( ++w->count == 8 ) {
            rs_byte(w->buf, w->acc & 0xFF);
            if ( (w->acc & 0xFF) == 0xFF ) {
                rs_byte(w->buf, 0);
            }
            w->acc = 0;
            w->count = 0;
        }
    }
}

static void
rs_symbol(struct rs_bits* w, const struct gpujpeg_table_huffman_encoder* table, unsigned symbol)
{
    RS_CHECK(table->size[symbol] > 0);
    rs_put(w, table->code[symbol], table->size[symbol]);
}

static int
rs_category(int value)
{
    int category = 0;
    for ( int magnitude = value < 0 ? -value : value; magnitude != 0; magnitude >>= 1 ) {
        category += 1;
    }
    return category;
}

static void
rs_value(struct rs_bits* w, int value, int category)
{
    rs_put(w, value < 0 ? (unsigned)(value + (1 << category) - 1) : (unsigned)value, category);
}

static unsigned
rs_rand(unsigned* state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int
rs_ac_value(unsigned* state, int max_category)
{
    const int category = 1 + (int)(rs_rand(state) % max_category);
    const int magnitude = (1 << (category - 1)) + (int)(rs_rand(state) % (1U << (category - 1)));
    return (rs_rand(state) & 1) != 0 ? -magnitude : magnitude;
}

/// generates zig-zag coefficients of a block: DC categories 0-11, empty, sparse (ZRL), dense, all-one value bits
static void
rs_generate_block(unsigned* state, int16_t z[64], int* predictor)
{
    int dc = *predictor;
    switch ( rs_rand(state) % 4 ) {
    case 0:
        break; // zero difference
    case 1:
        dc += (int)(rs_rand(state) % 61) - 30;
        break;
    case 2:
        dc = (int)(rs_rand(state) % 2047) - 1023;
        break;
    default:
        dc += (int)(rs_rand(state) % 7) - 3;
        break;
    }
    dc = dc < -1023 ? -1023 : dc > 1023 ? 1023 : dc;
    memset(z, 0, 64 * sizeof *z);
    z[0] = (int16_t)dc;
    *predictor = dc;
    switch ( rs_rand(state) % 6 ) {
    case 0: // end of block right after DC
        break;
    case 1: // long zero runs
        for ( int k = 1 + (int)(rs_rand(state) % 40); k < 64; k += 17 + (int)(rs_rand(state) % 30) ) {
            z[k] = (int16_t)rs_ac_value(state, 6);
        }
        break;
    case 2: // dense, longer than small chunks
        for ( int k = 1; k < 64; ++k ) {
            z[k] = (int16_t)rs_ac_value(state, 10);
        }
        break;
    case 3: // one value bits (0xFF bytes), last coefficient set
        for ( int k = 1; k < 64; k += 1 + (int)(rs_rand(state) % 3) ) {
            z[k] = 1023;
        }
        z[63] = 1023;
        break;
    default:
        for ( int n = (int)(rs_rand(state) % 24); n > 0; --n ) {
            z[1 + rs_rand(state) % 63] = (int16_t)rs_ac_value(state, 4);
        }
        break;
    }
}

static void
rs_encode_block(struct rs_bits* w, const int16_t z[64], int dc_diff, const struct gpujpeg_table_huffman_encoder* dc,
                const struct gpujpeg_table_huffman_encoder* ac)
{
    int category = rs_category(dc_diff);
    rs_symbol(w, dc, category);
    rs_value(w, dc_diff, category);
    int run = 0;
    for ( int k = 1; k < 64; ++k ) {
        if ( z[k] == 0 ) {
            run += 1;
            continue;
        }
        for ( ; run > 15; run -= 16 ) {
            rs_symbol(w, ac, 0xF0);
        }
        category = rs_category(z[k]);
        rs_symbol(w, ac, run << 4 | category);
        rs_value(w, z[k], category);
        run = 0;
    }
    if ( run > 0 ) {
        rs_symbol(w, ac, 0x00);
    }
}

static void
rs_dht(struct rs_buf* b, const struct gpujpeg_table_huffman_encoder* table, int index, int class)
{
    int count = 0;
    for ( int length = 1; length <= 16; ++length ) {
        count += table->bits[length];
    }
    rs_word(b, 0xFFC4);
    rs_word(b, 2 + 1 + 16 + count);
    rs_byte(b, class << 4 | index);
    for ( int length = 1; length <= 16; ++length ) {
        rs_byte(b, table->bits[length]);
    }
    for ( int i = 0; i < count; ++i ) {
        rs_byte(b, table->huffval[i]);
    }
}

static struct rs_jpeg
rs_write(const struct rs_jpeg_desc* d)
{
    // Annex K tables (index 0 luminance, 1 chrominance), optionally with the symbol order reversed
    struct gpujpeg_table_huffman_encoder table[2][2];
    for ( int index = 0; index < 2; ++index ) {
        for ( int class = 0; class < 2; ++class ) {
            struct gpujpeg_table_huffman_encoder* t = &table[index][class];
            gpujpeg_table_huffman_encoder_init(t,
                                               index == 0 ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE,
                                               class == 0 ? GPUJPEG_HUFFMAN_DC : GPUJPEG_HUFFMAN_AC);
            if ( d->reversed ) {
                int count = 0;
                for ( int length = 1; length <= 16; ++length ) {
                    count += t->bits[length];
                }
                for ( int i = 0; i < count / 2; ++i ) {
                    const unsigned char tmp = t->huffval[i];
                    t->huffval[i] = t->huffval[count - 1 - i];
                    t->huffval[count - 1 - i] = tmp;
                }
                gpujpeg_table_huffman_encoder_compute(t);
            }
        }
    }
    int selector[3];
    int max_h = 1;
    int max_v = 1;
    int mcu_blocks = 0;
    for ( int c = 0; c < d->comp_count; ++c ) {
        selector[c] = c == 0 || d->shared_table ? 0 : 1;
        max_h = d->sampling[c][0] > max_h ? d->sampling[c][0] : max_h;
        max_v = d->sampling[c][1] > max_v ? d->sampling[c][1] : max_v;
        mcu_blocks += d->sampling[c][0] * d->sampling[c][1];
    }
    const int mcu_count = ((d->width + 8 * max_h - 1) / (8 * max_h)) * ((d->height + 8 * max_v - 1) / (8 * max_v));

    struct rs_buf b = {0};
    rs_word(&b, 0xFFD8);
    rs_word(&b, 0xFFDB); // all-one quantization table 0
    rs_word(&b, 2 + 65);
    rs_byte(&b, 0);
    for ( int i = 0; i < 64; ++i ) {
        rs_byte(&b, 1);
    }
    rs_word(&b, d->sof1 ? 0xFFC1 : 0xFFC0);
    rs_word(&b, 8 + 3 * d->comp_count);
    rs_byte(&b, 8);
    rs_word(&b, d->height);
    rs_word(&b, d->width);
    rs_byte(&b, d->comp_count);
    for ( int c = 0; c < d->comp_count; ++c ) {
        rs_byte(&b, c + 1);
        rs_byte(&b, d->sampling[c][0] << 4 | d->sampling[c][1]);
        rs_byte(&b, 0);
    }
    const int used_tables = d->comp_count == 3 && !d->shared_table ? 2 : 1;
    for ( int index = 0; index < used_tables; ++index ) {
        rs_dht(&b, &table[index][0], index, 0);
        rs_dht(&b, &table[index][1], index, 1);
    }
    rs_word(&b, 0xFFDA);
    rs_word(&b, 6 + 2 * d->comp_count);
    rs_byte(&b, d->comp_count);
    for ( int c = 0; c < d->comp_count; ++c ) {
        rs_byte(&b, c + 1);
        rs_byte(&b, selector[c] << 4 | selector[c]);
    }
    rs_byte(&b, 0);
    rs_byte(&b, 63);
    rs_byte(&b, 0);

    struct rs_jpeg jpeg = {0};
    jpeg.scan_begin = b.size;
    jpeg.block_count = mcu_count * mcu_blocks;
    jpeg.blocks = malloc((size_t)jpeg.block_count * 64 * sizeof *jpeg.blocks);
    RS_CHECK(jpeg.blocks != NULL);
    struct rs_bits w = {&b, 0, 0};
    unsigned state = d->seed | 1U;
    int predictor[3] = {0, 0, 0};
    int16_t* z = jpeg.blocks;
    for ( int mcu = 0; mcu < mcu_count; ++mcu ) {
        for ( int c = 0; c < d->comp_count; ++c ) {
            for ( int i = 0; i < d->sampling[c][0] * d->sampling[c][1]; ++i, z += 64 ) {
                const int previous = predictor[c];
                rs_generate_block(&state, z, &predictor[c]);
                if ( d->ac_limit > 0 ) { // sparser, same random sequence
                    memset(z + d->ac_limit, 0, (64 - d->ac_limit) * sizeof *z);
                }
                rs_encode_block(&w, z, z[0] - previous, &table[selector[c]][0], &table[selector[c]][1]);
            }
        }
    }
    jpeg.pad_bits = (8 - w.count) % 8;
    rs_put(&w, d->padding == RS_PAD_ZEROS ? 0 : (1U << jpeg.pad_bits) - 1, jpeg.pad_bits);
    if ( d->padding == RS_PAD_EXTRA_BYTE ) {
        rs_byte(&b, 0);
    }
    jpeg.scan_size = b.size - jpeg.scan_begin;
    if ( d->dht_after_scan ) { // the reader ends a scan at APPn (or SOS, EOI)
        rs_word(&b, 0xFFEF);
        rs_word(&b, 2);
        rs_dht(&b, &table[0][0], 0, 0);
    }
    rs_word(&b, 0xFFD9);
    jpeg.data = b.data;
    jpeg.size = b.size;
    return jpeg;
}

static void
rs_jpeg_free(struct rs_jpeg* jpeg)
{
    free(jpeg->data);
    free(jpeg->blocks);
}

/// compares the device coefficients with the CPU Huffman decoder and those with the generated ones (if valid)
static void
rs_check_coefficients(struct gpujpeg_decoder* decoder, const struct rs_jpeg* jpeg, const char* name)
{
    struct gpujpeg_coder* coder = &decoder->coder;
    const size_t count = coder->data_size;
    int16_t* gpu = malloc(count * sizeof *gpu);
    RS_CHECK(gpu != NULL);
    RS_CHECK(cudaMemcpy(gpu, coder->d_data_quantized, count * sizeof *gpu, cudaMemcpyDeviceToHost) == cudaSuccess);
    if ( coder->data_quantized == NULL ) {
        RS_CHECK(gpujpeg_coder_allocate_cpu_huffman_buf(coder) == 0);
    }
    RS_CHECK(gpujpeg_huffman_cpu_decoder_decode(decoder) == 0);
    for ( size_t i = 0; i < count; ++i ) {
        if ( gpu[i] != coder->data_quantized[i] ) {
            fprintf(stderr, "%s: coefficient %zu of %zu is %d, CPU decoder %d!\n", name, i, count, gpu[i],
                    coder->data_quantized[i]);
            abort();
        }
    }
    if ( jpeg->blocks != NULL ) {
        RS_CHECK(jpeg->block_count == coder->block_count && (size_t)jpeg->block_count * 64 == count);
        for ( int i = 0; i < jpeg->block_count; ++i ) {
            const int16_t* block = coder->data_quantized + (coder->block_list[i] >> 8);
            for ( int k = 0; k < 64; ++k ) {
                RS_CHECK(block[gpujpeg_order_natural[k]] == jpeg->blocks[i * 64 + k]);
            }
        }
    }
    free(gpu);
}

/// decodes, checks the route (and for RESYNC_RETRY the status bits) and the coefficients
static bool
rs_decode(struct gpujpeg_decoder* decoder, const struct rs_jpeg* jpeg, enum gpujpeg_huffman_route route,
          unsigned status_bits, const char* name)
{
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    if ( gpujpeg_decoder_decode(decoder, jpeg->data, jpeg->size, &output) != 0 ) {
        fprintf(stderr, "%s: decoding failed!\n", name);
        abort();
    }
    const enum gpujpeg_huffman_route actual = decoder->huffman_route;
    bool ok = actual == route;
    if ( route == RS_ROUTE_RESYNC_ANY ) {
        ok = actual == GPUJPEG_HUFFMAN_ROUTE_RESYNC || actual == GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY;
    }
    if ( actual == GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY ) {
        ok = ok && (decoder->huffman_resync_status & status_bits) == status_bits;
    }
    if ( !ok ) {
        fprintf(stderr, "%s: route %d (status 0x%x), expected %d (status bits 0x%x)!\n", name, (int)actual,
                decoder->huffman_resync_status, (int)route, status_bits);
        abort();
    }
    rs_check_coefficients(decoder, jpeg, name);
    return actual == GPUJPEG_HUFFMAN_ROUTE_RESYNC;
}

struct rs_alloc_state
{
    int allocs;
    int live;
};

static void*
rs_device_alloc(void* user_data, size_t size, cudaStream_t stream)
{
    (void)stream;
    struct rs_alloc_state* s = user_data;
    void* ptr = NULL;
    if ( cudaMalloc(&ptr, size) != cudaSuccess ) {
        return NULL;
    }
    s->allocs += 1;
    s->live += 1;
    return ptr;
}

static void
rs_device_free(void* user_data, void* ptr, cudaStream_t stream)
{
    (void)stream;
    struct rs_alloc_state* s = user_data;
    s->live -= 1;
    cudaFree(ptr);
}

/// creates a decoder with the given limits (independent of the compile-time defaults)
static struct gpujpeg_decoder*
rs_decoder_create(struct rs_alloc_state* alloc, int chunk_bytes, int seq_chunks, int repair_rounds)
{
    struct gpujpeg_decoder_init_parameters param = gpujpeg_decoder_default_init_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    struct gpujpeg_allocator allocator = {rs_device_alloc, rs_device_free, alloc};
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create_with_allocator(&param, alloc != NULL ? &allocator : NULL);
    RS_CHECK(decoder != NULL);
    decoder->resync_param.chunk_bytes = chunk_bytes;
    decoder->resync_param.seq_chunks = seq_chunks;
    decoder->resync_param.repair_rounds = repair_rounds;
    decoder->resync_param.min_bytes = RS_MIN_BYTES;
    return decoder;
}

/// GPUJPEG-encoded interleaved 256x256 4:4:4 with restart markers
static void
rs_encode_markers(uint8_t* image, int restart_interval, int interleaved, uint8_t** jpeg, size_t* len)
{
    static struct gpujpeg_encoder* encoder;
    if ( encoder == NULL ) {
        encoder = gpujpeg_encoder_create(0);
        RS_CHECK(encoder != NULL);
    }
    if ( image == NULL ) { // cleanup
        gpujpeg_encoder_destroy(encoder);
        encoder = NULL;
        return;
    }
    struct gpujpeg_parameters param = gpujpeg_default_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    param.restart_interval = restart_interval;
    param.interleaved = interleaved;
    struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
    param_image.width = 256;
    param_image.height = 256;
    struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(image);
    RS_CHECK(gpujpeg_encoder_encode(encoder, &param, &param_image, &input, jpeg, len) == 0);
}

/**
 * Library default limits on a reused decoder: below min_bytes, only 256x256 interleaved 4:4:4 scans of 32 KiB up to
 * 64 KiB are decoded by resynchronization, other limits keep the CPU decoder for them.
 * @retval false not checked, the library was built with other default limits
 */
static bool
rs_default_policy(void)
{
    struct gpujpeg_decoder_init_parameters init = gpujpeg_decoder_default_init_parameters();
    init.verbose = GPUJPEG_LL_QUIET;
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create_with_params(&init);
    RS_CHECK(decoder != NULL);
    const struct gpujpeg_huffman_gpu_resync_param defaults = decoder->resync_param;
    if ( defaults.chunk_bytes != 512 || defaults.seq_chunks != 128 || defaults.repair_rounds != 128 ||
         defaults.min_bytes != 262144 ) {
        gpujpeg_decoder_destroy(decoder);
        return false;
    }
    // the first one is admitted, the others differ by AC coefficients (same seed), dimensions (same scan) or layout
    const struct rs_jpeg_desc cases[] = {
        {256, 256, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 20},
        {256, 256, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 1 }, // sparser
        {256, 256, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 0 }, // denser
        {255, 256, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 20},
        {256, 255, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 20},
        {256, 256, 3, {{2, 2}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 31, 40},
        {256, 256, 1, {{1, 1}},                 false, false, false, false, RS_PAD_ONES, 31, 0 },
    };
    const int band[] = {0, -1, 1, 0, 0, 0, 0}; // scan size below (-1), within (0) or above (1) 32 KiB up to 64 KiB
    enum {
        CASE_COUNT = sizeof cases / sizeof cases[0],
    };
    struct rs_jpeg jpeg[CASE_COUNT];
    for ( int i = 0; i < CASE_COUNT; ++i ) {
        jpeg[i] = rs_write(&cases[i]);
        const size_t size = jpeg[i].scan_size;
        RS_CHECK((size < 32 * 1024 ? -1 : size < 64 * 1024 ? 0 : 1) == band[i] && size < defaults.min_bytes);
        rs_decode(decoder, &jpeg[i], i == 0 ? GPUJPEG_HUFFMAN_ROUTE_RESYNC : GPUJPEG_HUFFMAN_ROUTE_CPU, 0,
                  "default limits");
        RS_CHECK(decoder->data_compressed_size == size);
        RS_CHECK(i != 0 || decoder->huffman_resync_status == 0);
    }
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "default limits, again");
    RS_CHECK(decoder->huffman_resync_status == 0);
    // a zero byte after the last block fails certification and triggers a CPU retry; then decode the valid scan
    // again on the same decoder
    struct rs_jpeg_desc extra_byte = cases[0];
    extra_byte.padding = RS_PAD_EXTRA_BYTE;
    struct rs_jpeg retried = rs_write(&extra_byte);
    RS_CHECK(retried.scan_size == jpeg[0].scan_size + 1 && retried.scan_size < 64 * 1024);
    rs_decode(decoder, &retried, GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY, GPUJPEG_HUFFMAN_RESYNC_PADDING,
              "default limits, extra byte");
    rs_jpeg_free(&retried);
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "default limits after retry");
    RS_CHECK(decoder->huffman_resync_status == 0);
    RS_CHECK(decoder->resync_param.chunk_bytes == defaults.chunk_bytes &&
             decoder->resync_param.seq_chunks == defaults.seq_chunks &&
             decoder->resync_param.repair_rounds == defaults.repair_rounds &&
             decoder->resync_param.min_bytes == defaults.min_bytes);

    decoder->resync_param.min_bytes = SIZE_MAX;
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_CPU, 0, "larger minimum");
    decoder->resync_param = defaults;
    decoder->resync_param.repair_rounds = 1024;
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_CPU, 0, "other repair rounds");
    decoder->resync_param = defaults;
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "default limits assigned");
    gpujpeg_decoder_destroy(decoder);
    for ( int i = 0; i < CASE_COUNT; ++i ) {
        rs_jpeg_free(&jpeg[i]);
    }
    return true;
}

void
test_huffman_resync();

void
test_huffman_resync()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    // gray, 4:4:4, 4:2:2 and 4:2:0 with odd sizes, Annex K and reversed tables
    const struct rs_jpeg_desc layouts[] = {
        {333, 211, 1, {{1, 1}},                 false, false, false, false, RS_PAD_ONES, 11},
        {400, 300, 3, {{1, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 12},
        {401, 299, 3, {{2, 1}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 13},
        {517, 389, 3, {{2, 2}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 14},
        {333, 211, 1, {{1, 1}},                 true,  false, false, false, RS_PAD_ONES, 15},
        {400, 300, 3, {{1, 1}, {1, 1}, {1, 1}}, true,  false, false, false, RS_PAD_ONES, 16},
        {401, 299, 3, {{2, 1}, {1, 1}, {1, 1}}, true,  false, false, false, RS_PAD_ONES, 17},
        {517, 389, 3, {{2, 2}, {1, 1}, {1, 1}}, true,  false, false, false, RS_PAD_ONES, 18},
    };
    enum {
        LAYOUT_COUNT = sizeof layouts / sizeof layouts[0],
    };
    struct rs_jpeg jpeg[LAYOUT_COUNT];
    for ( int i = 0; i < LAYOUT_COUNT; ++i ) {
        jpeg[i] = rs_write(&layouts[i]);
        RS_CHECK(jpeg[i].scan_size >= RS_MIN_BYTES);
    }
    const struct rs_jpeg* const gray = &jpeg[0];
    const struct rs_jpeg* const s420 = &jpeg[3];

    // A fresh decoder (the gray JPEG defines only its used tables) is reused across layouts and tables.
    // Reversed tables need a larger test-only repair budget to exercise successful replay.
    struct rs_alloc_state alloc = {0, 0};
    struct gpujpeg_decoder* decoder = rs_decoder_create(&alloc, 512, 128, 128);
    for ( int i = 0; i < LAYOUT_COUNT; ++i ) {
        decoder->resync_param.repair_rounds = layouts[i].reversed ? 1024 : 128;
        rs_decode(decoder, &jpeg[i], GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "layout replay");
    }
    decoder->resync_param.repair_rounds = 128;
    // These deterministic reversed-table fixtures exhaust the production budget; retry must remain exact.
    const int bounded_retry_layouts[] = {5, 7}; // reversed 4:4:4 and 4:2:0
    for ( unsigned i = 0; i < sizeof bounded_retry_layouts / sizeof bounded_retry_layouts[0]; ++i ) {
        rs_decode(decoder, &jpeg[bounded_retry_layouts[i]], GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY,
                  GPUJPEG_HUFFMAN_RESYNC_REPAIR_LIMIT, "reversed tables, bounded retry");
    }
    const int allocs = alloc.allocs;
    rs_decode(decoder, s420, GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "default limits, again");
    RS_CHECK(alloc.allocs == allocs);

    // transitions to and from the other routes on the same decoder
    static uint8_t image[256 * 256 * 3];
    for ( unsigned i = 0; i < sizeof image; ++i ) {
        image[i] = (uint8_t)(i / 3 % 256 + (i * 2654435761U >> 27));
    }
    const struct
    {
        int restart_interval;
        int interleaved;
        enum gpujpeg_huffman_route route;
    } markers[] = {
        {8,   1, GPUJPEG_HUFFMAN_ROUTE_GPU}, // 128 segments
        {128, 1, GPUJPEG_HUFFMAN_ROUTE_CPU}, // 8 segments
        {0,   0, GPUJPEG_HUFFMAN_ROUTE_CPU}, // 3 scans
    };
    for ( unsigned i = 0; i < sizeof markers / sizeof markers[0]; ++i ) {
        struct rs_jpeg encoded = {0};
        rs_encode_markers(image, markers[i].restart_interval, markers[i].interleaved, &encoded.data, &encoded.size);
        rs_decode(decoder, &encoded, markers[i].route, 0, "restart markers");
        rs_decode(decoder, gray, GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "after restart markers");
    }
    rs_encode_markers(NULL, 0, 0, NULL, NULL);

    // not eligible: legacy route (single segment, so CPU)
    const struct rs_jpeg_desc legacy[] = {
        {64,  64,  1, {{1, 1}},                 false, false, false, false, RS_PAD_ONES, 21}, // below min_bytes
        {333, 211, 1, {{1, 1}},                 false, false, true,  false, RS_PAD_ONES, 22}, // SOF1
        {333, 211, 1, {{1, 1}},                 false, false, false, true,  RS_PAD_ONES, 23}, // DHT after the scan
        {400, 300, 3, {{1, 2}, {1, 1}, {1, 1}}, false, false, false, false, RS_PAD_ONES, 24}, // 4:4:0
    };
    for ( unsigned i = 0; i < sizeof legacy / sizeof legacy[0]; ++i ) {
        struct rs_jpeg j = rs_write(&legacy[i]);
        RS_CHECK(i != 0 || j.scan_size < RS_MIN_BYTES);
        rs_decode(decoder, &j, GPUJPEG_HUFFMAN_ROUTE_CPU, 0, "not eligible");
        rs_jpeg_free(&j);
    }

    // not certified: decoded again on CPU
    struct rs_jpeg_desc desc = layouts[3];
    desc.padding = RS_PAD_EXTRA_BYTE;
    struct rs_jpeg j = rs_write(&desc);
    rs_decode(decoder, &j, GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY, GPUJPEG_HUFFMAN_RESYNC_PADDING, "extra byte");
    rs_jpeg_free(&j);
    desc.padding = RS_PAD_ZEROS;
    j = rs_write(&desc);
    rs_decode(decoder, &j, j.pad_bits > 0 ? GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY : GPUJPEG_HUFFMAN_ROUTE_RESYNC,
              j.pad_bits > 0 ? GPUJPEG_HUFFMAN_RESYNC_PADDING : 0, "zero padding");
    // corrupted data (no 0xFF created or removed): any route, still the CPU decoder's coefficients
    for ( size_t pos = j.scan_begin + j.scan_size / 3; pos < j.scan_begin + j.scan_size; ++pos ) {
        const uint8_t value = j.data[pos] ^ 0x5A;
        if ( j.data[pos] != 0xFF && j.data[pos - 1] != 0xFF && value != 0xFF ) {
            j.data[pos] = value;
            break;
        }
    }
    free(j.blocks);
    j.blocks = NULL;
    rs_decode(decoder, &j, RS_ROUTE_RESYNC_ANY, 0, "corrupted");
    rs_jpeg_free(&j);
    rs_decode(decoder, s420, GPUJPEG_HUFFMAN_ROUTE_RESYNC, 0, "after retries");
    gpujpeg_decoder_destroy(decoder);
    RS_CHECK(alloc.live == 0);

    // small limits: 32-byte chunks make empty chunks, chunks starting at stuffed zeros and many
    // sequences; when not synchronized within the rounds, the CPU decoder takes over
    const struct
    {
        int chunk_bytes;
        int seq_chunks;
        int repair_rounds;
    } limits[] = {
        {32, 32, 1024},
        {32, 96, 1024},
    };
    int stress_resync_count = 0;
    for ( unsigned l = 0; l < sizeof limits / sizeof limits[0]; ++l ) {
        int synchronized = 0;
        decoder = rs_decoder_create(NULL, limits[l].chunk_bytes, limits[l].seq_chunks, limits[l].repair_rounds);
        for ( int i = 0; i < LAYOUT_COUNT; ++i ) {
            synchronized += rs_decode(decoder, &jpeg[i], RS_ROUTE_RESYNC_ANY, 0, "small limits");
        }
        RS_CHECK(synchronized > 0);
        stress_resync_count += synchronized;
        gpujpeg_decoder_destroy(decoder);
    }
    // One-byte records for this dense grayscale fixture exceed the borrowed scratch capacity.
    decoder = rs_decoder_create(NULL, 1, 1024, 1024);
    rs_decode(decoder, &jpeg[0], GPUJPEG_HUFFMAN_ROUTE_CPU, 0, "scratch capacity");
    gpujpeg_decoder_destroy(decoder);
    int chunk_starts_stuffed = 0;
    for ( size_t pos = 32; pos < s420->scan_size; pos += 32 ) {
        chunk_starts_stuffed += s420->data[s420->scan_begin + pos - 1] == 0xFF;
    }

    // a single repair round cannot reach the true trajectory in later sequences
    decoder = rs_decoder_create(NULL, 32, 32, 1);
    rs_decode(decoder, s420, GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY, GPUJPEG_HUFFMAN_RESYNC_REPAIR_LIMIT, "repair limit");
    // chroma sharing the luma table: parsing cannot tell the MCU phase, so wrong phases must not be certified
    desc = layouts[1];
    desc.shared_table = true;
    j = rs_write(&desc);
    rs_decode(decoder, &j, RS_ROUTE_RESYNC_ANY, 0, "shared table, repair limit");
    gpujpeg_decoder_destroy(decoder);
    decoder = rs_decoder_create(NULL, 32, 32, 1024);
    rs_decode(decoder, &j, RS_ROUTE_RESYNC_ANY, 0, "shared table");
    gpujpeg_decoder_destroy(decoder);
    rs_jpeg_free(&j);

    for ( int i = 0; i < LAYOUT_COUNT; ++i ) {
        rs_jpeg_free(&jpeg[i]);
    }
    const bool default_policy = rs_default_policy();
    printf("Ok (%d of %zu small-chunk decodings synchronized, %d 32-byte chunks start at a stuffed zero, default "
           "small-scan limits %s)\n",
           stress_resync_count, (sizeof limits / sizeof limits[0]) * LAYOUT_COUNT, chunk_starts_stuffed,
           default_policy ? "checked" : "not checked (other compile-time defaults)");
}
