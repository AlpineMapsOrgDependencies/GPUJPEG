/**
 * @file
 * Copyright (c) 2011-2026, CESNET
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "../libgpujpeg/gpujpeg_decoder.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_postprocessor.h"
#include "gpujpeg_reader.h"
#include "gpujpeg_util.h"
#include "utils/image_delegate.h" //  for image_delegate_stbi_tga_set_rle
#include <stdint.h>               // for PTRDIFF_MAX, SIZE_MAX

/* Documented at declaration */
void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output)
{

    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
    output->data = custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture)
{
    output->type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    output->data = NULL;
    output->data_size = 0;
    output->texture = texture;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_custom_cuda(struct gpujpeg_decoder_output* output, uint8_t* d_custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    output->data = d_custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

static struct gpujpeg_decoder*
decoder_create(cudaStream_t stream, const struct gpujpeg_allocator* allocator)
{
    if ( !gpujpeg_allocator_is_valid(allocator) ) {
        return NULL;
    }
    struct gpujpeg_decoder* decoder = (struct gpujpeg_decoder*) calloc(1, sizeof(struct gpujpeg_decoder));
    if ( decoder == NULL )
        return NULL;

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 ) {
        free(decoder);
        return NULL;
    }
    if ( allocator != NULL ) {
        coder->allocator = *allocator;
    }

    // Set parameters
    gpujpeg_set_default_parameters(&coder->param);
    gpujpeg_image_set_default_parameters(&coder->param_image);
    coder->param_image.width = 0;
    coder->param_image.height = 0;
    coder->param.comp_count = 0;
    coder->param.restart_interval = 0;
    decoder->req_pixel_format = GPUJPEG_PIXFMT_AUTODETECT;
    decoder->req_color_space = GPUJPEG_CS_DEFAULT;
    decoder->resync_param = (struct gpujpeg_huffman_gpu_resync_param){
        GPUJPEG_HUFFMAN_GPU_RESYNC_CHUNK_BYTES, GPUJPEG_HUFFMAN_GPU_RESYNC_SEQ_CHUNKS,
        GPUJPEG_HUFFMAN_GPU_RESYNC_REPAIR_ROUNDS, GPUJPEG_HUFFMAN_GPU_RESYNC_MIN_BYTES};

    int result = 1;

    // Stream
    coder->stream = stream;

    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        if ( gpujpeg_coder_device_alloc(coder, (void**)&decoder->table_quantization[comp_type].d_table,
                                        64 * sizeof(uint16_t)) != 0 )
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            // gpujpeg_huffman_decoder_table_kernel() computes quick tables for 2 pair of Huffman tables, but eg. for grayscale only one pair is
            // present which causes the function potentially crash because computing from garbage values - memsetting to 0 fixes that
            if ( gpujpeg_coder_device_alloc(coder, (void**)&decoder->d_table_huffman[comp_type][huff_type],
                                            sizeof(struct gpujpeg_table_huffman_decoder)) != 0 ||
                 cudaSuccess != cudaMemsetAsync(decoder->d_table_huffman[comp_type][huff_type], 0,
                                                sizeof(struct gpujpeg_table_huffman_decoder), coder->stream) ) {
                result = 0;
            }
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation", result = 0);

    // Init huffman encoder
    if ((decoder->huffman_gpu_decoder = gpujpeg_huffman_gpu_decoder_init(coder)) == NULL) {
        result = 0;
    }

    if (result == 0) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }

    return decoder;
}

/* Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create(cudaStream_t stream)
{
    return decoder_create(stream, NULL);
}

struct gpujpeg_decoder_init_parameters
gpujpeg_decoder_default_init_parameters()
{
    return (struct gpujpeg_decoder_init_parameters){cudaStreamDefault, 0, false, false};
}
/**
 * Create JPEG decoder
 *
 * @sa gpujpeg_decoder_create
 * @return decoder structure if succeeds, otherwise NULL
 */
struct gpujpeg_decoder*
gpujpeg_decoder_create_with_params(const struct gpujpeg_decoder_init_parameters *params)
{
    return gpujpeg_decoder_create_with_allocator(params, NULL);
}

/* Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create_with_allocator(const struct gpujpeg_decoder_init_parameters* params,
                                      const struct gpujpeg_allocator* allocator)
{
    struct gpujpeg_decoder* decoder = decoder_create(params->stream, allocator);
    if ( decoder == NULL ) {
        return NULL;
    }
    decoder->coder.param.verbose = params->verbose;
    decoder->coder.param.perf_stats = params->perf_stats;
    decoder->ff_cs_itu601_is_709 = params->ff_cs_itu601_is_709;
    return decoder;
}

bool
gpujpeg_decoder_layout_differs(const struct gpujpeg_coder* coder, const struct gpujpeg_parameters* param,
                               const struct gpujpeg_image_parameters* param_image)
{
    int change = 0;
    change |= coder->param_image.width != param_image->width;
    change |= coder->param_image.width_padding != param_image->width_padding;
    change |= coder->param_image.height != param_image->height;
    change |= coder->param.comp_count != param->comp_count;
    change |= coder->param.restart_interval != param->restart_interval;
    change |= coder->param.interleaved != param->interleaved;
    change |= coder->param.color_space_internal != param->color_space_internal;
    change |= coder->param_image.color_space != param_image->color_space;
    change |= coder->param_image.pixel_format != param_image->pixel_format;
    for ( int comp = 0; comp < param->comp_count; comp++ ) {
        change |= coder->param.sampling_factor[comp].horizontal != param->sampling_factor[comp].horizontal;
        change |= coder->param.sampling_factor[comp].vertical != param->sampling_factor[comp].vertical;
    }
    return change != 0;
}

/* Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, const struct gpujpeg_parameters* param,
                     const struct gpujpeg_image_parameters* param_image)
{
    return gpujpeg_decoder_init_layout(decoder, param, param_image, 1, false);
}

int
gpujpeg_decoder_init_layout(struct gpujpeg_decoder* decoder, const struct gpujpeg_parameters* param,
                            const struct gpujpeg_image_parameters* param_image, int image_count, bool batch)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    coder->param.verbose = param->verbose;
    coder->param.perf_stats = param->perf_stats || param->verbose >= GPUJPEG_LL_STATUS;
    if ( param_image->width == 0 || param_image->height == 0 || param->comp_count == 0 ) {
        return 0;
    }

    // Check if (re)inialization is needed (also after a failed one)
    if ( !gpujpeg_decoder_layout_differs(coder, param, param_image) && coder->image_capacity >= image_count &&
         (!batch || coder->layout_batch) ) {
        return 0;
    }

    if ( coder->param.verbose >= GPUJPEG_LL_INFO &&
         (coder->param_image.width != 0 || coder->param_image.height != 0 || coder->param.comp_count != 0) ) {
        fprintf(stderr, "[GPUJPEG] [Info] Reinitializing decoder.\n");
    }

    if ( 0 == gpujpeg_coder_init_image(coder, param, param_image, image_count, batch, coder->stream) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init coder image!\n");
        return -1;
    }

    // Init postprocessor (also its copies of the component buffer pointers)
    if ( gpujpeg_postprocessor_decoder_init(&decoder->coder) != 0 ) {
        coder->image_capacity = 0;
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init postprocessor!\n");
        return -1;
    }

    gpujpeg_decoder_set_output_format(decoder, param_image->color_space, param_image->pixel_format);

    return 0;
}

/**
 * Huffman decoding on CPU and upload of all quantized coefficients (replacing any previous device content)
 *
 * @return 0 if succeeds, otherwise nonzero
 */
static int
decode_huffman_cpu(struct gpujpeg_decoder* decoder)
{
    struct gpujpeg_coder* coder = &decoder->coder;

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, return -1);
    if (coder->data_quantized == NULL) {
        if (gpujpeg_coder_allocate_cpu_huffman_buf(coder) != 0) {
            return -1;
        }
    }
    if ( 0 != gpujpeg_huffman_cpu_decoder_decode(decoder, 0, decoder->segment_count, 0) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
        return -1;
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, return -1);

    // Copy quantized data to device memory from cpu memory
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, coder->stream, return -1);
    cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t),
                    cudaMemcpyHostToDevice, coder->stream);
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, coder->stream, return -1);

    return 0;
}

/**
 * Uploads the tables committed on host by batch reading (gpujpeg_decoder_batch::quant_dirty, huffman_dirty)
 *
 * @return 0 if succeeds, otherwise nonzero
 */
static int
decoder_upload_tables(struct gpujpeg_decoder* decoder)
{
    for ( int index = 0; index < GPUJPEG_MAX_COMPONENT_COUNT; index++ ) {
        if ( (decoder->batch.quant_dirty & (1U << index)) != 0 &&
             gpujpeg_table_quantization_decoder_compute(&decoder->table_quantization[index], decoder->coder.stream) !=
                 0 ) {
            return -1;
        }
        for ( int type = 0; type < GPUJPEG_HUFFMAN_TYPE_COUNT; type++ ) {
            if ( (decoder->batch.huffman_dirty & (1U << (index * 2 + type))) != 0 &&
                 cudaMemcpyAsync(decoder->d_table_huffman[index][type], &decoder->table_huffman[index][type],
                                 sizeof(struct gpujpeg_table_huffman_decoder), cudaMemcpyHostToDevice,
                                 decoder->coder.stream) != cudaSuccess ) {
                return -1;
            }
        }
    }
    return 0;
}

/**
 * Uploads the tables committed by batch reading and waits for the stream (completing also other queued work), so
 * that device tables match the host ones for a following image that inherits them
 *
 * @return 0 if succeeds (the table masks are cleared), otherwise nonzero
 */
static int
decoder_flush_tables(struct gpujpeg_decoder* decoder)
{
    const int upload_rc = decoder_upload_tables(decoder);
    const cudaError_t sync_rc = cudaStreamSynchronize(decoder->coder.stream); // also after a failed upload
    if ( upload_rc != 0 || sync_rc != cudaSuccess ) {
        const cudaError_t last_error = cudaGetLastError();
        ERROR_MSG("Decoder table upload or stream completion failed: %s!\n",
                  cudaGetErrorString(sync_rc != cudaSuccess ? sync_rc : last_error));
        return -1;
    }
    decoder->batch.quant_dirty = 0;
    decoder->batch.huffman_dirty = 0;
    return 0;
}

/**
 * (Re)allocates internal raw buffers (pinned host and device) of at least size bytes
 *
 * @return 0 if succeeds, otherwise nonzero
 */
static int
decoder_ensure_raw(struct gpujpeg_coder* coder, size_t size)
{
    if ( size <= coder->data_raw_allocated_size ) {
        return 0;
    }
    coder->data_raw_allocated_size = 0;

    // (Re)allocate raw data in host memory
    if ( coder->data_raw != NULL ) {
        cudaFreeHost(coder->data_raw);
    }
    coder->data_raw = NULL;
    cudaMallocHost((void**)&coder->data_raw, size * sizeof(uint8_t));
    gpujpeg_cuda_check_error("Decoder raw data allocation", coder->data_raw = NULL; return -1);
    // (Re)allocate raw data in device memory
    gpujpeg_coder_device_free(coder, coder->d_data_raw_allocated);
    if ( gpujpeg_coder_device_alloc(coder, (void**)&coder->d_data_raw_allocated, size) != 0 ) {
        return -1;
    }
    coder->data_raw_allocated_size = size;
    return 0;
}

/* Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, size_t image_size, struct gpujpeg_decoder_output* output)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    int rc;
    bool use_cpu_huffman_decoder = false;

    coder->start_time = coder->param.perf_stats ? gpujpeg_get_time() : 0;

    // tables of a failed batch the image may inherit
    if ( (decoder->batch.quant_dirty | decoder->batch.huffman_dirty) != 0 && decoder_flush_tables(decoder) != 0 ) {
        return -1;
    }

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_stream, coder->param.perf_stats, coder->stream, return -1);

    // Read JPEG image data
    if (0 != (rc = gpujpeg_reader_read_image(decoder, image, image_size))) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return rc;
    }

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_stream, coder->param.perf_stats, coder->stream, return -1);

    // check if params is ok for GPU decoder
    for ( int i = 0; i < decoder->coder.param.comp_count; ++i ) {
        // packed_block_info_ptr holds only component type
        if ( decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] != decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC] ) {
            fprintf(stderr, "[GPUJPEG] [Warning] Using different table DC/AC indices (%d and %d) for component %d (ID %d)! Using Huffman CPU decoder. Please report to GPUJPEG developers.\n", decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC], decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC], i, decoder->comp_id[i]);
            use_cpu_huffman_decoder = true;
        }
        // only DC/AC tables 0 and 1 are processed gpujpeg_huffman_decoder_table_kernel()
        if ( decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] > 1 || decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC] > 1 ) {
            fprintf(stderr, "[GPUJPEG] [Warning] Using Huffman tables (%d, %d) implies extended process! Using Huffman CPU decoder. Please report to GPUJPEG developers.\n", decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC], decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC]);
            use_cpu_huffman_decoder = true;
        }
    }

    // a single segment without restart markers may be decoded on GPU by self-synchronization
    const bool resync = !use_cpu_huffman_decoder && gpujpeg_huffman_gpu_decoder_resync_possible(decoder);
    if ( resync ) {
        VERBOSE_MSG(decoder->coder.param.verbose, "Huffman has no restart markers. Using GPU decoder with "
                                                  "resynchronization.\n");
    }
    else if ( coder->segment_count < 32 ) { // the CPU Huffman implementation will be likely faster
        VERBOSE_MSG(decoder->coder.param.verbose, "Huffman has only %d segments. Using CPU decoder, which is slower!\n",
                    coder->segment_count);
        use_cpu_huffman_decoder = true;
    }
    decoder->huffman_route = use_cpu_huffman_decoder ? GPUJPEG_HUFFMAN_ROUTE_CPU
                             : resync                ? GPUJPEG_HUFFMAN_ROUTE_RESYNC
                                                     : GPUJPEG_HUFFMAN_ROUTE_GPU;

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if ( use_cpu_huffman_decoder ) {
        if ( decode_huffman_cpu(decoder) != 0 ) {
            return -1;
        }

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, coder->stream, return -1);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, coder->stream, return -1);

        // Reset huffman output
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), coder->stream);

        // Copy scan data to device memory
        gpujpeg_cuda_memcpy_async_partially_pinned(coder->d_data_compressed, coder->data_compressed, decoder->data_compressed_size,
                                                   cudaMemcpyHostToDevice, coder->stream,
                                                   coder->data_compressed_pinned_sz);
        gpujpeg_cuda_check_error("Decoder copy compressed data to memory", return -1);

        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment),
                        cudaMemcpyHostToDevice, coder->stream);
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, coder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, coder->stream, return -1);
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, return -1);
        // Perform huffman decoding
        rc = resync ? gpujpeg_huffman_gpu_decoder_resync_decode(decoder)
                    : gpujpeg_huffman_gpu_decoder_decode(decoder, false);
        if ( rc < 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, return -1);
        // not certified (eg. not synchronized), the CPU result replaces all coefficients (timers then show the CPU)
        if ( rc > 0 ) {
            VERBOSE_MSG(decoder->coder.param.verbose,
                        "Huffman GPU resynchronization failed (status 0x%x). Decoding again on CPU.\n",
                        decoder->huffman_resync_status);
            decoder->huffman_route = GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY;
            if ( decode_huffman_cpu(decoder) != 0 ) {
                return -1;
            }
        }
    }

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_dct_quantization, coder->param.perf_stats, coder->stream, return -1);

    // Perform IDCT and dequantization (own CUDA implementation)
    if ( 0 != gpujpeg_idct_gpu(decoder, 1) ) {
        return -1;
    }
    // gpujpeg_idct_cpu(decoder);

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_dct_quantization, coder->param.perf_stats, coder->stream, return -1);

    // Reallocate buffers if not enough size
    if ( decoder_ensure_raw(coder, coder->data_raw_size) != 0 ) {
        return -1;
    }

    // Select CUDA output buffer
    if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image should be directly decoded into custom CUDA buffer
        coder->d_data_raw = output->data;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE && output->texture->texture_callback_attach_opengl == NULL) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_map, coder->param.perf_stats, coder->stream, return -1);

        // Use OpenGL texture as decoding destination
        size_t data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);
        coder->d_data_raw = d_data;

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_map, coder->param.perf_stats, coder->stream, return -1);
    }
    else {
        // Use internal CUDA buffer as decoding destination
        coder->d_data_raw = coder->d_data_raw_allocated;
    }

    // Postprocessing
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_preprocessor, coder->param.perf_stats, coder->stream, return -1);
    rc = gpujpeg_postprocessor_decode(&decoder->coder, coder->stream, 1, 0);
    if (rc != GPUJPEG_NOERR) {
        return rc;
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_preprocessor, coder->param.perf_stats, coder->stream, return -1);

    // Wait for async operations before copying from the device
    cudaStreamSynchronize(coder->stream);

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_in_gpu, coder->param.perf_stats, coder->stream, return -1);

    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);
    output->param_image = decoder->coder.param_image;
    if (output->param_image.color_space == GPUJPEG_NONE) {
        output->param_image.color_space = decoder->coder.param.color_space_internal;
    }

    // Set decompressed image
    if (output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);

        // Copy decompressed image to host memory
        cudaMemcpyAsync(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, coder->stream);
        cudaStreamSynchronize(coder->stream);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);

        // Set output to internal buffer
        output->data = coder->data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);

        assert(output->data != NULL);

        // subtract width padding if user buffer hasn't allocated padding for the last line (unlikely)
        size_t data_raw_size = coder->data_raw_size - coder->param_image.width_padding;
        // Copy decompressed image to host memory
        cudaMemcpyAsync(output->data, coder->d_data_raw, data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                        coder->stream);
        cudaStreamSynchronize(coder->stream);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE) {
        // If OpenGL texture wasn't mapped and used directly for decoding into it
        if (output->texture->texture_callback_attach_opengl != NULL) {
            GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_map, coder->param.perf_stats, coder->stream, return -1);

            // Map OpenGL texture
            size_t data_size = 0;
            uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
            assert(data_size == coder->data_raw_size);

            GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_map, coder->param.perf_stats, coder->stream, return -1);

            GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);

            // Copy decompressed image to texture pixel buffer object device data
            cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

            GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, coder->stream, return -1);
        }

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_unmap, coder->param.perf_stats, coder->stream, return -1);

        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_unmap, coder->param.perf_stats, coder->stream, return -1);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER) {
        // Copy decompressed image to texture pixel buffer object device data
        output->data = coder->d_data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image was already directly decoded into custom CUDA buffer
        output->data = coder->d_data_raw;
    }
    else {
        // Unknown output type
        assert(0);
    }

    coder_process_stats(coder);
    if ( coder->param.verbose >= GPUJPEG_LL_STATUS ) {
        char buf[21];
        char* comp_size_delim = format_number_with_delim(output->data_size, buf, sizeof buf);
        PRINTF("Decompressed Size:%13s bytes %dx%d %s %s\n",comp_size_delim, output->param_image.width,
               output->param_image.height, gpujpeg_pixel_format_get_name(output->param_image.pixel_format),
               gpujpeg_color_space_get_name(output->param_image.color_space));
    }
    output->metadata = &decoder->metadata;

    return 0;
}

/**
 * Validates the arguments of gpujpeg_decoder_decode_batch() that do not depend on the images
 *
 * @retval @ref Errors
 */
static int
decoder_batch_check_args(struct gpujpeg_decoder* decoder, int count, uint8_t* const* images, const size_t* image_sizes,
                         const struct gpujpeg_decoder_output* output, int* error_index)
{
    if ( decoder == NULL || images == NULL || image_sizes == NULL || output == NULL || count < 1 ||
         count > GPUJPEG_BATCH_MAX_COUNT ) {
        ERROR_MSG("Invalid batch decoding arguments!\n");
        return GPUJPEG_ERROR;
    }
    switch ( output->type ) {
    case GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER:
    case GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER:
        break;
    case GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER:
    case GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER:
        if ( output->data == NULL ) {
            ERROR_MSG("Custom buffer for batch decoding not set!\n");
            return GPUJPEG_ERROR;
        }
        break;
    case GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE:
        ERROR_MSG("OpenGL texture output is not supported for batches!\n");
        return GPUJPEG_ERR_BATCH_UNSUPPORTED;
    default:
        ERROR_MSG("Invalid decoder output type %d!\n", (int)output->type);
        return GPUJPEG_ERROR;
    }
    if ( decoder->coder.preprocessor.flipped || decoder->coder.preprocessor.channel_remap != 0 ) {
        ERROR_MSG("Flipping and channel remapping are not supported for batches!\n");
        return GPUJPEG_ERR_BATCH_UNSUPPORTED;
    }
    // requested output format (the setters do not validate it), checked before the helpers assuming it valid
    const enum gpujpeg_pixel_format pixel_format = decoder->req_pixel_format;
    const enum gpujpeg_color_space color_space = decoder->req_color_space;
    if ( (gpujpeg_pixel_format_get_comp_count(pixel_format) == 0 && pixel_format != GPUJPEG_PIXFMT_AUTODETECT &&
          pixel_format != GPUJPEG_PIXFMT_NO_ALPHA && pixel_format != GPUJPEG_PIXFMT_STD &&
          pixel_format != GPUJPEG_PIXFMT_NATIVE) ||
         (color_space != GPUJPEG_NONE && color_space != GPUJPEG_CS_DEFAULT &&
          (color_space < GPUJPEG_RGB || color_space > GPUJPEG_YUV)) ||
         decoder->req_alignment < 0 ) {
        ERROR_MSG("Invalid decoder output format or alignment for batch decoding!\n");
        return GPUJPEG_ERROR;
    }
    if ( decoder->req_alignment > GPUJPEG_DECODER_BATCH_MAX_ALIGNMENT ) {
        ERROR_MSG("Output alignment %d is not supported for batches (at most %d)!\n", decoder->req_alignment,
                  GPUJPEG_DECODER_BATCH_MAX_ALIGNMENT);
        return GPUJPEG_ERR_BATCH_UNSUPPORTED;
    }
    for ( int i = 0; i < count; i++ ) {
        // SOI + EOI at least, image end representable as a pointer difference
        if ( images[i] == NULL || image_sizes[i] < 4 || image_sizes[i] > (size_t)PTRDIFF_MAX ) {
            ERROR_MSG("Invalid batch image %d (%zu bytes)!\n", i, image_sizes[i]);
            *error_index = i;
            return GPUJPEG_ERROR;
        }
    }
    return GPUJPEG_NOERR;
}

/* Documented at declaration */
int
gpujpeg_decoder_decode_batch(struct gpujpeg_decoder* decoder, int count, uint8_t* const* images,
                             const size_t* image_sizes, size_t output_stride, struct gpujpeg_decoder_output* output,
                             int* error_index)
{
    int ignored_error_index = -1;
    if ( error_index == NULL ) {
        error_index = &ignored_error_index;
    }
    *error_index = -1;
    int rc = decoder_batch_check_args(decoder, count, images, image_sizes, output, error_index);
    if ( rc != GPUJPEG_NOERR ) {
        return rc;
    }
    struct gpujpeg_coder* coder = &decoder->coder;
    struct gpujpeg_decoder_batch* batch = &decoder->batch;
    batch->reason = GPUJPEG_BATCH_REASON_NONE;
    if ( count > batch->metadata_capacity ) {
        struct gpujpeg_image_metadata* metadata = realloc(batch->metadata, count * sizeof *metadata);
        if ( metadata == NULL ) {
            ERROR_MSG("Cannot allocate batch metadata!\n");
            return GPUJPEG_ERROR;
        }
        batch->metadata = metadata;
        batch->metadata_capacity = count;
    }

    coder->start_time = coder->param.perf_stats ? gpujpeg_get_time() : 0;

    // tables of an earlier failed batch, reading may change them
    if ( (batch->quant_dirty | batch->huffman_dirty) != 0 && decoder_flush_tables(decoder) != 0 ) {
        return GPUJPEG_ERROR;
    }

    size_t image_size = 0; // raw bytes of an image
    size_t stride = 0;     // of the output
    size_t raw_stride = 0; // of the buffer the postprocessor writes
    size_t data_size = 0;  // coefficients of all images, also the ones read by the IDCT beyond the last image
    bool cpu_huffman = false;
    batch->active = true;
    batch->count = count;
    batch->init_failed = false;
    decoder->segment_count = 0;
    decoder->data_compressed_size = 0;

    // Read all images (layout of the batch built for the first one), no output written yet
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_stream, coder->param.perf_stats, coder->stream, rc = GPUJPEG_ERROR;
                               goto fail);
    for ( int i = 0; i < count; i++ ) {
        batch->image = i;
        rc = gpujpeg_reader_read_image(decoder, images[i], image_sizes[i]);
        if ( rc != GPUJPEG_NOERR ) {
            ERROR_MSG("Batch image %d could not be read!\n", i);
            *error_index = batch->init_failed ? -1 : i;
            goto fail;
        }
        if ( i == 0 ) {
            image_size = coder->data_raw_size;
            stride = output_stride != 0 ? output_stride : image_size;
            if ( image_size == 0 || stride < image_size || (size_t)(count - 1) > (SIZE_MAX - image_size) / stride ) {
                ERROR_MSG("Batch output stride %zu invalid for %d images of %zu bytes!\n", output_stride, count,
                          image_size);
                rc = GPUJPEG_ERROR;
                goto fail;
            }
            if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P1020 && coder->param_image.width % 2 != 0 ) {
                ERROR_MSG("Odd-width %s output is not supported for batches!\n",
                          gpujpeg_pixel_format_get_name(coder->param_image.pixel_format));
                batch->reason = GPUJPEG_BATCH_REASON_DOMAIN;
                rc = GPUJPEG_ERR_BATCH_UNSUPPORTED;
                goto fail;
            }
        }
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_stream, coder->param.perf_stats, coder->stream, rc = GPUJPEG_ERROR;
                              goto fail);

    // Huffman decoding route of gpujpeg_decoder_decode() for one image (equal for all), but without resynchronization
    cpu_huffman = coder->segment_count < 32;
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
        const int dc = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_DC];
        cpu_huffman = cpu_huffman || dc != decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_AC] || dc > 1;
    }
    decoder->huffman_route = cpu_huffman ? GPUJPEG_HUFFMAN_ROUTE_CPU : GPUJPEG_HUFFMAN_ROUTE_GPU;
    data_size = count * coder->image_data_stride; // the batch allocation holds that (gpujpeg_coder_init_image())
    rc = GPUJPEG_ERROR;                           // for the failures below
    if ( decoder_upload_tables(decoder) != 0 ) {
        goto fail;
    }
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, coder->stream, goto fail);
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, goto fail);
    if ( cpu_huffman ) {
        if ( coder->data_quantized == NULL && gpujpeg_coder_allocate_cpu_huffman_buf(coder) != 0 ) {
            goto fail;
        }
        // all uploaded coefficients defined (the decoder writes only the blocks, the IDCT reads beyond them)
        memset(coder->data_quantized, 0, data_size * sizeof(int16_t));
        for ( int i = 0; i < count; i++ ) {
            if ( gpujpeg_huffman_cpu_decoder_decode(decoder, i * coder->segment_count, coder->segment_count,
                                                    i * coder->image_data_stride) != 0 ) {
                ERROR_MSG("Huffman decoder failed for batch image %d!\n", i);
                *error_index = i;
                goto fail;
            }
        }
        if ( cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, data_size * sizeof(int16_t),
                             cudaMemcpyHostToDevice, coder->stream) != cudaSuccess ) {
            goto fail;
        }
    }
    else {
        // segments are final now; the Huffman decoder reads ahead, so initialized bytes follow the data
        memset(coder->data_compressed + decoder->data_compressed_size, 0, GPUJPEG_DECODER_COMPRESSED_TAIL);
        if ( cudaMemsetAsync(coder->d_data_quantized, 0, data_size * sizeof(int16_t), coder->stream) != cudaSuccess ||
             gpujpeg_cuda_memcpy_async_partially_pinned(coder->d_data_compressed, coder->data_compressed,
                                                        decoder->data_compressed_size + GPUJPEG_DECODER_COMPRESSED_TAIL,
                                                        cudaMemcpyHostToDevice, coder->stream,
                                                        coder->data_compressed_pinned_sz) != cudaSuccess ||
             cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment),
                             cudaMemcpyHostToDevice, coder->stream) != cudaSuccess ||
             gpujpeg_huffman_gpu_decoder_decode(decoder, true) != 0 ) {
            goto fail;
        }
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, coder->stream, goto fail);

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_dct_quantization, coder->param.perf_stats, coder->stream, goto fail);
    if ( gpujpeg_idct_gpu(decoder, count) != 0 ) {
        goto fail;
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_dct_quantization, coder->param.perf_stats, coder->stream, goto fail);

    // Pixels directly to custom CUDA output, otherwise to the internal buffer (packed for a custom host output)
    raw_stride = output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ? image_size : stride;
    if ( output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER ) {
        coder->d_data_raw = output->data;
    }
    else {
        if ( decoder_ensure_raw(coder, (count - 1) * raw_stride + image_size) != 0 ) {
            goto fail;
        }
        coder->d_data_raw = coder->d_data_raw_allocated;
    }
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_preprocessor, coder->param.perf_stats, coder->stream, goto fail);
    rc = gpujpeg_postprocessor_decode(coder, coder->stream, count, raw_stride);
    if ( rc != GPUJPEG_NOERR ) {
        goto fail;
    }
    rc = GPUJPEG_ERROR;
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_preprocessor, coder->param.perf_stats, coder->stream, goto fail);

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, coder->stream, goto fail);
    if ( output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER ) {
        if ( cudaMemcpyAsync(coder->data_raw, coder->d_data_raw, (count - 1) * raw_stride + image_size,
                             cudaMemcpyDeviceToHost, coder->stream) != cudaSuccess ) {
            goto fail;
        }
    }
    else if ( output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) {
        // without the padding of the last line, as gpujpeg_decoder_decode()
        if ( cudaMemcpy2DAsync(output->data, stride, coder->d_data_raw, image_size,
                               image_size - coder->param_image.width_padding, count, cudaMemcpyDeviceToHost,
                               coder->stream) != cudaSuccess ) {
            goto fail;
        }
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, coder->stream, goto fail);
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_in_gpu, coder->param.perf_stats, coder->stream, goto fail);

    // the only synchronization of the GPU route for CUDA outputs, all images are ready afterwards
    if ( cudaStreamSynchronize(coder->stream) != cudaSuccess ) {
        ERROR_MSG("Batch decoding failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
        goto fail;
    }
    batch->quant_dirty = 0; // uploaded and completed
    batch->huffman_dirty = 0;
    batch->active = false;

    if ( output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER ) {
        output->data = coder->data_raw;
    }
    else if ( output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER ) {
        output->data = coder->d_data_raw;
    }
    output->data_size = image_size;
    output->param_image = coder->param_image;
    if ( output->param_image.color_space == GPUJPEG_NONE ) {
        output->param_image.color_space = coder->param.color_space_internal;
    }
    output->metadata = batch->metadata;
    coder_process_stats(coder);
    return GPUJPEG_NOERR;

fail:
    // device tables consistent with the host ones, queued work completed
    if ( decoder_flush_tables(decoder) != 0 ) {
        *error_index = -1;
        rc = GPUJPEG_ERROR;
    }
    batch->active = false;
    return rc;
}

int
gpujpeg_decoder_get_stats(struct gpujpeg_decoder *decoder, struct gpujpeg_duration_stats *stats)
{
    return gpujpeg_coder_get_stats(&decoder->coder, stats);
}

void
gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder, enum gpujpeg_color_space color_space,
                                  enum gpujpeg_pixel_format pixel_format)
{
    decoder->req_color_space = color_space;
    decoder->req_pixel_format = pixel_format;
}

GPUJPEG_API int
gpujpeg_decoder_set_option(struct gpujpeg_decoder* decoder, const char *opt, const char* val)
{
    if ( decoder == NULL || opt == NULL || val == NULL ) {
        return GPUJPEG_ERROR;
    }
    if ( strcmp(opt, GPUJPEG_DEC_OPT_TGA_RLE_BOOL) == 0 ) {
        if ( strcmp(val, GPUJPEG_VAL_TRUE) != 0 && strcmp(val, GPUJPEG_VAL_FALSE) != 0 ) {
            ERROR_MSG("Unexpeceted value %s for " GPUJPEG_DEC_OPT_TGA_RLE_BOOL "\n", val);
            return GPUJPEG_ERROR;
        }
        image_delegate_stbi_tga_set_rle(strcmp(val, GPUJPEG_VAL_TRUE) == 0);
        return GPUJPEG_NOERR;
    }
    if ( strcmp(opt, GPUJPEG_DEC_OPT_FLIPPED_BOOL) == 0 ) {
        return gpujpeg_parse_bool_opt(&decoder->coder.preprocessor.flipped, val, GPUJPEG_DEC_OPT_FLIPPED_BOOL);
    }
    if ( strcmp(opt, GPUJPEG_DEC_OPT_CHANNEL_REMAP) == 0 ) {
        return gpujpeg_opt_set_channel_remap(&decoder->coder, val, GPUJPEG_DEC_OPT_CHANNEL_REMAP);
    }
    if ( strcmp(opt, GPUJPEG_DEC_OPT_CHROMA_FILTER_BOOL) == 0 ) {
        return gpujpeg_parse_bool_opt(&decoder->coder.preprocessor.chroma_filter, val,
                                      GPUJPEG_DEC_OPT_CHROMA_FILTER_BOOL);
    }
    if ( strcmp(opt, GPUJPEG_DEC_OPT_ALIGNMENT_BYTES_INT) == 0 ) {
        decoder->req_alignment = atoi(val);
        return GPUJPEG_NOERR;
    }
    ERROR_MSG("Invalid decoder option: %s!\n", opt);
    return GPUJPEG_ERROR;
}

GPUJPEG_API void
gpujpeg_decoder_print_options()
{
    printf("\t" GPUJPEG_DEC_OPT_TGA_RLE_BOOL "=[" GPUJPEG_VAL_FALSE "|" GPUJPEG_VAL_TRUE
           "] - set decoder option (not) to output RLE TGA\n");
    printf("\t" GPUJPEG_DEC_OPT_FLIPPED_BOOL "=[" GPUJPEG_VAL_FALSE "|" GPUJPEG_VAL_TRUE
           "] - vertically flip the output image\n");
    printf("\t" GPUJPEG_DEC_OPT_CHANNEL_REMAP "=XYZ[W] - ouput channel mapping, eg. '210F' for GBRX,\n"
        "\t\t'210' for GBR; special placeholders 'F' and 'Z' to set a channel to all-ones or all-zeros\n");
    printf("\t" GPUJPEG_DEC_OPT_CHROMA_FILTER_BOOL "=[" GPUJPEG_VAL_FALSE "|" GPUJPEG_VAL_TRUE
           "] - interpolate subsampled chroma (default) instead of replicating it\n");
    printf("\t" GPUJPEG_DEC_OPT_ALIGNMENT_BYTES_INT "=<num> - output alignment in bytes,\n"
        "\t\tshould not be used for gpujpegtool (cannot process)\n");
}

/* Documented at declaration */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder)
{
    assert(decoder != NULL);

    // queued work may still use the coder memory (eg. if decoding failed before its synchronization)
    cudaStreamSynchronize(decoder->coder.stream);
    coder_process_stats_overall(&decoder->coder);

    if (0 != gpujpeg_coder_deinit(&decoder->coder)) {
        return -1;
    }

    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        gpujpeg_coder_device_free(&decoder->coder, decoder->table_quantization[comp_type].d_table);
    }

    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            gpujpeg_coder_device_free(&decoder->coder, decoder->d_table_huffman[comp_type][huff_type]);
        }
    }

    if (decoder->huffman_gpu_decoder != NULL) {
        gpujpeg_huffman_gpu_decoder_destroy(decoder->huffman_gpu_decoder, &decoder->coder);
    }

    free(decoder->batch.metadata);
    free(decoder);

    return 0;
}

int
gpujpeg_decoder_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_parameters *param_image, struct gpujpeg_parameters *param, int *segment_count) {
    struct gpujpeg_image_info info = { 0 };
    const unsigned flags = segment_count == NULL ? 0 : GPUJPEG_COUNT_SEG_COUNT_REQ;
    const int verbose = param->verbose;
    const int rc = gpujpeg_reader_get_image_info(image, image_size, &info, verbose, flags);
    if ( rc != GPUJPEG_NOERR ) {
        return rc;
    };
    memcpy(param_image, &info.param_image, sizeof *param_image);
    memcpy(param, &info.param, sizeof *param);
    param->verbose = verbose;
    if ( segment_count != NULL ) {
        *segment_count = info.segment_count;
    }
    return GPUJPEG_NOERR;
}

GPUJPEG_API int
gpujpeg_decoder_get_image_info2(uint8_t* image, size_t image_size, struct gpujpeg_image_info* info, int verbose,
                                unsigned flags)
{
    return gpujpeg_reader_get_image_info(image, image_size, info, verbose, flags);
}

/* vi: set expandtab sw=4 : */
