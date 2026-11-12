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

#ifndef GPUJPEG_DECODER_INTERNAL_H
#define GPUJPEG_DECODER_INTERNAL_H

#include <stdbool.h>

#include "../libgpujpeg/gpujpeg_common.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_table.h"

/// Huffman decoder used for the last image (unit-test diagnostics)
enum gpujpeg_huffman_route {
    GPUJPEG_HUFFMAN_ROUTE_CPU,          ///< CPU (few segments or unsupported tables)
    GPUJPEG_HUFFMAN_ROUTE_GPU,          ///< GPU, one thread per segment
    GPUJPEG_HUFFMAN_ROUTE_RESYNC,       ///< GPU without restart markers (self-synchronization)
    GPUJPEG_HUFFMAN_ROUTE_RESYNC_RETRY, ///< GPU without restart markers failed, decoded again on CPU
};

/**
 * Private limits of the GPU Huffman decoder without restart markers (gpujpeg_huffman_gpu_decoder_resync_decode()),
 * set from compile-time defaults on decoder creation; unit tests use small values to cover chunk and repair limits.
 * Only exactly the shipped defaults (512, 128, 128, 262144) also decode 256x256 interleaved 4:4:4 and 4:2:2 scans of
 * at least 32 KiB and less than 64 KiB, with 32-byte chunks in sequences of 32 chunks
 * (gpujpeg_huffman_gpu_resync_scan_init()).
 */
struct gpujpeg_huffman_gpu_resync_param
{
    int chunk_bytes;   ///< entropy-coded (stuffed) bytes per chunk
    int seq_chunks;    ///< chunks repaired together by one thread block (at most 1024)
    int repair_rounds; ///< chunks each sequence repair may walk (exhaustion retries on CPU)
    size_t min_bytes;  ///< smaller scans use CPU unless the documented default exception applies
};

/**
 * JPEG decoder structure
 */
struct gpujpeg_decoder
{
    /// JPEG coder structure
    struct gpujpeg_coder coder;
    
    struct gpujpeg_huffman_gpu_decoder *huffman_gpu_decoder;

    uint8_t comp_id[GPUJPEG_MAX_COMPONENT_COUNT]; /// component IDs defined by SOF
    
    /// Quantization tables
    struct gpujpeg_table_quantization table_quantization[GPUJPEG_MAX_COMPONENT_COUNT];
    int comp_table_quantization_map[GPUJPEG_MAX_COMPONENT_COUNT]; ///< mapping component -> index to table_quantization
    
    /// Huffman coder tables
    struct gpujpeg_table_huffman_decoder table_huffman[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    // Huffman coder tables in device memory
    struct gpujpeg_table_huffman_decoder* d_table_huffman[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    int comp_table_huffman_map[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT]; ///< mapping component -> indices (AC/DC) to table_huffman
    
    /// Current segment count for decoded image
    int segment_count;
    
    /// Current data compressed size for decoded image
    size_t data_compressed_size;

    /// set by the reader for each image: the only scan is a complete sequential SOF0 scan of all components in
    /// SOF order without restart markers, using Huffman tables defined by this image
    bool huffman_resync_eligible;
    struct gpujpeg_huffman_gpu_resync_param resync_param;
    enum gpujpeg_huffman_route huffman_route;
    unsigned huffman_resync_status; ///< status bits of the last GPU decoding attempt without restart markers

    enum gpujpeg_pixel_format req_pixel_format;
    enum gpujpeg_color_space req_color_space;
    bool ff_cs_itu601_is_709; ///< if FFmpeg specific COM marker "CS=ITU601" present, interpret the data as
                              ///< limited-range BT.709 not BT.601
    int req_alignment;        ///< required output alignment in bytes

    /// metadata associated with last decoded image
    struct gpujpeg_image_metadata metadata;
};

#endif // GPUJPEG_DECODER_INTERNAL_H

