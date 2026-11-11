/**
 * @file
 * Copyright (c) 2011, CESNET z.s.p.o
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

#ifndef GPUJPEG_HUFFMAN_GPU_DECODER_H
#define GPUJPEG_HUFFMAN_GPU_DECODER_H

#include "gpujpeg_decoder_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_huffman_gpu_decoder;

// defaults of gpujpeg_decoder::resync_param (decoding without restart markers)
#ifndef GPUJPEG_HUFFMAN_GPU_RESYNC_CHUNK_BYTES
    #define GPUJPEG_HUFFMAN_GPU_RESYNC_CHUNK_BYTES 512
#endif
#ifndef GPUJPEG_HUFFMAN_GPU_RESYNC_SEQ_CHUNKS
    #define GPUJPEG_HUFFMAN_GPU_RESYNC_SEQ_CHUNKS 128
#endif
#ifndef GPUJPEG_HUFFMAN_GPU_RESYNC_REPAIR_ROUNDS
    #define GPUJPEG_HUFFMAN_GPU_RESYNC_REPAIR_ROUNDS 128
#endif
#ifndef GPUJPEG_HUFFMAN_GPU_RESYNC_MIN_BYTES
    #define GPUJPEG_HUFFMAN_GPU_RESYNC_MIN_BYTES 262144
#endif

/// gpujpeg_decoder::huffman_resync_status bits (reasons to decode the image again on CPU)
enum gpujpeg_huffman_gpu_resync_status_bits {
    GPUJPEG_HUFFMAN_RESYNC_REPAIR_LIMIT = 1 << 0, ///< chunk repair rounds exhausted
    GPUJPEG_HUFFMAN_RESYNC_ORDER = 1 << 1,        ///< block prefixes not anchored at 0 or not monotone
    GPUJPEG_HUFFMAN_RESYNC_START = 1 << 2,        ///< invalid chunk start state, position or MCU phase
    GPUJPEG_HUFFMAN_RESYNC_DECODE = 1 << 3,       ///< invalid code, run or DC predictor, or data overrun
    GPUJPEG_HUFFMAN_RESYNC_OWNERSHIP = 1 << 4,    ///< replayed block starting after its chunk
    GPUJPEG_HUFFMAN_RESYNC_ENDPOINT = 1 << 5,     ///< replay ended elsewhere than the chunk record
    GPUJPEG_HUFFMAN_RESYNC_PADDING = 1 << 6,      ///< more than up to 7 one bits after the last block
    GPUJPEG_HUFFMAN_RESYNC_INCOMPLETE = 1 << 7,   ///< not exactly one chunk completed the scan
};

/**
 * Init huffman decoder
 * 
 * @return 0 if succeeds, otherwise nonzero
 */
struct gpujpeg_huffman_gpu_decoder *
gpujpeg_huffman_gpu_decoder_init(const struct gpujpeg_coder* coder);

void
gpujpeg_huffman_gpu_decoder_destroy(struct gpujpeg_huffman_gpu_decoder *, const struct gpujpeg_coder* coder);

/**
 * Perform huffman decoding
 * 
 * @param encoder  Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_gpu_decoder_decode(struct gpujpeg_decoder* decoder);

/**
 * Checks if the image (parsed, before the copy to device) can be Huffman decoded on GPU without restart markers:
 * reader eligibility (gpujpeg_decoder::huffman_resync_eligible), grayscale 1x1 or interleaved 4:4:4, 4:2:2 or 4:2:0,
 * table selectors, baseline Huffman tables, size and resource limits (gpujpeg_decoder::resync_param)
 */
bool
gpujpeg_huffman_gpu_decoder_resync_possible(const struct gpujpeg_decoder* decoder);

/**
 * Huffman decoding of a scan without restart markers by self-synchronization (speculative chunk decoding, repair,
 * prefix scan and certified replay into coder->d_data_quantized), using coder->d_temp_huffman as scratch. Waits for
 * the result; on failure, gpujpeg_decoder::huffman_resync_status has the reason and the coefficients are incomplete.
 *
 * @pre gpujpeg_huffman_gpu_decoder_resync_possible(), compressed data and zeroed coefficients queued in coder stream
 * @retval 0 success
 * @retval 1 not decoded, decode on CPU instead
 * @retval -1 CUDA error or unmet precondition
 */
int
gpujpeg_huffman_gpu_decoder_resync_decode(struct gpujpeg_decoder* decoder);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_HUFFMAN_GPU_DECODER_H
