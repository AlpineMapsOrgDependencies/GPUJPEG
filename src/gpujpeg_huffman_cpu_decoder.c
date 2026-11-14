/**
 * @file
 * Copyright (c) 2011-2020, CESNET z.s.p.o
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

#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_util.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

/*
 * Sequential Huffman fast path adapted from libjpeg-turbo 3.2.1 jdhuff.c/jdhuff.h.
 * This code was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009-2011, 2016, 2018-2019, 2022, D. R. Commander.
 * Copyright (C) 2010-2011, 2015-2016, 2021, D. R. Commander.
 * Copyright (C) 2018, Matthias Räncker.
 * For conditions of distribution and use, see LICENSES/README.ijg.
 *
 * GPUJPEG adaptation (2026): block-local transactions, existing table and MCU
 * mapping, bounded segment input, checked tails and malformed-code errors.
 * No IJG source manager, dummy-bit recovery, or reconstruction is imported.
 */

struct cpu_huffman_table
{
    const struct gpujpeg_table_huffman_decoder* source;
    unsigned lookup[256]; // length in high byte, symbol in low byte; 9 means long code
};

struct gpujpeg_huffman_cpu_decoder
{
    struct gpujpeg_component* component;
    struct cpu_huffman_table table_dc[GPUJPEG_MAX_COMPONENT_COUNT];
    struct cpu_huffman_table table_ac[GPUJPEG_MAX_COMPONENT_COUNT];
    int get_bits;
    uint64_t get_buff; // shared by fast and checked paths; never narrowed on fallback
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    int comp_count;
    int scan_index;
    const uint8_t* data;
    size_t data_size;
};

static void
cpu_huffman_table_init(struct cpu_huffman_table* table, const struct gpujpeg_table_huffman_decoder* source)
{
    table->source = source;
    for ( int i = 0; i < 256; ++i ) {
        const int n = source->look_nbits[i];
        // Unused look_sym entries need not have been initialized by table_compute().
        table->lookup[i] = n ? ((unsigned)n << 8) | source->look_sym[i] : 9U << 8;
    }
}

// Fill only from this segment. A marker (including repeated FF fill bytes)
// terminates input, but already buffered bits may still complete the block.
static void
cpu_fill_bits(struct gpujpeg_huffman_cpu_decoder* coder)
{
    while ( coder->get_bits <= 56 && coder->data_size != 0 ) {
        unsigned c = *coder->data++;
        --coder->data_size;
        if ( c == 0xFF ) {
            do {
                if ( coder->data_size == 0 )
                    return;
                c = *coder->data++;
                --coder->data_size;
            } while ( c == 0xFF );
            if ( c != 0 ) {
                coder->data_size = 0;
                return;
            }
            c = 0xFF;
        }
        coder->get_buff = (coder->get_buff << 8) | c;
        coder->get_bits += 8;
    }
}

static inline int
cpu_get_bits(struct gpujpeg_huffman_cpu_decoder* coder, int n)
{
    if ( n == 0 )
        return 0;
    if ( coder->get_bits < n )
        cpu_fill_bits(coder);
    if ( coder->get_bits < n )
        return -1;
    coder->get_bits -= n;
    return (int)(coder->get_buff >> coder->get_bits) & ((1 << n) - 1);
}

static inline int
cpu_symbol(struct gpujpeg_huffman_cpu_decoder* coder, const struct cpu_huffman_table* table)
{
    int n = 1;
    if ( coder->get_bits < 8 )
        cpu_fill_bits(coder);
    if ( coder->get_bits >= 8 ) {
        unsigned entry = table->lookup[(coder->get_buff >> (coder->get_bits - 8)) & 255];
        n = (int)(entry >> 8);
        if ( n <= 8 ) {
            coder->get_bits -= n;
            return entry & 255;
        }
    }
    const struct gpujpeg_table_huffman_decoder* t = table->source;
    int code = cpu_get_bits(coder, n);
    if ( code < 0 )
        return -1;
    while ( code > t->maxcode[n] ) {
        if ( ++n > 16 )
            return -1;
        int bit = cpu_get_bits(coder, 1);
        if ( bit < 0 )
            return -1;
        code = (code << 1) | bit;
    }
    int index = t->valptr[n] + code - t->mincode[n];
    if ( code < t->mincode[n] || index < 0 || index >= 256 )
        return -1;
    return t->huffval[index];
}

static inline int
cpu_extend(int value, int n)
{
    return n && value < (1 << (n - 1)) ? value + 1 - (1 << n) : value;
}

// Adaptations of GET_BYTE, FILL_BIT_BUFFER_FAST and HUFF_DECODE_FAST. Six
// unrolled bytes refill a register-local 64-bit reservoir to 48..64 bits.
// No read is allowed to cross a marker; the caller retries from the block's
// original state. FF/00 stuffing consumes at most two bytes per data byte.
#define CPU_GET_BYTE()                                                                                                 \
    do {                                                                                                               \
        unsigned c = *buffer++;                                                                                        \
        if ( c == 0xFF && *buffer++ != 0 )                                                                             \
            return 1;                                                                                                  \
        get_buffer = (get_buffer << 8) | c;                                                                            \
        bits_left += 8;                                                                                                \
    } while ( 0 )
#define CPU_FILL_FAST()                                                                                                \
    do {                                                                                                               \
        if ( bits_left <= 16 ) {                                                                                       \
            CPU_GET_BYTE();                                                                                            \
            CPU_GET_BYTE();                                                                                            \
            CPU_GET_BYTE();                                                                                            \
            CPU_GET_BYTE();                                                                                            \
            CPU_GET_BYTE();                                                                                            \
            CPU_GET_BYTE();                                                                                            \
        }                                                                                                              \
    } while ( 0 )
#define CPU_GET_BITS(n) ((int)(get_buffer >> (bits_left -= (n))) & ((1 << (n)) - 1))
#define CPU_SYMBOL_FAST(symbol, table)                                                                                 \
    do {                                                                                                               \
        CPU_FILL_FAST();                                                                                               \
        unsigned entry = (table)->lookup[(get_buffer >> (bits_left - 8)) & 255];                                       \
        int nb = (int)(entry >> 8);                                                                                    \
        bits_left -= nb;                                                                                               \
        symbol = entry & 255;                                                                                          \
        if ( nb > 8 ) {                                                                                                \
            const struct gpujpeg_table_huffman_decoder* t = (table)->source;                                           \
            int code = (int)(get_buffer >> bits_left) & ((1 << nb) - 1);                                               \
            while ( code > t->maxcode[nb] ) {                                                                          \
                if ( ++nb > 16 )                                                                                       \
                    return -1;                                                                                         \
                code = (code << 1) | CPU_GET_BITS(1);                                                                  \
            }                                                                                                          \
            int index = t->valptr[nb] + code - t->mincode[nb];                                                         \
            if ( code < t->mincode[nb] || index < 0 || index >= 256 )                                                  \
                return -1;                                                                                             \
            symbol = t->huffval[index];                                                                                \
        }                                                                                                              \
    } while ( 0 )

// Headroom proof (also for malformed input): at most 64 symbols per block,
// each <=16 code bits + <=15 value bits = 1984 consumed bits. A refill adds
// 48 bits only at <=16, so <=63 unused bits can remain after a consumed bit.
// At most floor((1984+63)/8)=255 bytes are fetched, or 510 with stuffing.
// The upstream conservative 512-byte/block guard therefore bounds every
// read, including the second byte after FF. Runs and sizes are checked before
// consuming values. Return 1 requests checked retry; -1 means invalid data.
static int
cpu_decode_block_fast(struct gpujpeg_huffman_cpu_decoder* coder, int16_t* data, int* dc,
                      const struct cpu_huffman_table* table_dc, const struct cpu_huffman_table* table_ac)
{
    const uint8_t* buffer = coder->data;
    uint64_t get_buffer = coder->get_buff;
    int bits_left = coder->get_bits;
    int s;
    CPU_SYMBOL_FAST(s, table_dc);
    if ( s > 15 )
        return -1;
    if ( s ) {
        CPU_FILL_FAST();
        int value = CPU_GET_BITS(s);
        s = cpu_extend(value, s);
    }
    const int next_dc = *dc + s;
    if ( next_dc < INT16_MIN || next_dc > INT16_MAX )
        return -1;
    data[0] = (int16_t)next_dc;
    for ( int k = 1; k < 64; ++k ) {
        CPU_SYMBOL_FAST(s, table_ac);
        int run = s >> 4;
        s &= 15;
        if ( s ) {
            k += run;
            if ( k >= 64 )
                return -1;
            CPU_FILL_FAST();
            int value = CPU_GET_BITS(s);
            data[gpujpeg_order_natural[k]] = (int16_t)cpu_extend(value, s);
        }
        else {
            if ( run == 0 )
                break;
            if ( run != 15 || k + 16 > 64 )
                return -1;
            k += 15;
        }
    }
    coder->data_size -= (size_t)(buffer - coder->data);
    coder->data = buffer;
    coder->get_buff = get_buffer;
    coder->get_bits = bits_left;
    *dc = next_dc;
    return 0;
}

#undef CPU_SYMBOL_FAST
#undef CPU_GET_BITS
#undef CPU_FILL_FAST
#undef CPU_GET_BYTE

static int
gpujpeg_huffman_cpu_decoder_decode_block(struct gpujpeg_huffman_cpu_decoder* coder, int16_t* data, int* dc,
                                         const struct cpu_huffman_table* table_dc,
                                         const struct cpu_huffman_table* table_ac)
{
    memset(data, 0, 64 * sizeof *data);
    if ( coder->data_size >= 512 ) {
        int result = cpu_decode_block_fast(coder, data, dc, table_dc, table_ac);
        if ( result <= 0 )
            return result;
        // Reader and DC were not committed. Clear speculative coefficients,
        // including those a shorter checked decode would otherwise leave stale.
        memset(data, 0, 64 * sizeof *data);
    }
    int s = cpu_symbol(coder, table_dc);
    if ( s < 0 || s > 15 )
        return -1;
    int value = cpu_get_bits(coder, s);
    if ( value < 0 )
        return -1;
    const int next_dc = *dc + cpu_extend(value, s);
    if ( next_dc < INT16_MIN || next_dc > INT16_MAX )
        return -1;
    data[0] = (int16_t)next_dc;
    for ( int k = 1; k < 64; ++k ) {
        s = cpu_symbol(coder, table_ac);
        if ( s < 0 )
            return -1;
        int run = s >> 4;
        s &= 15;
        if ( s ) {
            k += run;
            if ( k >= 64 )
                return -1;
            value = cpu_get_bits(coder, s);
            if ( value < 0 )
                return -1;
            data[gpujpeg_order_natural[k]] = (int16_t)cpu_extend(value, s);
        } else {
            if ( run == 0 )
                break;
            if ( run != 15 || k + 16 > 64 )
                return -1;
            k += 15;
        }
    }
    *dc = next_dc;
    return 0;
}

/**
 * Decode one MCU
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_cpu_decoder_decode_mcu(struct gpujpeg_huffman_cpu_decoder* coder, int segment_index, int mcu_index)
{
    // Non-interleaving mode
    if ( coder->comp_count == 1 ) {
        // Get component for current scan
        struct gpujpeg_component* component = &coder->component[coder->scan_index];
 
        // Get component data for MCU
        int16_t* block = &component->data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
        
        // Get coder parameters
        int* dc = &coder->dc[coder->scan_index];
        const struct cpu_huffman_table* table_dc = &coder->table_dc[component->dc_huff_idx];
        const struct cpu_huffman_table* table_ac = &coder->table_ac[component->ac_huff_idx];

        // Encode 8x8 block
        if ( gpujpeg_huffman_cpu_decoder_decode_block(coder, block, dc, table_dc, table_ac) != 0 )
            return -1;
    } 
    // Interleaving mode
    else {
        assert(coder->scan_index == 0);
        for ( int comp = 0; comp < coder->comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];

            // Prepare mcu indexes
            int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
            int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
            // Compute base data index
            int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
            
            // For all vertical 8x8 blocks
            for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                // Compute base row data index
                assert((component->mcu_count_x * component->mcu_size_x) == component->data_width);
                int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                // For all horizontal 8x8 blocks
                for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                    // Compute 8x8 block data index
                    int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                    
                    // Get component data for MCU
                    int16_t* block = &component->data_quantized[data_index];
                    
                    // Get coder parameters
                    int* dc = &coder->dc[comp];
                    const struct cpu_huffman_table* table_dc = &coder->table_dc[component->dc_huff_idx];
                    const struct cpu_huffman_table* table_ac = &coder->table_ac[component->ac_huff_idx];

                    // Encode 8x8 block
                    if ( gpujpeg_huffman_cpu_decoder_decode_block(coder, block, dc, table_dc, table_ac) != 0 )
                        return -1;
                }
            }
        }
    }
    return 0;
}

/* Documented at declaration */
int
gpujpeg_huffman_cpu_decoder_decode(struct gpujpeg_decoder* decoder, int segment_begin, int segment_count,
                                   size_t image_offset)
{
    // Components viewing the coefficients of the image (compressed data indices of the segments are absolute)
    struct gpujpeg_component component[GPUJPEG_MAX_COMPONENT_COUNT];
    memcpy(component, decoder->coder.component, decoder->coder.param.comp_count * sizeof component[0]);
    for ( int comp = 0; comp < decoder->coder.param.comp_count; ++comp ) {
        component[comp].data_quantized += image_offset;
    }

    // Initialize huffman coder
    struct gpujpeg_huffman_cpu_decoder coder;
    coder.component = component;
    coder.scan_index = -1;

    // Derive only referenced tables: unused decoder slots may be uninitialized.
    unsigned initialized[2] = {0, 0};
    for ( int comp = 0; comp < decoder->coder.param.comp_count; ++comp ) {
        for ( int type = 0; type < 2; ++type ) {
            int index = decoder->comp_table_huffman_map[comp][type];
            if ( index < 0 || index >= GPUJPEG_MAX_COMPONENT_COUNT )
                return -1;
            if ( !(initialized[type] & (1U << index)) ) {
                cpu_huffman_table_init(type == GPUJPEG_HUFFMAN_DC ? &coder.table_dc[index] : &coder.table_ac[index],
                                       &decoder->table_huffman[index][type]);
                initialized[type] |= 1U << index;
            }
        }
        coder.component[comp].dc_huff_idx = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_DC];
        coder.component[comp].ac_huff_idx = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_AC];
    }

    // Set mcu component count
    if ( decoder->coder.param.interleaved == 1 )
        coder.comp_count = decoder->coder.param.comp_count;
    else
        coder.comp_count = 1;
    assert(coder.comp_count >= 1 && coder.comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Decode all segments
    for ( int segment_index = segment_begin; segment_index < segment_begin + segment_count; segment_index++ ) {
        // Get segment structure
        struct gpujpeg_segment* segment = &decoder->coder.segment[segment_index];
        
        // Change current scan index
        if ( coder.scan_index != segment->scan_index ) {
            coder.scan_index = segment->scan_index;
        }
        
        // Initialize huffman coder
        coder.get_buff = 0;
        coder.get_bits = 0;
        for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
            coder.dc[comp] = 0;
        coder.data = &decoder->coder.data_compressed[segment->data_compressed_index];
        coder.data_size = segment->data_compressed_size;
        
        // Decode segment MCUs
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            if ( gpujpeg_huffman_cpu_decoder_decode_mcu(&coder, segment->scan_segment_index, mcu_index) != 0 ) {
                fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed at block [%d, %d]!\n", segment_index, mcu_index);
                return -1;
            }
        }
    }

    return 0;
}

/* vi: set sw=4: */
