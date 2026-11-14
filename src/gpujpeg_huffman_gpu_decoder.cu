/**
 * @file
 * Copyright (c) 2011-2025, CESNET
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

#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_util.h"

/**
 * Entry of pre-built Huffman fast-decoding table.
 */
struct gpujpeg_table_huffman_decoder_entry {
    
    int value_nbits;
};

/** Number of code bits to be checked first (with high chance for the code to fit into this number of bits). */
#define QUICK_CHECK_BITS 10
#define QUICK_TABLE_ITEMS (4 * (1 << QUICK_CHECK_BITS))
// TODO: try to tweak QUICK table size and memory space

struct gpujpeg_huffman_gpu_decoder {
    /**
     * 4 pre-built tables for faster Huffman decoding (codewords up-to 16 bit length):
     *   - 0x00000 to 0x0ffff: luminance DC table
     *   - 0x10000 to 0x1ffff: luminance AC table
     *   - 0x20000 to 0x2ffff: chrominance DC table
     *   - 0x30000 to 0x3ffff: chrominance AC table
     *
     * Each entry consists of:
     *   - Number of bits of code corresponding to this entry (0 - 16, both inclusive) - bits 4 to 8
     *   - Number of run-length coded zeros before currently decoded coefficient + 1 (1 - 64, both inclusive) - bits 9 to 15
     *   - Number of bits representing the value of currently decoded coefficient (0 - 15, both inclusive) - bits 0 to 3
     * @code
     * bit #:    15                      9   8               4   3           0
     *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
     * value:  |      RLE zero count       |   code bit size   | value bit size|
     *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
     * @endcode
     */
    uint16_t *d_tables_full;

    /** Table with same format as the full table, except that all-zero-entry means that the full table should be consulted. */
    uint16_t *d_tables_quick;

    /** Natural order */
    int *d_order_natural;
};

#ifdef HUFFMAN_GPU_CONST_TABLES
/** Same table as above, but copied into constant memory */
__constant__ uint16_t gpujpeg_huffman_gpu_decoder_tables_quick_const[QUICK_TABLE_ITEMS];
 
/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_decoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE] = {GPUJPEG_ORDER_NATURAL_VALUES};
#else
/** Natural order in global memory (read-only, shared by all decoders) */
__device__ int gpujpeg_huffman_gpu_decoder_order_natural_global[GPUJPEG_ORDER_NATURAL_SIZE] = {
    GPUJPEG_ORDER_NATURAL_VALUES};
#endif

// /**
//  * Fill more bit to current get buffer
//  * 
//  * @param get_bits
//  * @param get_buff
//  * @param data
//  * @param data_size
//  * @return void
//  */
// __device__ inline void
// gpujpeg_huffman_gpu_decoder_decode_fill_bit_buffer(int & get_bits, int & get_buff, uint8_t* & data, int & data_size)
// {
//     while ( get_bits < 25 ) {
//         //Are there some data?
//         if( data_size > 0 ) { 
//             // Attempt to read a byte
//             //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
//             unsigned char uc = *data++;
//             data_size--;            
// 
//             // If it's 0xFF, check and discard stuffed zero byte
//             if ( uc == 0xFF ) {
//                 while ( uc == 0xFF ) {
//                     //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
//                     uc = *data++;
//                     data_size--;
//                 }
// 
//                 if ( uc == 0 ) {
//                     // Found FF/00, which represents an FF data byte
//                     uc = 0xFF;
//                 } else {                
//                     // There should be enough bits still left in the data segment;
//                     // if so, just break out of the outer while loop.
//                     //if (m_nGetBits >= nbits)
//                     if ( get_bits >= 0 )
//                         break;
//                 }
//             }
// 
//             get_buff = (get_buff << 8) | ((int) uc);
//             get_bits += 8;            
//         }
//         else
//             break;
//     }
// }

/**
 * Loads at least specified number of bits into the register
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_load_bits(
                const unsigned int required_bit_count, unsigned int & r_bit,
                unsigned int & r_bit_count, uint4 * const s_byte, unsigned int & s_byte_idx
) {
    // Add bytes until have enough
    while(r_bit_count < required_bit_count) {
        // Load byte value and posibly skip next stuffed byte if loaded byte's value is 0xFF
        const uint8_t byte_value = ((const uint8_t*)s_byte)[s_byte_idx++];
        if((uint8_t)0xFF == byte_value) {
            s_byte_idx++;
        }
        
        // Add newly loaded byte to the buffer, updating bit count
        r_bit = (r_bit << 8) + byte_value;
        r_bit_count += 8;
    }
}


/**
 * Get bits
 * 
 * @param nbits  Number of bits to get
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return bits
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_get_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count, 
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // update remaining bit count
    r_bit_count -= nbits;
    
    // return bits 
    return (r_bit >> r_bit_count) & ((1 << nbits) - 1);
}

/**
 * Gets bits without removing them from the buffer.
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_peek_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count,
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // return bits 
    return (r_bit >> (r_bit_count - nbits)) & ((1 << nbits) - 1);
}

/**
 * Removes some bits from the buffer (assumes that they are there)
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_discard_bits(const unsigned int nb, unsigned int, unsigned int & r_bit_count) {
    r_bit_count -= nb;
}

/**
 * Special Huffman decode:
 * (1) For codes with length > 8
 * (2) For codes with length < 8 while data is finished
 * 
 * @param table
 * @param min_bits
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_decode_special_decode(
                const struct gpujpeg_table_huffman_decoder* const table, int min_bits, unsigned int & r_bit,
                unsigned int & r_bit_count, uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = gpujpeg_huffman_gpu_decoder_get_bits(min_bits, r_bit, r_bit_count, s_byte, s_byte_idx);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while ( code > table->maxcode[l] ) {
        code <<= 1;
        code |= gpujpeg_huffman_gpu_decoder_get_bits(1, r_bit, r_bit_count, s_byte, s_byte_idx);
        l++;
    }

    // With garbage input we may reach the sentinel value l = 17.
    if ( l > 16 ) {
        // Fake a zero as the safest result
        return 0;
    }
    
    return table->huffval[table->valptr[l] + (int)(code - table->mincode[l])];
}

/**
 * To find dc or ac value according to code and its bit length s
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_value_from_category(int nbits, int code)
{
    // TODO: try to replace with __constant__ table lookup
    return code < ((1 << nbits) >> 1) ? (code + ((-1) << nbits) + 1) : code;
    
//     // Method 1: 
//     // On some machines, a shift and add will be faster than a table lookup.
//     // #define HUFF_EXTEND(x,s) ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x))
// 
//     // Method 2: Table lookup
//     // If (offset < half[category]), then value is below zero
//     // Otherwise, value is above zero, and just the offset 
//     // entry n is 2**(n-1)
//     const int half[16] =    { 
//         0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
//         0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000
//     };
// 
//     //start[i] is the starting value in this category; surely it is below zero
//     // entry n is (-1 << n) + 1
//     const int start[16] = { 
//         0, ((-1)<<1) + 1, ((-1)<<2) + 1, ((-1)<<3) + 1, ((-1)<<4) + 1,
//         ((-1)<<5) + 1, ((-1)<<6) + 1, ((-1)<<7) + 1, ((-1)<<8) + 1,
//         ((-1)<<9) + 1, ((-1)<<10) + 1, ((-1)<<11) + 1, ((-1)<<12) + 1,
//         ((-1)<<13) + 1, ((-1)<<14) + 1, ((-1)<<15) + 1 
//     };    
// 
//     return (code < half[nbits]) ? (code + start[nbits]) : code;    
}

/// coefficient index after an invalid code in checked decoding (beyond the block, not an end of block)
#define GPUJPEG_HUFFMAN_GPU_INVALID_COEFFICIENT_IDX 0x1000

/**
 * Decodes next coefficient, updating its output index
 * 
 * @tparam CHECKED  invalid code sets the index to GPUJPEG_HUFFMAN_GPU_INVALID_COEFFICIENT_IDX (without consuming it)
 * @param table
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
template <bool CHECKED = false>
__device__ inline int
gpujpeg_huffman_gpu_decoder_get_coefficient(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
                unsigned int & s_byte_idx, const unsigned int table_offset, unsigned int & coefficient_idx)
{
    // Peek next 16 bits and use them as an index into decoder table to find all the info.
    const unsigned int table_idx = table_offset + gpujpeg_huffman_gpu_decoder_peek_bits(16, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // Try the quick table first (use the full table only if not succeded with the quick table)
#ifdef HUFFMAN_GPU_CONST_TABLES
    unsigned int packed_info = gpujpeg_huffman_gpu_decoder_tables_quick_const[table_idx >> (16 - QUICK_CHECK_BITS)];
#else
    unsigned int packed_info = huffman_gpu_decoder.d_tables_quick[table_idx >> (16 - QUICK_CHECK_BITS)];
#endif
    if(0 == packed_info) {
        packed_info = huffman_gpu_decoder.d_tables_full[table_idx];
    }
    
    // invalid codes have zero length (the unchecked decoding treats them as an end of block)
    if ( CHECKED && ((packed_info >> 4) & 0x1F) == 0 ) {
        coefficient_idx = GPUJPEG_HUFFMAN_GPU_INVALID_COEFFICIENT_IDX;
        return 0;
    }

    // remove the right number of bits from the bit buffer
    gpujpeg_huffman_gpu_decoder_discard_bits((packed_info >> 4) & 0x1F, r_bit, r_bit_count);
    
    // update coefficient index by skipping run-length encoded zeros
    coefficient_idx += packed_info >> 9;
    
    // read coefficient bits and decode the coefficient from them
    const unsigned int value_nbits = packed_info & 0xF;
    const unsigned int value_code = gpujpeg_huffman_gpu_decoder_get_bits(value_nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // return deocded coefficient
    return gpujpeg_huffman_gpu_decoder_value_from_category(value_nbits, value_code);
}

/**
 * Decode one 8x8 block
 *
 * @tparam WRITE    store the coefficients (data_output unused otherwise)
 * @tparam CHECKED  fail on an invalid code, on a run other than an end of block crossing the block end (both
 *                  accepted by the CPU decoder differently) and on a window refill with no chunk left (the window
 *                  would otherwise be reread and the canonical position lost)
 * @return 0 if succeeds, otherwise nonzero
 */
template <bool WRITE = true, bool CHECKED = false>
__device__ inline int
gpujpeg_huffman_gpu_decoder_decode_block(
    struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
    int & dc, int16_t* const data_output, const unsigned int dc_table_offset, const unsigned int ac_table_offset,
    unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
    unsigned int & s_byte_idx, const uint4* & d_byte, unsigned int & d_byte_chunk_count)
{
    // TODO: try unified decoding of DC/AC coefficients
    
    // Index of next coefficient to be decoded (in ZIG-ZAG order)
    unsigned int coefficient_idx = 0;
    
    // Section F.2.2.1: decode the DC coefficient difference
    // Get the coefficient value (using DC coding table)
    int dc_coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient<CHECKED>(huffman_gpu_decoder, r_bit, r_bit_count, s_byte, s_byte_idx, dc_table_offset, coefficient_idx);
    if ( CHECKED && coefficient_idx > 64 ) {
        return -1;
    }

    // Convert DC difference to actual value, update last_dc_val
    dc = dc_coefficient_value += dc;

    // Output the DC coefficient (assumes gpujpeg_natural_order[0] = 0)
    // TODO: try to skip saving of zero coefficients
    if ( WRITE ) {
        data_output[0] = dc_coefficient_value;
    }
    
    // TODO: error check: coefficient_idx must still be 0 in valid codestreams
    coefficient_idx = 1;
    
    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    do {
        // Possibly load more bytes into shared buffer from global memory
        if(s_byte_idx >= 16) {
            if ( CHECKED && d_byte_chunk_count == 0 ) {
                return -1;
            }
            // Move remaining bytes to begin and update index of next byte
            s_byte[0] = s_byte[1];
            s_byte_idx -= 16;
            
            // Load another byte chunk from global memory only if there is one
            if(d_byte_chunk_count) {
                s_byte[1] = *(d_byte++);
                d_byte_chunk_count--;
            }
        }
        
        // decode next coefficient, updating its destination index
        const unsigned int previous_coefficient_idx = coefficient_idx;
        const int coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient<CHECKED>(huffman_gpu_decoder, r_bit, r_bit_count, s_byte, s_byte_idx, ac_table_offset, coefficient_idx);
        
        // stop with this block if have all coefficients
        if(coefficient_idx > 64) {
            // end of block advances by 64
            if ( CHECKED && coefficient_idx != previous_coefficient_idx + 64 ) {
                return -1;
            }
            break;
        }
        
        // save the coefficient   TODO: try to ommit saving 0 coefficients
        if ( WRITE ) {
#ifdef HUFFMAN_GPU_CONST_TABLES
            data_output[gpujpeg_huffman_gpu_decoder_order_natural[coefficient_idx - 1]] = coefficient_value;
#else
            data_output[huffman_gpu_decoder.d_order_natural[coefficient_idx - 1]] = coefficient_value;
#endif
        }
    } while(coefficient_idx < 64);
    
    return 0;
}


/**
 * Huffman decoder kernel
 * 
 * @return void
 */
template <bool SINGLE_COMP, int THREADS_PER_TBLOCK>
__global__ void
#if __CUDA_ARCH__ < 200
__launch_bounds__(THREADS_PER_TBLOCK, 2)
#else
__launch_bounds__(THREADS_PER_TBLOCK, 6)
#endif
gpujpeg_huffman_decoder_decode_kernel(
    struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count, 
    uint8_t* d_data_compressed,
    const uint64_t* d_block_list,
    int16_t* d_data_quantized
) {
    int segment_index = blockIdx.x * THREADS_PER_TBLOCK + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // Byte buffers in shared memory
    __shared__ uint4 s_byte_all[2 * THREADS_PER_TBLOCK]; // 32 bytes per thread
    uint4 * const s_byte = s_byte_all + 2 * threadIdx.x;
    
    // Last DC coefficient values   TODO: try to move into shared memory
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
        
    // Get aligned compressed data chunk pointer and load first 2 chunks of the data
    const unsigned int d_byte_begin_idx = segment->data_compressed_index;
    const unsigned int d_byte_begin_idx_aligned = d_byte_begin_idx & ~15; // loading 16byte chunks
    const uint4* d_byte = (uint4*)(d_data_compressed + d_byte_begin_idx_aligned);
    
    // Get number of remaining global memory byte chunks (not to read bytes out of buffer)
    const unsigned int d_byte_end_idx_aligned = (d_byte_begin_idx + segment->data_compressed_size + 15) & ~15;
    unsigned int d_byte_chunk_count = (d_byte_end_idx_aligned - d_byte_begin_idx_aligned) / 16;
    
    // Load first 2 chunks of compressed data into the shared memory buffer and remember index of first code byte (skipping bytes read due to alignment)
    s_byte[0] = d_byte[0];
    s_byte[1] = d_byte[1];
    d_byte += 2;
    d_byte_chunk_count = max(d_byte_chunk_count, 2) - 2;
    unsigned int s_byte_idx = d_byte_begin_idx - d_byte_begin_idx_aligned;
    
    // bits loaded into the register and their count
    unsigned int r_bit_count = 0;
    unsigned int r_bit = 0; // LSB-aligned
    
    // Non-interleaving mode
    if ( SINGLE_COMP ) {
        // Get component for current scan
        const struct gpujpeg_component* const component = d_component + segment->scan_index; 
        
        // Get huffman tables offset
        const unsigned int dc_table_offset = component->dc_huff_idx * 0x20000;
        const unsigned int ac_table_offset = component->ac_huff_idx * 0x20000 + 0x10000;

        // Size of MCUs in this segment's component
        const int component_mcu_size = component->mcu_size;
        
        // Pointer to first MCU's output block
        int16_t* block = component->d_data_quantized + segment->scan_segment_index * component->segment_mcu_count * component_mcu_size;
        
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Encode 8x8 block
            if ( gpujpeg_huffman_gpu_decoder_decode_block(huffman_gpu_decoder, dc[0], block, dc_table_offset, ac_table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count) != 0 )
                break;
            
            // advance to next block
            block += component_mcu_size;
        } 
    }
    // Interleaving mode
    else {
        // Pointer to segment's list of 8x8 blocks and their count
        const uint64_t* packed_block_info_ptr = d_block_list + segment->block_index_list_begin;
        
        // Encode all blocks
        for(int block_count = segment->block_count; block_count--;) {
            // Get pointer to next block input data and info about its color type
            const uint64_t packed_block_info = *(packed_block_info_ptr++);
            
            // Get coder parameters
            const int last_dc_idx = packed_block_info & 0x7f;
            
            // Get offset to right part of huffman table
            const unsigned int dc_huffman_table_offset = d_component[last_dc_idx].dc_huff_idx * 0x20000;
            const unsigned int ac_huffman_table_offset = d_component[last_dc_idx].ac_huff_idx * 0x20000 + 0x10000;
            
            // Source data pointer
            int16_t* block = d_data_quantized + (packed_block_info >> 8);
            
            // Encode 8x8 block
            gpujpeg_huffman_gpu_decoder_decode_block(huffman_gpu_decoder, dc[last_dc_idx], block, dc_huffman_table_offset, ac_huffman_table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count);
        }
        
        
//         // Encode MCUs in segment
//         for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
//             
//             
//             
//             
//             
//             
//             
//             
//             //assert(segment->scan_index == 0);
//             for ( int comp = 0; comp < comp_count; comp++ ) {
//                 struct gpujpeg_component* component = &d_component[comp];
// 
//                 // Prepare mcu indexes
//                 int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
//                 int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
//                 // Compute base data index
//                 int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
//                 
//                 // For all vertical 8x8 blocks
//                 for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
//                     // Compute base row data index
//                     int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
//                     // For all horizontal 8x8 blocks
//                     for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
//                         // Compute 8x8 block data index
//                         int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
//                         
//                         // Get component data for MCU
//                         int16_t* block = &component->d_data_quantized[data_index];
//                         
//                         // Get coder parameters
//                         int & component_dc = dc[comp];
//             
//                         // Get huffman tables offset
//                         const unsigned int table_offset = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00000 : 0x20000;
//                         
//                         // Encode 8x8 block
//                         gpujpeg_huffman_gpu_decoder_decode_block(component_dc, block, table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count);
//                     }
//                 }
//             }
//         }
    }
}

/**
 * Setup of one Huffman table entry for fast decoding.
 * @param bits  bits to extract one codeword from (first bit is bit #15, then #14, ... last is #0)
 * @param d_table_src  source (slow-decoding) table pointer
 * @param d_table_dest  destination (fast-decoding) table pointer
 */
__device__ void
gpujpeg_huffman_gpu_decoder_table_setup(
    struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
    const int bits, 
    const struct gpujpeg_table_huffman_decoder* const d_table_src,
    const int table_idx
) {
    // Decode one codeword from given bits to get following:
    //  - minimal number of bits actually needed to decode the codeword (up to 16 bits, 0 for invalid ones)
    //  - category ID represented by the codeword, consisting from:
    //      - number of run-length-coded preceding zeros (up to 16, or 63 for both special end-of block symbol or invalid codewords)
    //      - bit-size of the actual value of coefficient (up to 16, 0 for invalid ones)
    int code_nbits = 1, category_id = 0;

    // First, decode codeword length (This is per Figure F.16 in the JPEG spec.), at most 16 bits (an incomplete or
    // zeroed unused table has no code for some bit sequences)
    int code_value = bits >> 15; // only single bit initially
    while ( code_value > d_table_src->maxcode[code_nbits] ) {
        if ( ++code_nbits > 16 ) {
            break;
        }
        code_value = bits >> (16 - code_nbits); // not enough to decide => try more bits
    }

    // Invalid code (no symbol of at most 16 bits).
    if ( code_nbits > 16 ) {
        code_nbits = 0;
        // category ID remains 0 for invalid symbols from garbage input
    } else {
        category_id = d_table_src->huffval[d_table_src->valptr[code_nbits] + code_value - d_table_src->mincode[code_nbits]];
    }
    
    // decompose category number into 1 + number of run-length coded zeros and length of the value
    // (special category #0 contains all invalid codes and special end-of-block code -- all of those codes 
    // should terminate block decoding => use 64 run-length zeros and 0 value bits for such symbols)
    const int value_nbits = 0xF & category_id;
    const int rle_zero_count = category_id ? min(1 + (category_id >> 4), 64) : 64;
    
    // save all the info into the right place in the destination table
    const int packed_info = (rle_zero_count << 9) + (code_nbits << 4) + value_nbits;
    huffman_gpu_decoder.d_tables_full[(table_idx << 16) + bits] = packed_info;
    
    // some threads also save entries into the quick table
    const int dest_idx_quick = bits >> (16 - QUICK_CHECK_BITS);
    if(bits == (dest_idx_quick << (16 - QUICK_CHECK_BITS))) {
        // save info also into the quick table if number of required bits is less than quick 
        // check bit count, otherwise put 0 there to indicate that full table lookup consultation is needed
        huffman_gpu_decoder.d_tables_quick[(table_idx << QUICK_CHECK_BITS) + dest_idx_quick] = code_nbits <= QUICK_CHECK_BITS ? packed_info : 0;
    }
}

/**
 * Huffman decoder table setup kernel
 * (Based on the original table, this kernel prepares another table, which is more suitable for fast decoding.)
 */
__global__ void
gpujpeg_huffman_decoder_table_kernel(
                struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                const struct gpujpeg_table_huffman_decoder* const d_table_y_dc,
                const struct gpujpeg_table_huffman_decoder* const d_table_y_ac,
                const struct gpujpeg_table_huffman_decoder* const d_table_cbcr_dc,
                const struct gpujpeg_table_huffman_decoder* const d_table_cbcr_ac,
                const int table_mask
) {
    // Each thread uses all 4 Huffman tables (those in table_mask) to "decode" one symbol from its unique 16bits.
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( (table_mask & 1) != 0 ) {
        gpujpeg_huffman_gpu_decoder_table_setup(huffman_gpu_decoder, idx, d_table_y_dc, 0);
    }
    if ( (table_mask & 2) != 0 ) {
        gpujpeg_huffman_gpu_decoder_table_setup(huffman_gpu_decoder, idx, d_table_y_ac, 1);
    }
    if ( (table_mask & 4) != 0 ) {
        gpujpeg_huffman_gpu_decoder_table_setup(huffman_gpu_decoder, idx, d_table_cbcr_dc, 2);
    }
    if ( (table_mask & 8) != 0 ) {
        gpujpeg_huffman_gpu_decoder_table_setup(huffman_gpu_decoder, idx, d_table_cbcr_ac, 3);
    }
}

/* Documented at declaration */
struct gpujpeg_huffman_gpu_decoder *
gpujpeg_huffman_gpu_decoder_init(const struct gpujpeg_coder* coder)
{
    struct gpujpeg_huffman_gpu_decoder *huffman_gpu_decoder = (struct gpujpeg_huffman_gpu_decoder *) calloc(1, sizeof(struct gpujpeg_huffman_gpu_decoder));

#ifndef HUFFMAN_GPU_CONST_TABLES
    cudaGetSymbolAddress((void**)&huffman_gpu_decoder->d_order_natural, gpujpeg_huffman_gpu_decoder_order_natural_global);
    gpujpeg_cuda_check_error("Huffman GPU decoder natural order table address", gpujpeg_huffman_gpu_decoder_destroy(huffman_gpu_decoder, coder); return NULL);
#endif

    if ( gpujpeg_coder_device_alloc(coder, (void**)&huffman_gpu_decoder->d_tables_full,
                                    4 * (1 << 16) * sizeof(uint16_t)) != 0 ||
         gpujpeg_coder_device_alloc(coder, (void**)&huffman_gpu_decoder->d_tables_quick,
                                    QUICK_TABLE_ITEMS * sizeof(uint16_t)) != 0 ) {
        gpujpeg_huffman_gpu_decoder_destroy(huffman_gpu_decoder, coder);
        return NULL;
    }
    
    return huffman_gpu_decoder;
}

void
gpujpeg_huffman_gpu_decoder_destroy(struct gpujpeg_huffman_gpu_decoder *huffman_gpu_decoder, const struct gpujpeg_coder* coder)
{
    if (huffman_gpu_decoder == NULL) {
        return;
    }

    gpujpeg_coder_device_free(coder, huffman_gpu_decoder->d_tables_full);
    gpujpeg_coder_device_free(coder, huffman_gpu_decoder->d_tables_quick);
    free(huffman_gpu_decoder);
}

/**
 * Builds the fast decoding tables in table_mask (bit Huffman table index * 2 + AC) from the decoder's tables
 *
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_huffman_gpu_decoder_build_tables(struct gpujpeg_decoder* decoder, int table_mask)
{
    struct gpujpeg_coder* coder = &decoder->coder;

    // Setup GPU tables (one thread for each of 65536 entries)
    gpujpeg_huffman_decoder_table_kernel<<<256, 256, 0, coder->stream>>>(
        *decoder->huffman_gpu_decoder,
        decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC],
        table_mask
    );
    gpujpeg_cuda_check_error("Huffman decoder table setup failed", return -1);

#ifdef HUFFMAN_GPU_CONST_TABLES
    // Copy quick decoding table into constant memory
    cudaMemcpyToSymbolAsync(
        gpujpeg_huffman_gpu_decoder_tables_quick_const,
        decoder->huffman_gpu_decoder->d_tables_quick,
        sizeof(*decoder->huffman_gpu_decoder->d_tables_quick) * QUICK_TABLE_ITEMS,
        0,
        cudaMemcpyDeviceToDevice,
        coder->stream
    );
    gpujpeg_cuda_check_error("Huffman decoder table copy failed", return -1);
#endif

    return 0;
}

/* Documented at declaration */
int
gpujpeg_huffman_gpu_decoder_decode(struct gpujpeg_decoder* decoder, bool block_list)
{    
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    // assert(coder->param.restart_interval > 0);
    
    int comp_count = 1;
    if (coder->param.interleaved == 1) {
        comp_count = coder->param.comp_count;
    }
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Number of decoder kernel threads per each threadblock
    enum { THREADS_PER_TBLOCK = 192 };
    
    // Configure more Shared memory for both kernels
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_table_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_decode_kernel<true, THREADS_PER_TBLOCK>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_decode_kernel<false, THREADS_PER_TBLOCK>, cudaFuncCachePreferShared);
    
    if ( gpujpeg_huffman_gpu_decoder_build_tables(decoder, 0xF) != 0 ) {
        return -1;
    }

    for (int comp = 0; comp < coder->param.comp_count; comp++) {
        coder->component[comp].dc_huff_idx = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_DC];
        coder->component[comp].ac_huff_idx = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_AC];
    }
    // Copy updated components to device memory
    cudaMemcpyAsync(coder->d_component, coder->component, coder->param.comp_count * sizeof(struct gpujpeg_component),
                    cudaMemcpyHostToDevice, coder->stream);
    gpujpeg_cuda_check_error("Coder component copy", return -1);

    // Run decoding kernel
    dim3 thread(THREADS_PER_TBLOCK);
    dim3 grid(gpujpeg_div_and_round_up(decoder->segment_count, THREADS_PER_TBLOCK));
    if ( comp_count == 1 && !block_list ) {
        gpujpeg_huffman_decoder_decode_kernel<true, THREADS_PER_TBLOCK><<<grid, thread, 0, coder->stream>>>(
            *decoder->huffman_gpu_decoder,
            coder->d_component, 
            coder->d_segment, 
            comp_count,
            decoder->segment_count,
            coder->d_data_compressed,
            coder->d_block_list,
            coder->d_data_quantized
        );
    }
    else {
        gpujpeg_huffman_decoder_decode_kernel<false, THREADS_PER_TBLOCK><<<grid, thread, 0, coder->stream>>>(
            *decoder->huffman_gpu_decoder,
            coder->d_component, 
            coder->d_segment, 
            comp_count,
            decoder->segment_count,
            coder->d_data_compressed,
            coder->d_block_list,
            coder->d_data_quantized
        );
    }
    gpujpeg_cuda_check_error("Huffman decoding failed", return -1);
    
    return 0;
}

/*
 * Huffman decoding of a scan without restart markers (one segment) by self-synchronization, after Weissenberger and
 * Schmidt: Accelerating JPEG Decompression on GPUs (arXiv:2111.09219).
 *
 * The stuffed entropy-coded data is split into chunks of chunk_bytes. A chunk owns the blocks whose canonical start
 * position (stuffed bit position, never inside a stuffed zero) precedes the chunk end; a block may continue into the
 * next chunks, which are then empty. Decoding started at a guessed position usually falls into the true block
 * boundaries; from a block start with the same position and MCU block phase, two decodings are identical (DC
 * predictors affect values, not parsing).
 *  - sequence kernel: each chunk is decoded from a guessed start (exact for chunk 0) without storing, then repairs the
 *    following chunks of its sequence (seq_chunks chunks, one thread block)
 *  - link kernel: one thread block repairs across sequences, then replaces the chunk block counts and DC difference
 *    sums by their exclusive prefixes
 *  - replay kernel: decodes every chunk again from its predecessor's record, stores the coefficients through the block
 *    list and certifies the chain (see gpujpeg_huffman_gpu_resync_replay()); any failure or exhausted repair rounds
 *    are reported in the status and the image is decoded on CPU instead
 * Records and status live in coder->d_temp_huffman (used only by the encoder otherwise).
 */

enum {
    RESYNC_MAX_COMPS = 3,          ///< components of eligible scans
    RESYNC_STATUS_BYTES = 64,      ///< status size at the beginning of the scratch, records follow
    RESYNC_MAX_SEQUENCES = 1024,   ///< link kernel threads
    RESYNC_MAX_CHUNK_BYTES = 4096, ///< keeps the replayed DC predictors within int (see the replay)
    RESYNC_REPLAY_THREADS = 128,
};
#define RESYNC_MAX_BYTES (1U << 28)      ///< bit positions and block counts fit 32 bits
#define RESYNC_MAX_DC_PREFIX (1LL << 30) ///< replayed DC predictor seed limit
#define RESYNC_INVALID 0xFFFFFFFFU       ///< phase of a record whose trajectory failed

/** Chunk record */
struct gpujpeg_huffman_gpu_resync_record
{
    long long dc[RESYNC_MAX_COMPS]; ///< DC difference sums of the chunk's blocks (exclusive prefix after linking)
    unsigned int blocks;            ///< complete blocks starting in the chunk (exclusive prefix after linking)
    unsigned int end;               ///< canonical bit position after the chunk's blocks
    unsigned int phase;             ///< MCU block phase at end, RESYNC_INVALID if the trajectory failed in the chunk
    unsigned int unused;
};

/** Status at the beginning of the scratch */
struct gpujpeg_huffman_gpu_resync_status
{
    unsigned int repair;   ///< link kernel: GPUJPEG_HUFFMAN_RESYNC_REPAIR_LIMIT or 0 (constant during the replay)
    unsigned int total;    ///< link kernel: sum of the chunk block counts (constant during the replay)
    unsigned int error;    ///< replay: GPUJPEG_HUFFMAN_RESYNC_* bits (atomicOr)
    unsigned int complete; ///< replay: chunks completing the scan (atomicAdd)
};
static_assert(sizeof(struct gpujpeg_huffman_gpu_resync_status) <= RESYNC_STATUS_BYTES, "status overlaps records");

/** Block count and DC sums scanned by the link kernel in the reader windows */
struct gpujpeg_huffman_gpu_resync_sum
{
    unsigned long long blocks;
    long long dc[RESYNC_MAX_COMPS];
};
static_assert(sizeof(struct gpujpeg_huffman_gpu_resync_sum) == 2 * sizeof(uint4), "sum must fit a reader window");

/** Scan description (kernel parameter) */
struct gpujpeg_huffman_gpu_resync_scan
{
    const uint8_t* d_data;    ///< stuffed entropy-coded data, zeroed after size up to end_aligned
    unsigned int size;        ///< entropy-coded bytes
    unsigned int end_aligned; ///< 16-aligned size + 32 (initialized lookahead, never decoded as data)
    unsigned int chunk_bytes;
    unsigned int chunk_count;
    unsigned int seq_chunks;    ///< chunks per sequence (sequence kernel lanes per block)
    unsigned int repair_rounds; ///< link kernel rounds
    unsigned int block_count;   ///< blocks of the scan (including padding blocks)
    unsigned int mcu_blocks;    ///< blocks per MCU (1, 3, 4 or 6)
    unsigned int phase_comp;    ///< component of each MCU block phase (2 bits per phase)
    unsigned int comp_table;    ///< Huffman table index (0 or 1) of each component (bit per component)
    struct gpujpeg_huffman_gpu_resync_status* d_status;
    struct gpujpeg_huffman_gpu_resync_record* d_record;
};

/** Bit reader state (as the variables of gpujpeg_huffman_decoder_decode_kernel()) */
struct gpujpeg_huffman_gpu_resync_reader
{
    unsigned int r_bit;
    unsigned int r_bit_count;
    uint4* s_byte; ///< 32-byte window in shared memory
    unsigned int s_byte_idx;
    const uint4* d_byte; ///< next 16-byte chunk to load
    unsigned int d_byte_chunk_count;
};

/**
 * Starts reading at a canonical bit position (at most 8 * size). All loads stay below scan.end_aligned, which is
 * at least 32 bytes after the position's 16-byte chunk.
 */
__device__ inline void
gpujpeg_huffman_gpu_resync_seek(struct gpujpeg_huffman_gpu_resync_reader& rd,
                                const struct gpujpeg_huffman_gpu_resync_scan& scan, unsigned int bitpos)
{
    const unsigned int byte_idx = bitpos / 8;
    const unsigned int byte_idx_aligned = byte_idx & ~15U;
    rd.d_byte = (const uint4*)(scan.d_data + byte_idx_aligned);
    rd.s_byte[0] = rd.d_byte[0];
    rd.s_byte[1] = rd.d_byte[1];
    rd.d_byte += 2;
    rd.d_byte_chunk_count = (scan.end_aligned - byte_idx_aligned) / 16 - 2;
    rd.s_byte_idx = byte_idx - byte_idx_aligned;
    rd.r_bit = 0;
    rd.r_bit_count = 0;
    gpujpeg_huffman_gpu_decoder_get_bits(bitpos % 8, rd.r_bit, rd.r_bit_count, rd.s_byte, rd.s_byte_idx);
}

/**
 * Canonical bit position of the next unread bit. The window holds the 32 bytes before d_byte as long as every shift
 * loaded a chunk (checked decoding fails otherwise). The bit buffer keeps consumed bits, so its low byte is the last
 * byte loaded; a loaded 0xFF was followed by a skipped stuffed zero. At most 23 bits are buffered.
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_resync_cursor(const struct gpujpeg_huffman_gpu_resync_reader& rd, const uint8_t* d_data)
{
    unsigned int next_byte = (unsigned int)((const uint8_t*)rd.d_byte - d_data) - 32 + rd.s_byte_idx;
    unsigned int bits = rd.r_bit;
    unsigned int count = rd.r_bit_count;
    for ( ; count >= 8; count -= 8, bits >>= 8 ) { // unload the whole unread bytes
        next_byte -= (bits & 0xFF) == 0xFF ? 2 : 1;
    }
    if ( count == 0 ) {
        return next_byte * 8;
    }
    return (next_byte - ((bits & 0xFF) == 0xFF ? 2 : 1)) * 8 + 8 - count;
}

/**
 * Decodes one block with checks (see gpujpeg_huffman_gpu_decoder_decode_block()), updating dc and pos
 * @retval false invalid code or run, or data consumed beyond the entropy-coded data (lookahead is not data)
 */
template <bool WRITE>
__device__ inline bool
gpujpeg_huffman_gpu_resync_block(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                 const struct gpujpeg_huffman_gpu_resync_scan& scan,
                                 struct gpujpeg_huffman_gpu_resync_reader& rd, unsigned int comp, int& dc,
                                 int16_t* data_output, unsigned int& pos)
{
    const unsigned int table_offset = ((scan.comp_table >> comp) & 1) * 0x20000;
    if ( gpujpeg_huffman_gpu_decoder_decode_block<WRITE, true>(
             huffman_gpu_decoder, dc, data_output, table_offset, table_offset + 0x10000, rd.r_bit, rd.r_bit_count,
             rd.s_byte, rd.s_byte_idx, rd.d_byte, rd.d_byte_chunk_count) != 0 ) {
        return false;
    }
    pos = gpujpeg_huffman_gpu_resync_cursor(rd, scan.d_data);
    return pos <= scan.size * 8;
}

/**
 * Decodes (without storing) the blocks of a chunk from the reader state and phase. If a block fails, the record is
 * invalid but keeps the blocks completed before it.
 */
__device__ inline struct gpujpeg_huffman_gpu_resync_record
gpujpeg_huffman_gpu_resync_skip_chunk(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                      const struct gpujpeg_huffman_gpu_resync_scan& scan,
                                      struct gpujpeg_huffman_gpu_resync_reader& rd, unsigned int& phase,
                                      unsigned int chunk)
{
    struct gpujpeg_huffman_gpu_resync_record record = {};
    const unsigned int chunk_end = (chunk + 1) * scan.chunk_bytes * 8;
    unsigned int pos = gpujpeg_huffman_gpu_resync_cursor(rd, scan.d_data);
    while ( pos < chunk_end ) {
        const unsigned int comp = (scan.phase_comp >> (2 * phase)) & 3;
        int dc = 0;
        if ( !gpujpeg_huffman_gpu_resync_block<false>(huffman_gpu_decoder, scan, rd, comp, dc, NULL, pos) ) {
            record.phase = RESYNC_INVALID;
            return record;
        }
        record.dc[comp] += dc;
        record.blocks += 1;
        phase = phase + 1 == scan.mcu_blocks ? 0 : phase + 1;
    }
    record.end = pos;
    record.phase = phase;
    return record;
}

/// @retval true if a repaired chunk ends in the (valid) state of the chunk's previous record
__device__ inline bool
gpujpeg_huffman_gpu_resync_joined(const struct gpujpeg_huffman_gpu_resync_record& record,
                                  const struct gpujpeg_huffman_gpu_resync_record& previous)
{
    return record.phase != RESYNC_INVALID && record.phase == previous.phase && record.end == previous.end;
}

/**
 * Intra-sequence synchronization: lane t of block s decodes chunk s * seq_chunks + t from a guessed start, then
 * repairs the following chunks of the sequence, one per round, from its own state: a round's chunk is decoded,
 * compared with the chunk's record and the record always replaced (the counts depend on the start). Matching ends
 * the trajectory, which then joined the one of the record. Each record has one writer per round and a smaller origin
 * reaches it in a later round, so the sequence becomes consistent with its first chunk.
 * Launched with seq_chunks << LANE_SHIFT threads per block: lane t is thread t << LANE_SHIFT, the other threads never
 * decode but take part in every barrier.
 */
template <unsigned int LANE_SHIFT>
__global__ void __launch_bounds__(1024)
    gpujpeg_huffman_decoder_resync_sequence_kernel(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                                   struct gpujpeg_huffman_gpu_resync_scan scan)
{
    extern __shared__ uint4 s_resync_window[]; // 32 bytes per lane
    const unsigned int lane = threadIdx.x >> LANE_SHIFT;
    const bool lane_thread = (threadIdx.x & ((1U << LANE_SHIFT) - 1)) == 0;
    struct gpujpeg_huffman_gpu_resync_reader rd;
    rd.s_byte = s_resync_window + 2 * lane;
    const unsigned int seq_begin = blockIdx.x * scan.seq_chunks;
    const unsigned int seq_end = min(seq_begin + scan.seq_chunks, scan.chunk_count);
    unsigned int chunk = seq_begin + lane;
    unsigned int phase = 0;
    bool active = lane_thread && chunk < seq_end;
    if ( active ) {
        // the chunk's first byte or, if it is a stuffed zero, the next one (the reader ensures each 0xFF of the
        // segment is followed by a stuffed zero)
        unsigned int start = chunk * scan.chunk_bytes;
        if ( chunk != 0 && scan.d_data[start - 1] == 0xFF ) {
            start += 1;
        }
        gpujpeg_huffman_gpu_resync_seek(rd, scan, start * 8);
        const struct gpujpeg_huffman_gpu_resync_record record =
            gpujpeg_huffman_gpu_resync_skip_chunk(huffman_gpu_decoder, scan, rd, phase, chunk);
        scan.d_record[chunk] = record;
        active = record.phase != RESYNC_INVALID && chunk + 1 < seq_end;
    }
    // inactive threads keep taking part in the barriers
    while ( __syncthreads_or(active) ) {
        if ( active ) {
            chunk += 1;
            const struct gpujpeg_huffman_gpu_resync_record previous = scan.d_record[chunk];
            const struct gpujpeg_huffman_gpu_resync_record record =
                gpujpeg_huffman_gpu_resync_skip_chunk(huffman_gpu_decoder, scan, rd, phase, chunk);
            scan.d_record[chunk] = record;
            active = record.phase != RESYNC_INVALID && !gpujpeg_huffman_gpu_resync_joined(record, previous) &&
                     chunk + 1 < seq_end;
        }
    }
}

/**
 * Inter-sequence synchronization and prefix scan (single thread block, thread q for sequence q). Thread q >= 1
 * continues from the end record of sequence q - 1 as the sequence kernel left it (read before the first barrier) and
 * repairs chunk q * seq_chunks + r in round r as the sequence kernel. Chunks repaired in one round differ by
 * seq_chunks; a smaller origin reaches a chunk in a later round, so the trajectory from the anchored sequence 0
 * dominates. A trajectory still repairing after repair_rounds sets the repair limit status (decoded on CPU). Then the
 * block counts and DC sums of the chunks are replaced by their exclusive prefixes and the total is stored.
 */
__global__ void __launch_bounds__(1024)
    gpujpeg_huffman_decoder_resync_link_kernel(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                               struct gpujpeg_huffman_gpu_resync_scan scan)
{
    extern __shared__ uint4 s_resync_window[]; // 32 bytes per thread
    struct gpujpeg_huffman_gpu_resync_reader rd;
    rd.s_byte = s_resync_window + 2 * threadIdx.x;
    const unsigned int seq = threadIdx.x;
    const unsigned int seq_begin = min(seq * scan.seq_chunks, scan.chunk_count);
    const unsigned int seq_end = min(seq_begin + scan.seq_chunks, scan.chunk_count);
    if ( seq == 0 ) {
        scan.d_status->repair = 0;
        scan.d_status->error = 0;
        scan.d_status->complete = 0;
    }

    unsigned int chunk = seq_begin;
    unsigned int phase = 0;
    bool active = seq != 0 && seq_begin < seq_end;
    if ( active ) {
        const struct gpujpeg_huffman_gpu_resync_record seed = scan.d_record[seq_begin - 1];
        active = seed.phase != RESYNC_INVALID && seed.end <= scan.size * 8;
        if ( active ) {
            phase = seed.phase;
            gpujpeg_huffman_gpu_resync_seek(rd, scan, seed.end);
        }
    }
    for ( unsigned int round = 0; __syncthreads_or(active); ++round ) {
        if ( round == scan.repair_rounds ) { // uniform
            if ( seq == 0 ) {
                scan.d_status->repair = GPUJPEG_HUFFMAN_RESYNC_REPAIR_LIMIT;
            }
            break;
        }
        if ( active ) {
            const struct gpujpeg_huffman_gpu_resync_record previous = scan.d_record[chunk];
            const struct gpujpeg_huffman_gpu_resync_record record =
                gpujpeg_huffman_gpu_resync_skip_chunk(huffman_gpu_decoder, scan, rd, phase, chunk);
            scan.d_record[chunk] = record;
            active = record.phase != RESYNC_INVALID && !gpujpeg_huffman_gpu_resync_joined(record, previous) &&
                     chunk + 1 < scan.chunk_count;
            chunk += 1;
        }
    }
    __syncthreads();

    // exclusive prefix scan: each thread sums its sequence's chunks, the sums are scanned in the (no longer used)
    // reader windows, then each thread rewrites its chunks
    struct gpujpeg_huffman_gpu_resync_sum own = {};
    for ( unsigned int i = seq_begin; i < seq_end; ++i ) {
        own.blocks += scan.d_record[i].blocks;
        for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
            own.dc[comp] += scan.d_record[i].dc[comp];
        }
    }
    struct gpujpeg_huffman_gpu_resync_sum* const s_sum = (struct gpujpeg_huffman_gpu_resync_sum*)s_resync_window;
    s_sum[seq] = own;
    __syncthreads();
    for ( unsigned int offset = 1; offset < blockDim.x; offset *= 2 ) { // inclusive
        struct gpujpeg_huffman_gpu_resync_sum add = {};
        if ( seq >= offset ) {
            add = s_sum[seq - offset];
        }
        __syncthreads();
        s_sum[seq].blocks += add.blocks;
        for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
            s_sum[seq].dc[comp] += add.dc[comp];
        }
        __syncthreads();
    }
    struct gpujpeg_huffman_gpu_resync_sum prefix = s_sum[seq];
    if ( seq == blockDim.x - 1 ) {
        scan.d_status->total = (unsigned int)prefix.blocks;
    }
    prefix.blocks -= own.blocks;
    for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
        prefix.dc[comp] -= own.dc[comp];
    }
    for ( unsigned int i = seq_begin; i < seq_end; ++i ) {
        struct gpujpeg_huffman_gpu_resync_record* const record = &scan.d_record[i];
        const unsigned int blocks = record->blocks;
        record->blocks = (unsigned int)prefix.blocks;
        prefix.blocks += blocks;
        for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
            const long long dc = record->dc[comp];
            record->dc[comp] = prefix.dc[comp];
            prefix.dc[comp] += dc;
        }
    }
}

/// @retval true if at most 7 one bits (rest of one data byte, possibly followed by its stuffed zero) follow pos
__device__ inline bool
gpujpeg_huffman_gpu_resync_padding_valid(const struct gpujpeg_huffman_gpu_resync_scan& scan, unsigned int pos)
{
    if ( pos == scan.size * 8 ) {
        return true;
    }
    const unsigned int byte_idx = pos / 8;
    const unsigned int padding_bits = 8 - pos % 8;
    const unsigned int byte_value = scan.d_data[byte_idx];
    const unsigned int ones = (1U << padding_bits) - 1;
    return padding_bits < 8 && (byte_value & ones) == ones && byte_idx + (byte_value == 0xFF ? 2 : 1) == scan.size;
}

/**
 * Replays the blocks first..min(next, block_count) - 1 of a chunk (first < block_count, prefixes monotone) and
 * certifies them. By induction from the exact scan start, each chunk decodes the true blocks from the true state:
 * its start record must be valid with the phase of its block prefix, all its blocks must start in the chunk and
 * decode with checks, and a chunk followed by more blocks must end in its own record's state with the next chunk's
 * DC prefix. The chunk with the last block must be followed only by padding; the host requires exactly one such.
 *
 * @return GPUJPEG_HUFFMAN_RESYNC_* bits, 0 if succeeded
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_resync_replay(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                  const struct gpujpeg_huffman_gpu_resync_scan& scan, uint4* s_byte,
                                  const uint64_t* d_block_list, int16_t* d_data_quantized, unsigned int chunk,
                                  unsigned int first, unsigned int next)
{
    const struct gpujpeg_huffman_gpu_resync_record* const d_record = scan.d_record;
    // DC predictors after the previous chunks, small enough not to overflow in this chunk's at most
    // 4 * chunk_bytes + 1 blocks (each takes at least 2 bits)
    int dc[RESYNC_MAX_COMPS];
    for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
        const long long prefix = d_record[chunk].dc[comp];
        if ( chunk == 0 ? prefix != 0 : (prefix < -RESYNC_MAX_DC_PREFIX || prefix > RESYNC_MAX_DC_PREFIX) ) {
            return chunk == 0 ? GPUJPEG_HUFFMAN_RESYNC_ORDER : GPUJPEG_HUFFMAN_RESYNC_DECODE;
        }
        dc[comp] = (int)prefix;
    }
    // start in the exact scan start or in the previous chunk's end (which is in this chunk or after it)
    unsigned int pos = 0;
    unsigned int phase = 0;
    if ( chunk != 0 ) {
        const struct gpujpeg_huffman_gpu_resync_record previous = d_record[chunk - 1];
        if ( previous.phase == RESYNC_INVALID || previous.phase != first % scan.mcu_blocks ||
             previous.end < chunk * scan.chunk_bytes * 8 || previous.end > scan.size * 8 ) {
            return GPUJPEG_HUFFMAN_RESYNC_START;
        }
        pos = previous.end;
        phase = previous.phase;
    }
    struct gpujpeg_huffman_gpu_resync_reader rd;
    rd.s_byte = s_byte;
    gpujpeg_huffman_gpu_resync_seek(rd, scan, pos);

    const unsigned int chunk_end = (chunk + 1) * scan.chunk_bytes * 8;
    const unsigned int last = min(next, scan.block_count);
    for ( unsigned int block = first; block < last; ++block ) {
        if ( pos >= chunk_end ) {
            return GPUJPEG_HUFFMAN_RESYNC_OWNERSHIP;
        }
        const uint64_t packed_block_info = d_block_list[block];
        const unsigned int comp = packed_block_info & 0x7f;
        if ( comp >= RESYNC_MAX_COMPS ||
             !gpujpeg_huffman_gpu_resync_block<true>(huffman_gpu_decoder, scan, rd, comp, dc[comp],
                                                     d_data_quantized + (packed_block_info >> 8), pos) ) {
            return GPUJPEG_HUFFMAN_RESYNC_DECODE;
        }
        phase = phase + 1 == scan.mcu_blocks ? 0 : phase + 1;
    }

    if ( next < scan.block_count ) { // continued by the next chunk
        if ( chunk + 1 == scan.chunk_count ) {
            return GPUJPEG_HUFFMAN_RESYNC_INCOMPLETE; // block counts sum to less than the scan
        }
        const struct gpujpeg_huffman_gpu_resync_record own = d_record[chunk];
        if ( own.phase != phase || own.end != pos ) {
            return GPUJPEG_HUFFMAN_RESYNC_ENDPOINT;
        }
        for ( int comp = 0; comp < RESYNC_MAX_COMPS; ++comp ) {
            if ( dc[comp] != d_record[chunk + 1].dc[comp] ) {
                return GPUJPEG_HUFFMAN_RESYNC_ENDPOINT;
            }
        }
        return 0;
    }
    if ( !gpujpeg_huffman_gpu_resync_padding_valid(scan, pos) ) {
        return GPUJPEG_HUFFMAN_RESYNC_PADDING;
    }
    atomicAdd(&scan.d_status->complete, 1);
    return 0;
}

/**
 * Replay of chunk c by thread c << LANE_SHIFT (gpujpeg_huffman_gpu_resync_replay()), skipped if repair rounds were
 * exhausted; the other threads only return (there are no barriers) and each thread uses only its own window.
 * Every chunk checks that the block prefixes start at 0 and do not decrease.
 */
template <unsigned int LANE_SHIFT>
__global__ void
__launch_bounds__(RESYNC_REPLAY_THREADS)
    gpujpeg_huffman_decoder_resync_replay_kernel(struct gpujpeg_huffman_gpu_decoder huffman_gpu_decoder,
                                                 struct gpujpeg_huffman_gpu_resync_scan scan,
                                                 const uint64_t* d_block_list, int16_t* d_data_quantized)
{
    __shared__ uint4 s_byte_all[2 * RESYNC_REPLAY_THREADS]; // 32 bytes per thread
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int chunk = thread >> LANE_SHIFT;
    if ( (thread & ((1U << LANE_SHIFT) - 1)) != 0 || chunk >= scan.chunk_count || scan.d_status->repair != 0 ) {
        return;
    }
    const unsigned int first = scan.d_record[chunk].blocks;
    const unsigned int next = chunk + 1 < scan.chunk_count ? scan.d_record[chunk + 1].blocks : scan.d_status->total;
    unsigned int error = 0;
    if ( next < first || (chunk == 0 && first != 0) ) {
        error = GPUJPEG_HUFFMAN_RESYNC_ORDER;
    }
    else if ( first < scan.block_count ) {
        error = gpujpeg_huffman_gpu_resync_replay(huffman_gpu_decoder, scan, s_byte_all + 2 * threadIdx.x, d_block_list,
                                                  d_data_quantized, chunk, first, next);
    }
    if ( error != 0 ) {
        atomicOr(&scan.d_status->error, error);
    }
}

/**
 * @retval true if a Huffman table is a baseline table that the checked decoding reads as the CPU decoder: a prefix
 *              code of at most 256 symbols, DC categories 0-11, AC run/size with size 1-10, end of block or ZRL
 */
static bool
gpujpeg_huffman_gpu_resync_table_valid(const struct gpujpeg_table_huffman_decoder* table, bool ac)
{
    unsigned int count = 0;
    unsigned long kraft_sum = 0; // sum of 2^(16 - code length)
    for ( int length = 1; length <= 16; ++length ) {
        count += table->bits[length];
        kraft_sum += (unsigned long)table->bits[length] << (16 - length);
    }
    if ( count == 0 || count > 256 || kraft_sum > 1UL << 16 ) {
        return false;
    }
    for ( unsigned int i = 0; i < count; ++i ) {
        const unsigned int symbol = table->huffval[i];
        const unsigned int size = symbol & 0xF;
        if ( ac ? (size == 0 ? symbol != 0x00 && symbol != 0xF0 : size > 10) : symbol > 11 ) {
            return false;
        }
    }
    return true;
}

/**
 * Describes the scan for the decoding without restart markers (Huffman tables are not checked here)
 * @param small_tile  if not NULL, set (when eligible) to whether the shipped-default small-tile limits apply
 * @retval false if not eligible
 */
static bool
gpujpeg_huffman_gpu_resync_scan_init(const struct gpujpeg_decoder* decoder,
                                     struct gpujpeg_huffman_gpu_resync_scan* scan, bool* small_tile)
{
    const struct gpujpeg_coder* coder = &decoder->coder;
    const struct gpujpeg_huffman_gpu_resync_param* param = &decoder->resync_param;
    if ( !decoder->huffman_resync_eligible || coder->param.restart_interval != 0 || decoder->segment_count != 1 ||
         coder->segment_count != 1 ) {
        return false;
    }
    const struct gpujpeg_segment* segment = &coder->segment[0];
    const size_t size = decoder->data_compressed_size;
    if ( size == 0 || size > RESYNC_MAX_BYTES || segment->data_compressed_index != 0 ||
         segment->data_compressed_size != size || segment->block_index_list_begin != 0 ||
         segment->block_count != coder->block_count ) {
        return false;
    }

    // grayscale 1x1 or interleaved Y 1x1, 2x1 or 2x2 with Cb and Cr 1x1
    const struct gpujpeg_component_sampling_factor* sampling = coder->param.sampling_factor;
    unsigned int mcu_blocks = 0;
    if ( coder->param.comp_count == 1 && sampling[0].horizontal == 1 && sampling[0].vertical == 1 ) {
        mcu_blocks = 1;
    }
    else if ( coder->param.comp_count == 3 && coder->param.interleaved == 1 && sampling[0].vertical >= 1 &&
              sampling[0].vertical <= sampling[0].horizontal && sampling[0].horizontal <= 2 &&
              sampling[1].horizontal == 1 && sampling[1].vertical == 1 && sampling[2].horizontal == 1 &&
              sampling[2].vertical == 1 ) {
        mcu_blocks = sampling[0].horizontal * sampling[0].vertical + 2;
    }
    if ( mcu_blocks == 0 || coder->block_count < (int)mcu_blocks ) {
        return false;
    }
    unsigned int phase_comp = 0; // components of the first MCU's blocks in scan order
    for ( unsigned int phase = 0; phase < mcu_blocks; ++phase ) {
        phase_comp |= (unsigned int)(coder->block_list[phase] & 0x7f) << (2 * phase);
    }
    unsigned int comp_table = 0;
    for ( int comp = 0; comp < coder->param.comp_count; ++comp ) {
        const int index = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_DC];
        if ( index < 0 || index > 1 || index != decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_AC] ) {
            return false;
        }
        comp_table |= (unsigned int)index << comp;
    }

    // Smaller scans than min_bytes keep the CPU decoder, except with exactly the shipped defaults (other values, also
    // from compile-time overrides, keep their meaning): 256x256 interleaved 4:4:4 and 4:2:2 scans of at least 32 KiB
    // and less than 64 KiB use 32-byte chunks in sequences of 32 chunks, measured faster than the CPU decoder for
    // such dense tiles on RTX 4090 at quality 95 (4:2:0 and sparse tiles were not).
    int chunk_bytes = param->chunk_bytes;
    int seq_chunks = param->seq_chunks;
    bool small_limits = false;
    if ( size < param->min_bytes ) {
        const bool shipped_defaults = param->chunk_bytes == 512 && param->seq_chunks == 128 &&
                                      param->repair_rounds == 128 && param->min_bytes == 262144;
        if ( !shipped_defaults || coder->param_image.width != 256 || coder->param_image.height != 256 ||
             (mcu_blocks != 3 && mcu_blocks != 4) || size < 32 * 1024 || size >= 64 * 1024 ) {
            return false;
        }
        chunk_bytes = 32;
        seq_chunks = 32;
        small_limits = true;
    }

    // kernel limits, initialized lookahead and scratch (status and records) within the allocations
    if ( chunk_bytes < 1 || chunk_bytes > RESYNC_MAX_CHUNK_BYTES || seq_chunks < 1 || seq_chunks > 1024 ||
         param->repair_rounds < 1 ) {
        return false;
    }
    const size_t chunk_count = (size + chunk_bytes - 1) / chunk_bytes;
    const size_t seq_count = (chunk_count + seq_chunks - 1) / seq_chunks;
    const size_t end_aligned = ((size + 15) & ~(size_t)15) + 32;
    const size_t scratch_size = RESYNC_STATUS_BYTES + chunk_count * sizeof(struct gpujpeg_huffman_gpu_resync_record);
    if ( seq_count > RESYNC_MAX_SEQUENCES || end_aligned > coder->data_compressed_allocated_size ||
         scratch_size > coder->data_compressed_allocated_size || (uintptr_t)coder->d_temp_huffman % 16 != 0 ) {
        return false;
    }

    scan->d_data = coder->d_data_compressed;
    scan->size = (unsigned int)size;
    scan->end_aligned = (unsigned int)end_aligned;
    scan->chunk_bytes = (unsigned int)chunk_bytes;
    scan->chunk_count = (unsigned int)chunk_count;
    scan->seq_chunks = (unsigned int)seq_chunks;
    scan->repair_rounds = (unsigned int)param->repair_rounds;
    scan->block_count = (unsigned int)coder->block_count;
    scan->mcu_blocks = mcu_blocks;
    scan->phase_comp = phase_comp;
    scan->comp_table = comp_table;
    scan->d_status = (struct gpujpeg_huffman_gpu_resync_status*)coder->d_temp_huffman;
    scan->d_record = (struct gpujpeg_huffman_gpu_resync_record*)(coder->d_temp_huffman + RESYNC_STATUS_BYTES);
    if ( small_tile != NULL ) {
        *small_tile = small_limits;
    }
    return true;
}

/* Documented at declaration */
bool
gpujpeg_huffman_gpu_decoder_resync_possible(const struct gpujpeg_decoder* decoder)
{
    struct gpujpeg_huffman_gpu_resync_scan scan;
    if ( !gpujpeg_huffman_gpu_resync_scan_init(decoder, &scan, NULL) ) {
        return false;
    }
    for ( int comp = 0; comp < decoder->coder.param.comp_count; ++comp ) {
        const int index = decoder->comp_table_huffman_map[comp][GPUJPEG_HUFFMAN_DC];
        if ( !gpujpeg_huffman_gpu_resync_table_valid(&decoder->table_huffman[index][GPUJPEG_HUFFMAN_DC], false) ||
             !gpujpeg_huffman_gpu_resync_table_valid(&decoder->table_huffman[index][GPUJPEG_HUFFMAN_AC], true) ) {
            return false;
        }
    }
    return true;
}

/* Documented at declaration */
int
gpujpeg_huffman_gpu_decoder_resync_decode(struct gpujpeg_decoder* decoder)
{
    struct gpujpeg_coder* coder = &decoder->coder;
    struct gpujpeg_huffman_gpu_resync_scan scan;
    bool small_tile = false;
    if ( !gpujpeg_huffman_gpu_resync_scan_init(decoder, &scan, &small_tile) ) {
        return -1; // precondition
    }

    // lookahead after the data reads zeros, not leftovers of a previous image
    cudaMemsetAsync(coder->d_data_compressed + scan.size, 0, scan.end_aligned - scan.size, coder->stream);
    gpujpeg_cuda_check_error("Huffman decoder lookahead reset failed", return -1);

    // only the tables of the scan (the others may be undefined)
    int table_mask = 0;
    for ( int comp = 0; comp < coder->param.comp_count; ++comp ) {
        table_mask |= 3 << (2 * ((scan.comp_table >> comp) & 1));
    }
    if ( gpujpeg_huffman_gpu_decoder_build_tables(decoder, table_mask) != 0 ) {
        return -1;
    }

    const unsigned int seq_count = gpujpeg_div_and_round_up(scan.chunk_count, scan.seq_chunks);
    const unsigned int link_threads = gpujpeg_div_and_round_up(seq_count, 32) * 32;
    // the small-tile limits (sequences of 32 chunks, at most 2048 chunks) place sequence lanes 4 threads apart and
    // replay one chunk per warp (fewer diverging lanes per warp). The mapping improved 4:4:4 performance; the 4:2:2
    // path with these limits beat the CPU route. Other limits keep one chunk per thread.
    if ( small_tile ) {
        gpujpeg_huffman_decoder_resync_sequence_kernel<2>
            <<<seq_count, scan.seq_chunks << 2, scan.seq_chunks * 2 * sizeof(uint4), coder->stream>>>(
                *decoder->huffman_gpu_decoder, scan);
    }
    else {
        gpujpeg_huffman_decoder_resync_sequence_kernel<0>
            <<<seq_count, scan.seq_chunks, scan.seq_chunks * 2 * sizeof(uint4), coder->stream>>>(
                *decoder->huffman_gpu_decoder, scan);
    }
    gpujpeg_huffman_decoder_resync_link_kernel<<<1, link_threads, link_threads * 2 * sizeof(uint4), coder->stream>>>(
        *decoder->huffman_gpu_decoder, scan);
    if ( small_tile ) {
        gpujpeg_huffman_decoder_resync_replay_kernel<5>
            <<<gpujpeg_div_and_round_up(scan.chunk_count << 5, RESYNC_REPLAY_THREADS), RESYNC_REPLAY_THREADS, 0,
               coder->stream>>>(*decoder->huffman_gpu_decoder, scan, coder->d_block_list, coder->d_data_quantized);
    }
    else {
        gpujpeg_huffman_decoder_resync_replay_kernel<0>
            <<<gpujpeg_div_and_round_up(scan.chunk_count, RESYNC_REPLAY_THREADS), RESYNC_REPLAY_THREADS, 0,
               coder->stream>>>(*decoder->huffman_gpu_decoder, scan, coder->d_block_list, coder->d_data_quantized);
    }
    gpujpeg_cuda_check_error("Huffman decoding without restart markers failed", return -1);

    struct gpujpeg_huffman_gpu_resync_status status;
    cudaMemcpyAsync(&status, scan.d_status, sizeof status, cudaMemcpyDeviceToHost, coder->stream);
    cudaStreamSynchronize(coder->stream);
    gpujpeg_cuda_check_error("Huffman decoding without restart markers status", return -1);

    DEBUG_MSG(coder->param.verbose,
              "Huffman without restart markers: %u chunks of %u B in %u sequences, %zu B of Huffman temp buffer\n",
              scan.chunk_count, scan.chunk_bytes, seq_count,
              (size_t)RESYNC_STATUS_BYTES + scan.chunk_count * sizeof(struct gpujpeg_huffman_gpu_resync_record));
    decoder->huffman_resync_status =
        status.repair | status.error | (status.complete == 1 ? 0 : (unsigned)GPUJPEG_HUFFMAN_RESYNC_INCOMPLETE);
    return decoder->huffman_resync_status == 0 ? 0 : 1;
}

/* vi: set expandtab sw=4 : */
