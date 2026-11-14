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

#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>                          // for fprintf, stderr
#include <string.h>                         // for memcmp, strlen, memcpy

#include "../libgpujpeg/gpujpeg_decoder.h"
#include "../libgpujpeg/gpujpeg_type.h"     // for gpujpeg_color_space, gpuj...
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_exif.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_reader.h"
#include "gpujpeg_util.h"

#undef MIN
#define MIN(a, b)      (((a) < (b))? (a): (b))

#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112 || defined __STDC_NO_THREADS__)
#define _Thread_local __declspec(thread)
#endif

/** JPEG reader scan structure */
struct gpujpeg_reader_scan
{
    /// Global segment index
    int segment_index;
    /// Segment count in scan
    int segment_count;
};

/** JPEG reader structure */
struct gpujpeg_reader
{
    /// Parameters
    struct gpujpeg_parameters param;

    /// Parameters for image data
    struct gpujpeg_image_parameters param_image;

    /// Loaded component count
    int comp_count;

    /// Loaded scans
    struct gpujpeg_reader_scan scan[GPUJPEG_MAX_COMPONENT_COUNT];

    /// Loaded scans count
    int scan_count;

    /// Total segment count
    int segment_count;

    /// Total readed size
    size_t data_compressed_size;

    /// Segment info (every buffer is placed inside another header)
    uint8_t* segment_info[GPUJPEG_MAX_SEGMENT_INFO_HEADER_COUNT];
    /// Segment info buffers count (equals number of segment info headers)
    int segment_info_count;
    /// Segment info total buffers size
    int segment_info_size;

    const uint8_t *const image_end;
    const bool ff_cs_itu601_is_709;
    enum gpujpeg_color_space header_color_space;
    enum gpujpeg_header_type header_type;
    bool in_spiff;
    struct gpujpeg_image_metadata *metadata;
    const char *comment;

    // facts for gpujpeg_decoder::huffman_resync_eligible
    int sof_count;            ///< SOF0/SOF1 marker count
    bool sof1;                ///< SOF1 (extended sequential) read
    unsigned huffman_defined; ///< DC/AC tables defined by this image (bit table index * 2 + class)
    bool resync_scan;         ///< first scan eligible (cleared by a DHT after it)

    bool batch;              ///< batch rules (gpujpeg_decoder_batch::active)
    unsigned quant_defined;  ///< quantization tables defined by this image (bit table index)
    unsigned quant_used;     ///< quantization tables used by scans of this image (batch)
    unsigned huffman_used;   ///< Huffman tables used by scans of this image (batch, bits as huffman_defined)
    int segment_limit;       ///< segment index limit of the scan being read (exclusive)
    size_t compressed_limit; ///< usable size of the decoder buffer for compressed data
};

/**
 * Read byte from image data
 *
 * @param image
 * @return byte
 */
#define gpujpeg_reader_read_byte(image) \
    (uint8_t)(*(image)++)

/**
 * Read two-bytes from image data
 *
 * @param image
 * @return 2 bytes
 */
#define gpujpeg_reader_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

#define gpujpeg_reader_read_4byte(image) \
    (uint32_t)(((uint32_t) (image)[0]) << 24U | ((uint32_t) (image)[1] << 16U) | ((uint32_t) (image)[2] << 8U) | ((uint32_t) (image)[3])); \
    (image) += 4;

/**
 * Read marker from image data
 *
 * @param image
 * @return marker code or -1 if failed
 */
static int
gpujpeg_reader_read_marker(uint8_t** image, const uint8_t* image_end, int verbose)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to read marker from JPEG data (end of data)\n");
        return -1;
    }

    uint8_t byte = gpujpeg_reader_read_byte(*image);
    if( byte != 0xFF ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to read marker from JPEG data (0xFF was expected but 0x%X was presented)\n", byte);
        return -1;
    }
    int marker = gpujpeg_reader_read_byte(*image);
    DEBUG2_MSG(verbose, "Read marker %s\n", gpujpeg_marker_name(marker));
    return marker;
}

/**
 * Skip marker content (read length and that much bytes - 2)
 *
 * @param image
 * @return 0 on success
 */
static int
gpujpeg_reader_skip_marker_content(uint8_t** image, const uint8_t* image_end)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to skip marker content (end of data)\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length < 2 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Invalid marker length %d\n", length);
        return -1;
    }
    length -= 2;

    if(length > image_end - *image) {
        fprintf(stderr, "[GPUJPEG] [Error] Marker content goes beyond data size\n");
        return -1;
    }
    *image += length;
    return 0;
}

/**
 * Rejects a batch image (gpujpeg_decoder_decode_batch())
 *
 * @return rc
 */
static int
gpujpeg_reader_batch_reject(struct gpujpeg_decoder* decoder, enum gpujpeg_batch_reason reason, int rc, const char* what)
{
    decoder->batch.reason = reason;
    ERROR_MSG("Batch image %d: %s!\n", decoder->batch.image, what);
    return rc;
}

/**
 * @param length  length field of the marker
 */
static int
gpujpeg_reader_read_jfif(uint8_t** image, const uint8_t* image_end, int length)
{
    if(image_end - *image < 9) {
        fprintf(stderr, "[GPUJPEG] [Error] JFIF segment goes beyond end of data\n");
        return -1;
    }

    int version_major = gpujpeg_reader_read_byte(*image);
    int version_minor = gpujpeg_reader_read_byte(*image);
    int pixel_units = gpujpeg_reader_read_byte(*image);
    int pixel_xdpu = gpujpeg_reader_read_2byte(*image);
    int pixel_ydpu = gpujpeg_reader_read_2byte(*image);
    int thumbnail_width = gpujpeg_reader_read_byte(*image);
    int thumbnail_height = gpujpeg_reader_read_byte(*image);
    (void) pixel_units, (void) pixel_xdpu, (void) pixel_ydpu;

    if ( version_major != 1 || ( version_minor < 0 || version_minor > 2) ) {
        fprintf(stderr, "[GPUJPEG] [Error] JFIF marker version should be 1.00 to 1.02 but %d.%02d was presented!\n", version_major, version_minor);
        return -1;
    }

    int thumbnail_length = thumbnail_width * thumbnail_height * 3;
    int expected_length = 16 + thumbnail_length;
    if ( length != expected_length ) {
        fprintf(stderr, "[GPUJPEG] [Warning] JFIF marker length should be %d (with thumbnail %dx%d) but %d was presented!\n",
                expected_length, thumbnail_width, thumbnail_height, length);
    }

    *image += length - 16;
    return 0;
}

/**
 * @param length  length field of the marker
 */
static int
gpujpeg_reader_skip_jfxx(uint8_t** image, int length)
{
    *image += length - 7;

    return 0;
}

/**
 * Read segment info for following scan
 *
 * @param decoder
 * @param image
 * @param length length of the marker
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_segment_info(struct gpujpeg_reader* reader, uint8_t** image, const uint8_t* image_end, int length)
{
    if(image_end - *image < 1) {
        fprintf(stderr, "[GPUJPEG] [Error] Segment info goes beyond end of data\n");
        return -1;
    }

    int scan_index = (int)gpujpeg_reader_read_byte(*image);

    int data_size = length - 3;
    if ( scan_index != reader->scan_count ) {
        fprintf(stderr, "[GPUJPEG] [Warning] %s marker (segment info) scan index should be %d but %d was presented!"
                " (marker not a segment info?)\n",
                gpujpeg_marker_name(GPUJPEG_MARKER_SEGMENT_INFO), reader->scan_count, scan_index);
        *image += data_size;
        return -1;
    }

    reader->segment_info[reader->segment_info_count] = *image;
    reader->segment_info_count++;
    reader->segment_info_size += data_size;

    *image += data_size;

    return 0;
}

/**
 * Read application info block from image
 *
 * @param image
 * @retval 0 if JFIF was successfully parsed
 * @retval 1 if JFXX was read
 * @retval -1 on error (no/unexpected tag or malformed JFIF/JFXX)
 */
static int
gpujpeg_reader_read_app0(uint8_t** image, const uint8_t* image_end, enum gpujpeg_header_type *header_type, int verbose)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read APP0 length\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);

    if(image_end - *image < length) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 goes beyond end of data\n");
        return -1;
    }

    if(length < 7)
    {
        // Just skip
        *image += length - 2;
        return -1;
    }

    char marker[5];
    marker[0] = gpujpeg_reader_read_byte(*image);
    marker[1] = gpujpeg_reader_read_byte(*image);
    marker[2] = gpujpeg_reader_read_byte(*image);
    marker[3] = gpujpeg_reader_read_byte(*image);
    marker[4] = gpujpeg_reader_read_byte(*image);

    if ( marker[4] == '\0' ) {
        if ( strcmp(marker, "JFIF") == 0 ) {
            *header_type = GPUJPEG_HEADER_JFIF;
            return gpujpeg_reader_read_jfif(image, image_end, length);
        }
        if ( strcmp(marker, "JFXX") == 0 ) {
            int ret = gpujpeg_reader_skip_jfxx(image, length);
            return ret == 0 ? 1 : ret;
        }
        VERBOSE_MSG(verbose, "APP0 marker identifier is not supported '%s'!\n", marker);
    } else {
        fprintf(stderr, "[GPUJPEG] [Warning] APP0 marker identifier either not terminated or not present!\n");
    }
    // Something else - skip contents
    *image += length - 2 - 5;
    return -1;
}

static void
gpujpeg_reader_read_app1(uint8_t** image, struct gpujpeg_reader *reader)
{
    if ( reader->image_end - *image < 2 ) {
        ERROR_MSG("Unexpected end of APP1 marker!\n");
        return;
    }

    uint8_t type_tag[50];
    unsigned i = 0;
    const uint8_t* ptr = *image + 2;
    while ( *ptr != '\0' && ptr < reader->image_end && i < sizeof type_tag  - 1 ) {
        type_tag[i++] = *ptr++;
    }
    type_tag[i] = '\0';
    if ( strcmp((char *) type_tag, "Exif") == 0 ) {
        reader->header_color_space = GPUJPEG_YCBCR_BT601_256LVLS;
        reader->header_type = GPUJPEG_HEADER_EXIF;
        gpujpeg_exif_parse(image, reader->image_end, reader->param.verbose, reader->metadata);
        return;
    }
    WARN_MSG("Skipping unsupported APP1 marker \"%s\"!\n", type_tag);
    gpujpeg_reader_skip_marker_content(image, reader->image_end);
}

/**
 * Read Adobe APP13 marker (used as a segment info by GPUJPEG)
 *
 * @param decoder decoder state
 * @param image   JPEG data
 * @return        0 if succeeds, otherwise nonzero
 *
 * @todo
 * Segment info should have an identifying header like other application markers
 * (see http://www.ozhiker.com/electronics/pjmt/jpeg_info/app_segments.html)
 */
static int
gpujpeg_reader_read_app13(struct gpujpeg_reader* reader, uint8_t** image, const uint8_t* image_end)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read APP13 length (end of data)\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length <= 3 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP13 marker (segment info) length should be greater than 3 but %d was presented!\n",
                length);
        return -1;
    }

    if(image_end - *image < length) {
        fprintf(stderr, "[GPUJPEG] [Error] APP13 marker data goes beyond end of data\n");
        return -1;
    }

    char *ignored_hdrs[] = { "Adobe_Photoshop2.5", "Photoshop 3.0", "Adobe_CM" };
    for (size_t i = 0; i < sizeof ignored_hdrs / sizeof ignored_hdrs[0]; ++i) {
        if ((size_t) length >= 2 + sizeof ignored_hdrs[i] - 1 && memcmp((char *) *image, ignored_hdrs[i], sizeof ignored_hdrs[i] - 1) == 0) {
            if ( reader->param.verbose ) {
                fprintf(stderr, "[GPUJPEG] [Warning] Skipping unsupported %s APP13 marker!\n", ignored_hdrs[i]);
            }
            *image += length - 2;
            return 0;
        }
    }

    // forward compatibility - expect future GPUJPEG version to tag the segment info marker,
    // which is currently not written
    char gpujpeg_str[] = "GPUJPEG"; // with terminating '\0'
    if ((size_t) length >= 2 + sizeof gpujpeg_str && memcmp((char *) image, gpujpeg_str, sizeof gpujpeg_str) == 0) {
        *image += sizeof gpujpeg_str;
        length -= sizeof gpujpeg_str;
        return gpujpeg_reader_read_segment_info(reader, image, image_end, length);
    }
    // suppose segment info marker - current GPUJPEG doesn't tag it
    gpujpeg_reader_read_segment_info(reader, image, image_end, length); // ignore return code since we are
                                                                          // not sure if it is ours
    return 0;
}

static int
gpujpeg_reader_read_spiff_header(uint8_t** image, int verbose, enum gpujpeg_color_space *color_space, _Bool *in_spiff)
{
    int version = gpujpeg_reader_read_2byte(*image); // version
    int profile_id = gpujpeg_reader_read_byte(*image); // profile ID
    int comp_count = gpujpeg_reader_read_byte(*image); // component count
    int width = gpujpeg_reader_read_4byte(*image); // width
    int height = gpujpeg_reader_read_4byte(*image); // height
    int spiff_color_space = gpujpeg_reader_read_byte(*image);
    int bps = gpujpeg_reader_read_byte(*image); // bits per sample
    int compression = gpujpeg_reader_read_byte(*image);
    int pixel_units = gpujpeg_reader_read_byte(*image); // resolution units
    int pixel_xdpu = gpujpeg_reader_read_4byte(*image); // vertical res
    int pixel_ydpu = gpujpeg_reader_read_4byte(*image); // horizontal res
    (void) profile_id, (void) comp_count, (void) width, (void) height, (void) pixel_units, (void) pixel_xdpu, (void) pixel_ydpu;

    if (version != SPIFF_VERSION) {
        VERBOSE_MSG(verbose, "Unknown SPIFF version %d.%d.\n", version >> 8, version & 0xFF);
    }
    if (bps != 8) {
        ERROR_MSG("Wrong bits per sample %d, only 8 is supported.\n", bps);
    }
    if (compression != SPIFF_COMPRESSION_JPEG) {
            ERROR_MSG("Unexpected compression index %d, expected %d (JPEG)\n", compression, SPIFF_COMPRESSION_JPEG);
            return -1;
    }

    switch (spiff_color_space) {
        case 1: // NOLINT
            *color_space = GPUJPEG_YCBCR_BT709;
            break;
        case 2: // NOLINT
            break;
        case 3: // NOLINT
        case 8: /* grayscale */ // NOLINT
            *color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            break;
        case 4: // NOLINT
            *color_space = GPUJPEG_YCBCR_BT601;
            break;
        case 10: // NOLINT
            *color_space = GPUJPEG_RGB;
            break;
        default:
            ERROR_MSG("Unsupported or unrecongnized SPIFF color space %d!\n", spiff_color_space);
            return -1;
    }

    DEBUG_MSG(verbose, "APP8 SPIFF parsed succesfully, internal color space: %s\n", gpujpeg_color_space_get_name(*color_space));
    *in_spiff = 1;

    return 0;
}

/** @retval -1 error
 *  @retval  0 no error */
static int
gpujpeg_reader_read_spiff_directory(uint8_t** image, struct gpujpeg_reader *reader, int length)
{
    if ( length < 8 ) { // ELEN at least 8
        ERROR_MSG("APP8 SPIFF directory too short (%d bytes)\n", length);
        *image += length - 2;
        return -1;
    }
    uint32_t tag = gpujpeg_reader_read_4byte(*image);
    DEBUG2_MSG(reader->param.verbose, "Read SPIFF tag 0x%x with length %d.\n", tag, length);
    if ( tag == SPIFF_ENTRY_TAG_EOD && length == SPIFF_ENTRY_TAG_EOD_LENGHT ) {
        int marker_soi = gpujpeg_reader_read_marker(image, reader->image_end, reader->param.verbose);
        if ( marker_soi != GPUJPEG_MARKER_SOI ) {
            VERBOSE_MSG(reader->param.verbose, "SPIFF entry 0x1 should be followed directly with SOI.\n");
            return -1;
        }
        DEBUG2_MSG(reader->param.verbose, "SPIFF EOD presented.\n");
        reader->in_spiff = false;
        return 0;
    }

    if ( tag >> 24U != 0 ) { // given by the standard
        VERBOSE_MSG(reader->param.verbose, "Erroneous SPIFF tag 0x%x (first byte should be 0).", tag);
        *image += length - 6;
        return 0;
    }
    if ( tag == SPIFF_ENTRY_TAG_ORIENATAION ) {
        int rotation = gpujpeg_reader_read_byte(*image);
        bool flip = gpujpeg_reader_read_byte(*image);
        reader->metadata->vals[GPUJPEG_METADATA_ORIENTATION].orient.rotation = rotation;
        reader->metadata->vals[GPUJPEG_METADATA_ORIENTATION].orient.flip = flip;
        reader->metadata->vals[GPUJPEG_METADATA_ORIENTATION].set = 1;
        DEBUG_MSG(reader->param.verbose, "SPIFF CW rotation: %d deg%s\n", rotation * 90, flip ? ", mirrored" : "");
        *image += 2; // 2 bytes reserved
        return 0;
    }
    DEBUG2_MSG(reader->param.verbose, "Unhandled SPIFF tag 0x%x with length %d presented.\n", tag, length);
    *image += length - 6;
    return 0;
}

/**
 * Read APP8 marker
 *
 * Obtains colorspace from APP8.
 *
 * @param         decoder     decoder state
 * @param[in,out] image       JPEG data
 * @param[out]    color_space detected color space
 * @return        0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_app8(uint8_t** image, struct gpujpeg_reader* reader)
{
    if ( reader->image_end - *image < 2 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read APP8 marker length (end of data)\n");
        return -1;
    }
    int length = gpujpeg_reader_read_2byte(*image);
    length -= 2;
    if ( reader->image_end - *image < length ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP8 marker goes beyond end of data\n");
        return -1;
    }

    if ( reader->in_spiff ) {
        return gpujpeg_reader_read_spiff_directory(image, reader, length + 2);
    }

    if (length + 2 != SPIFF_MARKER_LEN) {
        VERBOSE_MSG(reader->param.verbose, "APP8 segment length is %d, expected 32 for SPIFF.\n", length + 2);
        *image += length;
        return 0;
    }

    const char spiff_marker_name[6] = { 'S', 'P', 'I', 'F', 'F', '\0' };
    char marker_name[sizeof spiff_marker_name];
    for (unsigned i = 0; i < sizeof marker_name; i++) {
        marker_name[i] = gpujpeg_reader_read_byte(*image);
        if (!isprint(marker_name[i])) {
            marker_name[i] = '\0';
        }
    }
    length -= sizeof marker_name;

    if (strcmp(marker_name, spiff_marker_name) != 0) {
        VERBOSE_MSG(reader->param.verbose, "APP8 marker identifier should be 'SPIFF\\0' but '%-6.6s' was presented!\n",
                    marker_name);
        *image += length - 2;
        return 0;
    }

    reader->header_type = GPUJPEG_HEADER_SPIFF;
    return gpujpeg_reader_read_spiff_header(image, reader->param.verbose, &reader->header_color_space,
                                            &reader->in_spiff);
}

/**
 * Reads Adobe APP14 marker
 *
 * Marker length should be 14:
 * https://www.adobe.com/content/dam/acom/en/devnet/postscript/pdfs/5116.DCT_Filter.pdf#G3.851943
 *
 * @todo
 * Some Adobe markers have length 38, see:
 * https://github.com/CESNET/GPUJPEG/issues/70
 * For now we are just skipping the rest.
 *
 * @return        0 if succeeds, >0 if unsupported features were found, <0 on error
 */
static int
gpujpeg_reader_read_adobe_header(uint8_t** image, const uint8_t* image_end, enum gpujpeg_color_space* color_space,
                                 int verbose)
{
    if(image_end - *image < 7) {
        fprintf(stderr, "[GPUJPEG] [Error] APP14 marker goes beyond end of data\n");
        return -1;
    }

    int version = gpujpeg_reader_read_2byte(*image);
    int flags0 = gpujpeg_reader_read_2byte(*image);
    int flags1 = gpujpeg_reader_read_2byte(*image);
    int color_transform = gpujpeg_reader_read_byte(*image);
    DEBUG_MSG(verbose, "APP14 Adobe header - version: %d, flags0: %d, flags1: %d, color_transform: %d\n", version,
              flags0, flags1, color_transform);

    if (color_transform == 0) {
        *color_space = GPUJPEG_RGB;
    } else if (color_transform == 1) {
        *color_space = GPUJPEG_YCBCR_BT601_256LVLS;
    } else if (color_transform == 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Unsupported YCCK color transformation was presented!\n");
        return 1;
    } else {
        fprintf(stderr, "[GPUJPEG] [Error] Unsupported color transformation value '%d' was presented in APP14 marker!\n", color_transform);
        return 1;
    }

    return 0;
}

/**
 * Read Adobe APP14 marker (used for RGB images by GPUJPEG)
 *
 * Obtains colorspace from APP14.
 *
 * @param decoder decoder state
 * @param image   JPEG data
 * @return        0 if succeeds, >0 if unsupported features were found, <0 on error
 */
static int
gpujpeg_reader_read_app14(uint8_t** image, const uint8_t* image_end, enum gpujpeg_color_space* color_space,
                          enum gpujpeg_header_type* header_type, int verbose)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read APP14 marker size (end of data)\n");
        return -1;
    }

    int length = gpujpeg_reader_read_2byte(*image);

    if ( length - 2 > image_end - *image ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP14 segment goes beyond end of data\n");
        return -1;
    }

    const char adobe_tag[] = { 'A', 'd', 'o' ,'b', 'e' };
    if (length >= APP14_ADOBE_MARKER_LEN && strncmp((char *) *image, adobe_tag, sizeof adobe_tag) == 0) {
        *image += sizeof adobe_tag;
        *header_type = GPUJPEG_HEADER_ADOBE;
        int rc = gpujpeg_reader_read_adobe_header(image, image_end, color_space, verbose);
        *image += length - APP14_ADOBE_MARKER_LEN;
        if (length > APP14_ADOBE_MARKER_LEN) {
            fprintf(stderr, "[GPUJPEG] [Warning] APP14 Adobe marker length should be 14 but %d was presented!\n", length);
            rc = rc < 0 ? rc : 1;
        }
        return rc;
    }

    fprintf(stderr, "[GPUJPEG] [Warning] Unknown APP14 marker %dB (%dB) long was presented: ", length, length - 2);
    length -= 2;
    while (length > 0 && isprint(**image)) {
        putc(*(*image)++, stderr);
        length--;
    }
    putc('\n', stderr);
    while (length-- > 0) {
        (*image)++;
    }

    return 0;
}

static int
gpujpeg_reader_read_com(uint8_t** image, struct gpujpeg_reader* reader)
{
    if ( reader->image_end - *image < 2 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read com length\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);

    if ( length < 2 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Invalid COM length %d\n", length);
        return -1;
    }
    if ( length - 2 > reader->image_end - *image ) {
        fprintf(stderr, "[GPUJPEG] [Error] COM goes beyond end of data\n");
        return -1;
    }
    if ( length == 2 ) {
        return 0;
    }

    const char cs_itu601[] = "CS=ITU601";
    const size_t com_length = length - 2; // check both with '\0' and without:
    if ( (com_length == sizeof cs_itu601 || com_length == sizeof cs_itu601 - 1) &&
         strncmp((char*)*image, cs_itu601, com_length) == 0 ) {
        reader->header_color_space = reader->ff_cs_itu601_is_709 ? GPUJPEG_YCBCR_BT709 : GPUJPEG_YCBCR_BT601;
    }

    if ((*image)[com_length - 1] == '\0') {
        reader->comment = (char*)*image;
    } else {
        DEBUG_MSG(reader->param.verbose, "Not storing non-NULL-terminated COM: %.*s", (int)com_length, *image);
    }

    *image += length - 2;

    return 0;
}

/**
 * Read quantization table definition block from image
 *
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_dqt(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader, uint8_t** image,
                        const uint8_t* image_end)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read dqt length\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    length -= 2;

    if(length > image_end - *image) {
        fprintf(stderr, "[GPUJPEG] [Error] DQT marker goes beyond end of data\n");
        return -1;
    }

    if ( (length % 65) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] DQT marker length should be 65 but %d was presented!\n", length);
        return -1;
    }

    for (;length > 0; length-=65) {
        int index = gpujpeg_reader_read_byte(*image);

        int Pq = index >> 4; // precision - 0 for 8-bit, 1 for 16-bit
        int Tq = index & 0xF; // destination (0-3)
        if (Pq != 0 && Pq != 1) {
            fprintf(stderr, "[GPUJPEG] [Error] DQT marker Pq should be 0 or 1 (8 or 16-bit) but %d was presented!\n", Pq);
            return -1;
        }
        if (Pq != 0) {
            fprintf(stderr, "[GPUJPEG] [Error] Unsupported DQT Pq 1 (16-bit table) presented!\n");
            return -1;
        }
        if ( Tq >= GPUJPEG_MAX_COMPONENT_COUNT ) {
            fprintf(stderr, "[GPUJPEG] [Error] DQT marker Tq should be 0-3 but %d was presented!\n", Tq);
            return -1;
        }

        uint8_t table_raw[64];
        for ( int i = 0; i < 64; i++ ) {
            table_raw[i] = gpujpeg_reader_read_byte(*image);
        }

        struct gpujpeg_table_quantization* table = &decoder->table_quantization[Tq];
        const unsigned bit = 1U << Tq;
        if ( reader->batch && (reader->quant_used & bit) != 0 &&
             memcmp(table->table_raw, table_raw, sizeof table_raw) != 0 ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLE_SCOPE, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                               "quantization table changed after its use");
        }
        memcpy(table->table_raw, table_raw, sizeof table_raw);
        reader->quant_defined |= bit;
        if ( reader->batch ) {
            decoder->batch.quant_dirty |= bit; // uploaded after reading all images
        }
        else {
            // Prepare quantization table for read raw table
            gpujpeg_table_quantization_decoder_compute(table, decoder->coder.stream);
        }

        if (decoder->coder.param.verbose >= GPUJPEG_LL_DEBUG2) {
            PRINTF("Quantization table 0x%02x (%d-bit, dst: %d):\n", index, (Pq + 1) * 8, Tq);
            gpujpeg_table_quantization_print(table);
        }
    }
    return 0;
}

static const char *array_serialize(int comp_count, const uint8_t *comp_id) {
    _Thread_local static char buffer[1024] = "[";
    if (comp_count >= 1) {
        snprintf(buffer + strlen(buffer), sizeof buffer - strlen(buffer), "%" PRIu8, comp_id[0]);
    }
    for (int i = 1; i < comp_count; ++i) {
        snprintf(buffer + strlen(buffer), sizeof buffer - strlen(buffer), ",%" PRIu8, comp_id[i]);
    }
    strncat(buffer, "]", sizeof(buffer) - strlen(buffer) - 1);
    return buffer;
}

/**
 * checks component ID to determine if JPEG is in YCbCr (component IDs 1, 2, 3) or RGB (component IDs 'R', 'G', 'B')
 */
static enum gpujpeg_color_space gpujpeg_reader_process_cid(int comp_count, uint8_t *comp_id, enum gpujpeg_color_space header_color_space) {
    static_assert(GPUJPEG_MAX_COMPONENT_COUNT >= 3, "An array of at least 3 components expected");
    static const uint8_t ycbcr_ids[] = { 1, 2, 3 };
    static const uint8_t rgb_ids[] = { 'R', 'G', 'B' };
    static const uint8_t bg_rgb_ids[] = { 'r', 'g', 'b' }; // big gamut sRGB (see ILG libjpeg - seemingly handled as above)
    if (comp_count < 3) {
        return GPUJPEG_NONE;
    }
    if (header_color_space == GPUJPEG_NONE) {
        // avoid comparing alpha
        if (memcmp(comp_id, ycbcr_ids, sizeof ycbcr_ids) == 0) {
            if (comp_count == 4 && comp_id[3] != 4) {
                fprintf(stderr, "[GPUJPEG] [Warning] Unexpected 3rd channel id %d!\n", comp_id[3]);
            }
            return GPUJPEG_YCBCR_BT601_256LVLS;
        }
        if (memcmp(comp_id, rgb_ids, sizeof rgb_ids) == 0 || memcmp(comp_id, bg_rgb_ids, sizeof bg_rgb_ids) == 0) {
            if (comp_count == 4 && toupper(comp_id[3]) != 'R') {
                fprintf(stderr, "[GPUJPEG] [Warning] Unexpected 3rd channel id %d!\n", comp_id[3]);
            }
            return GPUJPEG_RGB;
        }
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 unexpected component id %s was presented!\n", array_serialize(comp_count, comp_id));
        return GPUJPEG_NONE;
    }
    if (header_color_space >= GPUJPEG_YCBCR_BT601 && header_color_space <= GPUJPEG_YCBCR_BT709
            && memcmp(comp_id, ycbcr_ids, sizeof ycbcr_ids) != 0) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker component id should be %s but %s was presented!\n", array_serialize(sizeof ycbcr_ids, ycbcr_ids), array_serialize(comp_count, comp_id));
    }
    if (header_color_space == GPUJPEG_RGB
            && memcmp(comp_id, rgb_ids, sizeof rgb_ids) != 0 && memcmp(comp_id, bg_rgb_ids, sizeof bg_rgb_ids) != 0) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker component id should be %s but %s was presented!\n", array_serialize(sizeof rgb_ids, rgb_ids), array_serialize(comp_count, comp_id));
    }
    return GPUJPEG_NONE;
}

static void
sof0_dump(int comp_count, const struct gpujpeg_component_sampling_factor* sampling_factor, const uint8_t* id,
          const int* map)
{
    PRINTF("SOF0 subsampling:");
    for (int comp = 0; comp < comp_count; ++comp) {
        PRINTF(" %dx%d", sampling_factor[comp].horizontal, sampling_factor[comp].vertical);
    }
    PRINTF("\nSOF0 component quantization tab usage:");
    for (int comp = 0; comp < comp_count; ++comp) {
        PRINTF(" %" PRIu8 "->%d", id[comp], map[comp]);
    }
    PRINTF("\n");
}

/**
 * Read start of frame block from image
 *
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_sof0(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image,
                         enum gpujpeg_color_space header_color_space, enum gpujpeg_header_type header_type,
                         int quant_map[GPUJPEG_MAX_COMPONENT_COUNT], uint8_t comp_id[GPUJPEG_MAX_COMPONENT_COUNT],
                         uint8_t** image, const uint8_t* image_end)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read SOF0 size (end of data)\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length < 8 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker length should be at least 8 but %d was presented!\n", length);
        return -1;
    }
    length -= 2;

    if(length > image_end - *image) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 goes beyond end of data\n");
        return -1;
    }

    int precision = (int)gpujpeg_reader_read_byte(*image);
    param_image->height = (int)gpujpeg_reader_read_2byte(*image);
    param_image->width = (int)gpujpeg_reader_read_2byte(*image);
    param->comp_count = (int)gpujpeg_reader_read_byte(*image);
    length -= 6;

    if ( param->comp_count == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 has 0 components!\n");
        return -1;
    }
    if ( param->comp_count > GPUJPEG_MAX_COMPONENT_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 has %d components but JPEG can contain at most %d components\n",
                param->comp_count, (int)GPUJPEG_MAX_COMPONENT_COUNT);
        return -1;
    }
    if ( precision != 8 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker precision should be 8 but %d was presented!\n", precision);
        return -1;
    }
    if ( param->comp_count == 2 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 with 2 components is not supported!\n");
        return -1;
    }
    if ( length < 3 * param->comp_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker is too short for %d components!\n", param->comp_count);
        return -1;
    }

    for ( int comp = 0; comp < param->comp_count; comp++ ) {
        int id = (int)gpujpeg_reader_read_byte(*image);
        comp_id[comp] = id;

        int sampling = (int)gpujpeg_reader_read_byte(*image);
        param->sampling_factor[comp].horizontal = (sampling >> 4) & 15;
        param->sampling_factor[comp].vertical = (sampling) & 15;
        if ( param->sampling_factor[comp].horizontal == 0 || param->sampling_factor[comp].vertical == 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker contains zero sampling factor!\n");
            return -1;
        }

        int table_index = (int)gpujpeg_reader_read_byte(*image);
        if ( table_index < 0 || table_index > 3 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker contains unexpected quantization table index %d!\n", table_index);
            return -1;
        }
        quant_map[comp] = table_index;
        length -= 3;
    }

    if (param->verbose >= GPUJPEG_LL_DEBUG2) {
        sof0_dump(param->comp_count, param->sampling_factor, comp_id, quant_map);
    }

    // Deduce color space if not known from headers
    enum gpujpeg_color_space detected_color_space = gpujpeg_reader_process_cid(param->comp_count, comp_id, header_color_space);
    if (header_color_space == GPUJPEG_NONE && detected_color_space != GPUJPEG_NONE) {
        VERBOSE_MSG(param->verbose, "Deduced color space %s.\n", gpujpeg_color_space_get_name(detected_color_space));
        param->color_space_internal = detected_color_space;
    }
    if ( header_type == GPUJPEG_HEADER_ADOBE && param->color_space_internal == GPUJPEG_RGB && param->comp_count == 1 ) {
        param->color_space_internal = GPUJPEG_YCBCR_BT601_256LVLS;
    }

    // Check length
    if ( length > 0 ) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker contains %d more bytes than needed!\n", length);
        *image += length;
    }

    return 0;
}

static void
huff_table_dump(int Th, int Tc, const struct gpujpeg_table_huffman_decoder* table)
{
    const char* comp_type = "(unknown)";
    switch ( Th ) {
    case 0:
        comp_type = "lum";
        break;
    case 1:
        comp_type = "chr";
        break;
    }
    PRINTF("table index 0x%02x (Tc: %d /%s/, Th: %d /%s/):\n", Th | (Tc << 4), Tc, Tc == 0 ? "DC" : "AC", Th, comp_type);
    int hi = 0;
    for ( unsigned i = 1; i < sizeof table->bits / sizeof table->bits[0]; ++i ) {
        PRINTF("values per %2u bits - count: %3hhu, list:", i, table->bits[i]);
        for ( int j = hi; j < hi + table->bits[i]; ++j ) {
            PRINTF(" %3hhu", table->huffval[j]);
        }
        hi += table->bits[i];
        PRINTF("\n");
    }
    PRINTF("total: %d\n\n", hi);
}

/// @return count of Huffman codes (symbols) with the given counts per code length
static int
gpujpeg_reader_huffman_count(const uint8_t bits[17])
{
    int count = 0;
    for ( int length = 1; length <= 16; length++ ) {
        count += bits[length];
    }
    return count;
}

/**
 * Checks that the counts per code length define canonical Huffman codes (JPEG Annex C) without an all-ones code
 */
static bool
gpujpeg_reader_huffman_codes_valid(const uint8_t bits[17])
{
    unsigned code = 0; // first code of the length
    for ( int length = 1; length <= 16; length++ ) {
        if ( code + bits[length] >= (1U << length) ) {
            return false;
        }
        code = (code + bits[length]) << 1;
    }
    return true;
}

/**
 * Read huffman table definition block from image
 *
 * @param decoder
 * @param reader   tables defined by the image (reader->huffman_defined) are updated
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_dht(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader, uint8_t** image,
                        const uint8_t* image_end)
{
    if(image_end - *image < 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read DHT size (end of data)\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    length -= 2;

    if(length > image_end - *image) {
        fprintf(stderr, "[GPUJPEG] [Error] DHT goes beyond end of data\n");
        return -1;
    }

    while (length > 0) {
        if ( length < 17 ) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading bit counts!\n");
            return -1;
        }
        int index = gpujpeg_reader_read_byte(*image);
        int Tc = index >> 4; // class - 0 = DC, 1 = AC
        int Th = index & 0xF; // destination (0-1 baseline, 0-3 extended), usually 0 = luminance, 1 = chrominance
        if (Tc != 0 && Tc != 1) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker Tc should be 0 or 1 but %d was presented!\n", Tc);
            return -1;
        }
        if ( Th >= GPUJPEG_MAX_COMPONENT_COUNT ) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker Th should be 0-3 but %d was presented!\n", Th);
            return -1;
        }

        // Read in bits[] and huffval, validated before they are stored
        uint8_t bits[17] = {0};
        for ( int i = 1; i <= 16; i++ ) {
            bits[i] = gpujpeg_reader_read_byte(*image);
        }
        length -= 17;
        const int count = gpujpeg_reader_huffman_count(bits);
        if ( count > 256 || count > length ) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading %d huffman values!\n", count);
            return -1;
        }
        uint8_t huffval[256];
        for ( int i = 0; i < count; i++ ){
            huffval[i] = gpujpeg_reader_read_byte(*image);
        }
        length -= count;
        if ( !gpujpeg_reader_huffman_codes_valid(bits) ) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker defines invalid Huffman codes!\n");
            return -1;
        }

        struct gpujpeg_table_huffman_decoder* table = &decoder->table_huffman[Th][Tc];
        const unsigned bit = 1U << (Th * 2 + Tc);
        if ( reader->batch && (reader->huffman_used & bit) != 0 &&
             (memcmp(table->bits, bits, sizeof bits) != 0 || memcmp(table->huffval, huffval, count) != 0) ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLE_SCOPE, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                               "Huffman table changed after its use");
        }
        memcpy(table->bits, bits, sizeof bits);
        memcpy(table->huffval, huffval, count);
        // Compute huffman table for read values
        gpujpeg_table_huffman_decoder_compute(table);
        reader->huffman_defined |= bit;

        if (decoder->coder.param.verbose >= GPUJPEG_LL_DEBUG2) {
            huff_table_dump(Th, Tc, table);
        }

        if ( reader->batch ) {
            decoder->batch.huffman_dirty |= bit; // uploaded after reading all images
            continue;
        }
        // Copy table to device memory
        cudaMemcpyAsync(decoder->d_table_huffman[Th][Tc], table, sizeof(struct gpujpeg_table_huffman_decoder),
                        cudaMemcpyHostToDevice, decoder->coder.stream);
        gpujpeg_cuda_check_error("Decoder copy huffman table ", return -1);
    }
    return 0;
}

/**
 * Read restart interval block from image
 *
 * @param[in,out] out_restart_interval output value (checked if changed)
 * @param image
 * @retval @ref Errors
 */
static int
gpujpeg_reader_read_dri(int *out_restart_interval, uint8_t** image, const uint8_t* image_end)
{
    if(image_end - *image < 4) {
        fprintf(stderr, "[GPUJPEG] [Error] Could not read DRI (end of data)\n");
        return -1;
    }

    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length != 4 ) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker length should be 4 but %d was presented!\n", length);
        return -1;
    }

    int restart_interval = gpujpeg_reader_read_2byte(*image);
    if ( restart_interval == *out_restart_interval ) {
        return 0;
    }

    if ( *out_restart_interval != 0
            && *out_restart_interval != restart_interval ) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker can't redefine restart interval (%d to %d)!\n",
               *out_restart_interval, restart_interval );
        fprintf(stderr, "This may be caused when more DRI markers are presented which is not supported!\n");
        return GPUJPEG_ERR_RESTART_CHANGE;
    }

    *out_restart_interval = restart_interval;

    return 0;
}

/**
 * Stores a closed segment of the scan being read: its descriptor and entropy-coded data
 *
 * @param data   segment data in the image
 * @param index  destination of the data in the decoder buffer for compressed data
 * @return 0 if succeeds, otherwise nonzero (more segments or data than the decoder can hold)
 */
// inlined into the loop of gpujpeg_reader_read_scan_content_by_parsing() (compilers may keep it a call otherwise)
#if defined _MSC_VER
static __forceinline int
#elif defined __GNUC__
static inline __attribute__((always_inline)) int
#else
static inline int
#endif
gpujpeg_reader_put_segment(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader,
                           struct gpujpeg_reader_scan* scan, int scan_index, const uint8_t* data, size_t index,
                           size_t size)
{
    const int segment_index = scan->segment_index + scan->segment_count;
    if ( segment_index >= reader->segment_limit ) {
        if ( reader->batch ) {
            decoder->batch.reason = GPUJPEG_BATCH_REASON_SEGMENTS;
        }
        fprintf(stderr, "[GPUJPEG] [Error] JPEG scan contains more segments than expected!\n");
        return -1;
    }
    if ( index > reader->compressed_limit || size > reader->compressed_limit - index ) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG scan data exceed the decoder buffer!\n");
        return -1;
    }
    struct gpujpeg_segment* segment = &decoder->coder.segment[segment_index];
    segment->scan_index = scan_index;
    segment->scan_segment_index = scan->segment_count;
    segment->data_compressed_index = index;
    segment->data_compressed_size = size;
    memcpy(&decoder->coder.data_compressed[index], data, size);
    scan->segment_count++;
    return 0;
}

/**
 * Read scan content by parsing byte-by-byte
 *
 * @param decoder
 * @param image
 * @param image_end
 * @param scan
 * @param scan_index
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_scan_content_by_parsing(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader,
                                            uint8_t** image, const uint8_t* image_end, struct gpujpeg_reader_scan* scan,
                                            int scan_index)
{
    // destination offset of the data read (the source position minus skipped markers)
    size_t data_compressed_offset = reader->data_compressed_size;

    // First segment in scan (its descriptor is written when the segment is closed)
    uint8_t * segment_data_start = *image;
    size_t segment_data_index = data_compressed_offset;

    // Read scan data
    int result = -1;
    uint8_t previous_marker = GPUJPEG_MARKER_RST0 - 1;
    do {
        uint8_t *ret = memchr(*image, 0xFF, image_end - *image);
        if (ret == NULL || ret == image_end - 1) {
            data_compressed_offset += image_end - *image;
            *image = (uint8_t *) image_end;
            break;
        }
        data_compressed_offset += ret - *image + 2;
        *image = ret + 1;
        uint8_t marker = *(*image)++;
        // Check zero byte
        if (marker == 0) {
            continue;
        }
        // Check restart marker
        if ( marker >= GPUJPEG_MARKER_RST0 && marker <= GPUJPEG_MARKER_RST7 ) {
            // Check expected marker
            uint8_t expected_marker = (previous_marker < GPUJPEG_MARKER_RST7) ? (previous_marker + 1) : GPUJPEG_MARKER_RST0;
            if ( expected_marker != marker ) {
                fprintf(stderr, "[GPUJPEG] [Error] Expected marker 0x%X but 0x%X was presented!\n", expected_marker, marker);
                if ( reader->batch ) {
                    decoder->batch.reason = GPUJPEG_BATCH_REASON_SEGMENTS;
                    return -1;
                }

                // Skip bytes to expected marker
                int found_expected_marker = 0;
                size_t skip_count = 0;
                while ( *image < image_end ) {
                    uint8_t *ret = memchr(*image, 0xFF, image_end - *image);
                    if (ret == NULL || ret == image_end - 1) {
                        break;
                    }
                    skip_count += ret - *image + 2;
                    *image = ret + 1;
                    marker = *(*image)++;
                    if ( marker == expected_marker ) {
                        fprintf(stderr, "[GPUJPEG] [Recovery] Skipping %zd bytes of data until marker 0x%X was found!\n", skip_count, expected_marker);
                        found_expected_marker = 1;
                        break;
                    }
                    if ( marker == GPUJPEG_MARKER_EOI || marker == GPUJPEG_MARKER_SOS ) {
                        // Go back last marker (will be read again by main read cycle)
                        *image -= 2;
                        break;
                    }
                }

                // If expected marker was not found to end of stream
                if ( found_expected_marker == 0 ) {
                    fprintf(stderr, "[GPUJPEG] [Error] No marker 0x%X was found until end of current scan!\n", expected_marker);
                    continue;
                }
            }
            // Set previous marker
            previous_marker = marker;

            // Close the segment
            data_compressed_offset -= 2;
            if ( gpujpeg_reader_put_segment(decoder, reader, scan, scan_index, segment_data_start, segment_data_index,
                                            data_compressed_offset - segment_data_index) != 0 ) {
                return -1;
            }

            // Start new segment in scan
            segment_data_start = *image;
            segment_data_index = data_compressed_offset;
        }
        // Check scan end (a batch image may also define tables, restart interval or comments between scans, checked
        // by the outer reader)
        else if ( marker == GPUJPEG_MARKER_EOI || marker == GPUJPEG_MARKER_SOS ||
                  (marker >= GPUJPEG_MARKER_APP0 && marker <= GPUJPEG_MARKER_APP15) ||
                  (reader->batch && (marker == GPUJPEG_MARKER_DHT || marker == GPUJPEG_MARKER_DQT ||
                                     marker == GPUJPEG_MARKER_DRI || marker == GPUJPEG_MARKER_COM)) ) {
            *image -= 2;

            // Close the last segment
            data_compressed_offset -= 2;
            if ( data_compressed_offset ==
                 segment_data_index ) { // skip FFMPEG empty segments after last RST before EOF (FF bug #8412)
                VERBOSE_MSG(decoder->coder.param.verbose, "Empty segment detected!\n");
            }
            else if ( gpujpeg_reader_put_segment(decoder, reader, scan, scan_index, segment_data_start,
                                                 segment_data_index,
                                                 data_compressed_offset - segment_data_index) != 0 ) {
                return -1;
            }

            // Add scan segment count to decoder segment count
            reader->segment_count += scan->segment_count;

            // Successfully read end of scan, so the result is OK
            result = 0;
            break;
        }
        else {
            fprintf(stderr, "[GPUJPEG] [Error] JPEG scan contains unexpected marker 0x%X!\n", marker);
            return -1;
        }
    } while( *image < image_end );

    reader->data_compressed_size = data_compressed_offset;

    if ( result == -1) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data unexpected ended while reading SOS marker!\n");
    }
    return result;
}

/**
 * Read scan content by segment info contained inside the JPEG stream
 *
 * @param decoder
 * @param image
 * @param image_end
 * @param scan
 * @param scan_index
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_scan_content_by_segment_info(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader,
                                                 uint8_t** image, const uint8_t* image_end,
                                                 struct gpujpeg_reader_scan* scan, int scan_index)
{
    // Calculate segment count
    int segment_count = reader->segment_info_size / 4 - 1;

    // Read first record from segment info, which means beginning of the first segment
    int scan_start = (reader->segment_info[0][0] << 24)
        + (reader->segment_info[0][1] << 16)
        + (reader->segment_info[0][2] << 8)
        + (reader->segment_info[0][3] << 0);

    // Read all segments from segment info
    for ( int segment_index = 0; segment_index < segment_count; segment_index++ ) {
        // Determine header index
        int header_index = ((segment_index + 1) * 4) / GPUJPEG_MAX_HEADER_SIZE;

        // Determine header data index
        int header_data_index = ((segment_index + 1) * 4) % GPUJPEG_MAX_HEADER_SIZE;

        // Determine segment ending in the scan
        int scan_end = (reader->segment_info[header_index][header_data_index + 0] << 24)
            + (reader->segment_info[header_index][header_data_index + 1] << 16)
            + (reader->segment_info[header_index][header_data_index + 2] << 8)
            + (reader->segment_info[header_index][header_data_index + 3] << 0);

        // Setup segment
        struct gpujpeg_segment* segment = &decoder->coder.segment[scan->segment_index + segment_index];
        segment->scan_index = scan_index;
        segment->scan_segment_index = segment_index;
        segment->data_compressed_index = reader->data_compressed_size + scan_start;
        segment->data_compressed_size = reader->data_compressed_size + scan_end - segment->data_compressed_index;

        // If segment is not last it contains restart marker at the end so remove it
        if ( (segment_index + 1) < segment_count ) {
            segment->data_compressed_size -= 2;
        }

        // Move info for next segment
        scan_start = scan_end;
    }

    // Set segment count in the scan
    scan->segment_count = segment_count;

    // Increase number of segment count in reader
    reader->segment_count += scan->segment_count;

    if(*image + scan_start >= image_end) {
        fprintf(stderr, "[GPUJPEG] [Error] scan data goes beyond end of data\n");
        return -1;
    }

    // Copy scan data to buffer
    memcpy(&decoder->coder.data_compressed[reader->data_compressed_size], *image, scan_start);
    *image += scan_start;
    reader->data_compressed_size += scan_start;

    // Reset segment info, for next scan it has to be loaded again from other header
    reader->segment_info_count = 0;
    reader->segment_info_size = 0;

    return 0;
}

static void
sos_check_dump(int verbose, int comp_count, int Ss, int Se, int Ah, int Al)
{
    bool invalid_val = false;
    if (Ss != 0 || Se != 63 || Ah != 0 || Al != 0) {
        WARN_MSG("Some of SOS parameters not valid for sequential DCT.\n");
        invalid_val = true;
    }
    if (!invalid_val && verbose < GPUJPEG_LL_DEBUG2) {
        return;
    }
    printf("SOS components=%d Ss=%d Se=%d Ah=%d Al=%d\n", comp_count, Ss, Se, Ah, Al);
}

/**
 * Batch: checks the frame of an image before its first scan, builds the layout of the batch for the first image
 * and compares the frame of the other ones with it
 *
 * @retval @ref Errors
 */
static int
gpujpeg_reader_batch_layout(struct gpujpeg_decoder* decoder, const struct gpujpeg_reader* reader)
{
    const struct gpujpeg_parameters* param = &reader->param;
    // resolved output (the requested one was checked by gpujpeg_decoder_decode_batch()), concerns the whole call
    if ( gpujpeg_pixel_format_get_comp_count(reader->param_image.pixel_format) == 0 ||
         reader->param_image.width_padding < 0 ) {
        decoder->batch.init_failed = true;
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_DOMAIN, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                           "output format not supported");
    }
    // sampling factors 1 to 4 dividing the maximal ones (expected by the postprocessor)
    int max_h = 1;
    int max_v = 1;
    for ( int comp = 0; comp < param->comp_count; comp++ ) {
        max_h = param->sampling_factor[comp].horizontal > max_h ? param->sampling_factor[comp].horizontal : max_h;
        max_v = param->sampling_factor[comp].vertical > max_v ? param->sampling_factor[comp].vertical : max_v;
    }
    for ( int comp = 0; comp < param->comp_count; comp++ ) {
        const int h = param->sampling_factor[comp].horizontal;
        const int v = param->sampling_factor[comp].vertical;
        if ( max_h > 4 || max_v > 4 || max_h % h != 0 || max_v % v != 0 ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_DOMAIN, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                               "sampling factors not supported");
        }
    }
    if ( decoder->batch.image != 0 ) {
        if ( gpujpeg_decoder_layout_differs(&decoder->coder, param, &reader->param_image) ) {
            return gpujpeg_reader_batch_reject(
                decoder, GPUJPEG_BATCH_REASON_LAYOUT, GPUJPEG_ERR_BATCH_MISMATCH,
                "size, sampling, restart interval or format differ from the first image");
        }
        return GPUJPEG_NOERR;
    }
    // Save requested pixfmt + color_space
    const enum gpujpeg_color_space saved_color_space = decoder->req_color_space;
    const enum gpujpeg_pixel_format saved_pixel_format = decoder->req_pixel_format;
    const int rc = gpujpeg_decoder_init_layout(decoder, param, &reader->param_image, decoder->batch.count, true);
    gpujpeg_decoder_set_output_format(decoder, saved_color_space, saved_pixel_format);
    if ( rc != 0 ) {
        decoder->batch.init_failed = true;
        return GPUJPEG_ERROR;
    }
    return GPUJPEG_NOERR;
}

/**
 * Read start of scan block from image
 *
 * @param decoder
 * @param image
 * @param image_end
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_sos(struct gpujpeg_decoder* decoder, struct gpujpeg_reader* reader, uint8_t** image,
                        const uint8_t* image_end)
{
    if ( image_end - *image < 3 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS goes beyond end of data\n");
        return -1;
    }

    const int length = (int)gpujpeg_reader_read_2byte(*image);
    const int comp_count = (int)gpujpeg_reader_read_byte(*image);
    if (length != comp_count * 2 + 6) {
        fprintf(stderr, "[GPUJPEG] [Error] Wrong SOS length (expected %d, got %d)\n", comp_count * 2 + 6, length);
        return -1;
    }
    if ( length - 3 > image_end - *image ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS goes beyond end of data\n");
        return -1;
    }
    if ( reader->batch ) {
        // a single scan of all components or one scan per component (component order checked below)
        if ( reader->sof_count != 1 || reader->scan_count >= reader->param.comp_count ||
             (reader->scan_count > 0 && (comp_count != 1 || reader->param.interleaved == 1)) ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                               "unsupported frame or scan structure");
        }
        const uint8_t* const scan_param = *image + 2 * comp_count; // Ss, Se, Ah/Al
        if ( scan_param[0] != 0 || scan_param[1] != 63 || scan_param[2] != 0 ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERROR,
                                               "SOS parameters not valid for sequential DCT");
        }
    }
    // Not interleaved mode
    if ( comp_count == 1 ) {
        reader->param.interleaved = 0;
    }
    // Interleaved mode
    else if ( comp_count == reader->param.comp_count ) {
        if ( reader->comp_count != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported for multiple scans!\n", comp_count);
            return -1;
        }
        reader->param.interleaved = 1;
    }
    // Unknown mode
    else {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported (should be 1 or equals to total component count)!\n", comp_count);
        return -1;
    }

    // Check maximum component count
    const bool first_scan = reader->comp_count == 0;
    reader->comp_count += comp_count;
    if ( reader->comp_count > reader->param.comp_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count for all scans %d exceeds maximum component count %d!\n",
            reader->comp_count, reader->param.comp_count);
    }

    // Collect the component-spec parameters
    bool resync_scan = reader->scan_count == 0 && comp_count == reader->param.comp_count;
    int scan_comp[GPUJPEG_MAX_COMPONENT_COUNT]; // component indices of the scan
    for ( int comp = 0; comp < comp_count; comp++ )
    {
        int comp_id = (int)gpujpeg_reader_read_byte(*image);
        int table = (int)gpujpeg_reader_read_byte(*image);
        int table_dc = (table >> 4) & 15;
        int table_ac = table & 15;

        int component_index = -1;
        for ( int i = 0; i < reader->param.comp_count; ++i ) {
            if (decoder->comp_id[i] == comp_id) {
                component_index = i;
                break;
            }
        }
        if ( component_index == -1 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Unexpected component ID '%d' present SOS marker (not defined by SOF marker)!\n", comp_id);
            return -1;
        }
        if ( table_dc >= GPUJPEG_MAX_COMPONENT_COUNT || table_ac >= GPUJPEG_MAX_COMPONENT_COUNT ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker Huffman table indices (%d, %d) should be 0-3!\n", table_dc,
                    table_ac);
            return -1;
        }
        if ( reader->batch && component_index != (comp_count == 1 ? reader->scan_count : comp) ) {
            return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                               "scan components not in frame order");
        }
        scan_comp[comp] = component_index;

        decoder->comp_table_huffman_map[component_index][GPUJPEG_HUFFMAN_DC] = table_dc;
        decoder->comp_table_huffman_map[component_index][GPUJPEG_HUFFMAN_AC] = table_ac;
        // components in SOF order, tables defined by this image before the scan
        resync_scan = resync_scan && component_index == comp && table_dc <= 1 && table_ac <= 1 &&
                      (reader->huffman_defined >> (table_dc * 2 + GPUJPEG_HUFFMAN_DC) & 1U) != 0 &&
                      (reader->huffman_defined >> (table_ac * 2 + GPUJPEG_HUFFMAN_AC) & 1U) != 0;

        if ( decoder->coder.param.verbose >= GPUJPEG_LL_DEBUG2 ) {
            printf("SOS component #%d table DC: %d table AC: %d\n", comp_id, table_dc, table_ac);
        }
    }

    // Collect the additional scan parameters Ss, Se, Ah/Al.
    if ( image_end - *image < 3 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS goes beyond end of data\n");
        return -1;
    }

    int Ss = (int)gpujpeg_reader_read_byte(*image);
    int Se = (int)gpujpeg_reader_read_byte(*image);
    int Ax = (int)gpujpeg_reader_read_byte(*image);
    int Ah = (Ax >> 4) & 15;
    int Al = (Ax) & 15;
    sos_check_dump(decoder->coder.param.verbose, comp_count, Ss, Se, Ah, Al);

    // Check maximum scan count
    if ( reader->scan_count >= GPUJPEG_MAX_COMPONENT_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker reached maximum number of scans (3)!\n");
        return -1;
    }

    // Tables of the scan: defined by the image (batch) and not empty
    for ( int comp = 0; comp < comp_count; comp++ ) {
        const int quant = decoder->comp_table_quantization_map[scan_comp[comp]];
        const int table_dc = decoder->comp_table_huffman_map[scan_comp[comp]][GPUJPEG_HUFFMAN_DC];
        const int table_ac = decoder->comp_table_huffman_map[scan_comp[comp]][GPUJPEG_HUFFMAN_AC];
        const unsigned huffman = 1U << (table_dc * 2 + GPUJPEG_HUFFMAN_DC) | 1U << (table_ac * 2 + GPUJPEG_HUFFMAN_AC);
        if ( reader->batch ) {
            if ( (reader->quant_defined & 1U << quant) == 0 || (reader->huffman_defined & huffman) != huffman ) {
                return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLE_SCOPE,
                                                   GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                                   "table used but not defined before by the image");
            }
            reader->quant_used |= 1U << quant;
            reader->huffman_used |= huffman;
        }
        if ( gpujpeg_reader_huffman_count(decoder->table_huffman[table_dc][GPUJPEG_HUFFMAN_DC].bits) == 0 ||
             gpujpeg_reader_huffman_count(decoder->table_huffman[table_ac][GPUJPEG_HUFFMAN_AC].bits) == 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker uses an empty Huffman table!\n");
            return -1;
        }
    }

    // We must init decoder before data is loaded into it
    if ( first_scan ) {
        if ( reader->param_image.width == 0 || reader->param_image.height == 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Image size %dx%d is not supported (missing SOF or DNL)!\n",
                    reader->param_image.width, reader->param_image.height);
            return -1;
        }
        if ( reader->batch ) {
            const int rc = gpujpeg_reader_batch_layout(decoder, reader);
            if ( rc != 0 ) {
                return rc;
            }
        }
        else {
            // Save requested pixfmt + color_space
            const enum gpujpeg_color_space saved_color_space = decoder->req_color_space;
            const enum gpujpeg_pixel_format saved_pixel_format = decoder->req_pixel_format;
            // Init decoder
            const int rc = gpujpeg_decoder_init(decoder, &reader->param, &reader->param_image);
            gpujpeg_decoder_set_output_format(decoder, saved_color_space, saved_pixel_format);
            if ( rc != GPUJPEG_NOERR ) {
                return -1;
            }
        }
        decoder->coder.init_end_time = decoder->coder.param.perf_stats ? gpujpeg_get_time() : 0;
        reader->segment_limit = decoder->coder.segment_count;
        reader->compressed_limit = decoder->coder.data_compressed_allocated_size > GPUJPEG_DECODER_COMPRESSED_TAIL
                                       ? decoder->coder.data_compressed_allocated_size - GPUJPEG_DECODER_COMPRESSED_TAIL
                                       : 0;
    }
    if ( reader->batch && reader->param.restart_interval != decoder->coder.param.restart_interval ) {
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                           "restart interval changed between scans");
    }

    int scan_index = reader->scan_count;

    // Get scan structure
    struct gpujpeg_reader_scan* scan = &reader->scan[scan_index];
    reader->scan_count++;
    // Scan segments begin at the end of previous scan segments or from zero index
    scan->segment_index = reader->segment_count;
    scan->segment_count = 0;
    int expected_segments = 0; // batch: exactly
    if ( reader->batch ) {
        expected_segments = reader->param.interleaved == 1 ? decoder->coder.segment_count
                                                           : decoder->coder.component[scan_comp[0]].segment_count;
        reader->segment_limit = scan->segment_index + expected_segments;
    }

    // Read scan content
    if ( reader->segment_info_count > 0 ) { // never in batch
        // Read scan content by segment info contained in special header
        if ( gpujpeg_reader_read_scan_content_by_segment_info(decoder, reader, image, image_end, scan, scan_index) != 0 )
            return -1;
    }
    else {
        // Read scan content byte-by-byte
        const uint8_t* const scan_begin = *image;
        if ( gpujpeg_reader_read_scan_content_by_parsing(decoder, reader, image, image_end, scan, scan_index) != 0 )
            return -1;
        // sequential SOF0 scan copied as one segment without restart markers or skipped bytes
        reader->resync_scan = resync_scan && reader->sof_count == 1 && !reader->sof1 && Ss == 0 && Se == 63 &&
                              Ax == 0 && reader->param.restart_interval == 0 && scan->segment_count == 1 &&
                              reader->data_compressed_size == (size_t)(*image - scan_begin);
    }
    if ( reader->batch && scan->segment_count != expected_segments ) {
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SEGMENTS, GPUJPEG_ERROR,
                                           "restart segment count of a scan differs from the frame");
    }

    return 0;
}

/**
 * Common handling for gpujpeg_reader_read_image() and gpujpeg_reader_get_image_info()
 *
 * @retval -1 error
 * @retval @ref Errors
 */
static int
gpujpeg_reader_read_common_markers(uint8_t** image, int marker, struct gpujpeg_reader *reader)
{
    int rc = 0;
    switch (marker)
    {
        case GPUJPEG_MARKER_APP0:
            if ( gpujpeg_reader_read_app0(image, reader->image_end, &reader->header_type, reader->param.verbose) == 0 ) {
                reader->header_color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            }
            break;
        case GPUJPEG_MARKER_APP1:
            gpujpeg_reader_read_app1(image, reader);
            break;
        case GPUJPEG_MARKER_APP8:
            if ( gpujpeg_reader_read_app8(image, reader) != 0 ) {
                return -1;
            }
            break;
        case GPUJPEG_MARKER_APP14:
            if ( gpujpeg_reader_read_app14(image, reader->image_end, &reader->header_color_space, &reader->header_type,
                                           reader->param.verbose) < 0 ) {
                return -1;
            }
            break;
        case GPUJPEG_MARKER_APP2:
        case GPUJPEG_MARKER_APP3:
        case GPUJPEG_MARKER_APP4:
        case GPUJPEG_MARKER_APP5:
        case GPUJPEG_MARKER_APP6:
        case GPUJPEG_MARKER_APP7:
        case GPUJPEG_MARKER_APP9:
        case GPUJPEG_MARKER_APP10:
        case GPUJPEG_MARKER_APP11:
        case GPUJPEG_MARKER_APP12:
        case GPUJPEG_MARKER_APP15:
            if ( reader->param.verbose > 0 ) {
                fprintf(stderr, "[GPUJPEG] [Warning] JPEG data contains not supported %s marker\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            }
            if ( gpujpeg_reader_skip_marker_content(image, reader->image_end) != 0 ) {
                return -1;
            }
            break;
        case GPUJPEG_MARKER_DRI:
            if ( (rc = gpujpeg_reader_read_dri(&reader->param.restart_interval, image, reader->image_end)) != 0 ) {
                return rc;
            }
            break;

        case GPUJPEG_MARKER_SOF2:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF2 (Progressive with Huffman coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF3:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF3 (Lossless with Huffman coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF5:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF5 (Differential sequential with Huffman coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF6:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF6 (Differential progressive with Huffman coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF7:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF7 (Extended lossless with Arithmetic coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_JPG:
            fprintf(stderr, "[GPUJPEG] [Error] Marker JPG (Reserved for JPEG extensions ) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF10:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF10 (Progressive with Arithmetic coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF11:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF11 (Lossless with Arithmetic coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF13:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF13 (Differential sequential with Arithmetic coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF14:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF14 (Differential progressive with Arithmetic coding) is not supported!\n");
            return -1;
        case GPUJPEG_MARKER_SOF15:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF15 (Differential lossless with Arithmetic coding) is not supported!\n");
            return -1;

        case GPUJPEG_MARKER_COM:
            if ( gpujpeg_reader_read_com(image, reader) != 0 ) {
                return -1;
            }
            break;
        default:
            return 1;
    }
    return 0;
}

static _Bool sampling_factor_compare(int count, struct gpujpeg_component_sampling_factor *a,
    struct gpujpeg_component_sampling_factor *b) {
    for (int comp = 0; comp < count; comp++) {
        if (a[comp].vertical != b[comp].vertical ||
                a[comp].horizontal != b[comp].horizontal) {
            return 0;
        }
    }
    return 1;
}

static int
gcd(int a, int b)
{
    assert(b != 0);
    int c = a % b;
    while (c != 0) {
        a = b;
        b = c;
        c = a % b;
    }

    return (b);
}

static enum gpujpeg_pixel_format
get_native_pixel_format(struct gpujpeg_parameters* param)
{
    if ( param->comp_count == 4 ) {
        return GPUJPEG_4444_U8_P0123;
    }

    assert(param->comp_count == 3);

    // reduce [2, 2; 1, 2; 1, 2] (FFmpeg) to [2, 1; 1, 1; 1, 1]
    int horizontal_gcd = param->sampling_factor[0].horizontal;
    int vertical_gcd = param->sampling_factor[0].vertical;
    for ( int i = 1; i < 3; ++i ) {
        horizontal_gcd = gcd(horizontal_gcd, param->sampling_factor[i].horizontal);
        vertical_gcd = gcd(vertical_gcd, param->sampling_factor[i].vertical);
    }
    for ( int i = 0; i < 3; ++i ) {
        param->sampling_factor[i].horizontal /= horizontal_gcd;
        param->sampling_factor[i].vertical /= vertical_gcd;
    }

    // handle normal 4:4:4, 4:2:2 or 4:2:0
    if ( param->sampling_factor[1].horizontal == 1 && param->sampling_factor[1].vertical == 1 &&
         param->sampling_factor[2].horizontal == 1 && param->sampling_factor[2].vertical == 1 ) {
        int sum = param->interleaved << 16 | param->sampling_factor[0].horizontal << 8 |
                  param->sampling_factor[0].vertical; // NOLINT
        switch ( sum ) {
            // clang-format off
            case 1<<16 | 1<<8 | 1: return GPUJPEG_444_U8_P012; break;   // NOLINT
            case 0<<16 | 1<<8 | 1: return GPUJPEG_444_U8_P0P1P2; break; // NOLINT
            case 1<<16 | 2<<8 | 1: return GPUJPEG_422_U8_P1020; break;  // NOLINT
            case 0<<16 | 2<<8 | 1: return GPUJPEG_422_U8_P0P1P2; break; // NOLINT
            case 1<<16 | 2<<8 | 2: // we have only one pixfmt for 420, so use for both          NOLINT
            case 0<<16 | 2<<8 | 2: return GPUJPEG_420_U8_P0P1P2; break; // NOLINT
            default: break;
            // clang-format on
        }
    }

    // fallback
    return param->interleaved ? GPUJPEG_444_U8_P012 : GPUJPEG_444_U8_P0P1P2;
}

static enum gpujpeg_pixel_format
adjust_pixel_format(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image) {
    assert(param_image->pixel_format == GPUJPEG_PIXFMT_AUTODETECT || param_image->pixel_format == GPUJPEG_PIXFMT_NO_ALPHA ||
           param_image->pixel_format == GPUJPEG_PIXFMT_STD || param_image->pixel_format == GPUJPEG_PIXFMT_NATIVE);
    GPUJPEG_ASSERT(param->comp_count != 2);

    if ( param->comp_count == 1 ) {
        return GPUJPEG_U8;
    }

    if ( param_image->pixel_format == GPUJPEG_PIXFMT_NATIVE ) {
        return get_native_pixel_format(param);
    }

    if (param_image->pixel_format == GPUJPEG_PIXFMT_STD && param_image->color_space != GPUJPEG_RGB) {
        struct gpujpeg_parameters tmp;
        gpujpeg_parameters_chroma_subsampling(&tmp, GPUJPEG_SUBSAMPLING_420);
        if (sampling_factor_compare(3, param->sampling_factor, tmp.sampling_factor)) {
            return GPUJPEG_420_U8_P0P1P2;
        }
        gpujpeg_parameters_chroma_subsampling(&tmp, GPUJPEG_SUBSAMPLING_422);
        if (sampling_factor_compare(3, param->sampling_factor, tmp.sampling_factor)) {
            return GPUJPEG_422_U8_P0P1P2;
        }
        return GPUJPEG_444_U8_P0P1P2;
    }

    if ( param->comp_count == 3 ) {
        return GPUJPEG_444_U8_P012;
    }
    assert(param->comp_count == 4);
    return param_image->pixel_format == GPUJPEG_PIXFMT_NO_ALPHA ? GPUJPEG_444_U8_P012 : GPUJPEG_4444_U8_P0123;
}

/**
 * Adjust the unbound/unset parameters to match JPEG image properties.
 *
 * The main reason is to adnust the @ref gpujpeg_parameters and @ref gpujpeg_image_parameters
 * to be consistend and/or match user requirements.
 */
static void
adjust_params(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image,
              enum gpujpeg_color_space color_space_internal, unsigned req_alignment_b)
{
    static_assert(GPUJPEG_PIXFMT_AUTODETECT < 0, "enum gpujpeg_pixel_format type should be signed");
    if ( param_image->color_space == GPUJPEG_NONE) {
        param_image->color_space = color_space_internal;
    }

    if ( param_image->color_space == GPUJPEG_CS_DEFAULT ) {
        if ( param_image->pixel_format == GPUJPEG_U8 ||
             (param_image->pixel_format <= GPUJPEG_PIXFMT_AUTODETECT && param->comp_count == 1) ) {
            param_image->color_space = GPUJPEG_YCBCR_JPEG;
        }
        else {
            param_image->color_space = GPUJPEG_RGB;
        }
    }
    if ( param_image->pixel_format <= GPUJPEG_PIXFMT_AUTODETECT ) {
        param_image->pixel_format = adjust_pixel_format(param, param_image);
    }
    if (req_alignment_b != 0) {
        const unsigned linesize_b = gpujpeg_pixel_format_get_unit_size(param_image->pixel_format) * param_image->width;
        const unsigned aligned_linesize_b = (linesize_b + req_alignment_b - 1) / req_alignment_b * req_alignment_b;
        param_image->width_padding = (int) (aligned_linesize_b - linesize_b);
    }
}

/**
 * Batch: checks the scans of a read image and compares its components and used tables with the first image (or
 * records them for the first image)
 *
 * @retval @ref Errors
 */
static int
gpujpeg_reader_batch_finish(struct gpujpeg_decoder* decoder, const struct gpujpeg_reader* reader)
{
    struct gpujpeg_decoder_batch* batch = &decoder->batch;
    const int comp_count = reader->param.comp_count;
    if ( reader->sof_count != 1 || reader->scan_count == 0 ||
         reader->scan_count != (reader->param.interleaved == 1 ? 1 : comp_count) ) {
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                           "not a scan of all components or one scan per component");
    }
    if ( batch->image == 0 ) {
        memcpy(batch->comp_id, decoder->comp_id, comp_count * sizeof batch->comp_id[0]);
        memcpy(batch->quant_map, decoder->comp_table_quantization_map, comp_count * sizeof batch->quant_map[0]);
        memcpy(batch->huffman_map, decoder->comp_table_huffman_map, comp_count * sizeof batch->huffman_map[0]);
        batch->quant_used = reader->quant_used;
        batch->huffman_used = reader->huffman_used;
    }
    else if ( memcmp(batch->comp_id, decoder->comp_id, comp_count * sizeof batch->comp_id[0]) != 0 ||
              memcmp(batch->quant_map, decoder->comp_table_quantization_map, comp_count * sizeof batch->quant_map[0]) !=
                  0 ||
              memcmp(batch->huffman_map, decoder->comp_table_huffman_map, comp_count * sizeof batch->huffman_map[0]) !=
                  0 ) {
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_COMPONENTS, GPUJPEG_ERR_BATCH_MISMATCH,
                                           "component IDs or table selectors differ from the first image");
    }
    else if ( batch->quant_used != reader->quant_used || batch->huffman_used != reader->huffman_used ) {
        return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLES, GPUJPEG_ERR_BATCH_MISMATCH,
                                           "used tables differ from the first image");
    }
    // contents of the used tables (as defined, unchanged since their use)
    for ( int index = 0; index < GPUJPEG_MAX_COMPONENT_COUNT; index++ ) {
        const uint8_t* const quant = decoder->table_quantization[index].table_raw;
        if ( (reader->quant_used & 1U << index) != 0 ) {
            if ( batch->image == 0 ) {
                memcpy(batch->quant[index], quant, sizeof batch->quant[index]);
            }
            else if ( memcmp(batch->quant[index], quant, sizeof batch->quant[index]) != 0 ) {
                return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLES, GPUJPEG_ERR_BATCH_MISMATCH,
                                                   "quantization table differs from the first image");
            }
        }
        for ( int type = 0; type < GPUJPEG_HUFFMAN_TYPE_COUNT; type++ ) {
            const struct gpujpeg_table_huffman_decoder* table = &decoder->table_huffman[index][type];
            const int count = gpujpeg_reader_huffman_count(table->bits);
            if ( (reader->huffman_used & 1U << (index * 2 + type)) == 0 ) {
                continue;
            }
            if ( batch->image == 0 ) {
                memcpy(batch->huffman_bits[index][type], table->bits, sizeof batch->huffman_bits[index][type]);
                memcpy(batch->huffman_val[index][type], table->huffval, count);
            }
            else if ( memcmp(batch->huffman_bits[index][type], table->bits, sizeof batch->huffman_bits[index][type]) !=
                          0 ||
                      memcmp(batch->huffman_val[index][type], table->huffval, count) != 0 ) {
                return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_TABLES, GPUJPEG_ERR_BATCH_MISMATCH,
                                                   "Huffman table differs from the first image");
            }
        }
    }
    return GPUJPEG_NOERR;
}

/* Documented at declaration */
int
gpujpeg_reader_read_image(struct gpujpeg_decoder* decoder, uint8_t* image, size_t image_size)
{
    const bool batch = decoder->batch.active;
    int rc = 0;

    // Setup reader
    struct gpujpeg_reader reader = {
        .param = decoder->coder.param,
        .param_image = decoder->coder.param_image,
        .image_end = image + image_size,
        .ff_cs_itu601_is_709 = decoder->ff_cs_itu601_is_709,
        .metadata = batch ? &decoder->batch.metadata[decoder->batch.image] : &decoder->metadata,
        .batch = batch,
    };
    if ( batch ) {
        // as a new decoder would, independently of the previous images and calls
        gpujpeg_set_default_parameters(&reader.param);
        reader.param.verbose = decoder->coder.param.verbose;
        reader.param.perf_stats = decoder->coder.param.perf_stats;
        reader.param.comp_count = 0;
        gpujpeg_image_set_default_parameters(&reader.param_image);
        // after the previous images
        reader.segment_count = decoder->batch.image * decoder->coder.segment_count;
        reader.data_compressed_size = decoder->data_compressed_size;
    }
    reader.param.restart_interval = 0;
    reader.param_image.pixel_format = decoder->req_pixel_format;
    reader.param_image.color_space = decoder->req_color_space;
    decoder->huffman_resync_eligible = false;
    for (unsigned i = 0; i < GPUJPEG_METADATA_COUNT; ++i) {
        reader.metadata->vals[i].set = false;
    }

    // Check first SOI marker
    int marker_soi = gpujpeg_reader_read_marker(&image, reader.image_end, decoder->coder.param.verbose);
    if ( marker_soi != GPUJPEG_MARKER_SOI ) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker_soi));
        return -1;
    }

    int eoi_presented = 0;
    while ( eoi_presented == 0 ) {
        // Read marker
        int marker = gpujpeg_reader_read_marker(&image, reader.image_end, decoder->coder.param.verbose);
        if ( marker == -1 ) {
            return -1;
        }

        // Read more info according to the marker
        rc = gpujpeg_reader_read_common_markers(&image, marker, &reader);
        if ( rc < 0 ) {
            return rc;
        }
        if (rc == 0) { // already processed
            continue;
        }
        switch (marker)
        {
        case GPUJPEG_MARKER_APP13:
            if ( reader.batch ) { // segment info not used, batch scans are always parsed
                if ( gpujpeg_reader_skip_marker_content(&image, reader.image_end) != 0 ) {
                    return -1;
                }
                break;
            }
            if ( gpujpeg_reader_read_app13(&reader, &image, reader.image_end) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_DQT:
            if ( (rc = gpujpeg_reader_read_dqt(decoder, &reader, &image, reader.image_end)) != 0 )
                return rc;
            break;

        case GPUJPEG_MARKER_SOF1: // Extended sequential with Huffman coder
            fprintf(stderr, "[GPUJPEG] [Warning] Reading SOF1 as it was SOF0 marker (should work but verify it)!\n");
            reader.sof1 = true;
            /* fall through */
        case GPUJPEG_MARKER_SOF0: // Baseline
            if ( reader.batch && reader.sof_count != 0 ) {
                return gpujpeg_reader_batch_reject(decoder, GPUJPEG_BATCH_REASON_SCANS, GPUJPEG_ERR_BATCH_UNSUPPORTED,
                                                   "more frames (SOF markers)");
            }
            reader.sof_count += 1;
            if (reader.header_color_space != GPUJPEG_NONE) {
                reader.param.color_space_internal = reader.header_color_space;
            }
            if ( gpujpeg_reader_read_sof0(&reader.param, &reader.param_image, reader.header_color_space, reader.header_type,
                                          decoder->comp_table_quantization_map, decoder->comp_id, &image,
                                          reader.image_end) != 0 ) {
                return -1;
            }
            adjust_params(&reader.param, &reader.param_image, reader.param.color_space_internal,
                          decoder->req_alignment);
            break;

        case GPUJPEG_MARKER_DHT:
            if ( (rc = gpujpeg_reader_read_dht(decoder, &reader, &image, reader.image_end)) != 0 )
                return rc;
            reader.resync_scan = reader.resync_scan && reader.scan_count == 0;
            break;

        case GPUJPEG_MARKER_SOS:
            if ( (rc = gpujpeg_reader_read_sos(decoder, &reader, &image, reader.image_end)) != 0 )
                return rc;
            break;

        case GPUJPEG_MARKER_EOI:
            eoi_presented = 1;
            break;

        case GPUJPEG_MARKER_DAC:
        case GPUJPEG_MARKER_DNL:
            fprintf(stderr, "[GPUJPEG] [Warning] JPEG data contains not supported %s marker\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            gpujpeg_reader_skip_marker_content(&image, reader.image_end);
            break;

        default:
            fprintf(stderr, "[GPUJPEG] [Error] JPEG data contains not supported %s marker!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            gpujpeg_reader_skip_marker_content(&image, reader.image_end);
            return -1;
        }
    }

    if ( !reader.batch && reader.segment_count != decoder->coder.segment_count ) {
        fprintf(stderr, "[GPUJPEG] [Warning] %d segments read, expected %d. Broken JPEG?\n",
                reader.segment_count, decoder->coder.segment_count);
    }

    // Check EOI marker
    if ( eoi_presented == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should end with EOI marker!\n");
        return -1;
    }
    if ( reader.batch && (rc = gpujpeg_reader_batch_finish(decoder, &reader)) != 0 ) {
        return rc;
    }

    // Set decoder parameters (for a batch the totals of the images read so far)
    decoder->segment_count = reader.segment_count;
    decoder->data_compressed_size = reader.data_compressed_size;
    decoder->huffman_resync_eligible = !reader.batch && reader.resync_scan && reader.scan_count == 1 &&
                                       reader.sof_count == 1 && reader.param.restart_interval == 0 &&
                                       reader.segment_count == decoder->coder.segment_count;

    if ( !reader.batch && decoder->segment_count > decoder->coder.segment_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder can't decode image that has segment count %d (maximum segment count for specified parameters is %d)!\n",
            decoder->segment_count, decoder->coder.segment_count);
        return -1;
    }

    return 0;
}

/* Documented at declaration */
int
gpujpeg_reader_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_info *info, int verbose, unsigned flags)
{
    // gpujpeg_reader_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_parameters *param_image,
    // struct gpujpeg_parameters *param, int *segment_count)
    int segments = 0;
    int unused[4];
    uint8_t unused2[4];

    struct gpujpeg_reader reader = {
        .param.verbose = verbose,
        .param_image.pixel_format = GPUJPEG_PIXFMT_NATIVE,
        .image_end = image + image_size,
        .metadata = &info->metadata,
    };

    // Check first SOI marker
    int marker_soi = gpujpeg_reader_read_marker(&image, reader.image_end, verbose);
    if (marker_soi != GPUJPEG_MARKER_SOI) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker_soi));
        return -1;
    }

    int eoi_presented = 0;
    while (eoi_presented == 0) {
        // Read marker
        int marker = gpujpeg_reader_read_marker(&image, reader.image_end, verbose);
        if (marker == -1) {
            return -1;
        }

        // Read more info according to the marker
        int rc = gpujpeg_reader_read_common_markers(&image, marker, &reader);
        if ( rc < 0 ) {
            return rc;
        }
        if (rc == 0) { // already processed
            continue;
        }
        switch (marker)
        {
        case GPUJPEG_MARKER_SOF0: // Baseline
        case GPUJPEG_MARKER_SOF1: // Extended sequential with Huffman coder
        {
            reader.param.color_space_internal = reader.header_color_space;
            if ( gpujpeg_reader_read_sof0(&reader.param, &reader.param_image, reader.header_color_space, reader.header_type,
                                          unused, unused2, &image,
                                          reader.image_end) != 0 ) {
                return -1;
            }
            adjust_params(&reader.param, &reader.param_image, reader.param.color_space_internal, 0);
            break;
        }
        case GPUJPEG_MARKER_RST0:
        case GPUJPEG_MARKER_RST1:
        case GPUJPEG_MARKER_RST2:
        case GPUJPEG_MARKER_RST3:
        case GPUJPEG_MARKER_RST4:
        case GPUJPEG_MARKER_RST5:
        case GPUJPEG_MARKER_RST6:
        case GPUJPEG_MARKER_RST7:
        case GPUJPEG_MARKER_SOS:
        {
            segments++;
            if (marker == GPUJPEG_MARKER_SOS) {
                if(image_size < 3) {
                    fprintf(stderr, "[GPUJPEG] [Error] SOS ends prematurely\n");
                    return -1;
                }

                int length = (int) gpujpeg_reader_read_2byte(image); // length
                (void) length; // ifdef NDEBUG unused variable
                assert(length > 3);
                int comp_count = (int) gpujpeg_reader_read_byte(image); // comp count in the segment
                if (comp_count > 1) {
                    reader.param.interleaved = 1;
                }
                if ( (flags & GPUJPEG_COUNT_SEG_COUNT_REQ) == 0 ) { // if not counting segments, we can skip the rest
                    eoi_presented = 1;
                }
            }

            while(1) {
                if(image_size < 1) {
                    fprintf(stderr, "[GPUJPEG] [Error] SOS ends prematurely\n");
                    return -1;
                }

                if(*image != 0xff)
                {
                    image++;
                    image_size--;
                    continue;
                }

                // 0xff, so check if next byte is 0x00
                if(image_size < 2) {
                    fprintf(stderr, "[GPUJPEG] [Error] SOS ends prematurely\n");
                    return -1;
                }

                if(image[1] == 0x00)
                {
                    image += 2;
                    image_size -= 2;
                    continue;
                }

                // 0xff not followed by 0x00, so break here
                break;
            }
            break;
        }
        case GPUJPEG_MARKER_EOI:
        {
            eoi_presented = 1;
            break;
        }
        default:
            gpujpeg_reader_skip_marker_content(&image, reader.image_end);
            break;
        }
    }

    info->param = reader.param;
    info->param_image = reader.param_image;
    info->segment_count = segments;
    info->header_type = reader.header_type;
    info->comment = reader.comment;

    return 0;
}

/* vim: set expandtab sw=4: */
