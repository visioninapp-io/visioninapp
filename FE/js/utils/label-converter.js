/**
 * Label format conversion utilities
 * Converts annotations to various training formats (YOLO, COCO, etc.)
 */

/**
 * Convert annotations to YOLO format
 * YOLO format: class_id center_x center_y width height (normalized 0-1)
 *
 * @param {Array} annotations - Array of annotation objects
 * @param {Object} labelClassesMap - Map of class names to class IDs
 * @returns {string} YOLO format label content
 */
function convertToYOLO(annotations, labelClassesMap) {
    if (!annotations || annotations.length === 0) {
        return '';
    }

    const lines = [];

    for (const ann of annotations) {
        try {
            // Get class ID from class name
            // Support multiple attribute names for compatibility
            const className = ann.className || ann.class_name || ann.label_class_name;
            const yoloIndex = labelClassesMap.get(className)?.yolo_index;

            if (yoloIndex === undefined || yoloIndex === null) {
                console.warn(`[LabelConverter] Missing yolo_index for class: ${className}`);
                continue;
            }

            // Get geometry (bbox)
            // Support both direct properties and nested geometry object
            let x, y, width, height;

            if (ann.geometry) {
                // Nested geometry object (from DB)
                const geometry = ann.geometry;
                if (!geometry.x || !geometry.y || !geometry.width || !geometry.height) {
                    console.warn(`[LabelConverter] Invalid geometry:`, ann);
                    continue;
                }
                x = geometry.x;
                y = geometry.y;
                width = geometry.width;
                height = geometry.height;
            } else if (ann.x !== undefined && ann.y !== undefined && ann.width !== undefined && ann.height !== undefined) {
                // Direct properties (from frontend state)
                x = ann.x;
                y = ann.y;
                width = ann.width;
                height = ann.height;
            } else {
                console.warn(`[LabelConverter] Invalid annotation structure:`, ann);
                continue;
            }

            // YOLO expects normalized coordinates (0-1)
            // If already normalized (is_normalized = true), use as-is
            // Otherwise, normalize (shouldn't happen, but handle it)

            // YOLO uses center coordinates
            const center_x = x + (width / 2);
            const center_y = y + (height / 2);

            // Format: class_id center_x center_y width height
            // All values should be between 0 and 1
            const line = `${yoloIndex} ${center_x.toFixed(6)} ${center_y.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`;
            lines.push(line);

        } catch (error) {
            console.error(`[LabelConverter] Error converting annotation:`, error, ann);
        }
    }

    return lines.join('\n');
}

/**
 * Convert annotations to COCO format
 * @param {Array} annotations - Array of annotation objects
 * @param {Object} imageInfo - Image information {id, filename, width, height}
 * @param {Object} labelClassesMap - Map of class names to class info
 * @returns {Object} COCO format object
 */
function convertToCOCO(annotations, imageInfo, labelClassesMap) {
    const cocoAnnotations = [];

    for (let i = 0; i < annotations.length; i++) {
        const ann = annotations[i];
        const className = ann.class_name || ann.label_class_name;
        const classInfo = labelClassesMap.get(className);

        if (!classInfo) {
            console.warn(`[LabelConverter] Unknown class: ${className}`);
            continue;
        }

        const geometry = ann.geometry;

        // Convert normalized to absolute coordinates for COCO
        const x = geometry.x * imageInfo.width;
        const y = geometry.y * imageInfo.height;
        const width = geometry.width * imageInfo.width;
        const height = geometry.height * imageInfo.height;

        cocoAnnotations.push({
            id: ann.id || i + 1,
            image_id: imageInfo.id,
            category_id: classInfo.id,
            bbox: [x, y, width, height],
            area: width * height,
            iscrowd: 0,
            segmentation: []
        });
    }

    return {
        images: [{
            id: imageInfo.id,
            file_name: imageInfo.filename,
            width: imageInfo.width,
            height: imageInfo.height
        }],
        annotations: cocoAnnotations,
        categories: Array.from(labelClassesMap.values()).map(cls => ({
            id: cls.id,
            name: cls.name,
            supercategory: 'object'
        }))
    };
}

/**
 * Get image filename without extension
 * @param {string} filename - Full filename (e.g., "image1.jpg")
 * @returns {string} Filename without extension (e.g., "image1")
 */
function getFilenameWithoutExtension(filename) {
    if (!filename) return '';
    const lastDotIndex = filename.lastIndexOf('.');
    return lastDotIndex > 0 ? filename.substring(0, lastDotIndex) : filename;
}

/**
 * Generate label filename from image filename
 * @param {string} imageFilename - Image filename (e.g., "image1.jpg")
 * @param {string} format - Label format ('yolo', 'json')
 * @returns {string} Label filename (e.g., "image1.txt")
 */
function getLabelFilename(imageFilename, format = 'yolo') {
    const baseName = getFilenameWithoutExtension(imageFilename);
    const extension = format === 'yolo' ? 'txt' : 'json';
    return `${baseName}.${extension}`;
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        convertToYOLO,
        convertToCOCO,
        getLabelFilename,
        getFilenameWithoutExtension
    };
}
