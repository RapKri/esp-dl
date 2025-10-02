#include "coco_detect.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "bsp/esp-bsp.h"

extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[] asm("_binary_bus_jpg_end");
const char *TAG = "yolo11n";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== Custom YOLO11 Detection Starting ===");
    ESP_LOGI(TAG, "Free heap: %u bytes", (unsigned int)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free PSRAM: %u bytes", (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    
#if CONFIG_COCO_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)bus_jpg_start, .data_len = (size_t)(bus_jpg_end - bus_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ESP_LOGI(TAG, "Starting object detection...");
    ESP_LOGI(TAG, "Image decoded: width=%d, height=%d", img.width, img.height);
    
    COCODetect *detect = new COCODetect();
    
    ESP_LOGI(TAG, "Detector created, running inference...");
    ESP_LOGI(TAG, "Running inference on image (%dx%d)...", img.width, img.height);
    
    auto &detect_results = detect->run(img);
    ESP_LOGI(TAG, "Inference completed. Found %d objects.", (int)detect_results.size());
    
    if (detect_results.size() == 0) {
        ESP_LOGI(TAG, "No objects detected! Try lowering confidence threshold.");
    }
    
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[category: %d, score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.category,
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }
    for (const auto &res : detect_results) {
        // YOLO Format: class center_x center_y width height (alle normalisiert 0.0-1.0)
        float x1 = (float)res.box[0] / 640.0f;
        float y1 = (float)res.box[1] / 640.0f;
        float x2 = (float)res.box[2] / 640.0f;
        float y2 = (float)res.box[3] / 640.0f;
        
        float center_x = (x1 + x2) / 2.0f;
        float center_y = (y1 + y2) / 2.0f;
        float width = x2 - x1;
        float height = y2 - y1;
        
        ESP_LOGI(TAG,
                 "%d %f %f %f %f",
                 res.category,
                 center_x,
                 center_y, 
                 width,
                 height);
    }
    delete detect;
    heap_caps_free(img.data);

#if CONFIG_COCO_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
    
    ESP_LOGI(TAG, "=== Detection completed successfully ===");
    ESP_LOGI(TAG, "Final free heap: %u bytes", (unsigned int)esp_get_free_heap_size());
}

/*
I (27735) yolo11n: Inference completed. Found 10 objects.
I (27735) yolo11n: [category: 0, score: 0.015906, x1: 225, y1: 227, x2: 273, y2: 288]
I (27735) yolo11n: [category: 0, score: 0.002801, x1: 338, y1: 269, x2: 466, y2: 388]
I (27735) yolo11n: [category: 0, score: 0.002183, x1: 233, y1: 231, x2: 469, y2: 386]
I (27745) yolo11n: [category: 0, score: 0.001325, x1: 0, y1: 254, x2: 29, y2: 310]
I (27755) yolo11n: [category: 0, score: 0.001032, x1: 227, y1: 228, x2: 273, y2: 256]
I (27765) yolo11n: [category: 0, score: 0.001032, x1: 292, y1: 241, x2: 469, y2: 389]
I (27765) yolo11n: [category: 0, score: 0.000911, x1: 578, y1: 462, x2: 639, y2: 639]
I (27775) yolo11n: [category: 0, score: 0.000804, x1: 1, y1: 0, x2: 121, y2: 96]
I (27785) yolo11n: [category: 0, score: 0.000804, x1: 44, y1: 231, x2: 131, y2: 384]
I (27795) yolo11n: [category: 0, score: 0.000710, x1: 20, y1: 0, x2: 179, y2: 33]

I (27846) yolo11n: Inference completed. Found 10 objects.
I (27846) yolo11n: [category: 0, score: 0.015906, x1: 225, y1: 227, x2: 273, y2: 287]
I (27856) yolo11n: [category: 0, score: 0.006693, x1: 326, y1: 258, x2: 468, y2: 384]
I (27856) yolo11n: [category: 0, score: 0.001927, x1: 262, y1: 235, x2: 469, y2: 384]
I (27866) yolo11n: [category: 0, score: 0.001170, x1: 0, y1: 246, x2: 28, y2: 316]
I (27876) yolo11n: [category: 0, score: 0.000911, x1: 0, y1: 0, x2: 122, y2: 89]
I (27886) yolo11n: [category: 0, score: 0.000804, x1: 9, y1: 0, x2: 143, y2: 32]
I (27886) yolo11n: [category: 0, score: 0.000804, x1: 49, y1: 0, x2: 226, y2: 33]
I (27896) yolo11n: [category: 0, score: 0.000804, x1: 0, y1: 0, x2: 83, y2: 125]
I (27906) yolo11n: [category: 0, score: 0.000804, x1: 0, y1: 32, x2: 57, y2: 179]
I (27906) yolo11n: [category: 0, score: 0.000804, x1: 578, y1: 466, x2: 639, y2: 639]

*/