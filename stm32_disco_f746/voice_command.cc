#include "voice_command_model.h"
#include "LCD_DISCO_F746NG.h"

void respond_to_command(tflite::ErrorReporter *error_reporter, int32_t current_time, const char *found_command, uint8_t score, bool is_new_command) {
    if (is_new_command) {
        error_reporter->Report("Heard %s (%d) @%dms", found_command, score, current_time);
        if (score > 200) {
            BSP_LCD_DisplayStringAt(0, LINE(8), (uint8_t *)found_command, CENTER_MODE);
        }
    }

    if (*found_command == 'y') {
        BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t *)"Yes", CENTER_MODE);
    } else if (*found_command == 'n') {
        BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t *)"No", CENTER_MODE);
    } else if (*found_command == 's') {
        BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t *)"Stop", CENTER_MODE);
    } else if (*found_command == 'g') {
        BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t *)"Go", CENTER_MODE);
    } else {
        BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t *)"Silence", CENTER_MODE);
    }
}