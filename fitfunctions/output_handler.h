#ifndef TENSORFLOW_LITE_MICRO_FITFUNCTIONS_OUTPUT_HANDLER_H_
#define TENSORFLOW_LITE_MICRO_FITFUNCTIONS_OUTPUT_HANDLER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

void HandleOutput(tflite::ErrorReporter * error_reporter, float x_value, float y_value);

#endif
