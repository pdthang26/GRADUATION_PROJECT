#include "string.h"
#include <stdio.h>
#include <stdint.h>
#include "math.h"
#include "stm32f1xx_hal.h"
uint32_t convert8byteToUint32_t(uint8_t *arr, uint8_t startByte, uint8_t stopByte); 
void convertUint32_tTo8byte(uint32_t value, uint8_t *arr, uint8_t startByte, uint8_t stopByte);
void convertFloatTo8Byte(float value, uint8_t *arr, uint8_t startByte, uint8_t stopByte );
float convert8ByteToFloat (uint8_t *arr, uint8_t startByte, uint8_t stopByte);
float map(float inValue, float inMax, float inMin,float outMax, float outMin );
void convertUint16_tTo8byte(uint16_t value, uint8_t *arr, uint8_t startByte, uint8_t stopByte);
uint16_t convert8byteToUint16_t(uint8_t *arr, uint8_t startByte, uint8_t stopByte);
void CharToNum (uint16_t *SaveNum, uint8_t *DataIn, uint8_t Index);
