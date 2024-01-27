#include "convert_lib.h"

uint32_t convert8byteToUint32_t(uint8_t *arr, uint8_t startByte, uint8_t stopByte)
{
	uint32_t result = 0;
	uint8_t j = 3;
	for (int i = startByte; i <= stopByte; i++) 
	{
		result |= ((uint32_t)arr[i] << (8 * j));
		j--;
	}
	return result;
}

void convertUint32_tTo8byte(uint32_t value, uint8_t *arr, uint8_t startByte, uint8_t stopByte)
{
	uint8_t j = 3;
	
	for (int i = startByte; i <= stopByte; i++) 
	{
		arr[i] = (value >> (8 * j)) & 0xFF;
		j--;
	}
}

void convertUint16_tTo8byte(uint16_t value, uint8_t *arr, uint8_t startByte, uint8_t stopByte)
{
	uint8_t j = 1;
	
	for (int i = startByte; i <= stopByte; i++) 
	{
		arr[i] = (value >> (8 * j)) & 0xFF;
		j--;
	}
}

uint16_t convert8byteToUint16_t(uint8_t *arr, uint8_t startByte, uint8_t stopByte)
{
	uint16_t result = 0;
	uint8_t j = 1;
	for (int i = startByte; i <= stopByte; i++) 
	{
		result |= ((uint16_t)arr[i] << (8 * j));
		j--;
	}
	return result;
}


void convertFloatTo8Byte(float value, uint8_t *arr, uint8_t startByte, uint8_t stopByte) {
    uint8_t j = stopByte - startByte;
    uint32_t num;
    memcpy(&num, &value, sizeof(value));
    
    for (int i = startByte; i <= stopByte; i++) {
        arr[i] = (num >> (8 * j)) & 0xFF;
        j--;
    }
}

float convert8ByteToFloat(uint8_t *arr, uint8_t startByte, uint8_t stopByte) {
    uint8_t j = stopByte - startByte;
    uint32_t value = 0;
    
    for (int i = startByte; i <= stopByte; i++) {
        value |= ((uint32_t)arr[i] << (8 * j));
        j--;
    }

    float num;
    memcpy(&num, &value, sizeof(num));

    return num;
}


float map(float inValue, float inMax, float inMin,float outMax, float outMin )
{
	if(inValue > inMax) 
	{
		return outMax;
	}
	else if (inValue < inMin)
	{
		return outMin;
	}
	else
	{
		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin;
	}
}



