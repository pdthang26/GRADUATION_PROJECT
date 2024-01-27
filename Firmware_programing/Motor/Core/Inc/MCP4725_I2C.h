#ifndef __MCP4725_I2C_H
#define __MCP4725_I2C_H

#include "stm32f1xx_hal.h"

#define FAST_MODE                  0x00
#define WRITE_DAC_REG_ONLY         0x02
#define WRITE_DAC_REG_AND_EEPROM   0x03



#define POW_DOWN_NOMAL             0x00
#define POW_DOWN_1K_OHM            0x01
#define POW_DOWN_100K_OHM          0x02
#define POW_DOWN_500K_OHM          0x03


typedef struct
{
	I2C_HandleTypeDef* I2C;
	uint8_t ADDRESS;
	uint8_t ByteSet;
	uint8_t modeWrite;
	uint8_t modePowDown;
	uint8_t highByteDAC;
	uint8_t lowByteDAC;
	
}MCP4725_I2C;
void MCP4725_I2C_Init(MCP4725_I2C* MCP4725, I2C_HandleTypeDef* hi2c, uint8_t Address, uint8_t ModeWrite, uint8_t ModePowDown);
void MCP4725_I2C_SetValueDAC(MCP4725_I2C* MCP4725, uint16_t Value);


#endif
