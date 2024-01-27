/******************************************************************************************************************
Huong dan su dung:
- Su dung thu vien HAL
- Khoi tao bien LCD: CLCD_I2C_Name LCD1;
- Khoi tao LCD do: CLCD_I2C_Init(&LCD1,&hi2c1,0x4e,20,4);
- Su dung cac ham phai truyen dia chi cua LCD do: 
CLCD_I2C_SetCursor(&LCD1, 0, 1);
CLCD_I2C_WriteString(&LCD1,"hello anh em ");
******************************************************************************************************************/
#include "MCP4725_I2C.h"
uint8_t error = 0;
//************************** Low Level Function ****************************************************************//
void MCP4725_I2C_Init(MCP4725_I2C* MCP4725, I2C_HandleTypeDef* hi2c, uint8_t Address, uint8_t ModeWrite, uint8_t ModePowDown)
{
	MCP4725->ADDRESS = Address;
	MCP4725->I2C = hi2c;
	MCP4725->modeWrite = ModeWrite;
	MCP4725->modePowDown = ModePowDown;
	if(MCP4725->modeWrite == FAST_MODE)
	{
		MCP4725->ByteSet= ModeWrite<<6 | ModePowDown<<4;
	}
	else if(MCP4725->modeWrite == WRITE_DAC_REG_ONLY || MCP4725->modeWrite ==WRITE_DAC_REG_AND_EEPROM)
	{
		MCP4725->ByteSet= ModeWrite<<5 | ModePowDown<<1;
	}
}
void MCP4725_I2C_SetValueDAC(MCP4725_I2C* MCP4725, uint16_t Value)
{
	uint8_t Data_I2C_1[2];
	uint8_t Data_I2C_2[3];
	
	if(MCP4725->modeWrite == FAST_MODE)
	{
		MCP4725->ByteSet &=~0x0F;
		MCP4725->ByteSet = MCP4725->ByteSet | Value>>8;
		MCP4725->highByteDAC = (uint8_t)Value;
		Data_I2C_1[0] = MCP4725->ByteSet;
		Data_I2C_1[1] = MCP4725->highByteDAC;
		HAL_I2C_Master_Transmit(MCP4725->I2C, MCP4725->ADDRESS, Data_I2C_1 , sizeof(Data_I2C_1), HAL_MAX_DELAY);
		
	}
	else if(MCP4725->modeWrite == WRITE_DAC_REG_ONLY || MCP4725->modeWrite ==WRITE_DAC_REG_AND_EEPROM)
	{
		MCP4725->highByteDAC = (uint8_t) (Value>>4);
		MCP4725->lowByteDAC =  (uint8_t)((Value&0x000F)<<4);
		Data_I2C_2[0] = MCP4725->ByteSet;
		Data_I2C_2[1] = MCP4725->highByteDAC;
		Data_I2C_2[2] = MCP4725->lowByteDAC;
		if(HAL_I2C_Master_Transmit(MCP4725->I2C, MCP4725->ADDRESS, (uint8_t *)Data_I2C_2 , sizeof(Data_I2C_2), HAL_MAX_DELAY) != HAL_OK)
		{
			error = 1;
		}
		
	}
}