#ifndef _SR04_H
#define _SR04_H

#include "string.h"
#include <stdio.h>
#include <stdint.h>
#include "math.h"
#include "stm32f1xx_hal.h"



typedef struct
{
	TIM_HandleTypeDef *htim_SR04;
	TIM_TypeDef* TIMx;
	GPIO_TypeDef* GPIOx_TRIG;
	uint16_t PIN_TRIG;
	GPIO_TypeDef* GPIOx_ECHO;
	uint16_t PIN_ECHO;
	uint16_t pulseValue;
	float distance_mm;
	uint8_t error;
	
}SR04_Name;



void Init_SR04x(SR04_Name* SR04_x,TIM_HandleTypeDef *htim, TIM_TypeDef* TIMx, GPIO_TypeDef* GPIOx_trig, uint16_t pinTrig, GPIO_TypeDef* GPIOx_echo,uint16_t pinEcho);

void readPulseDuration (SR04_Name* SR04_x);

float readDistance (SR04_Name* SR04_x,uint32_t timeout);




#endif