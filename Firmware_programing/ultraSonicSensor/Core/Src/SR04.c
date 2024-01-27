//=============================================================================================
/*
+ Lua chon timer voi gia tri dem la micro giay 
+ Khai bao struct luu gia tri
VD:
SR04_Name SR04_1;

+ Khai bao timer va GPIO cua chan triger va echo
VD:
Init_SR04x(&SR04_1,&htim1,TIM1, GPIOA, GPIO_PIN_12, GPIOB,GPIO_PIN_12);
						---------------    ----------------    -----------------
						cau hinh timer					chan triger			chan echo

+ Doc do rong xung echo voi khi o muc cao (mac dinh)
VD:
readPulseDurationSR04 (SR04_1,     100);
												----		  ------
												struct			thoi gian
												luu gtri		cho toi da
												
+ Doc gia tri do dai don vi cm
VD:
  readDistance(SR04_1,     100);
								----		  ------
								struct			thoi gian
							luu gtri		cho toi da


*/
//=============================================================================================
#include "SR04.h"

float velocityOfUltrasonic = 0.343; //(mm/mirosecond)
	
uint32_t state = 1;


void Init_SR04x(SR04_Name* SR04_x,TIM_HandleTypeDef *htim, TIM_TypeDef* TIMx, GPIO_TypeDef* GPIOx_trig, uint16_t pinTrig, GPIO_TypeDef* GPIOx_echo,uint16_t pinEcho)
{
	
	SR04_x->htim_SR04 = htim;
	SR04_x->TIMx = TIMx;
	SR04_x->GPIOx_TRIG = GPIOx_trig;
	SR04_x->GPIOx_ECHO = GPIOx_echo;
	SR04_x->PIN_TRIG = pinTrig;
	SR04_x->PIN_ECHO = pinEcho;
	HAL_TIM_Base_Start(SR04_x->htim_SR04);
}

void readPulseDurationSR04 (SR04_Name* SR04_x,uint32_t timeout)
{
	uint32_t startTime = HAL_GetTick();
  uint32_t timeoutTime = startTime + timeout;
  uint32_t lastTime;
  uint32_t pulse = 0;
  uint32_t inputState = 0;
	uint32_t pulseWidth=0;
    
	if (state == 1)
	{
			inputState = GPIO_PIN_SET;
	}
	else if (state == 0)
	{
			inputState = GPIO_PIN_RESET;
	}
	(SR04_x->TIMx)->CNT = 0;
	HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,0);
	while((SR04_x->TIMx) ->CNT < 2);
	HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,1);
	while((SR04_x->TIMx) ->CNT < 22);
	HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,0);
	
	
	while (HAL_GPIO_ReadPin(SR04_x->GPIOx_ECHO, SR04_x->PIN_ECHO) != inputState)
	{
			if (HAL_GetTick() > timeoutTime)
			{
				SR04_x->error = 1;
				break;
			}
			else SR04_x -> error = 0;
	}
	
	(SR04_x->TIMx)->CNT = 0;
	while (HAL_GPIO_ReadPin(SR04_x->GPIOx_ECHO, SR04_x->PIN_ECHO) == inputState)
	{
			if (HAL_GetTick() > timeoutTime)
			{
				SR04_x->error = 2;
				break;
			}
			else SR04_x -> error = 0;
	}
	if(SR04_x -> error ==0)  SR04_x ->pulseValue  = (SR04_x->TIMx)->CNT ;
}

float readDistance (SR04_Name* SR04_x,uint32_t timeout)
{
	uint8_t FLAG_TIMEOUT = 0;
	uint32_t startTime = HAL_GetTick();
  uint32_t timeoutTime = startTime + timeout;
  uint32_t inputState = 0;
	
    
    if (state == 1)
    {
        inputState = GPIO_PIN_SET;
    }
    else if (state == 0)
    {
        inputState = GPIO_PIN_RESET;
    }
		(SR04_x->TIMx)->CNT = 0;
		HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,0);
		while((SR04_x->TIMx) ->CNT < 2);
		HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,1);
		while((SR04_x->TIMx) ->CNT < 22);
		HAL_GPIO_WritePin(SR04_x->GPIOx_TRIG, SR04_x->PIN_TRIG,0);
		
		
    while (HAL_GPIO_ReadPin(SR04_x->GPIOx_ECHO, SR04_x->PIN_ECHO) != inputState)
    {
        if (HAL_GetTick() > timeoutTime)
        {
					FLAG_TIMEOUT = 1;
					break;
        }
    }

    (SR04_x->TIMx)->CNT = 0;
    while (HAL_GPIO_ReadPin(SR04_x->GPIOx_ECHO, SR04_x->PIN_ECHO) == inputState)
    {
        if (HAL_GetTick() > timeoutTime)
        {
					FLAG_TIMEOUT = 1;
					break;
        }

		}
		
		if((SR04_x->TIMx)->CNT <35000 && FLAG_TIMEOUT == 0 )  
		{
			SR04_x ->pulseValue  = (SR04_x->TIMx)->CNT ;
//			HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
			SR04_x -> error = 0;
		}
		else if( (SR04_x->TIMx)->CNT >= 35000 || FLAG_TIMEOUT ==1)
		{
			SR04_x -> error = 1;
			SR04_x ->pulseValue  = 34986;
		}

		SR04_x ->distance_mm = SR04_x ->pulseValue * velocityOfUltrasonic/2;
		
		
		return SR04_x ->distance_mm;

	
	
}
