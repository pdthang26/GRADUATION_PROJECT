/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "CLCD_I2C.h"
#include "convert_lib.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define diameter            0.4 //(m)
#define PI                  3.14159
#define OFFSET							3
#define IDLE								0
#define AUTO								1
#define MANUAL							2
#define MASTER_ID      			0x281
#define SLAVE_ID1   				0x012
#define SLAVE_ID2   				0x274
#define alpha 0.2
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

CAN_HandleTypeDef hcan;

I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;

/* USER CODE BEGIN PV */
CLCD_I2C_Name LCD1;
float adcValue;
float aileValue;
uint8_t mode = OFFSET, btnState, changeMode;
char lcdRPM[16];
char lcdEncoderValue[16];


// CAN protocol variable
CAN_HandleTypeDef     CanHandle;
CAN_TxHeaderTypeDef   TxHeader;
CAN_RxHeaderTypeDef   RxHeader;
uint8_t               TxData[8];
uint8_t               RxData[8];
uint32_t              TxMailbox;


// Encoder varible
int32_t encoderValue = 0;
uint16_t encoderGet = 0;
int32_t last_encoderValue = 0;
const float sampleTime = 0.01; // in seconds
const float pulsesPerRevolution = 400*4; // pulse per revolution
float rpm = 0; // velocity in RPM
float mps = 0; // velocity in m/s
float angular_vel = 0; // angular velocity in rad/s
int direction; // FORWARD is 1 and REVERSE is -1
float posInRad =0, posInMeter = 0;
int count=-1;




//PID 
//float Kp = 0.8;
//float Ki = 0;
//float Kd = 2;
//float Ts = 0.01; // 10ms
//float prev_error = 0.0;
//float integral = 0.0;

float Kp = 0.1 ,Kd = 0.01;
float Ts = 0.01; // 10ms

float input, output, lastError;
uint16_t test =10000;


// Khai bao bien cho PWM
uint16_t pwm_value = 0;
uint16_t pwmValueCW = 0;
uint16_t pwmValueCCW = 0;
uint32_t valueIn = 0;
uint8_t vel = 0;

float setpoint ;
int offsetCount;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_I2C1_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM1_Init(void);
static void MX_CAN_Init(void);
/* USER CODE BEGIN PFP */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim);
void dc_motor_control(float setpoint, float input);
float pid_controller(float setpoint, float input);
void WriteCAN(uint16_t ID,uint8_t *data);
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
	
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_I2C1_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM1_Init();
  MX_CAN_Init();
  /* USER CODE BEGIN 2 */
	// start analog to digital convert
	HAL_ADC_Start(&hadc1);
	HAL_GPIO_WritePin(GPIOC,GPIO_PIN_13,1);
	HAL_Delay(20000);
  //Start timer

	
	HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1); // khoi dong PWM tai channel 1
	HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2); // khoi dong PWM tai channel 2
	// Enable motor
	HAL_GPIO_WritePin(GPIOB,GPIO_PIN_0,1);// active for run Clockwise direction
	HAL_GPIO_WritePin(GPIOB,GPIO_PIN_1,1);// active for run Counter Clockwise direction

	
	CLCD_I2C_Init(&LCD1,&hi2c1,0x4E,16,2);
	
	// Initial CAN protocol
	HAL_CAN_Start(&hcan);
	HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING);

  TxHeader.DLC = 8;
  TxHeader.ExtId = 0;
  TxHeader.IDE = CAN_ID_STD;
  TxHeader.RTR = CAN_RTR_DATA;
  TxHeader.TransmitGlobalTime = DISABLE;
	
	
	//alpha = sampleTime/(N+sampleTime);
	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		switch (mode)
		{
			case OFFSET:
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 50000);
//			
				while(HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_5)==1){}
				count=-1;
				__HAL_TIM_SET_COUNTER(&htim2, 0);
				HAL_TIM_Base_Start_IT(&htim1);// khoi dong ngat thoi gian lay mau
				HAL_TIM_Encoder_Start(&htim2,TIM_CHANNEL_ALL);// khoi dong bo doc encoder tai timer2
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
				HAL_Delay(2000);
				mode = IDLE;
				break;
			case IDLE:
				sprintf(lcdRPM,"ctrl:%.0d :%d  ",valueIn,vel );
				sprintf(lcdEncoderValue,"encoder:%d   ",encoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,0);
				CLCD_I2C_WriteString(&LCD1,lcdEncoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,1);
				CLCD_I2C_WriteString(&LCD1,lcdRPM);
				if(encoderValue>18000)
				{
					HAL_Delay(3000);
					mode = AUTO;
				}
				break;
			case AUTO:
				if(changeMode!=mode)
				{
					vel=0;
					CLCD_I2C_Clear(&LCD1);
					CLCD_I2C_SetCursor(&LCD1, 0,0);
					CLCD_I2C_WriteString(&LCD1, "    AUTO MODE   ");
					HAL_Delay(2000);
					changeMode=mode;
				}
				setpoint = valueIn;
//				setpoint = test;
				sprintf(lcdRPM,"ctrl:%.0d :%d  ",valueIn,vel );
				sprintf(lcdEncoderValue,"encoder:%d   ",encoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,0);
				CLCD_I2C_WriteString(&LCD1,lcdEncoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,1);
				CLCD_I2C_WriteString(&LCD1,lcdRPM);
				
				break;
			case MANUAL:
				if(changeMode!=mode)
				{
					CLCD_I2C_Clear(&LCD1);
					CLCD_I2C_SetCursor(&LCD1, 0,0);
					CLCD_I2C_WriteString(&LCD1, "   MANUAL MODE  ");
					changeMode=mode;
					HAL_Delay(2000);
				}
				btnState =  HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_4)<<1 | HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_3);
				adcValue = (float)(HAL_ADC_GetValue(&hadc1)/4095.0);
				sprintf(lcdRPM,"ADC:%.2f ",adcValue*100);
				sprintf(lcdEncoderValue,"encoder:%d ",encoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,0);
				CLCD_I2C_WriteString(&LCD1,lcdEncoderValue);
				CLCD_I2C_SetCursor(&LCD1, 0,1);
				CLCD_I2C_WriteString(&LCD1,lcdRPM);
				if((btnState&0x01) ==0)
				{
					pwmValueCW = (uint16_t)(65535 * adcValue);
					pwmValueCCW = 0;
					
				}
				else if((btnState>>1&0x01) ==0)
				{				
					pwmValueCW = 0;
					pwmValueCCW = (uint16_t)(65535 * adcValue);
				}
				else 
				{
					pwmValueCW = 0;
					pwmValueCCW = 0;
				}
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwmValueCW);
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwmValueCCW);


				break;
		}
		

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief CAN Initialization Function
  * @param None
  * @retval None
  */
static void MX_CAN_Init(void)
{

  /* USER CODE BEGIN CAN_Init 0 */

  /* USER CODE END CAN_Init 0 */

  /* USER CODE BEGIN CAN_Init 1 */

  /* USER CODE END CAN_Init 1 */
  hcan.Instance = CAN1;
  hcan.Init.Prescaler = 18;
  hcan.Init.Mode = CAN_MODE_NORMAL;
  hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan.Init.TimeSeg1 = CAN_BS1_2TQ;
  hcan.Init.TimeSeg2 = CAN_BS2_1TQ;
  hcan.Init.TimeTriggeredMode = DISABLE;
  hcan.Init.AutoBusOff = DISABLE;
  hcan.Init.AutoWakeUp = DISABLE;
  hcan.Init.AutoRetransmission = DISABLE;
  hcan.Init.ReceiveFifoLocked = DISABLE;
  hcan.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CAN_Init 2 */
	  CAN_FilterTypeDef canfilterconfig;

  canfilterconfig.FilterActivation = CAN_FILTER_ENABLE;
  canfilterconfig.FilterBank = 10;
  canfilterconfig.FilterFIFOAssignment = CAN_RX_FIFO0;
  canfilterconfig.FilterIdHigh = 0x0000;
  canfilterconfig.FilterIdLow = 0x0000;
  canfilterconfig.FilterMaskIdHigh = 0x0000;
  canfilterconfig.FilterMaskIdLow = 0x0000;
  canfilterconfig.FilterMode = CAN_FILTERMODE_IDMASK;
  canfilterconfig.FilterScale = CAN_FILTERSCALE_32BIT;
  canfilterconfig.SlaveStartFilterBank = 13;

  HAL_CAN_ConfigFilter(&hcan, &canfilterconfig);

  /* USER CODE END CAN_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 399;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 899;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 65535;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0|GPIO_PIN_1, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PA1 PA2 */
  GPIO_InitStruct.Pin = GPIO_PIN_1|GPIO_PIN_2;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PA3 PA4 */
  GPIO_InitStruct.Pin = GPIO_PIN_3|GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : PA5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI1_IRQn);

  HAL_NVIC_SetPriority(EXTI2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI2_IRQn);

}

/* USER CODE BEGIN 4 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM1) 
		{
			HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
			direction = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim2) ? -1 : 1;// checking the direction of the car
			encoderGet = __HAL_TIM_GET_COUNTER(&htim2);// get value from encoder of timer 2
			if (__HAL_TIM_GET_FLAG(&htim2, TIM_FLAG_UPDATE) != RESET) // check if timer 2 overflowed
			{
				__HAL_TIM_CLEAR_FLAG(&htim2, TIM_FLAG_UPDATE); // clear overflow flag
				if (direction == 1) count++; // increment count if car is moving forward
				else count--; // decrement count if car is moving backward
			}
			encoderValue = encoderGet + (count*65535);
			angular_vel = (encoderValue-last_encoderValue)*PI/8;
			
			TxData[3] = 'A';
			convertFloatTo8Byte(angular_vel, TxData, 4, 7 );
			WriteCAN(MASTER_ID,TxData);
			
			last_encoderValue = encoderValue;
			if ( mode == IDLE)
			{
				vel=100;
				dc_motor_control(20000, encoderValue);
			}
			else if (mode == AUTO && changeMode==mode)
			{
				dc_motor_control(setpoint, encoderValue);
			}
		
    }
}



// Traditional PID
float pid_controller(float setPoint, float current) {
	
//    float error = setpoint - input; // Calculating error
//    
//    float output = Kp * error  ; // Tính giá tr? d?u ra c?a hàm di?u khi?n

//    // Luu tr? sai s? và tích phân
//    
//		if (output > 100)
//        output = 100;
//    else if (output < - 100)
//        output = -100;
//		
//    return output;
	int output;
	float error = setPoint - current;
	float d_term = Kd*(error-lastError)/Ts;
	
	static float d_term_f, last_d_term_f;
	
	d_term_f = alpha*d_term + (1-alpha)*last_d_term_f;
	last_d_term_f = d_term_f;
	
	float PID = Kp*error + d_term_f;
	
	
	if(PID > 100.0)	output =100;
	else if(PID < -100.0) output = -100;
	else output = PID;
	lastError = error;
	return output;
	
	
}

// dieu khien dong co dua tren pid
void dc_motor_control(float setpoint, float input)
{
		output = pid_controller(setpoint, input);
			
			// tính gia tri PWM tu gia tri dieu khien PID va xuat xung PWM tai chan PB6
		if (output <0)
		{
			pwmValueCCW = (uint16_t)(-output *0.01* (vel*350));
			pwmValueCW = 0;
		}
		else if (output>0)
		{
			pwmValueCW = (uint16_t)(output *0.01* (vel*350));
			pwmValueCCW = 0;
		}
		else
		{
			pwmValueCCW = 0;
			pwmValueCW = 0;
		}
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwmValueCW);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwmValueCCW);
	
}
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	if (GPIO_Pin==GPIO_PIN_1) 
	{
		mode = AUTO;
		output = 0;
	}
	
	else if (GPIO_Pin==GPIO_PIN_2) mode = MANUAL;
}

void WriteCAN(uint16_t ID,uint8_t *data)
{
	uint8_t dataOut[8];
	TxHeader.StdId = ID;
	dataOut[0] = data[0];
	dataOut[1] = data[1];
	dataOut[2] = data[2];
	dataOut[3] = data[3];
	dataOut[4] = data[4];
	dataOut[5] = data[5];
	dataOut[6] = data[6];
	dataOut[7] = data[7];
	HAL_CAN_AddTxMessage(&hcan, &TxHeader, dataOut, &TxMailbox);
	
}
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
	if(HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &RxHeader, RxData)== HAL_OK)
	{
		if(RxHeader.StdId==SLAVE_ID2)
		{
			HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
			vel = RxData[3];
			valueIn = convert8byteToUint32_t(RxData, 4,7);
		}
	}
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
