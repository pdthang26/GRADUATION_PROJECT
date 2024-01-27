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
#include "convert_lib.h"
#include "CLCD_I2C.h"

#include "MCP4725_I2C.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define diameter                0.4 //(m)
#define PI                      3.14159
#define MASTER_ID      			    0x281
#define BACK_WHEEL_ID  				  0x012
#define FRONT_WHEEL_ID  				0x274
#define BRAKE 							    0x222

#define AUTO   							    0x01
#define MANUAL 							    0x02

#define ACCELERATION   					0x01
#define DECELERATION 						0x02

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

CAN_HandleTypeDef hcan;

I2C_HandleTypeDef hi2c2;

TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;

/* USER CODE BEGIN PV */
//LCD variable
CLCD_I2C_Name LCD1;
float adcValue;
uint8_t throValue;
char row1[16];
char row2[16];
// MCP4725 variable
MCP4725_I2C MCP4725;



// CAN protocol variable
CAN_HandleTypeDef     CanHandle;
CAN_TxHeaderTypeDef   TxHeader;
CAN_RxHeaderTypeDef   RxHeader;
uint8_t               TxData[8];
uint8_t               RxData[8];
uint8_t               RxDataThro[8];
uint32_t              TxMailbox;



// encoder variable
int32_t encoderValue = 0;
uint16_t encoderGet = 0;
int32_t last_encoderValue = 0;
const float sampleTime = 0.01; // in seconds
const float pulsesPerRevolution = 312; // pulse per round
float rpm = 0; // velocity in RPM
float mps = 0; // velocity in m/s
int direction; // FORWARD is 1 and REVERSE is -1
float posInRad =0, posInMeter = 0;
int count=-1;
float resolutions;

// Variable PID
//float Kp = 1.5;
//float Ki = 0;
//float Kd = 0;
//float integral = 0.0;
//float derivative = 0.0;
//float last_error = 0.0;
//float output = 0.0;

uint16_t out;
uint8_t pre_setpoint = 0;
float pre_mps;

//double E, E1, E2;
//double alpha, beta, gamma;
//double Output, LastOutput;

// Khai bao bien cho PWM
uint16_t DAC_value = 0; 
uint8_t mode =0;
uint8_t btnState;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_CAN_Init(void);
static void MX_ADC1_Init(void);
static void MX_I2C2_Init(void);
/* USER CODE BEGIN PFP */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim);
void dc_motor_control(float setpoint, float input, char dir);
float pid_controller(float setpoint, float input, float dt);
void WriteCAN(uint16_t ID,uint8_t *data);
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan);
uint32_t convert8byteToInt(uint8_t *arr, uint8_t startByte, uint8_t stopByte); 
void convertFloatTo8Byte(float value, uint8_t *arr, uint8_t startByte, uint8_t stopByte );
float convert8ByteToFloat (uint8_t *arr, uint8_t startByte, uint8_t stopByte);
float map(float inValue, float inMax, float inMin,float outMax, float outMin );
uint16_t pwm_generation(float speed);
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
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_CAN_Init();
  MX_ADC1_Init();
  MX_I2C2_Init();
  /* USER CODE BEGIN 2 */
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13,1);
	// Init LCD 
	CLCD_I2C_Init(&LCD1,&hi2c2,0x4E,16,2);
	
	// init MCP4725 DAC
	MCP4725_I2C_Init(&MCP4725, &hi2c2, 0xC0, WRITE_DAC_REG_ONLY, POW_DOWN_NOMAL);
	MCP4725_I2C_SetValueDAC(&MCP4725, 0);// set initial value of DAC is 0
	
	// init timer and encoder reader
	HAL_TIM_Encoder_Start(&htim2,TIM_CHANNEL_ALL);// khoi dong bo doc encoder tai timer2
	HAL_TIM_Base_Start_IT(&htim3);// khoi dong ngat thoi gian lay mau
	
	
	// Start reading ADC 
	HAL_ADC_Start(&hadc1);

	
	//initial CAN protocol
	HAL_CAN_Start(&hcan);
	HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING);

  TxHeader.DLC = 8;
  TxHeader.ExtId = 0;
  TxHeader.IDE = CAN_ID_STD;
  TxHeader.RTR = CAN_RTR_DATA;
  TxHeader.TransmitGlobalTime = DISABLE;
	
	HAL_GPIO_WritePin(GPIOA,GPIO_PIN_2,1);
	HAL_GPIO_WritePin(GPIOA,GPIO_PIN_3,1);
	
	
	

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		btnState = HAL_GPIO_ReadPin(GPIOA,GPIO_PIN_5)<<3|HAL_GPIO_ReadPin(GPIOA,GPIO_PIN_6)<<2
							|HAL_GPIO_ReadPin(GPIOB,GPIO_PIN_0)<<1|HAL_GPIO_ReadPin(GPIOB,GPIO_PIN_1);
		if((btnState>>3&0x01) == 0)
		{
			throValue = RxDataThro[7];
			mode = AUTO;
			sprintf(row1,"dis:%.2f :%d  ", posInMeter,throValue);
			sprintf(row2,"vel:%.2f :%d  ",mps, out);
		}
		else if((btnState>>2&0x01) == 0)
		{	
			TxData[6] = 'R';
			TxData[7] = 0;
			WriteCAN(BRAKE,TxData);
			
			mode = MANUAL;
			adcValue = (float)(HAL_ADC_GetValue(&hadc1)/4095.0);
			DAC_value = adcValue*4095;
			sprintf(row1,"DAC:%.1f %d    ",adcValue, DAC_value);
			sprintf(row2,"POS:%.1f %.1f ",posInMeter,mps);
			
			if((btnState>>1&0x01) == 0)
			{
				HAL_GPIO_WritePin(GPIOA,GPIO_PIN_2,1);
				HAL_GPIO_WritePin(GPIOA,GPIO_PIN_3,1);
			}
			else if((btnState&0x01) == 0)
			{
				HAL_GPIO_WritePin(GPIOA,GPIO_PIN_2,0);
				HAL_GPIO_WritePin(GPIOA,GPIO_PIN_3,0);
			}
			MCP4725_I2C_SetValueDAC(&MCP4725, DAC_value);
			
		}
		
		
		
		CLCD_I2C_SetCursor(&LCD1, 0,0);
		CLCD_I2C_WriteString(&LCD1,row1);
		CLCD_I2C_SetCursor(&LCD1, 0,1);
		CLCD_I2C_WriteString(&LCD1,row2);
			
		
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
  sConfig.Channel = ADC_CHANNEL_4;
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
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.ClockSpeed = 100000;
  hi2c2.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

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

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 799;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 899;
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
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

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
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2|GPIO_PIN_3, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PA2 PA3 */
  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PA5 PA6 */
  GPIO_InitStruct.Pin = GPIO_PIN_5|GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM3) 
		{
			direction = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim2) ? -1 : 1;// checking the direction of the car
			encoderGet = __HAL_TIM_GET_COUNTER(&htim2);// get value from encoder of timer 2
			if (__HAL_TIM_GET_FLAG(&htim2, TIM_FLAG_UPDATE) != RESET) // check if timer 2 overflowed
			{
				__HAL_TIM_CLEAR_FLAG(&htim2, TIM_FLAG_UPDATE); // clear overflow flag
				if (direction == 1)  // increment count if car is moving forward
				{
					count++;
				}
				else 
				{
					count--;
				}
			}
			encoderValue = encoderGet + (count*65535);
			resolutions = (encoderValue - last_encoderValue) / pulsesPerRevolution;// calculating the number of resolutions
			rpm = resolutions / sampleTime  * 60  ;// calculating the value of velocity in RPM
			mps = (resolutions * diameter * PI)/ sampleTime;// calculating the value of velocity in m/s
//			posInRad = encoderValue * 0.017453293f ; //calculating the value of position in rad
			posInMeter = (encoderValue / pulsesPerRevolution) * diameter * PI;
			last_encoderValue = encoderValue;	
			

			if(mode == AUTO) 
			{
				dc_motor_control(RxDataThro[7],posInMeter,RxDataThro[6]);
			}
			convertFloatTo8Byte(mps, TxData, 4, 7);
			TxData[3] = 'V';
			WriteCAN(MASTER_ID,TxData);
			convertFloatTo8Byte(posInMeter, TxData, 4, 7);
			TxData[3] = 'P';
			WriteCAN(MASTER_ID,TxData);

			
    }
}

// tính toán PID
//float pid_controller(float setpoint, float input, float dt)
//{
//	float E = setpoint - input;
//	alpha = 2 *sampleTime*Kp + Ki* sampleTime * sampleTime + 2*Kd;
//  beta = sampleTime * sampleTime * Ki - 4 * Kd - 2 * sampleTime * Kp;
//  gamma = 2 * Kd;
//  //Output = (alpha * E + beta * E1 + gamma * E2 + 2 * sampleTime * LastOutput) / (2 * sampleTime);
//	Output = Kp*E;
//  LastOutput = Output;
//  E2 = E1;
//  E1 = E;
//	if (Output > 100.0)
//			Output = 100.0;
//	else if (Output < -100.0)
//			output = -100.0;
//	return Output;
//}



// dieu khien dong co dua tren pid
void dc_motor_control(float setpoint, float input, char dir)
{
	static uint8_t flag_state = 0;
	
	switch (dir)
	{
		case 'T':
			HAL_GPIO_WritePin(GPIOA,GPIO_PIN_2,1);
			HAL_GPIO_WritePin(GPIOA,GPIO_PIN_3,1);
			break;
		case 'L':
			HAL_GPIO_WritePin(GPIOA,GPIO_PIN_2,0);
			HAL_GPIO_WritePin(GPIOA,GPIO_PIN_3,0);
			break;
	}
	if(setpoint==0)
	{
		MCP4725_I2C_SetValueDAC(&MCP4725, 0);
		if(mps ==0) TxData[6] = 'S';
		else TxData[6] = 'R';
		TxData[7] = 50;
		WriteCAN(BRAKE,TxData);
	}
	else 
	{
		out = (uint16_t)map(setpoint, 100, 0, 4095, 2000 );
		MCP4725_I2C_SetValueDAC(&MCP4725, out);
		
		
		// phat hien xe co giam toc hay khong giam toc khi co yeu cau
		//=================================================================================================================================
		uint8_t difference = (uint8_t)setpoint - pre_setpoint;// xem su thay doi cua tín hieu dau vao
		
		if(difference>0) flag_state = ACCELERATION;// neu gtri dau vao sau lon hon gtri dau vao truoc thi la tang toc
		else if (difference<0) flag_state = DECELERATION;// neu gtri dau vao sau nho hon gtri dau vao truoc thi la giam toc
		// trang thai giam toc
		if (flag_state == DECELERATION) 
		{			
			if (mps < pre_mps) TxData[6] = 'G'; // neu toc do truoc do be hon toc do hien tai (tang toc) thi gui tin hieu yeu cau giam toc
			else TxData[6] = 'R'; // neu toc neu toc do truoc do be hon toc do hien tai(giam toc) hoac bang (on dinh) thi ngung yeu cau giam toc
		}
		// trang thai tang toc
		else if (flag_state == ACCELERATION) TxData[6] = 'R';// trang thai tang toc thi khong gui tin hieu yeu cau giam toc
		//=================================================================================================================================
		
		
		TxData[7] = 0;
		WriteCAN(BRAKE,TxData);
	}
	pre_setpoint = (uint8_t)setpoint;
	pre_mps = mps;
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
		if(RxHeader.StdId==BACK_WHEEL_ID)
		{
			HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
			RxDataThro[7] = RxData[7];
			RxDataThro[6] = RxData[6];
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
