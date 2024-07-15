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
#include "stdio.h"
#include "convert_lib.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define MASTER_ID      			0x281
#define BACK   							0x012
#define FRONT   						0x274
#define BRAKE								0x222
#define ULTRASONIC_ID       0x112

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CAN_HandleTypeDef hcan;

UART_HandleTypeDef huart3;
DMA_HandleTypeDef hdma_usart3_rx;
DMA_HandleTypeDef hdma_usart3_tx;

/* USER CODE BEGIN PV */
// controller variable

uint32_t back_adc_p = 0;
uint32_t front_adc_p = 0;
uint32_t brake_adc_p = 0;

uint8_t countSend =0 ;
uint8_t count;
uint8_t buffer;
uint8_t dataBuff[9];
//char data_y[20];
//char data_v[20];
//char data_d[20];
//char data_a[20];
//char data_us[21];
char DMAdataSend[21];

typedef struct 
{
	char Dir;
	uint16_t val;
}dataIn;

dataIn dataInBackWheel;
dataIn dataInFrontWheel;
dataIn dataInBrake;

typedef struct  
{
	char type;
	float value;
}dataCAN;

typedef struct  
{
	char type;
	uint32_t value;
}dataCANfront;

dataCAN dataCANvel = {.type = 'V'};
dataCAN dataCANpos = {.type = 'P'};
dataCAN dataCANyaw = {.type = 'Y'};
dataCANfront dataCANangularVel = {.type = 'A'};

typedef struct  
{
	char type;
	uint16_t outside;
	uint16_t center;
}ultraSonicDataCAN;

ultraSonicDataCAN ultraSonicLeft = {.type = 'L'};
ultraSonicDataCAN ultraSonicRight = {.type = 'R'};

// CAN protocol variable
CAN_HandleTypeDef     CanHandle;
CAN_TxHeaderTypeDef   TxHeader;
CAN_RxHeaderTypeDef   RxHeader;
uint8_t              	TxBack[8];
uint8_t              	TxFront[8];
uint8_t								TxBrake[8];
uint8_t               RxData[8];
uint32_t              TxMailbox;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_CAN_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */

void WriteCAN(uint16_t ID,uint8_t *data);
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan);
void assignUint32_tChar8byte(uint32_t value, char* buffer);
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart);
void clearBuffer (uint8_t *buff);
void updateData (uint8_t checkType,uint8_t *data);


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
  MX_DMA_Init();
  MX_CAN_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */

	HAL_CAN_Start(&hcan);
	HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING);

  TxHeader.DLC = 8;
  TxHeader.ExtId = 0;
  TxHeader.IDE = CAN_ID_STD;
  TxHeader.RTR = CAN_RTR_DATA;
  TxHeader.TransmitGlobalTime = DISABLE;


	


	HAL_UART_Receive_DMA(&huart3, &buffer, 1);
	HAL_UART_Transmit_DMA(&huart3, (uint8_t*) DMAdataSend, sizeof(DMAdataSend));
	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		
	
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
  canfilterconfig.FilterBank = 0;
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
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_IRQn);
  /* DMA1_Channel3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel3_IRQn);

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
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PA0 PA1 PA2 PA3
                           PA4 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */





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
	
	if (HAL_CAN_AddTxMessage(&hcan, &TxHeader, dataOut, &TxMailbox) != HAL_OK)
  {
    Error_Handler();
  }
}
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
	if(HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &RxHeader, RxData)== HAL_OK)
	{
		if(RxHeader.StdId==MASTER_ID)
		{
			
			if(RxData[3]== dataCANvel.type)
			{
				dataCANvel.value = convert8ByteToFloat(RxData,4,7);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_2);
			}
			
			else if(RxData[3]== dataCANpos.type) 
			{
				dataCANpos.value = convert8ByteToFloat(RxData,4,7);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_1);
			}
			
			else if(RxData[3]== dataCANyaw.type) 
			{
				dataCANyaw.value = convert8ByteToFloat(RxData,4,7);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_0);
			}
			else if(RxData[3]== ultraSonicLeft.type)
			{
				ultraSonicLeft.outside = convert8byteToUint16_t(RxData,6,7);
				ultraSonicLeft.center = convert8byteToUint16_t(RxData,4,5);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_3);
			}
			else if(RxData[3]== ultraSonicRight.type)
			{
				ultraSonicRight.outside = convert8byteToUint16_t(RxData,6,7);
				ultraSonicRight.center = convert8byteToUint16_t(RxData,4,5);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_3);
			}
			else if(RxData[3]== dataCANangularVel.type)
			{
				dataCANangularVel.value = convert8byteToUint32_t(RxData,4,7);
				HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_4);
			}
		}
	}
}


void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
		for(int i=0;i<21;i++)
		{
			DMAdataSend[i]= 0;
		}
		switch (countSend)
    {
    	case 0:
				sprintf(DMAdataSend,"\nY%.1f",dataCANyaw.value);
    		break;
    	case 2:
				sprintf(DMAdataSend,"\nD%.1f",dataCANpos.value);
    		break;
    	case 4:
				sprintf(DMAdataSend,"\nV%.1f",dataCANvel.value);
    		break;
			case 6:
				sprintf(DMAdataSend,"\nA%d",dataCANangularVel.value);
    		break;
			case 8:
//				sprintf(DMAdataSend,"\nU%d,%d,%d,%d",ultraSonicLeft.outside,ultraSonicLeft.center,ultraSonicRight.center,ultraSonicRight.outside);
    		break;
    }
		countSend++;
		if (countSend >10)countSend =0;
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if(huart->Instance == USART3) 
	{
		if (buffer != 10)
		{
			dataBuff[count] = buffer;
			count++;
		}
		else if (buffer == 10)
		{
			updateData (dataBuff[0], dataBuff);
			clearBuffer(dataBuff);
		}
	}
}

void clearBuffer (uint8_t *buff)
{
	for(int i=0;i<9;i++)
  {
		buff[i]= 0;
  }
	count=0;
}
void updateData (uint8_t checkType, uint8_t *data)
{
	switch (checkType)
  {
  	case 'B':
			dataInBackWheel.Dir = dataBuff[1];
			CharToNum (&dataInBackWheel.val, dataBuff, 2);
			TxBack[6] = dataInBackWheel.Dir;
			TxBack[7] = dataInBackWheel.val;
			WriteCAN(BACK,TxBack);
  		break;
		case 'F':
			CharToNum (&dataInFrontWheel.val, dataBuff, 2);
			TxFront[3] = dataBuff[1];
			convertUint32_tTo8byte(dataInFrontWheel.val,TxFront,4,7);
			WriteCAN(FRONT,TxFront);
  		break;
		case 'P':
			CharToNum (&dataInBrake.val, dataBuff, 1);
  		break;
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
