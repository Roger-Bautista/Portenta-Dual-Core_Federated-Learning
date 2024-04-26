// If your target is limited in memory remove this macro to save 10K RAM

#define ONCOMPUTER 0
#include "nn.h"


#include "mbed.h"
#include <vector>

using namespace mbed;
using namespace rtos;



#define T_BEAM_V10 // ttgo-t-beam
// #define T_BEAM_LORA_32 // ttgo-lora32-v1

#if defined(T_BEAM_LORA_32)
#define LED LED_BUILTIN
#define LED_ON      HIGH
#define LED_OFF     LOW
#elif defined(T_BEAM_V10)
#define LED 4
#define LED_ON      LOW
#define LED_OFF     HIGH
#endif

static u32 memoryUsed = 0;
static f32 lr = 0.001f;

struct train_data{
    M input;
    M target;
};

struct train_data_cnn{
    M3 input;
    M target;
};

#define OUTPUT_SIZE 25

// CNN DEFINES
#define SIZE 28
#define CHANNELS 1
#define KERNEL_SIZE 5
#define OUT_CHANNELS 8

//SHARED MEMORY DEFINES
#define USING_SRAM4               false

#define SRAM3_START_ADDRESS       ((uint32_t) 0x30040000)
#define SRAM4_START_ADDRESS       ((uint32_t) 0x38000000)

#if USING_SRAM4
  // Using AHB SRAM4 at 0x38000000
  #define SRAM_START_ADDRESS        SRAM4_START_ADDRESS

  //#define ARRAY_SIZE                16000
  const uint16_t ARRAY_SIZE = 16000-5;
  
  // Max 64K - 8 bytes (1 * uint32_t + 2 * uint8_t)
  #if ( ARRAY_SIZE > (65536 - 8) / 4 )
    #error ARRAY_SIZE must be < 16382
  #endif
#else
  // Using AHB SRAM3 at 0x30040000
  #define SRAM_START_ADDRESS        SRAM3_START_ADDRESS

  //#define ARRAY_SIZE                8000
  const uint16_t ARRAY_SIZE = 8000-5;        
  
  // Max 32K - 8 bytes (1 * uint32_t + 2 * uint8_t)
  #if ( ARRAY_SIZE > (32768 - 8) / 4 )
    #error ARRAY_SIZE must be < 8190
  #endif
#endif

#ifndef BUFF_CORES_SIZE
	#define BUFF_CORES_SIZE	32
#endif


typedef struct {
    unsigned int led;
    unsigned int nextLed;
    unsigned int otherLed;
    boolean update;
	// Flags to lock reading or writing
	unsigned int buff4to7_size;
	unsigned int buff7to4_size;

	int status_CM7_to_CM4;	// CM4 semaphor flag -> 1 - Can Read | 0 -> Can write - Default
	int status_CM4_to_CM7;	// CM7 semaphor flag -> 1 - Can read | 0 -> Can Write - Default
    int printSerial;

	f32 buff4to7[ARRAY_SIZE/2];	// Buffer to transfer from core 4 to core 7
	f32 buff7to4[ARRAY_SIZE/2];	// Buffer to transfer from core 7 to core 4

	// Stored buffer sizes. MUST BE LESS THAN ARRAY_SIZE/2
} shared_data_TypeDef;

// FF added fct declarations for platformio
void initNetworkModel();
void sendFloat (float arg);
void sendInt (int arg);
//void read_bias(Conv2D* l);
//void read_weights(Layer* l);


#define shared_data ((shared_data_TypeDef *)SRAM_START_ADDRESS)

static unsigned int buffer_size_limited_4to7, buffer_size_limited_7to4;

void core_share_init() {
	shared_data->status_CM7_to_CM4 = false;
	shared_data->status_CM4_to_CM7 = false;
    shared_data->printSerial = false;
}




void MPU_Config()
{

  /*
  MPU - The MPU is an optional component for the memory protection. Including the MPU
        in the STM32 microcontrollers (MCUs) makes them more robust and reliable.
        The MPU must be programmed and enabled before using it. If the MPU is not enabled,
        there is no change in the memory system behavior.

  HAL - The STM32 Hardware Abstraction Layer (HAL) provides a simple, generic multi-instance
        set of APIs (application programming interfaces) to interact with the upper layers
        like the user application, libraries and stacks.

  */
  MPU_Region_InitTypeDef MPU_InitStruct;

  /* Disable the MPU */
  HAL_MPU_Disable();

  /////////////
  
  /* Configure the MPU attributes as WT for SDRAM */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;

  // Base address of the region to protect
#if USING_SRAM4
  MPU_InitStruct.BaseAddress = SRAM4_START_ADDRESS;             // For SRAM4 only
  // Size of the region to protect, 64K for SRAM4
  MPU_InitStruct.Size = MPU_REGION_SIZE_64KB;                   // Important to access more memory
#else
  MPU_InitStruct.BaseAddress = SRAM3_START_ADDRESS;             // For SRAM3 only
  // Size of the region to protect, only 32K for SRAM3
  MPU_InitStruct.Size = MPU_REGION_SIZE_32KB;                   // Important to access more memory
#endif

  // Region access permission type
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;

  // Shareability status of the protected region
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;

  /////////////

  // Optional

  // Bufferable status of the protected region
  //MPU_InitStruct.IsBufferable = MPU_ACCESS_BUFFERABLE;
  // Cacheable status of the protected region
  //MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
  // Number of the region to protect
  //MPU_InitStruct.Number = MPU_REGION_NUMBER7;
  // TEX field level
  //MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  // number of the subregion protection to disable
  //MPU_InitStruct.SubRegionDisable = 0x00;
  // instruction access status
  //MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;

  /////////////

  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  /* Enable the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

struct buffer_struct{
    unsigned int size;
    f32 *buffer;

    buffer_struct(unsigned int size, f32 *buffer) : buffer(buffer), size(size) {};
    buffer_struct(const buffer_struct& obj){
        size = obj.size;
        buffer = new f32[obj.size];
        for (int i = 0; i < obj.size; i++){
            buffer[i] = obj.buffer[i];
        }
    };

    inline buffer_struct operator=(buffer_struct obj){
        Serial.print("CCCC");
        size = obj.size;
        buffer = new f32[obj.size];
        for (int i = 0; i < obj.size; i++){
            Serial.print(obj.buffer[i]);
            buffer[i] = obj.buffer[i];
        }
        Serial.print("CCCC");
    };
};

#ifdef CORE_CM4
/*
 * Get data from M4 to M7
 */

boolean read_to_M7(f32* buffer) {
    boolean return_value = false;
	if (shared_data->status_CM4_to_CM7) {		// if M4 to M7 buffer has data

        for(unsigned int n = 0; n < shared_data->buff4to7_size; ++n) {
        buffer[n] = shared_data->buff4to7[n];	// Transfer data
		
        return_value = true;
        }
        shared_data->status_CM4_to_CM7 = false; // Unlock buffer and marks that it can write again
	}
    return return_value;
}


/*
 * Send data from M7 to M4
 */

void  write_to_M4(f32 buffer[], unsigned int buffer_size) {

	if (!shared_data->status_CM7_to_CM4) {	// if M7 to M4 buffer is not locked

		buffer_size_limited_7to4 = (buffer_size > ARRAY_SIZE/2) ? ARRAY_SIZE/2 : buffer_size;

		shared_data->buff7to4_size = buffer_size_limited_7to4;
		for (unsigned int n = 0; n < buffer_size_limited_7to4; ++n) {
			shared_data->buff7to4[n] = buffer[n];	// Transfer data
        }
		shared_data->status_CM7_to_CM4 = true;

	}
    return;
}


void print_from_M4(f32 buffer[], unsigned int buffer_size) {
	if (!shared_data->status_CM7_to_CM4) {	// if M4 to M7 buffer is not locked
        buffer_size_limited_7to4 = (buffer_size > ARRAY_SIZE/2) ? ARRAY_SIZE/2 : buffer_size;

		shared_data->buff7to4_size = buffer_size_limited_7to4;
		for (unsigned int n = 0; n < buffer_size_limited_7to4; ++n) {
			shared_data->buff7to4[n] = buffer[n];	// Transfer data
		}
		shared_data->status_CM7_to_CM4 = true;
        shared_data->printSerial = true;

	}
    return;
}

train_data buffer_to_train_data(f32* buffer, char& typeOfOperation){

    f32 printBuffer[ARRAY_SIZE/4];
    

    typeOfOperation = char(buffer[0]);
    
    //sample.input.cols = buffer[1];
    //sample.input.rows = buffer[2];
    unsigned int input_size = buffer[1] * buffer[2];
    //sample.target.cols = buffer[input_size+3];
    //sample.target.rows = buffer[input_size+4];
    unsigned int target_size = buffer[input_size+3] * buffer[input_size+4];

    train_data sample = {};
    sample.input = M::zeros(buffer[2], buffer[1]);
    sample.target = M::zeros(buffer[input_size+4], buffer[input_size+3]);

    for(unsigned int n = 0; n<input_size; n++){
        printBuffer[0] = buffer[n+3];
        print_from_M4(printBuffer, 1);
        delay(200);
        sample.input.data[n] = buffer[n+3];
        printBuffer[0] = sample.input.data[n];
        print_from_M4(printBuffer, 1);
        delay(200);
    }
    
    for(unsigned int n = 0; n<target_size; n++){
        printBuffer[0] = buffer[n+input_size+5];
        print_from_M4(printBuffer, 1);
        delay(200);
        sample.target.data[n] = buffer[n+input_size+5];
        printBuffer[0] = sample.target.data[n];
        print_from_M4(printBuffer, 1);
        delay(200);
    }

    return sample;
}


//Order of Buffer info: error, matrix.cols, matrix.rows, matrix.data
buffer_struct M_to_buffer(M matrix, f32 error){
    unsigned int matrix_size = matrix.rows * matrix.cols;
    f32 buffer[matrix_size + 3];
    buffer[0] = error;
    buffer[1] = matrix.cols;
    buffer[2] = matrix.rows;
    for (unsigned int n=0; n<matrix_size; n++){
        buffer[n+3] = matrix.data[n];
    }

    buffer_struct result(matrix_size + 3, buffer);
    return result;
}

#endif



#ifdef CORE_CM7

/*
 * Get data from M7 to M4
 */

boolean read_to_M4(f32 *buffer) {
    boolean return_value = false;
    if (shared_data->status_CM7_to_CM4) {	// if M7 to M4 buffer has data
        if (shared_data->printSerial){
            for(unsigned int n = 0; n < shared_data->buff7to4_size; ++n){
                Serial.print(shared_data->buff7to4[n]);
            }
            shared_data->printSerial = false;
            shared_data->status_CM7_to_CM4 = false;
        }
		else {
           for(unsigned int n = 0; n < shared_data->buff7to4_size; ++n) {
			*(buffer+n) = shared_data->buff7to4[n];	// Transfer data
		}
		
        return_value = true;
        }
        shared_data->status_CM7_to_CM4 = false;
		
	}
    return return_value;
}


/*
 * Send data from M4 to M7
 */

void write_to_M7(f32* buffer, unsigned int buffer_size) {
	if (!shared_data->status_CM4_to_CM7) {	// if M4 to M7 buffer is not locked
		buffer_size_limited_4to7 = (buffer_size > ARRAY_SIZE/2) ? ARRAY_SIZE/2 : buffer_size;
        
		shared_data->buff4to7_size = buffer_size_limited_4to7;
		for (unsigned int n = 0; n < buffer_size_limited_4to7; ++n) {
			shared_data->buff4to7[n] = buffer[n];	// Transfer data
		}
		shared_data->status_CM4_to_CM7 = true;

	}
    return;
}

//Order of Buffer info: typeOfOperation, input.cols, input.rows, input.data, target.cols, target.rows, target.data
/*buffer_struct train_data_to_buffer(char typeOfOperation, train_data sample){
    /*
    sample.input.data
    sample.input.cols
    sample.input.rows
    
    sample.target.data
    sample.target.cols
    sample.target.rows
    

    unsigned int input_size = sample.input.cols * sample.input.rows;
    unsigned int target_size = sample.target.cols * sample.target.rows;

    f32 buffer[input_size + target_size + 5];

    buffer[0] = typeOfOperation;
    buffer[1] = sample.input.cols;
    buffer[2] = sample.input.rows;
    
    for(unsigned int n=0; n < input_size; n++){
        buffer[n+3] = sample.input.data[n];
    }

    buffer[input_size + 3] = sample.target.cols;
    buffer[input_size + 4] = sample.target.rows;
    for(unsigned int n=0; n < target_size; n++){
        buffer[input_size+5+n] = sample.target.data[n];
    }
    buffer_struct result(input_size + target_size + 5, buffer);
    return result;
}
*/

void train_data_to_buffer(char typeOfOperation, train_data sample, int& size, f32* buffer){
    /*
    sample.input.data
    sample.input.cols
    sample.input.rows
    
    sample.target.data
    sample.target.cols
    sample.target.rows
    */

    unsigned int input_size = sample.input.cols * sample.input.rows;
    unsigned int target_size = sample.target.cols * sample.target.rows;

    buffer[0] = typeOfOperation;
    buffer[1] = sample.input.cols;
    buffer[2] = sample.input.rows;
    
    for(unsigned int n=0; n < input_size; n++){
        buffer[n+3] = sample.input.data[n];
    }

    buffer[input_size + 3] = sample.target.cols;
    buffer[input_size + 4] = sample.target.rows;
    for(unsigned int n=0; n < target_size; n++){
        buffer[input_size+5+n] = sample.target.data[n];
    }
    //buffer_struct result(input_size + target_size + 5, buffer);
    size = input_size + target_size + 5;
    return;
}

M buffer_to_M(f32* buffer, f32& error){
    error = buffer[0];
    M data;
    data.cols = buffer[1];
    data.rows = buffer[2];
    unsigned int matrix_size = data.cols * data.rows;
    for(unsigned int n=0; n<matrix_size; n++){
        data.data[n] = buffer[n+3];
    }
    return data;
}

void print_from_M4(f32 buffer[], unsigned int buffer_size) {
    return;
}

#endif




static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal


// /** Audio buffers, pointers and selectors */
// typedef struct {
//     int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
//     uint8_t buf_ready;
//     uint32_t buf_count;
//     uint32_t n_samples;
// } inference_t;

// static inference_t inference;

// typedef uint8_t scaledType;

// static scaledType microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
//     numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
//     return 0;
// }

static Layer *l1;
static Layer *l2;
//static Layer* l1;




void read_weights(Layer* l) {
    //Serial.println("Receiving weights..");
    float* weights = l->w.data;

    for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        weights[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}


void send_weights(MaxPooling*l){
    float* weights = l->grad_input.data;

    for (u32 i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_weights(Conv2D*l){
    float* weights = l->kernels.data;

    for (u32 i = 0; i < l->kernels.d1 * l->kernels.d2 * l->kernels.d3 * l->kernels.d4; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_bias(Conv2D*l){
    float* weights = l->bias.data;

    for (u16 i = 0; i < l->bias.rows * l->bias.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_weights(Layer *l){
    float* weights = l->w.data;

    for (u16 i = 0; i < l->w.rows * l->w.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_bias(Layer *l){
    float* weights = l->b.data;

    for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}


void read_bias(Layer* l) {
    //Serial.println("Receiving bias..");
    float* bias = l->b.data;
    for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        bias[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
    //     Serial.print(l->b.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->b.std());
}

void read_weights(Conv2D* l) {
    //Serial.println("Receiving weights..");
    float* weights = l->kernels.data;
    for (uint16_t i = 0; i < l->kernels.d1 * l->kernels.d2 * l->kernels.d3 * l->kernels.d4; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        weights[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}

void read_weights(MaxPooling* l) {
    //Serial.println("Receiving weights..");
    // float* weights = l->grad_input.data;
    // for (uint16_t i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
    //     //Serial.write('n');
    //     while (Serial.available() < 4);

    //     char bytes[4];
    //     Serial.readBytes(bytes, 4);
    //     //weights[i] = *reinterpret_cast<float*>(bytes);
    // }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}

void read_bias(Conv2D* l) {
    //Serial.println("Receiving bias..");
    float* bias = l->bias.data;
    for (uint16_t i = 0; i < l->bias.rows * l->bias.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        bias[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
    //     Serial.print(l->b.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->b.std());
}

void read_bias(MaxPooling* l) {
}


void read_layer_weights(Layer* l){
    read_bias(l);
    read_weights(l);
}

void read_layer_weights(Conv2D* l){
    read_bias(l);
    read_weights(l);
}

void read_layer_weights(MaxPooling* l){
    read_bias(l);
    read_weights(l);
}

void send_layer_weights(Layer* l){
    send_weights(l);
    send_bias(l);
}

void send_layer_weights(MaxPooling* l){
    send_weights(l);
    //send_bias(l);
}

void sendLayerMetaData(Layer* l){
    sendInt(-1); // Dense layer
    sendInt(l->w.rows);
    sendInt(l->w.cols);
}

void sendLayerMetaData(MaxPooling* l){
    sendInt(-2); // Max pool layer
    sendInt(l->grad_input.d1);
    sendInt(l->grad_input.d2);
    sendInt(l->grad_input.d3);
}


void sendLayerMetaData(Conv2D* l){
    sendInt(-3); // CNN
    sendInt(l->kernels.d1);
    sendInt(l->kernels.d2);
    sendInt(l->kernels.d3);
    sendInt(l->kernels.d4);
}

// SEND GRADIENTS
void send_gradients(Conv2D*l){
    float* weights = l->dkernels.data;

    for (u32 i = 0; i < l->dkernels.d1 * l->dkernels.d2 * l->dkernels.d3 * l->dkernels.d4; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradient_bias(Conv2D*l){
    float* weights = l->db.data;

    for (u16 i = 0; i < l->db.rows * l->db.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradient_bias(Layer*l){
    float* weights = l->db.data;

    for (u16 i = 0; i < l->db.rows * l->db.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradients(Layer *l){
    float* weights = l->dw.data;

    for (u16 i = 0; i < l->dw.rows * l->dw.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
        // sendFloat(weights[i]);
    }
}

void send_gradients(MaxPooling *l){
    float* weights = l->grad_input.data;

    for (u32 i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}


void send_layer_gradients(Layer* l){
    send_gradients(l);
    send_gradient_bias(l);
}

void send_layer_gradients(MaxPooling* l){
    send_gradients(l);
    //send_gradient_bias(l);
}

void send_layer_weights(Conv2D* l){
    send_weights(l);
    send_bias(l);
}

void send_layer_gradients(Conv2D* l){
    send_gradients(l);
    send_gradient_bias(l);
}

void initNetworkModel(){
    Serial.println("Start receiving model");
    char signal;
    do {
        signal = Serial.read();
        Serial.println(memoryUsed);
    } while(signal != 's'); // s -> START

    Serial.println("start");
    Serial.write("i");

    // READ LEARNING RATE
    while(Serial.available()<4);
    char bytes[4];
    Serial.readBytes(bytes, 4);
    memcpy(&lr, bytes, sizeof(f32));


    // How many layers?
    sendInt(2);
    sendLayerMetaData(l1);
    sendLayerMetaData(l2);
    //sendLayerMetaData(l1);


    // TODO : Change to read_weights(l1): this will read bias and weights within the same function.
    read_layer_weights(l1);
    //read_layer_weights(pl1);
    read_layer_weights(l2);
    //read_layer_weights(l1);
    // read_bias(l1);
    // read_weights(l1);

    // read_bias(l2);
    // read_weights(l2);
}

train_data receive_sample(u32 input_size){
    train_data data = {};
    data.input = M::zeros(1, input_size);
    data.target = M::zeros(1, OUTPUT_SIZE);

    f32* input = data.input.data;
    f32* target = data.target.data;

    for(u16 i=0;i<input_size;++i){
        // Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
        
    }
    
    for(u16 i=0;i<OUTPUT_SIZE;++i){
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        target[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}


train_data_cnn receive_sample_cnn(u32 height, u32 width, u32 channels, u32 output_size){
    train_data_cnn data = {};
    data.input = M3::zeros(height, width, channels);
    data.target = M::zeros(1, output_size);

    f32* input = data.input.data;
    f32* target = data.target.data;


    for(u32 i=0;i<height*width*channels;++i){
        // Serial.write('n');
        //sendInt(i);
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
    }
    
    for(u16 i=0;i<output_size;++i){
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        target[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

void sendM(M arg)
{
    for(u32 i=0;i<arg.cols * arg.rows;++i){
        Serial.write('n');
        sendFloat(arg.data[i]);
    }
}

void sendInferenceResult(M arg, f32 loss)
{
    for(u32 i=0;i<arg.cols * arg.rows;++i){
        Serial.write('n');
        sendFloat(arg.data[i]);
    }

    Serial.write('n');
    sendFloat(loss);
}

void sendFloat (float arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

void sendInt (int arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}


M train(M input, M target, f32& loss){
    f32 printBuffer[ARRAY_SIZE/4];
    printBuffer[0] = f32(5);
    print_from_M4(printBuffer, 1);
    delay(200);
    printBuffer[0] = input.cols;
    print_from_M4(printBuffer, 1);
    delay(200);
    unsigned int input_size = input.cols * input.rows;
    printBuffer[0] = input.rows;
    print_from_M4(printBuffer, 1);
    delay(200);
    printBuffer[0] = target.cols;
    print_from_M4(printBuffer, 1);
    delay(200);
    unsigned int target_size = target.cols * target.rows;
    printBuffer[0] = target.rows;
    print_from_M4(printBuffer, 1);
    delay(200);
    for(unsigned int n = 0; n<input_size; n++){
        printBuffer[0] = input.data[n];
        print_from_M4(printBuffer, 1);
        delay(200);
    }
    
    for(unsigned int n = 0; n<target_size; n++){
        printBuffer[0] = target.data[n];
        print_from_M4(printBuffer, 1);
        delay(200);
    }
    l1->resetGradients();
    l2->resetGradients();
    M a = Tanh(l1->forward(input));
    M b = Tanh(l2->forward(a));
    M _d4 = MsePrime(input, b);
    M _d1 = l2->backward(_d4) * TanhPrime(a);

    // accumulate gradients
    l1->dw += l1->getDelta(_d1, input);
    l1->db += _d1;

    l2->dw += l2->getDelta(_d4, a);
    l2->db += _d4;

    l1->UpdateWeights(lr);
    l2->UpdateWeights(lr);
    loss = Mse(input, b);
    return b;
}

M predict(M input, M target, f32& loss){
    M a = Tanh(l1->forward(input));
    M b = Tanh(l2->forward(a));
    
    // calculate error
    loss = Mse(input, b);
    return b;
}

#ifdef CORE_CM4

M train_for_M4(f32* buffer, f32& loss){
    f32 printBuffer[ARRAY_SIZE/4];
    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);

    unsigned int buffersize = shared_data->buff4to7_size;
    for(int i = 0; i < buffersize; i++){
            printBuffer[0] = buffer[i];
            print_from_M4(printBuffer, 1);
            delay(200);
        }
    
    M input = M::zeros(buffer[2], buffer[1]);
    unsigned int input_size = buffer[1] * buffer[2];
    for(unsigned int i = 0; input_size; i++){
        printBuffer[0] = input.data[i];
        print_from_M4(printBuffer, 1);
        delay(200);
    }
    for(unsigned int i = 0; input_size; i++){
        input.data[i] = buffer[i+3];
        printBuffer[0] = input.data[i];
        print_from_M4(printBuffer, 1);
        delay(200);
    }

    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);


    l1->resetGradients();
    l2->resetGradients();

    
    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);


    M a = Tanh(l1->forward(input));

    
    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);

    M b = Tanh(l2->forward(a));

    
    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);

    M _d4 = MsePrime(input, b);

    
    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);

    M _d1 = l2->backward(_d4) * TanhPrime(a);


    printBuffer[0] = f32(7);
    print_from_M4(printBuffer, 1);
    delay(200);

    // accumulate gradients
    l1->dw += l1->getDelta(_d1, input);
    l1->db += _d1;

    l2->dw += l2->getDelta(_d4, a);
    l2->db += _d4;

    l1->UpdateWeights(lr);
    l2->UpdateWeights(lr);
    loss = Mse(input, b);
    return b;
}

void setup(){
    InitMemory(1024 * 430);
    
    l1 = Layer::create(25, 30);
    l2 = Layer::create(30, 25);
    /*
    Serial.begin(9600);


    memoryUsed = MemoryArena.Used;
    

    //pinMode(LEDR, OUTPUT);
    //pinMode(LEDG, OUTPUT);
    //pinMode(LEDB, OUTPUT);
    pinMode(LED, OUTPUT);

    //digitalWrite(LEDR, HIGH);
    //digitalWrite(LEDG, HIGH);
    //digitalWrite(LEDB, HIGH);
    digitalWrite(LED, LED_ON);
    digitalWrite(LED, LED_OFF);

    // put your setup code here, to run once:
    randomSeed(0);

    initNetworkModel();
    digitalWrite(LED_BUILTIN, LED_OFF);    // OFF   
    */
}

void loop(){
    f32 printBuffer[ARRAY_SIZE/4];
    if(shared_data->status_CM4_to_CM7){
        unsigned int buffersize = shared_data->buff4to7_size;
        f32 buffer[buffersize];
        int read = false;
        while(!read){
            read = read_to_M7(buffer);
        }
        /*
        for(int i = 0; i < buffersize; i++){
            printBuffer[0] = buffer[i];
            print_from_M4(printBuffer, 1);
            delay(200);
        }

        printBuffer[0] = f32(7);
        print_from_M4(printBuffer, 1);
        delay(200);*/

        char inByte;
        M output;
        f32 error = 0;
        //train_data sample = buffer_to_train_data(buffer, inByte);






        inByte = char(buffer[0]);
        
        //sample.input.cols = buffer[1];
        //sample.input.rows = buffer[2];
        unsigned int input_size = buffer[1] * buffer[2];
        //sample.target.cols = buffer[input_size+3];
        //sample.target.rows = buffer[input_size+4];
        unsigned int target_size = buffer[input_size+3] * buffer[input_size+4];

        train_data sample = {};
        sample.input = M::zeros(buffer[2], buffer[1]);
        sample.target = M::zeros(buffer[input_size+4], buffer[input_size+3]);

        for(unsigned int n = 0; n<input_size; n++){
            printBuffer[0] = buffer[n+3];
            print_from_M4(printBuffer, 1);
            delay(200);
            sample.input.data[n] = buffer[n+3];
            printBuffer[0] = sample.input.data[n];
            print_from_M4(printBuffer, 1);
            delay(200);
        }
        
        for(unsigned int n = 0; n<target_size; n++){
            printBuffer[0] = buffer[n+input_size+5];
            print_from_M4(printBuffer, 1);
            delay(200);
            sample.target.data[n] = buffer[n+input_size+5];
            printBuffer[0] = sample.target.data[n];
            print_from_M4(printBuffer, 1);
            delay(200);
        }


        printBuffer[0] = sample.input.cols;
        print_from_M4(printBuffer, 1);
        delay(200);
        printBuffer[0] = sample.input.rows;
        print_from_M4(printBuffer, 1);
        delay(200);
        printBuffer[0] = sample.target.cols;
        print_from_M4(printBuffer, 1);
        delay(200);
        printBuffer[0] = sample.target.rows;
        print_from_M4(printBuffer, 1);
        delay(200);

        for(unsigned int n = 0; n<input_size; n++){
            printBuffer[0] = sample.input.data[n];
            print_from_M4(printBuffer, 1);
            delay(200);
        }
        
        for(unsigned int n = 0; n<target_size; n++){
            printBuffer[0] = sample.target.data[n];
            print_from_M4(printBuffer, 1);
            delay(200);
        }

        if(inByte == 't'){
            error = 0;
            //output = train(sample.input, sample.target, error);
            output = train_for_M4(buffer, error);
        }
        else if(inByte == 'p'){
            error = 0;
            output = predict(sample.input, sample.target, error);
        }
        printBuffer[0] = 7;
        print_from_M4(printBuffer, 1);
        delay(200);
        buffer_struct bufferAux = M_to_buffer(output, error);
        printBuffer[0] = 6;
        print_from_M4(printBuffer, 1);
        delay(200);
        write_to_M4(bufferAux.buffer, bufferAux.size);
    }
    /*
    if(shared_data->status_CM4_to_CM7){
        unsigned int buffersize = shared_data->buff4to7_size;
        f32 buffer[buffersize];
        boolean read = 0;
        printBuffer[0] = 9;
        print_from_M4(printBuffer, 1);
        while(!read){
            read = read_to_M7(buffer);
        }
        printBuffer[0] = 9;
        print_from_M4(printBuffer, 1);
        char inByte;
        M output;
        f32 error = 0;
        train_data sample = buffer_to_train_data(buffer, inByte);
        if(inByte == 't'){
            error = 0;
            output = train(sample.input, sample.target, error);
        }
        else if(inByte == 'p'){
            error = 0;
            output = predict(sample.input, sample.target, error);
        }
        buffer_struct bufferAux = M_to_buffer(output, error);
        write_to_M4(bufferAux.buffer, bufferAux.size);
    }*/
}

#endif

#ifdef CORE_CM7

void initSharedData(){
  shared_data->update = true;
  shared_data->led = LEDG;
  shared_data->nextLed = LEDB;
  shared_data->otherLed = LEDR;
  shared_data->update = false;
}

void setup(){
    MPU_Config();
    bootM4();
    initSharedData();

    
    core_share_init();
    Serial.begin(9600);
    InitMemory(1024 * 430);

    l1 = Layer::create(25, 30);
    l2 = Layer::create(30, 25);

    memoryUsed = MemoryArena.Used;

   //pinMode(LEDR, OUTPUT);
    //pinMode(LEDG, OUTPUT);
    //pinMode(LEDB, OUTPUT);
    pinMode(LED, OUTPUT);

    //digitalWrite(LEDR, HIGH);
    //digitalWrite(LEDG, HIGH);
    //digitalWrite(LEDB, HIGH);
    digitalWrite(LED, LED_ON);
    digitalWrite(LED, LED_OFF);

    // put your setup code here, to run once:
    randomSeed(0);

    initNetworkModel();
    digitalWrite(LED_BUILTIN, LED_OFF);    // OFF   

    digitalWrite(LEDR, LED_OFF);
    digitalWrite(LEDB, LED_OFF);
    digitalWrite(LEDG, LED_OFF);
    
}


void loop(){
    // put your main code here, to run repeatedly:
    if(Serial.available() > 0){
        char inByte = Serial.read();
        if(inByte == 't' || true){
            //train_data_cnn sample = receive_sample_cnn(SIZE,SIZE,CHANNELS, 10);
            uint begin = millis();
            train_data sample = receive_sample(25);
            //buffer_struct aux = train_data_to_buffer(inByte, sample);
            int size;
            f32 aux[250] = {0};
            train_data_to_buffer(inByte, sample, size, aux);
            /*Serial.print("CCCC");
            Serial.print(size);
            Serial.print("BBBB");
            for (int i = 0; i < size; i++){
                Serial.print(aux[i]);
            }
            Serial.print("BBBB");/**/

            write_to_M7(aux, size);
            int read = false;
            while(!read){
                read = read_to_M4(aux);
            }
            Serial.print(8);
            f32 error = 0;
            M output = buffer_to_M(aux,error);
            uint end = millis();
            //sendFloat(error);
            sendInferenceResult(output, error);
            uint elapsed_secs = uint(end - begin);
             byte * data = (byte *) &elapsed_secs; 
            Serial.write(data, sizeof (elapsed_secs));
        }
    }
    /*
    
    if(Serial.available() > 0){
        char inByte = Serial.read();
        if(inByte == 't' || true){
            //train_data_cnn sample = receive_sample_cnn(SIZE,SIZE,CHANNELS, 10);
            train_data sample = receive_sample(25);
            uint begin = millis();
            buffer_struct aux = train_data_to_buffer(inByte, sample);
            write_to_M7(aux.buffer, aux.size);
            //M output = train(sample.input, sample.target, error); //-> CP7 
            boolean read = 0;
            while(!read){
                read = read_to_M4(aux.buffer);
            }
            f32 error = 0;
            M output = buffer_to_M(aux.buffer,error);
            uint end = millis();
            //sendFloat(error);
            sendInferenceResult(output, error);
            uint elapsed_secs = uint(end - begin);
             byte * data = (byte *) &elapsed_secs; 
            Serial.write(data, sizeof (elapsed_secs));
        } else if(inByte == 'p'){
            //train_data_cnn sample = receive_sample_cnn(SIZE,SIZE,CHANNELS, 10);

            train_data sample = receive_sample(25);
            uint begin = millis();
            //M input = receive_sample_inference();
            buffer_struct aux = train_data_to_buffer(inByte, sample);
            sendInt(2);
            write_to_M7(aux.buffer, aux.size);
            //M output = predict(sample.input, sample.target, loss); //-> CP7 
            boolean read = 0;
            while(!read){
                read = read_to_M4(aux.buffer);
            }
            float loss = 0.0;
            M output = buffer_to_M(aux.buffer,loss);
            uint end = millis();
            sendInferenceResult(output, loss);
            MemoryArena.Used = memoryUsed;
            uint elapsed_secs = uint(end - begin);
             byte * data = (byte *) &elapsed_secs; 
            Serial.write(data, sizeof (elapsed_secs));
        } else if(inByte == 'f'){
            // START FEDERATED LEARNING
        } else if(inByte == 'g'){
            send_layer_weights(l1);
            //send_layer_weights(pl1);
            send_layer_weights(l2);
            //send_layer_weights(l1);
            // send_weights(l1);
            // send_bias(l1);
            // send_weights(l2);
            // send_bias(l2);
        } else if(inByte == 'r'){
            // ALWAYS READ BIAS FIRST!!!
            read_layer_weights(l1);
            //read_layer_weights(pl1);
            read_layer_weights(l2);
            //read_layer_weights(l1);

            // read_bias(l1);
            // read_weights(l1);

            // read_bias(l2);
            // read_weights(l2);
        }
    }
    MemoryArena.Used = memoryUsed;
    */
}
#endif