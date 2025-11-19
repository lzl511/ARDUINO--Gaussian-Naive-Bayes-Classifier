#include <math.h>

const int NumFeatures = 2;
const int NumClasses = 2;
const int NumTestSamples = 171;

#define CPU_FREQ_MHZ 480 

const float MODEL_MEANS[NumClasses][NumFeatures] = {
  // Feature 0 (Mean Smoothness), Feature 1 (Worst Texture)
  { 0.102219, 28.941027 }, // Class 0: Malignant 
  { 0.092781, 23.766905 }  // Class 1: Benign 
};

const float MODEL_VARS[NumClasses][NumFeatures] = {
  // Feature 0 (Mean Smoothness), Feature 1 (Worst Texture)
  { 0.000137, 27.148619 }, // Class 0: Malignant 
  { 0.000179, 32.458108 }  // Class 1: Benign 
};

const float MODEL_PRIORS[NumClasses] = {
  0.367, // Class 0: Malignant Prior
  0.633  // Class 1: Benign Prior
};

float LOG_PRIORS[NumClasses];
float LOG_VAR_CONST[NumClasses][NumFeatures]; 
float INV_TWO_VAR[NumClasses][NumFeatures];   

const float TestData[171][2] = {
  {0.1028, 21.1}, {0.08123, 34.24}, {0.0904, 19.35}, {0.07376, 16.94}, {0.0924, 25.05},
  {0.07515, 18.2}, {0.08757, 22.13}, {0.1091, 26.38}, {0.1054, 22.91}, {0.09003, 19.27},
  {0.08817, 17.5}, {0.08352, 17.4}, {0.08363, 17.37}, {0.1169, 30.38}, {0.1053, 28.14},
  {0.07466, 23.89}, {0.07813, 18.26}, {0.1007, 27.0}, {0.09258, 35.64}, {0.123, 27.78},
  {0.1227, 35.34}, {0.08855, 34.69}, {0.09832, 17.24}, {0.08099, 17.6}, {0.09076, 20.79},
  {0.1278, 23.75}, {0.1035, 24.57}, {0.1109, 31.64}, {0.08477, 27.49}, {0.103, 30.36},
  {0.08713, 26.14}, {0.09578, 24.75}, {0.1122, 33.47}, {0.09898, 34.49}, {0.0944, 27.26},
  {0.1102, 32.16}, {0.1447, 23.99}, {0.1092, 26.0}, {0.08388, 28.39}, {0.0842, 45.41},
  {0.1007, 26.84}, {0.1215, 32.72}, {0.1273, 30.73}, {0.08313, 28.46}, {0.0832, 36.91},
  {0.09231, 24.04}, {0.08043, 26.34}, {0.1186, 21.4}, {0.0915, 26.37}, {0.1064, 34.01},
  {0.09883, 20.43}, {0.09597, 32.82}, {0.07963, 15.73}, {0.12, 34.01}, {0.08472, 17.58},
  {0.1197, 32.09}, {0.09742, 31.03}, {0.08142, 23.31}, {0.07941, 22.0}, {0.08597, 12.87},
  {0.108, 29.43}, {0.09342, 23.87}, {0.115, 18.16}, {0.1059, 27.2}, {0.1005, 26.83},
  {0.08302, 20.21}, {0.07721, 19.23}, {0.1184, 17.33}, {0.08445, 23.6}, {0.07355, 22.35},
  {0.08402, 27.83}, {0.1082, 24.85}, {0.1141, 33.48}, {0.09136, 24.38}, {0.1178, 39.42},
  {0.09933, 25.47}, {0.1025, 30.29}, {0.1012, 28.68}, {0.1036, 18.45}, {0.08151, 20.29},
  {0.116, 29.41}, {0.0974, 29.94}, {0.1044, 31.56}, {0.1286, 31.72}, {0.1005, 19.93},
  {0.09984, 19.68}, {0.1132, 18.24}, {0.1036, 25.41}, {0.09405, 29.72}, {0.1133, 25.07},
  {0.07683, 22.25}, {0.08682, 16.85}, {0.08588, 26.36}, {0.09916, 25.63}, {0.08365, 18.93},
  {0.08772, 25.05}, {0.08837, 31.73}, {0.06828, 21.8}, {0.08386, 27.06}, {0.1026, 23.05},
  {0.1016, 28.81}, {0.0926, 36.27}, {0.08098, 30.92}, {0.09831, 30.88}, {0.09267, 21.18},
  {0.0784, 31.67}, {0.09138, 17.04}, {0.09905, 33.81}, {0.1046, 21.9}, {0.09586, 28.36},
  {0.08098, 18.22}, {0.08464, 19.16}, {0.08759, 29.15}, {0.08637, 25.72}, {0.1006, 28.07},
  {0.1175, 25.4}, {0.07561, 19.74}, {0.08791, 34.23}, {0.1089, 30.39}, {0.07497, 26.56},
  {0.1044, 27.78}, {0.09783, 15.67}, {0.1045, 20.92}, {0.06576, 25.26}, {0.07903, 31.31},
  {0.08045, 25.34}, {0.09469, 47.16}, {0.08108, 19.29}, {0.1006, 16.82}, {0.08685, 28.88},
  {0.09345, 22.46}, {0.1094, 32.85}, {0.09423, 19.05}, {0.1425, 26.5}, {0.08458, 21.58},
  {0.1, 39.16}, {0.08371, 19.31}, {0.08946, 25.62}, {0.1066, 22.65}, {0.1007, 20.86},
  {0.09714, 29.89}, {0.09401, 30.9}, {0.0909, 27.84}, {0.08421, 25.82}, {0.1024, 26.29},
  {0.1236, 32.19}, {0.09524, 22.47}, {0.1003, 22.88}, {0.07818, 19.69}, {0.1634, 16.38},
  {0.08217, 25.73}, {0.09752, 15.4}, {0.09037, 20.88}, {0.09699, 24.39}, {0.09373, 15.97},
  {0.07734, 16.47}, {0.1003, 16.67}, {0.09812, 44.87}, {0.09488, 35.27}, {0.08983, 22.81},
  {0.1071, 33.82}, {0.1075, 25.16}, {0.08752, 33.33}, {0.1248, 15.38}, {0.1015, 33.58},
  {0.08284, 25.5}, {0.09384, 20.5}, {0.1002, 29.16}, {0.07618, 21.75}, {0.1165, 25.21},
  {0.1037, 28.65}
};

const int TestLabels[171] = {
  1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1,
  0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0,
  1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
  0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
  0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0
};

void initCycleCounter() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; 
  DWT->CYCCNT = 0;                                
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;           
}

uint32_t getCycles() {
  return DWT->CYCCNT;
}

void precomputeConstants() {
  for (int c = 0; c < NumClasses; c++) {
    LOG_PRIORS[c] = log(MODEL_PRIORS[c]);
    
    for (int f = 0; f < NumFeatures; f++) {
      float v = MODEL_VARS[c][f];
      if (v < 1e-9) v = 1e-9;
      LOG_VAR_CONST[c][f] = -0.5f * log(2.0f * PI * v);
      INV_TWO_VAR[c][f] = 1.0f / (2.0f * v);
    }
  }
}

int predict(const float sample[]) {
  float logProbs[NumClasses];

  for (int c = 0; c < NumClasses; c++) {
    logProbs[c] = LOG_PRIORS[c];
    for (int f = 0; f < NumFeatures; f++) {
      float diff = sample[f] - MODEL_MEANS[c][f];
      logProbs[c] += LOG_VAR_CONST[c][f] - (diff * diff * INV_TWO_VAR[c][f]);
    }
  }
  return (logProbs[0] > logProbs[1]) ? 0 : 1;
}

void printResult(const char* label, int prediction, int trueLabel, uint32_t cycles) {
  float time_ns = (float)cycles * (1000.0f / CPU_FREQ_MHZ); 
  
  Serial.print(label);
  Serial.print(" -> Pred: ");
  Serial.print(prediction);
  Serial.print(" (True: ");
  Serial.print(trueLabel);
  Serial.print(") | Cycles: ");
  Serial.print(cycles);
  Serial.print(" | Time: ");
  Serial.print(time_ns, 2);
  Serial.println(" ns");
}

void setup() {
  Serial.begin(115200);
  while (!Serial); 
  delay(2000);

  initCycleCounter(); 
  precomputeConstants(); 
  
  int truePos = 0, falseNeg = 0, falsePos = 0, trueNeg = 0;

  for (int i = 0; i < NumTestSamples; i++) {
    char labelBuffer[20];
    sprintf(labelBuffer, "Sample %d", i + 1);
    uint32_t startCycles = getCycles();
    int prediction = predict(TestData[i]);
    uint32_t endCycles = getCycles();
    uint32_t durationCycles = endCycles - startCycles;
    printResult(labelBuffer, prediction, TestLabels[i], durationCycles);
    if (TestLabels[i] == 1 && prediction == 1) truePos++;      
    else if (TestLabels[i] == 1 && prediction == 0) falseNeg++; 
    else if (TestLabels[i] == 0 && prediction == 1) falsePos++; 
    else if (TestLabels[i] == 0 && prediction == 0) trueNeg++;  
  }
  
  Serial.println("\n=== Confusion Matrix ===");
  Serial.println("Predicted   |  0(Mal) |  1(Ben) ");
  Serial.println("   True     |---------|---------");
  Serial.print("   0(Mal)   |   "); Serial.print(trueNeg); Serial.print("    |   "); Serial.println(falsePos);
  Serial.print("   1(Ben)   |   "); Serial.print(falseNeg); Serial.print("    |   "); Serial.println(truePos);
  
  float accuracy = (float)(truePos + trueNeg) / NumTestSamples * 100.0;
  Serial.print("\nAccuracy: ");
  Serial.print(accuracy);
  Serial.println("%");
}

void loop() {

}