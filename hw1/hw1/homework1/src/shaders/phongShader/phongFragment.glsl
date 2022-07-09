#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 100
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

//判定是否使用bias
float isBias = 0.0;

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

//泊松盘采样
void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

//均匀分布采样
void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

//方便切换采样类型，type大于1时是泊松盘采样，否则是均匀随机采样
void Samples(const in vec2 randomSeed, float type){
  if(type > 1.0) poissonDiskSamples(randomSeed);
  else uniformDiskSamples(randomSeed);
}

//即PCSS第一步，获取遮挡物的平均深度
float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver, float type) {
	Samples(uv, type);
  float totalDepth = 0.0;
  int blockCount = 0;

  for(int i = 0; i < NUM_SAMPLES; ++i){
    vec2 simpleUV = uv + poissonDisk[i] / 2048.0 * 50.0;
    float shadowMapDepth = unpack(vec4(texture2D(shadowMap, simpleUV).rgb, 1.0));
    if(zReceiver > (shadowMapDepth + 0.003)){
      totalDepth += shadowMapDepth;
      blockCount++;
    }
  }
  
  //如果没有遮挡
  if(blockCount == 0) return -1.0;
  //完全被遮挡住
  if(blockCount == NUM_SAMPLES) return 2.0;

  return totalDepth / float(blockCount);
}

float Bias(){
  //通过增加一个偏移量（即遮挡物与阴影之间距离小于偏移量
  //时视作未遮挡）解决Shadow Map中的自遮挡问题。
  //计算光线与当前像素法线的夹角，夹角越大，bias越大。
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  
  return max(0.005*(1.0-dot(normal, lightDir)), 0.005);
}

float PCF(sampler2D shadowMap, vec4 coords, float type, float simpleScale) {
  float bias = Bias();
  bias = 0.0;
  float visibility = 0.0;
  float shadingPointDepth = coords.z;
  //这里是对随机分布正则化
  //因为泊松盘采样的返回值在[-1, 1]，需要映射到shadowMap的坐标中
  float filterSize = 1.0 * simpleScale / 2048.0;
  Samples(coords.xy, type);
  for(int i = 0; i < NUM_SAMPLES; ++i){
    vec2 texcoords = poissonDisk[i] * filterSize + coords.xy;
    float chosenDepth = unpack(vec4(texture2D(shadowMap, texcoords).xyz, 1.0));
    visibility += chosenDepth < shadingPointDepth - 0.001 ? 0.0 : 1.0;
  }
  return visibility / float(NUM_SAMPLES);
}

float PCSS(sampler2D shadowMap, vec4 coords, float type){

  // STEP 1: avgblocker depth
  float zBlocker = findBlocker(shadowMap, coords.xy, coords.z, type);
  // STEP 2: penumbra size
  float penumberaScale = (coords.z - zBlocker) / zBlocker;
  // STEP 3: filtering
  return PCF(shadowMap, coords, type, penumberaScale * 15.0);
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  float bias = 0.0;
  if(isBias > 0.0) bias = Bias();
  //获取纹理的2D坐标，是一个RGBA值
  vec4 rgbaDepth = texture2D(shadowMap, shadowCoord.xy);
  //使用unpack函数将RGBA转为深度值
  float depth = unpack(rgbaDepth);
  //渲染的点的z就是它coord的z
  float shadingPointDepth = shadowCoord.z;
  //判断点是否被遮挡，返回0/1
  if(shadingPointDepth - bias < depth) return 1.0;
  return 0.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}


void main(void) {
  //设置渲染效果
  float visibility;
  vec4 shadowCoord = vPositionFromLight;
  shadowCoord = shadowCoord * 0.5 + 0.5;
  //使用Shadow Map，但是不设置bias
  //visibility = useShadowMap(uShadowMap, shadowCoord);
  //使用Shadow Map，设置bias
  //isBias = 1.0; visibility = useShadowMap(uShadowMap, shadowCoord);
  //使用PCF阴影抗锯齿,采用泊松盘
  //visibility = PCF(uShadowMap, shadowCoord, 2.0, 5.0);
  //使用PCF，采用均匀随机采样
  //visibility = PCF(uShadowMap, shadowCoord, 0.0, 5.0);
  //使用PCSS生成软阴影，使用泊松盘
  visibility = PCSS(uShadowMap, shadowCoord, 2.0);
  //PCSS使用均匀采样
  //visibility = PCSS(uShadowMap, shadowCoord, 0.0);

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}