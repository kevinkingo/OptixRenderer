/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixVox: a sample that renders a subset of the VOX file format from MagicaVoxel @ ephtracy.
// Demonstrates non-triangle geometry, and naive random path tracing.
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <time.h>
#include <limits.h>

#include <opencv2/opencv.hpp>

#include <time.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <cmath>
#include <map>


#include "inout/readXML.h"
#include "structs/cameraInput.h"
#include "lightStructs.h"
#include "inout/rgbe.h"
#include "inout/relativePath.h"
#include "Camera.h"
#include "sutil.h"
#include "shapeStructs.h"

#include "creator/createAreaLight.h"
#include "creator/createCamera.h"
#include "creator/createContext.h"
#include "creator/createEnvmap.h"
#include "creator/createGeometry.h"
#include "creator/createMaterial.h"
#include "creator/createPointFlashLight.h"
#include "sampler/sampler.h" 
#include "postprocessing/filter.h"
#include "stdio.h"

using namespace optix;

long unsigned vertexCount(const std::vector<shape_t>& shapes)
{
    long unsigned vertexSum = 0;
    for(int i = 0; i < shapes.size(); i++){
        shape_t shape = shapes[i];
        int vertexNum = shape.mesh.positions.size() / 3;
        vertexSum += vertexNum;
    }
    return vertexSum;
}

void boundingBox(
        Context& context,
        const std::vector<shape_t>& shapes
        )
{
    float3 vmin, vmax;
    for(int i = 0; i < shapes.size(); i++){
        shape_t shape = shapes[i];
        int vertexNum = shape.mesh.positions.size() / 3;
        for(int j = 0; j < vertexNum; j++){
            float vx = shape.mesh.positions[3*j];
            float vy = shape.mesh.positions[3*j+1];
            float vz = shape.mesh.positions[3*j+2];
            float3 v = make_float3(vx, vy, vz);
            if(j == 0 && i == 0){
                vmin = v;
                vmax = v;
            }
            else{
                vmin = fminf(vmin, v);
                vmax = fmaxf(vmax, v);
            }
        }
    }
    float infiniteFar = length(vmax - vmin) * 10;
    std::cout<<"The length of diagonal of bouncding box: "<<(infiniteFar / 10) <<std::endl;
    printf("Max X: %.3f Max Y: %.3f Max Z: %.3f \n", vmax.x, vmax.y, vmax.z);
    printf("Min X: %.3f Min Y: %.3f Min Z: %.3f \n", vmin.x, vmin.y, vmin.z);
    context["infiniteFar"] -> setFloat(infiniteFar);
    context["scene_epsilon"]->setFloat(infiniteFar / 1e6);
}

float clip(float a, float min, float max){
    a = (a < min) ? min : a;
    a = (a > max) ? max : a;
    return a;
}


bool writeBufferToFile(std::string fileName, float* imgData, int width, int height, int mode)
{   
    std::string suffix;
    std::size_t pos = fileName.find_last_of(".");
    if(pos == std::string::npos ){
        suffix = std::string("");
    }
    else{
        suffix = fileName.substr(pos+1, fileName.length() );
    }

    // if(mode == 5){
    //     std::ofstream depthOut(fileName, std::ios::out|std::ios::binary);
    //     depthOut.write((char*)&height, sizeof(int) );
    //     depthOut.write((char*)&width, sizeof(int) );

    //     float* image = new float[width * height];
    //     for(int i = 0; i < height; i++){
    //         for(int j = 0; j < width; j++){
    //             image[i*width + j] = imgData[3 * ( (height-1-i) * width +j ) ];
    //         }
    //     }
    //     depthOut.write( (char*)image, sizeof(float) * width * height);
    //     depthOut.close();
    //     delete [] image;

    //     return true;
    // }

    if(suffix == std::string("hdr")) {
        FILE* imgOut = fopen(fileName.c_str(), "w");
        if(imgOut == NULL){
            std::cout<<"Wrong: can not open the output file!"<<std::endl;
            return false;
        }
        float* image = new float[width * height * 3];
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                for(int ch = 0; ch < 3; ch ++){
                    image[3*(i * width + j) + ch] = imgData[3*( (height-1-i) * width + j) + ch];
                }
            }
        }
        RGBE_WriteHeader(imgOut, width, height, NULL);
        RGBE_WritePixels(imgOut, image, width * height);
        fclose(imgOut);
        delete [] image;
    }
    else{
        cv::Mat image(height, width, CV_8UC3);
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                int ind = 3 * ( (height - 1 -i) * width + j);
                float r = imgData[ind];
                float g = imgData[ind + 1]; 
                float b = imgData[ind + 2];
                if(mode == 0 || mode == 1){
                    r = pow(r, 1.0f/2.2f);
                    g = pow(g, 1.0f/2.2f);
                    b = pow(b, 1.0f/2.2f);
                }
                r = clip(r, 0.0f, 1.0f);
                g = clip(g, 0.0f, 1.0f);
                b = clip(b, 0.0f, 1.0f);
                image.at<cv::Vec3b>(i, j)[0] = (unsigned char)(b * 255);
                image.at<cv::Vec3b>(i, j)[1] = (unsigned char)(g * 255);
                image.at<cv::Vec3b>(i, j)[2] = (unsigned char)(r * 255);
            }
        } 
        cv::imwrite(fileName.c_str(), image);
    }

    return true;
}


bool loadConfigFile(
    std::string fileName,
    int& renderNum, int& renderWidth, int& renderHeight, std::string& sampleType, int& sampleNum,
    std::vector<int>& modes, std::vector<int>& camIds, std::vector<int>& lightIds, std::vector<std::string>& outputFilenames,
    std::vector<CameraInput>& cameraInputs, std::vector<Envmap>& envmaps, std::vector<Point>& points
)
{    
    std::ifstream camIn(fileName.c_str(), std::ios::in);
    if(!camIn ){
        std::cout<<"Wrong: can not load the camera file "<<fileName<<" !"<<std::endl;
        return false;
    }
    // Render config
    camIn >> renderNum >> renderWidth >> renderHeight >> sampleType >> sampleNum; 
    for (int i = 0; i < renderNum; i++) {
        int mode, camId, lightId;
        std::string outputFilename;

        camIn >> mode >> camId >> lightId >> outputFilename;
        modes.push_back(mode);
        camIds.push_back(camId);
        lightIds.push_back(lightId);
        outputFilenames.push_back(outputFilename);
    }

    // Camera config
    int camNum;
    camIn >> camNum;
    for (int i = 0; i < camNum; i++) {
        CameraInput camInput;
        camIn >> camInput.cameraType;
        camIn >> camInput.origin[0] >> camInput.origin[1] >> camInput.origin[2];
        camIn >> camInput.target[0] >> camInput.target[1] >> camInput.target[2];
        camIn >> camInput.up[0] >> camInput.up[1] >> camInput.up[2];
        camIn >> camInput.fov;
        camInput.width = renderWidth;
        camInput.height = renderHeight;
        cameraInputs.push_back(camInput);
    }

    // Light config
    int lightNum;
    camIn >> lightNum;
    for(int i = 0; i < lightNum; i++){
        std::string lightType;
        camIn >> lightType;
        if (lightType == std::string("envmap")) {
            Envmap env;
            camIn >> env.scale;
            camIn >> env.fileName;
            envmaps.push_back(env);
        }
        else if (lightType == std::string("point")) {
            Point p;
            camIn >> p.intensity.x >> p.intensity.y >> p.intensity.z;
            camIn >> p.position.x >> p.position.y >> p.position.z;
            points.push_back(p);
        }
    }
    return true;
}


int main( int argc, char** argv )
{
    bool use_pbo  = false;

    std::string fileName;
    std::string configFile("");
    int maxIteration = -1;
    std::vector<int> gpuIds;
    float noiseLimit = 0.11;
    int vertexLimit = 150000;
    float intensityLimit = 0.05;
    bool noiseLimitEnabled = false;
    bool vertexLimitEnabled = false;
    bool intensityLimitEnabled = false;
    int maxPathLength = 7;
    int rrBeginLength = 5;
    
    bool isForceOutput = false;
    bool isMedianFilter = false;

    Context context = 0;

    for(int i = 0; i < argc; i++){
        if(i == 0){
            continue;
        }
        else if(std::string(argv[i]) == std::string("-f") ){
            if(i == argc - 1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            fileName = std::string(argv[++i] ); 
        }
        else if(std::string(argv[i] ) == std::string("-c") ){
            if(i == argc - 1){
                std::cout<<"Missing inut variable"<<std::endl;
                exit(1);
            }
            configFile = std::string(argv[++i] );
        }
        else if(std::string(argv[i] ) == std::string("--gpuIds") ){
            if(i == argc - 1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            while(i + 1 <= argc-1 && argv[i+1][0] != '-'){
                gpuIds.push_back(atoi(argv[++i] ) );
            }
        }
        else if(std::string(argv[i] ) == std::string("--noiseLimit") ){
            if(i == argc - 1){
                std::cout<<"Missing  input variable"<<std::endl;
                exit(1);
            }
            std::vector<float> flArr = parseFloatStr(std::string(argv[++i] ) );
            noiseLimit = flArr[0];
            noiseLimitEnabled = true;
            std::cout<<"Warning: noiseLimit "<<noiseLimit<<"is enabled!"<<std::endl;
        }
        else if(std::string(argv[i] ) == std::string("--vertexLimit") ){
            if(i == argc - 1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            vertexLimit = atoi(argv[++i] );
            vertexLimitEnabled = true; 
            std::cout<<"Warning: vertexLimit "<<vertexLimit<<"is enabled!"<<std::endl;
        }
        else if(std::string(argv[i] ) == std::string("--intensityLimit") ){
            if(i == argc - 1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            std::vector<float> flArr = parseFloatStr(std::string(argv[++i] ) );
            intensityLimit = flArr[0];
            intensityLimitEnabled = true;
        }
        else if(std::string(argv[i] ) == std::string("--forceOutput") ){
            isForceOutput = true;   
        }
        else if(std::string(argv[i] ) == std::string("--rrBeginLength") ){
            if(i == argc - 1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            rrBeginLength = atoi(argv[++i] );
        }
        else if(std::string(argv[i] ) == std::string("--maxPathLength") ){
            if(i == argc-1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            maxPathLength = atoi(argv[++i] );
        } 
        else if(std::string(argv[i] ) == std::string("--maxIteration") ){
            if(i == argc-1){
                std::cout<<"Missing input variable"<<std::endl;
                exit(1);
            }
            maxIteration = atoi(argv[++i] );
        } 
        else if(std::string(argv[i] ) == std::string("--medianFilter") ){ 
            isMedianFilter = true;
        }
        else{
            std::cout<<"Unrecognizable input command"<<std::endl;
            exit(1);
        }
    }
    
    char fileNameNew[PATH_MAX+1 ];
    char* isRealPath = realpath(fileName.c_str(), fileNameNew );
    if(isRealPath == NULL){
        std::cout<<"Wrong: Fail to transform the realpath of XML to true path."<<std::endl;
        return false;
    }
    fileName = std::string(fileNameNew );
    std::cout<<"Input file name: "<<fileName<<std::endl;

    std::vector<shape_t> shapes;
    std::vector<material_t> materials;
    bool isXml = readXML(fileName, shapes, materials);
    if(!isXml ) return false;

    long unsigned vertexNum = vertexCount(shapes );
    std::cout<<"Material num: "<<materials.size() << std::endl;
    std::cout<<"Shape num: "<<shapes.size() <<std::endl;
    std::cout<<"Vertex num: "<<vertexNum <<std::endl;
    if(vertexLimitEnabled && vertexNum > vertexLimit){
        std::cout<<"Warning: the model is too huge, will be skipped!"<<std::endl;
        return 0;
    }

    // Config File
    // The begining of the file should be the number of camera we are going to HAVE
    // Then we just load the camera view one by one, in the order of origin, target and up
    int renderNum, renderWidth, renderHeight;
    std::string sampleType;
    int sampleNum;
    std::vector<int> modes, camIds, lightIds;
    std::vector<std::string> outputFilenames;

    std::vector<CameraInput> cameraInputs;
    std::vector<Envmap> envmaps;
    std::vector<Point> points;

    if(configFile != std::string("") ){
        configFile = relativePath(fileName, configFile );
        bool isLoad = loadConfigFile(
            configFile,
            renderNum, renderWidth, renderHeight, sampleType, sampleNum,
            modes, camIds, lightIds, outputFilenames,
            cameraInputs, envmaps, points);
        if (!isLoad ) return false;
    }
    else{
        std::cout<<"Warning: wrong config file"<<std::endl;
        return 0;
    }

    
    float* imgData = new float[renderWidth * renderHeight * 3];
    for(int i = 0; i < renderNum; i++){ 

        createContext(context, use_pbo, renderWidth, renderHeight, rrBeginLength);
        if(gpuIds.size() != 0){
            std::cout<<"GPU Num: "<<gpuIds.size()<<std::endl;
            context -> setDevices(gpuIds.begin(), gpuIds.end() );
        }
        boundingBox(context, shapes);

        // Filename
        std::string outputFileName = relativePath(fileName, outputFilenames[i] );
        std::ifstream f(outputFileName.c_str() );
        if(f.good() && !isForceOutput ) {
            std::cout<<"Warning: "<<outputFileName<<" already exists. Will be skipped."<<std::endl;
            continue;
        }
        else{
            std::cout<<"Output Image: "<<outputFileName<<std::endl;
        }

        // Render config
        unsigned maxDepth = maxPathLength;
        unsigned sqrt_num_samples = (unsigned )(sqrt(float(sampleNum )) + 1.0);
        if(sqrt_num_samples == 0){
            sqrt_num_samples = 1;
        }
        if(modes[i] > 0){
            maxDepth = 1;
            sqrt_num_samples = 1;
        }

        context["max_depth"]->setInt(maxDepth);
        context["sqrt_num_samples"] -> setUint(sqrt_num_samples);

        // Geometry Programs
        std::string miss_path( ptxPath("envmap.cu") );
        if(modes[i] == 0){
            Program miss_program = context->createProgramFromPTXFile(miss_path, "envmap_miss");
            context->setMissProgram(0, miss_program);
        }
        else{
            Program miss_program = context->createProgramFromPTXFile(miss_path, "miss");
            context->setMissProgram(0, miss_program);
        }

        std::cout << "Create Geometry" << std::endl;
        createGeometry(context, shapes, materials, modes[i]);

        // Camera
        if(cameraInputs[camIds[i]].cameraType == std::string("perspective") ){
            context["cameraMode"] -> setInt(0);
        }
        else if(cameraInputs[camIds[i]].cameraType == std::string("envmap") ){
            context["cameraMode"] -> setInt(1);
        }
        else if(cameraInputs[camIds[i]].cameraType == std::string("hemisphere") ){
            context["cameraMode"] -> setInt(2);
        }
        else if(cameraInputs[camIds[i]].cameraType == std::string("orthographic") ){
            context["cameraMode"] -> setInt(3);
            context["fov"] -> setFloat(cameraInputs[camIds[i]].fov);
        }
        else{ 
            std::cout<<"Wrong: unrecognizable camera type!"<<std::endl;
            exit(1);
        }
        std::cout << "Create Camera" << std::endl;
        createCamera(context, cameraInputs[camIds[i]]);

        // bool isFindFlash = false;
        // for(int j = 0; j < points.size(); j++){
        //     if(points[j].isFlash == true){
        //         isFindFlash = true;
        //         points[j].position.x = cameraInput.origin[0];
        //         points[j].position.y = cameraInput.origin[1];
        //         points[j].position.z = cameraInput.origin[2];
        //     }
        // }
        // if(isFindFlash ){
        //     updatePointLight(context, points);            
        // }
        std::cout << "Create Light" << std::endl;
        createPointLight(context, points);
        createAreaLightsBuffer(context, shapes);

        if (modes[i] == 0) {
            createEnvmap(context, envmaps[lightIds[i]]); 
        } 
        else {
            context["isEnvmap"] -> setInt(0);
            // Create the texture sampler 
            cv::Mat emptyMat = cv::Mat::zeros(1, 1, CV_32FC3);
            createEnvmapBuffer(context, emptyMat, emptyMat, 1, 1);
        }

        context->validate();
        clock_t t;
        t = clock();
        
        if(intensityLimitEnabled == true && modes[i] == 0){
            independentSampling(context, renderWidth, renderHeight, imgData, 4);
            float meanIntensity = 0;
            int pixelNum = renderWidth * renderHeight * 3;
            for(int i = 0; i < pixelNum; i++){
                meanIntensity += imgData[i];
            }
            meanIntensity /= pixelNum;
            std::cout<<"Mean Intensity of image: "<<meanIntensity<<std::endl;
            if(meanIntensity < 0.05){
                std::cout<<"Warning: the image is too dark, will be skipped"<<std::endl;
                continue;
            }
        }

        std::cout<<"Start to render: "<<i+1<<"/"<<renderNum<<std::endl;
        if(sampleType == std::string("independent") || modes[i] != 0){
            independentSampling(context, renderWidth, renderHeight, imgData, sampleNum);
        }
        else if(sampleType == std::string("adaptive") ) {            
            if(maxIteration >= 1){
                cameraInputs[i].adaptiveSampler.maxIteration =  maxIteration; 
            }

            bool isTooNoisy = adaptiveSampling(context, renderWidth, renderHeight, sampleNum, imgData, 
                    noiseLimit, noiseLimitEnabled, 
                    cameraInputs[i].adaptiveSampler.maxIteration, 
                    cameraInputs[i].adaptiveSampler.noiseThreshold
                    );
            std::cout<<"Sample Num: "<<sampleNum<<std::endl;
            if(isTooNoisy){
                std::cout<<"This image will not be output!"<<std::endl;
                continue;
            }
        }
        t = clock() - t;
        std::cout<<"Time: "<<float(t) / CLOCKS_PER_SEC<<'s'<<std::endl;
        
        if(isMedianFilter && modes[i] == 0 ){
            medianFilter(imgData, renderWidth, renderHeight, 1);
        }
         
        bool isWrite = writeBufferToFile(
            outputFileName, 
            imgData, renderWidth, renderHeight, modes[i]
        );
        destroyContext(context );
    }
    delete [] imgData;

    return 0;
}
