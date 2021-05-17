#ifndef CREATECONTEXT_HEADER
#define CREATECONTEXT_HEADER

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <string>
#include "utils/ptxPath.h"

using namespace optix;


void destroyContext(Context& context );

void createContext( 
        Context& context, 
        bool use_pbo, 
        unsigned width, unsigned height, 
        unsigned rr_begin_depth = 3);

#endif
