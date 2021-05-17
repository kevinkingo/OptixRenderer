#include "creator/createContext.h"

void destroyContext(Context& context )
{
    if( context ){
        context->destroy();
        context = 0;
    }
}

void createContext( 
        Context& context, 
        bool use_pbo, 
        unsigned width, unsigned height, 
        unsigned rr_begin_depth)
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 600 );

    context["rr_begin_depth"] -> setUint(rr_begin_depth);
    Buffer outputBuffer = context -> createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, width, height);
    context["output_buffer"]->set( outputBuffer );

    // Ray generation program 
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 0.0f, 0.0f, 0.0f );


    // Set light sampling program 
    const std::string area_path = ptxPath("areaLight.cu");
    Program sampleAreaLight = context -> createProgramFromPTXFile(area_path, "sampleAreaLight");
    context["sampleAreaLight"]->set(sampleAreaLight );

    const std::string env_path = ptxPath("envmap.cu");
    Program sampleEnvLight = context -> createProgramFromPTXFile(env_path, "sampleEnvironmapLight");
    context["sampleEnvironmapLight"]->set(sampleEnvLight );
        
}
