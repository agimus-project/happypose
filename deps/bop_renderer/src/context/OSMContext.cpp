#include "Context.h"
#include <vector>

#include "GL/osmesa.h"

struct OSMUserData {
    OSMesaContext ctx;
    int width, height;
    unsigned char *buffer;

    OSMUserData(): ctx(nullptr), width(0), height(0), buffer(nullptr) {}
};


#define DATA (*reinterpret_cast<OSMUserData*>(userData))


Context::Context(): userData(nullptr) {}


Context::~Context() {
    if (userData != nullptr)
        delete reinterpret_cast<OSMUserData*>(userData);
}


bool Context::init(int width, int height) {
    struct Attribute {
        int key, value;
        Attribute(int k, int v):key(k),value(v) {}
    };

    this->userData = new OSMUserData;

    std::vector<Attribute> attribs;

    // General settings
    attribs.push_back(Attribute(OSMESA_FORMAT, OSMESA_RGBA));
    attribs.push_back(Attribute(OSMESA_DEPTH_BITS, 24));
    attribs.push_back(Attribute(OSMESA_STENCIL_BITS, 0));
    attribs.push_back(Attribute(OSMESA_ACCUM_BITS, 0));
    // Version settings
    attribs.push_back(Attribute(OSMESA_PROFILE, OSMESA_CORE_PROFILE));
    attribs.push_back(Attribute(OSMESA_CONTEXT_MAJOR_VERSION, 3));
    attribs.push_back(Attribute(OSMESA_CONTEXT_MINOR_VERSION, 3));
    // End delimiter
    attribs.push_back(Attribute(0, 0));

    DATA.ctx = OSMesaCreateContextAttribs(reinterpret_cast<int*>(attribs.data()), nullptr);

    if (DATA.ctx == nullptr) {
        errorString += "Context creation failed!\n";
        return false;
    }

    // Allocate memory - we will not need it but whatever
    DATA.width = width;
    DATA.height = height;
    DATA.buffer = new unsigned char[width*height*4];

    return true;
}


void Context::makeCurrent() {
    OSMesaMakeCurrent(DATA.ctx, DATA.buffer, GL_UNSIGNED_BYTE, DATA.width, DATA.height);
}


void Context::deinit() {
    OSMesaDestroyContext(DATA.ctx);
    if (DATA.buffer != nullptr) {
        delete [] DATA.buffer;
        DATA.buffer = nullptr;
    }
}


CtxProcAddress Context::getProcAddress(const char *name) {
    return OSMesaGetProcAddress(name);
}


const std::string &Context::getError() {
    return errorString;
}
