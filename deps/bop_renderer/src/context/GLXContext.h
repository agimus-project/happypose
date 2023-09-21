#pragma once
#include "AbstractContext.h"

class GlxContext: public AbstractContext {
public:
    bool initContext(int width, int height, int oglMinor, int oglMajor);
    void releaseContext();
    
    void makeCurrent();
};
