#pragma once

#include <string>

typedef void (*CtxProcAddress)();

class Context {
public:
    Context();
    ~Context();

    bool init(int width, int height);
    void makeCurrent();
    void deinit();

    static CtxProcAddress getProcAddress(const char *name);

    const std::string &getError();
private:
    std::string errorString;

    void *userData;
};
