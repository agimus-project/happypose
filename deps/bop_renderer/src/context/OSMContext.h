#pragma once

#include <string>

class Context {
public:
    bool init(int width, int height);
    void makeCurrent();
    void deinit();

    const std::string &getError();
private:
    std::string errorString;
};
